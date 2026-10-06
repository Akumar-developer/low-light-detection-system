import argparse, shutil, time
import cv2, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

SRC = Path("data/exdark_yolo")
MAX_SIDE = 1280   # some ExDark images are huge; YOLO resizes to 640 anyway
JPG_Q = [cv2.IMWRITE_JPEG_QUALITY, 95]


def cap(img):
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def clahe(img, clip=3.0, grid=8):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def denoise(img):
    # applied AFTER enhancement: brightening amplifies sensor noise
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 15)


def _classical(job):
    src, dst, method = job
    img = cv2.imread(str(src))
    if img is None:
        return 0
    img = cap(img)
    out = clahe(img)
    if method == "clahe_dn":
        out = denoise(out)
    cv2.imwrite(str(dst), out, JPG_Q)
    return 1


def mirror_dirs(method):
    dst = Path(f"data/exdark_{method}")
    for s in ("train", "val", "test"):
        (dst / "images" / s).mkdir(parents=True, exist_ok=True)
        shutil.copytree(SRC / "labels" / s, dst / "labels" / s, dirs_exist_ok=True)
    y = (SRC / "data.yaml").read_text().replace(str(SRC.resolve()), str(dst.resolve()))
    (dst / "data.yaml").write_text(y)
    return dst


def jobs_for(dst):
    for s in ("train", "val", "test"):
        for p in (SRC / "images" / s).iterdir():
            yield p, dst / "images" / s / p.name


def run_zerodce(dst, weights, device):
    import torch, torch.nn as nn, torch.nn.functional as F

    class DCENet(nn.Module):
        def __init__(self, n=32):
            super().__init__()
            c = lambda i, o: nn.Conv2d(i, o, 3, 1, 1, bias=True)
            self.e_conv1, self.e_conv2 = c(3, n), c(n, n)
            self.e_conv3, self.e_conv4 = c(n, n), c(n, n)
            self.e_conv5, self.e_conv6 = c(2 * n, n), c(2 * n, n)
            self.e_conv7 = c(2 * n, 24)

        def forward(self, x):
            x1 = F.relu(self.e_conv1(x))
            x2 = F.relu(self.e_conv2(x1))
            x3 = F.relu(self.e_conv3(x2))
            x4 = F.relu(self.e_conv4(x3))
            x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
            x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
            r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
            for ri in torch.split(r, 3, dim=1):      # 8 curve iterations
                x = x + ri * (x * x - x)
            return x

    dev = torch.device(device)
    net = DCENet().to(dev)
    net.load_state_dict(torch.load(weights, map_location=dev))
    net.eval()
    for src, out in tqdm(list(jobs_for(dst))):
        img = cv2.imread(str(src))
        if img is None:
            continue
        img = cap(img)
        t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().div(255)
        t = t.permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            y = net(t).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        res = cv2.cvtColor((y * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out), res, JPG_Q)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["clahe", "clahe_dn", "zerodce"])
    ap.add_argument("--weights", default="weights/zerodce.pth")
    ap.add_argument("--device", default="cpu")   # cuda on Colab, mps on Mac
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    dst = mirror_dirs(a.method)
    t0 = time.time()
    if a.method == "zerodce":
        run_zerodce(dst, a.weights, a.device)
    else:
        jobs = [(s, d, a.method) for s, d in jobs_for(dst)]
        with ProcessPoolExecutor(a.workers) as ex:
            list(tqdm(ex.map(_classical, jobs, chunksize=16), total=len(jobs)))
    print(f"{a.method}: done in {time.time() - t0:.0f}s -> {dst}")