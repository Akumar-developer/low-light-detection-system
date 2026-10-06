# src/convert_exdark.py
import shutil, cv2
from pathlib import Path
from collections import Counter

IMG_ROOT = Path("data/ExDark")
ANNO_ROOT = Path("data/ExDark_Anno")
OUT = Path("data/exdark_yolo")
CLASSES = ["Bicycle","Boat","Bottle","Bus","Car","Cat",
           "Chair","Cup","Dog","Motorbike","People","Table"]
CID = {c.lower(): i for i, c in enumerate(CLASSES)}
SPLIT = {"1": "train", "2": "val", "3": "test"}

imgs = {p.name: p for p in IMG_ROOT.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
annos = {p.name: p for p in ANNO_ROOT.rglob("*.txt")}
print(f"images indexed: {len(imgs)}, annotation files: {len(annos)}")

for s in SPLIT.values():
    (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)

rows = Path("data/Groundtruth/imageclasslist.txt").read_text().splitlines()[1:]
stats = Counter()
for row in rows:
    parts = row.split()
    if len(parts) < 5:
        continue
    name, split = parts[0], SPLIT[parts[-1]]
    img_path, anno = imgs.get(name), annos.get(name + ".txt")
    if img_path is None or anno is None:
        stats["skipped_missing"] += 1
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        stats["skipped_unreadable"] += 1
        continue
    h, w = img.shape[:2]

    out_lines = []
    for ln in anno.read_text().splitlines():
        if ln.startswith("%") or not ln.strip():
            continue
        f = ln.split()
        cls = CID.get(f[0].lower())
        if cls is None:
            stats["unknown_class"] += 1
            continue
        l, t, bw, bh = map(float, f[1:5])
        x1, y1 = max(l, 0), max(t, 0)
        x2, y2 = min(l + bw, w), min(t + bh, h)
        if x2 <= x1 or y2 <= y1:
            stats["degenerate_box"] += 1
            continue
        out_lines.append(f"{cls} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                         f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")

    shutil.copy(img_path, OUT / "images" / split / name)
    (OUT / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(out_lines))
    stats[split] += 1

print(dict(stats))

Path(OUT / "data.yaml").write_text(
    f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
    f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))