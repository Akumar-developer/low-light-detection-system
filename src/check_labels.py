# src/check_labels.py
import cv2, random
from pathlib import Path
root = Path("data/exdark_yolo")
for p in random.sample(list((root/"images/train").glob("*")), 5):
    img = cv2.imread(str(p)); h, w = img.shape[:2]
    lab = root/"labels/train"/(p.stem + ".txt")
    for ln in lab.read_text().splitlines():
        c, xc, yc, bw, bh = map(float, ln.split())
        x1, y1 = int((xc-bw/2)*w), int((yc-bh/2)*h)
        x2, y2 = int((xc+bw/2)*w), int((yc+bh/2)*h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(f"check_{p.stem}.jpg", img)