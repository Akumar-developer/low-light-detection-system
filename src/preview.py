# src/preview.py
import cv2, numpy as np, random
from pathlib import Path
names = [p.name for p in random.sample(list(Path("data/exdark_yolo/images/test").iterdir()), 4)]
rows = []
for n in names:
    ims = [cv2.imread(f"data/{d}/images/test/{n}") for d in
           ("exdark_yolo", "exdark_clahe", "exdark_clahe_dn", "exdark_zerodce")]
    ims = [cv2.resize(i, (320, 240)) for i in ims]
    print(n, [round(float(i.mean()), 1) for i in ims])   # mean brightness per variant
    rows.append(np.hstack(ims))
cv2.imwrite("preview.jpg", np.vstack(rows))