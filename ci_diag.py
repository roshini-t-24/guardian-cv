import glob
import hashlib

import cv2

for f in sorted(glob.glob("videos/clip*_*.mp4"))[:3]:
    raw = hashlib.sha256(open(f, "rb").read()).hexdigest()[:12]
    cap = cv2.VideoCapture(f)
    h = hashlib.sha256()
    for _ in range(30):
        ok, fr = cap.read()
        if not ok:
            break
        h.update(fr.tobytes())
    print(f"{f}  file={raw}  decoded30={h.hexdigest()[:12]}")