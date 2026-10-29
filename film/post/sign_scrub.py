"""Scrub real signage from PLATEAU's facade photos before the city uses them (10/28).

The film shows no real brands or shop names, but the photo textures carry them. Rather than
dropping the photos (the user's call, 10/28: keep the real textures and change only what is a
problem), every texture is read with OCR (EasyOCR, Japanese and English) and each region with
text is blurred hard, padded by a margin. The originals are moved to `<folder>/_original/` first
(nothing is deleted), and a JSON report lists every file and box changed.

Logos without lettering are not found this way. At PLATEAU's aerial-photo resolution (a pixel
is several centimetres to tens of centimetres) they are rarely legible; a building that still
shows one is fixed by hand.

    python sign_scrub.py --root <extracted bldg/lod2> --report scrub.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True, help="a folder of PLATEAU *_obj folders with materials_textures/")
    p.add_argument("--report", required=True)
    p.add_argument("--min-conf", type=float, default=0.25)
    p.add_argument("--pad", type=float, default=0.25, help="margin round each text box, as a fraction of its height")
    args = p.parse_args()

    import cv2
    import easyocr
    import numpy as np

    reader = easyocr.Reader(["ja", "en"], gpu=True, verbose=False)
    files = sorted(Path(args.root).glob("*/materials_textures/*.jpg"))
    report, changed = [], 0
    for k, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None or min(img.shape[:2]) < 24:
            continue
        hits = [(box, text, conf) for box, text, conf in reader.readtext(img) if conf >= args.min_conf and len(text.strip()) >= 2]
        if not hits:
            continue
        backup = f.parent / "_original" / f.name
        backup.parent.mkdir(exist_ok=True)
        if not backup.exists():
            shutil.copy2(f, backup)
        boxes = []
        for box, text, conf in hits:
            pts = np.array(box, dtype=np.float32)
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            m = (y1 - y0) * args.pad
            x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
            x1, y1 = int(min(img.shape[1], x1 + m)), int(min(img.shape[0], y1 + m))
            region = img[y0:y1, x0:x1]
            if region.size == 0:
                continue
            kx = max(3, (x1 - x0) // 3 * 2 + 1)
            ky = max(3, (y1 - y0) // 3 * 2 + 1)
            img[y0:y1, x0:x1] = cv2.GaussianBlur(region, (kx | 1, ky | 1), 0)
            boxes.append({"box": [x0, y0, x1, y1], "text": text, "conf": round(float(conf), 3)})
        cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        report.append({"file": str(f.relative_to(args.root)), "boxes": boxes})
        changed += 1
        if k % 500 == 0:
            print(f"[scrub] {k}/{len(files)} read, {changed} changed", flush=True)
    Path(args.report).write_text(json.dumps({"files_read": len(files), "files_changed": changed, "changes": report},
                                            ensure_ascii=False, indent=1))
    print(f"SCRUB_DONE {changed} of {len(files)} textures had text blurred", flush=True)


if __name__ == "__main__":
    main()
