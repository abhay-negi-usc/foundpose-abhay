import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def is_int_stem(p: Path) -> bool:
    try:
        int(p.stem)
        return True
    except Exception:
        return False


def count_instances_from_mask(mask_path: Path) -> int:
    """
    Count connected components in a binary mask.
    - If your mask may contain holes/noise, consider erode/dilate before CC.
    """
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(mask_path)

    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0).astype(np.uint8)

    # Connected components: label 0 is background
    num_labels, _ = cv2.connectedComponents(m, connectivity=8)
    inst_count = max(0, num_labels - 1)
    return inst_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-dir", required=True, type=str)
    ap.add_argument("--mask-dir", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--scene-id", type=int, default=0)
    ap.add_argument("--obj-id", type=int, default=1)
    ap.add_argument("--ext", type=str, default=".png", help="image extension to scan (e.g., .png, .jpg)")
    args = ap.parse_args()

    rgb_dir = Path(args.rgb_dir)
    mask_dir = Path(args.mask_dir)
    out_path = Path(args.out)

    rgb_paths = sorted(rgb_dir.glob(f"*{args.ext}"))
    if not rgb_paths:
        raise RuntimeError(f"No images found in {rgb_dir} with ext {args.ext}")

    # Decide image_id mapping:
    # - If stems are integers (000123), use that directly (BOP-style).
    # - Else, enumerate in sorted order.
    use_stem_as_id = all(is_int_stem(p) for p in rgb_paths)

    targets = []
    for idx, rgb_p in enumerate(rgb_paths):
        im_id = int(rgb_p.stem) if use_stem_as_id else idx

        # mask must match by stem (allow any common mask ext)
        # try: same extension first, else fall back to png/jpg
        cand = [
            mask_dir / f"{rgb_p.stem}{rgb_p.suffix}",
            mask_dir / f"{rgb_p.stem}.png",
            mask_dir / f"{rgb_p.stem}.jpg",
            mask_dir / f"{rgb_p.stem}.jpeg",
        ]
        mask_p = next((c for c in cand if c.exists()), None)
        if mask_p is None:
            raise FileNotFoundError(f"No mask found for {rgb_p.name} (looked for {', '.join(str(c) for c in cand)})")

        inst_count = count_instances_from_mask(mask_p)
        if inst_count == 0:
            # If your masks are always single-instance, you might prefer forcing to 1.
            # But keeping 0 can help catch misaligned/missing masks.
            print(f"[WARN] mask has 0 foreground components: {mask_p}")

        targets.append(
            {
                "scene_id": args.scene_id,
                "im_id": im_id,
                "obj_id": args.obj_id,
                "inst_count": int(inst_count),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(targets, f, indent=2)

    print(f"Wrote {len(targets)} targets to: {out_path}")
    if not use_stem_as_id:
        print("[NOTE] Filenames were not integer-stemmed; im_id was assigned by sorted order (0..N-1).")


if __name__ == "__main__":
    main()

# example usage 
"""
python scripts/make_test_targets_from_masks.py \ 
---rgb-dir /home/omey/abhay_ws/bop_datasets/templates/v1/BNC_hole/1/rgb/ \ 
--mask-dir /home/omey/abhay_ws/bop_datasets/templates/v1/BNC_hole/1/mask/ \
--out /home/omey/abhay_ws/bop_datasets/templates/v1/BNC_hole/1/rgb/

"""