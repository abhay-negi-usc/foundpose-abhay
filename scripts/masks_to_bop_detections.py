#!/usr/bin/env python3
"""
Convert per-image binary masks into a detections JSON that FoundPose can read,
AND write dummy depth images (all zeros) aligned to each RGB.

Assumptions:
- images_dir has RGB images
- masks_dir has binary masks with same basename as RGB
- Masks are 0/255 or 0/1 (non-zero treated as foreground)
- We assign scene_id=0 and image_id=enumeration index over sorted RGB files
  (unless you change mapping)

Outputs:
1) detections JSON:
   <BOP_PATH>/detections/cnos-fastsam/cnos-fastsam_<dataset_name>-test.json

2) dummy depth PNGs (uint16, all zeros):
   <BOP_PATH>/<dataset_name>/test/000000/depth/<image_id:06d>.png
"""

import os
import json
from pathlib import Path

import numpy as np
import cv2


# ----------------------------
# CONFIG (edit these)
# ----------------------------
CFG = {
    "dataset_name": "BNC_hole",
    "obj_id": 1,

    "images_dir": "/home/omey/abhay_ws/bop_datasets/BNC_hole/test/000000/rgb/",
    "masks_dir":  "/home/omey/abhay_ws/bop_datasets/BNC_hole/test/000000/mask/",

    "image_exts": [".png", ".jpg", ".jpeg"],

    "min_fg_pixels": 50,
    "default_scene_id": 0,

    # BOP root
    "bop_path": os.environ.get("BOP_PATH", ""),

    # NEW: write dummy depth images?
    "write_dummy_depth": True,

    # NEW: where to write dummy depths (BOP-style)
    # This matches FoundPose's typical BOP split layout.
    "depth_out_dir": None,  # if None, will use <BOP_PATH>/<dataset>/test/000000/depth

    # NEW: depth image format
    # BOP commonly stores depth as uint16 PNG (e.g., in mm). We'll write zeros in uint16.
    "depth_dtype": "uint16",  # "uint16" recommended
}


def coco_rle_encode(binary_mask_hw: np.ndarray) -> dict:
    """
    COCO RLE encode a binary mask.

    Returns dict: {"size":[H,W], "counts": <ascii str> OR list}
    """
    try:
        from pycocotools import mask as mask_utils  # type: ignore

        m = np.asfortranarray(binary_mask_hw.astype(np.uint8))
        rle = mask_utils.encode(m)
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}

    except Exception:
        # Pure-python fallback: uncompressed counts list in Fortran order.
        m = np.asfortranarray(binary_mask_hw.astype(np.uint8))
        h, w = m.shape
        pixels = m.reshape(-1, order="F")
        counts = []
        prev = 0
        run_len = 0
        for p in pixels:
            if p != prev:
                counts.append(run_len)
                run_len = 0
                prev = int(p)
            run_len += 1
        counts.append(run_len)
        return {"size": [h, w], "counts": counts}


def mask_to_bbox_xywh(mask_hw: np.ndarray):
    ys, xs = np.nonzero(mask_hw)
    if len(xs) == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    w = int(x1 - x0 + 1)
    h = int(y1 - y0 + 1)
    return [x0, y0, w, h]


def find_mask_for_image(mask_dir: Path, stem: str) -> Path | None:
    for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def write_dummy_depth(depth_dir: Path, image_id: int, height: int, width: int, dtype: str = "uint16"):
    depth_dir.mkdir(parents=True, exist_ok=True)
    if dtype == "uint16":
        depth = np.zeros((height, width), dtype=np.uint16)
    elif dtype == "float32":
        depth = np.zeros((height, width), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported depth_dtype: {dtype}")

    out_path = depth_dir / f"{image_id:06d}.png"

    # For float32 we still write as 16-bit PNG zeros unless you really need EXR.
    # Best is uint16.
    if depth.dtype == np.float32:
        depth_to_write = depth.astype(np.uint16)
    else:
        depth_to_write = depth

    ok = cv2.imwrite(str(out_path), depth_to_write)
    if not ok:
        raise RuntimeError(f"Failed to write depth png: {out_path}")


def main():
    images_dir = Path(CFG["images_dir"])
    masks_dir = Path(CFG["masks_dir"])
    bop_path = Path(CFG["bop_path"]) if CFG["bop_path"] else None

    assert images_dir.exists(), f"images_dir not found: {images_dir}"
    assert masks_dir.exists(), f"masks_dir not found: {masks_dir}"
    assert bop_path is not None and bop_path.exists(), (
        "Set CFG['bop_path'] or export BOP_PATH to your BOP datasets root."
    )

    # Where to write dummy depth images
    if CFG["depth_out_dir"] is None:
        depth_out_dir = bop_path / CFG["dataset_name"] / "test" / f"{CFG['default_scene_id']:06d}" / "depth"
    else:
        depth_out_dir = Path(CFG["depth_out_dir"])

    # Collect image files
    img_paths = []
    for ext in CFG["image_exts"]:
        img_paths += list(images_dir.glob(f"*{ext}"))
    img_paths = sorted(img_paths)

    detections = []
    scene_id = int(CFG["default_scene_id"])
    obj_id = int(CFG["obj_id"])

    kept = 0
    skipped_missing_mask = 0
    skipped_empty = 0

    for image_id, img_path in enumerate(img_paths):
        stem = img_path.stem

        mask_path = find_mask_for_image(masks_dir, stem)
        if mask_path is None:
            skipped_missing_mask += 1
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            skipped_missing_mask += 1
            continue

        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_bin = (mask > 0).astype(np.uint8)

        if int(mask_bin.sum()) < int(CFG["min_fg_pixels"]):
            skipped_empty += 1
            continue

        bbox = mask_to_bbox_xywh(mask_bin)
        if bbox is None:
            skipped_empty += 1
            continue

        rle = coco_rle_encode(mask_bin)

        det = {
            "scene_id": scene_id,
            "image_id": int(image_id),      # NOTE: infer_pose_util expects image_id
            "category_id": obj_id,          # NOTE: infer_pose_util expects category_id
            "score": 1.0,
            "time": 0.0,
            "bbox": bbox,                   # [x, y, w, h]
            "segmentation": rle,            # COCO RLE (modal mask)
        }
        detections.append(det)
        kept += 1

        # NEW: write dummy depth aligned to RGB dimensions
        if CFG["write_dummy_depth"]:
            rgb = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if rgb is None:
                raise RuntimeError(f"Failed to read RGB: {img_path}")
            h, w = rgb.shape[:2]
            write_dummy_depth(
                depth_dir=depth_out_dir,
                image_id=image_id,
                height=h,
                width=w,
                dtype=CFG["depth_dtype"],
            )

    # Write detections JSON
    out_dir = bop_path / "detections" / "cnos-fastsam"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cnos-fastsam_{CFG['dataset_name']}-test.json"

    with open(out_path, "w") as f:
        json.dump(detections, f)

    print(f"[OK] Wrote {len(detections)} detections to:\n  {out_path}")
    print(f"Kept: {kept}")
    print(f"Skipped (no mask): {skipped_missing_mask}")
    print(f"Skipped (empty/small): {skipped_empty}")

    if CFG["write_dummy_depth"]:
        print(f"[OK] Wrote dummy depth PNGs to:\n  {depth_out_dir}")


if __name__ == "__main__":
    main()
