#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


def load_ply_vertices(ply_path: Path) -> np.ndarray:
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    with ply_path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {ply_path}")
            s = line.decode("ascii", errors="ignore").strip()
            header_lines.append(s)
            if s == "end_header":
                break

        fmt = None
        num_verts = None
        prop_names = []
        prop_types = []
        in_vert = False

        for s in header_lines:
            if s.startswith("format "):
                fmt = s.split()[1]
            elif s.startswith("element vertex "):
                num_verts = int(s.split()[-1])
                in_vert = True
                prop_names = []
                prop_types = []
            elif s.startswith("element "):
                if in_vert and not s.startswith("element vertex "):
                    in_vert = False
            elif in_vert and s.startswith("property "):
                parts = s.split()
                if len(parts) >= 3:
                    prop_types.append(parts[1])
                    prop_names.append(parts[2])

        if fmt is None or num_verts is None:
            raise ValueError(f"Failed to parse PLY header (format/num_verts missing): {ply_path}")

        if "x" not in prop_names or "y" not in prop_names or "z" not in prop_names:
            raise ValueError(f"PLY vertex properties missing x/y/z: {ply_path} (props={prop_names})")

        ix, iy, iz = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")

        if fmt == "ascii":
            # reopen as text and skip header
            with ply_path.open("r", encoding="utf-8", errors="ignore") as tf:
                for _ in header_lines:
                    tf.readline()
                verts = np.empty((num_verts, 3), dtype=np.float64)
                for i in range(num_verts):
                    parts = tf.readline().strip().split()
                    if len(parts) < max(ix, iy, iz) + 1:
                        raise ValueError(f"Bad vertex line {i} in {ply_path}: {parts}")
                    verts[i, 0] = float(parts[ix])
                    verts[i, 1] = float(parts[iy])
                    verts[i, 2] = float(parts[iz])
            return verts

        if fmt != "binary_little_endian":
            raise ValueError(
                f"Unsupported PLY format '{fmt}' in {ply_path}. "
                "This script supports ascii and binary_little_endian."
            )

        ply2np = {
            "char": np.int8, "uchar": np.uint8,
            "int8": np.int8, "uint8": np.uint8,
            "short": np.int16, "ushort": np.uint16,
            "int16": np.int16, "uint16": np.uint16,
            "int": np.int32, "uint": np.uint32,
            "int32": np.int32, "uint32": np.uint32,
            "float": np.float32, "float32": np.float32,
            "double": np.float64, "float64": np.float64,
        }

        dtype_fields = []
        for t, n in zip(prop_types, prop_names):
            if t not in ply2np:
                raise ValueError(f"Unsupported PLY property type '{t}' in {ply_path}")
            dtype_fields.append((n, ply2np[t]))
        vert_dtype = np.dtype(dtype_fields)

        data = np.fromfile(f, dtype=vert_dtype, count=num_verts)
        verts = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
        return verts


def aabb_info(verts: np.ndarray) -> dict:
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    size = vmax - vmin
    return {
        "min_x": float(vmin[0]),
        "min_y": float(vmin[1]),
        "min_z": float(vmin[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }


def exact_max_pairwise_distance(points: np.ndarray, chunk: int = 2048) -> float:
    P = points.shape[0]
    max_d2 = 0.0
    for i in range(0, P, chunk):
        a = points[i:i+chunk]
        diff = a[:, None, :] - points[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        max_d2 = max(max_d2, float(np.max(d2)))
    return float(math.sqrt(max_d2))


def approx_diameter(verts: np.ndarray, num_samples: int = 5000, seed: int = 0) -> float:
    N = verts.shape[0]
    if N == 0:
        return 0.0
    if N <= num_samples:
        return exact_max_pairwise_distance(verts)

    rng = np.random.default_rng(seed)
    idx0 = int(rng.integers(0, N))
    chosen = np.empty((num_samples,), dtype=np.int64)
    chosen[0] = idx0

    d2 = np.sum((verts - verts[idx0]) ** 2, axis=1)
    for i in range(1, num_samples):
        idx = int(np.argmax(d2))
        chosen[i] = idx
        new_d2 = np.sum((verts - verts[idx]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)

    return exact_max_pairwise_distance(verts[chosen])


def infer_obj_id_from_filename(p: Path) -> int | None:
    m = re.match(r"obj_(\d+)\.ply$", p.name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=str, default="")
    ap.add_argument("--obj-id", type=int, default=1)
    ap.add_argument("--models-dir", type=str, default="")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--diam-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Will write to: {out_path.resolve()}")

    models_info = {}

    try:
        if args.models_dir:
            models_dir = Path(args.models_dir)
            ply_paths = sorted(models_dir.glob("obj_*.ply"))
            if not ply_paths:
                raise FileNotFoundError(f"No obj_*.ply found in {models_dir}")
            for ply_path in ply_paths:
                obj_id = infer_obj_id_from_filename(ply_path)
                if obj_id is None:
                    continue
                print(f"[INFO] Loading {ply_path} (obj_id={obj_id})")
                verts = load_ply_vertices(ply_path)
                info = aabb_info(verts)
                info["diameter"] = float(approx_diameter(verts, args.diam_samples, args.seed))
                models_info[str(obj_id)] = info

        elif args.ply:
            ply_path = Path(args.ply)
            print(f"[INFO] Loading {ply_path} (obj_id={args.obj_id})")
            verts = load_ply_vertices(ply_path)
            info = aabb_info(verts)
            info["diameter"] = float(approx_diameter(verts, args.diam_samples, args.seed))
            models_info[str(args.obj_id)] = info

        else:
            raise ValueError("Provide either --ply or --models-dir")

        with out_path.open("w") as f:
            json.dump(models_info, f, indent=2, sort_keys=True)

        print(f"[OK] Wrote models_info for {len(models_info)} object(s).")
        print(f"[OK] File exists now? {out_path.exists()}  size={out_path.stat().st_size if out_path.exists() else 'NA'} bytes")

    except Exception as e:
        print("[ERROR] Failed before writing output.")
        raise


if __name__ == "__main__":
    main()
