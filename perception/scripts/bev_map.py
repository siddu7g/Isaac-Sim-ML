#!/usr/bin/env python3
import os, re, json, argparse, math
from glob import glob
import numpy as np
from PIL import Image
import cv2

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def idx_from_any_name(path: str):
    m = re.findall(r"(\d+)", os.path.basename(path))
    return int(m[-1]) if m else None

def rot2d_yaw_deg(yaw_deg: float):
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s],[s, c]], dtype=np.float32)

def read_semantic_ids_png(png_path: str):
    """
    Convert RGBA semantic PNG to integer IDs:
      0 = other/background
      1 = floor
      2 = nvidia_box
    """
    img = np.array(Image.open(png_path))

    # Ensure RGBA
    if img.ndim == 2:
        # already single-channel; treat as IDs
        return img.astype(np.int32)
    if img.shape[2] == 3:
        # add opaque alpha if needed
        alpha = 255 * np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=2)

    # Your RGBA mapping
    FLOOR = np.array([93, 220, 11, 255], dtype=np.uint8)
    BOX   = np.array([243, 69, 141, 255], dtype=np.uint8)

    ids = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)

    floor_mask = np.all(img == FLOOR, axis=2)
    box_mask   = np.all(img == BOX, axis=2)

    ids[floor_mask] = 1
    ids[box_mask]   = 2
    return ids


def build_frame_index(input_dir: str):
    dep = glob(os.path.join(input_dir, "distance_to_camera_*.npy"))
    sem = glob(os.path.join(input_dir, "semantic_segmentation_*.png"))
    meta = glob(os.path.join(input_dir, "meta", "frame_*.json"))
    labj = glob(os.path.join(input_dir, "semantic_segmentation_labels_*.json"))

    if not dep:  raise FileNotFoundError("No distance_to_camera_*.npy found.")
    if not sem:  raise FileNotFoundError("No semantic_segmentation_*.png found.")
    if not meta: raise FileNotFoundError("No meta/frame_*.json found (pose required).")
    if not labj: raise FileNotFoundError("No semantic_segmentation_labels_*.json found (label mapping required).")

    dep_map  = {idx_from_any_name(p): p for p in dep if idx_from_any_name(p) is not None}
    sem_map  = {idx_from_any_name(p): p for p in sem if idx_from_any_name(p) is not None}
    meta_map = {idx_from_any_name(p): p for p in meta if idx_from_any_name(p) is not None}

    common = sorted(set(dep_map) & set(sem_map) & set(meta_map))
    if not common:
        raise RuntimeError("Could not match depth/semantic/meta by frame index.")

    # Use first labels json for mapping
    labels_json = sorted(labj, key=lambda p: idx_from_any_name(p) or 0)[0]
    return [(i, dep_map[i], sem_map[i], meta_map[i]) for i in common], labels_json

def parse_label_ids(labels_json_path: str):
    """
    Tries to parse replicator semantic_segmentation_labels_*.json.
    The exact schema can vary by version; we handle common patterns.
    Returns dict: {label_name: id_int}
    """
    j = load_json(labels_json_path)

    # Common patterns
    out = {}

    if isinstance(j, dict):
        if "mapping" in j and isinstance(j["mapping"], dict):
            for k, v in j["mapping"].items():
                try: out[str(k)] = int(v)
                except: pass
            if out: return out

        for key in ("labels", "classes", "classLabels", "semanticLabels"):
            if key in j and isinstance(j[key], list):
                for item in j[key]:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("class") or item.get("name") or item.get("label")
                    idx  = item.get("id") or item.get("index") or item.get("value")
                    if name is None or idx is None:
                        continue
                    try:
                        out[str(name)] = int(idx)
                    except:
                        continue
                if out:
                    return out

    # Fallback: try to find any dict entries with string->int
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and isinstance(v, int):
                    out[k] = v
                else:
                    walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    walk(j)
    return out

def rays_from_pixels(K, H, W, stride):
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])

    vs = np.arange(0, H, stride, dtype=np.int32)
    us = np.arange(0, W, stride, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)

    uu_f = uu.astype(np.float32).reshape(-1)
    vv_f = vv.astype(np.float32).reshape(-1)

    x = (uu_f - cx) / fx
    y = (vv_f - cy) / fy
    z = np.ones_like(x, dtype=np.float32)

    rays = np.stack([x, y, z], axis=1)
    rays /= (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-9)

    uv = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1).astype(np.int32)
    return uv, rays.astype(np.float32)

def project_frame_to_ground(dist_cam, sem_ids, uv, ray_cam_unit, ned, yaw_deg,
                            max_dist=60.0, cam_x_is_east=True, cam_y_is_north=False,
                            keep_ids=None):
    if dist_cam.ndim == 3:
        dist_cam = dist_cam[:, :, 0]

    u = uv[:,0]; v = uv[:,1]
    d = dist_cam[v, u].astype(np.float32)
    valid = np.isfinite(d) & (d > 0.2) & (d < max_dist)

    if not np.any(valid):
        return None

    u = u[valid]; v = v[valid]
    rays = ray_cam_unit[valid]
    labels = sem_ids[v, u].astype(np.int32)

    if keep_ids is not None:
        keep_mask = np.isin(labels, keep_ids)
        if not np.any(keep_mask):
            return None
        rays = rays[keep_mask]
        labels = labels[keep_mask]

    xh = rays[:,0]; yh = rays[:,1]; zh = rays[:,2]
    good = zh > 1e-3
    if not np.any(good):
        return None
    xh = xh[good]; yh = yh[good]; zh = zh[good]; labels = labels[good]

    pre_E = xh if cam_x_is_east else -xh
    pre_N = yh if cam_y_is_north else -yh

    R = rot2d_yaw_deg(float(yaw_deg))
    NE = (R @ np.stack([pre_N, pre_E], axis=0)).astype(np.float32)
    dir_N, dir_E = NE[0,:], NE[1,:]

    n0 = float(ned["n"]); e0 = float(ned["e"]); d0 = float(ned["d"])

    t = (0.0 - d0) / zh
    good2 = t > 0.0
    if not np.any(good2):
        return None

    t = t[good2]; labels = labels[good2]
    dir_N = dir_N[good2]; dir_E = dir_E[good2]

    N = n0 + t * dir_N
    E = e0 + t * dir_E
    return N.astype(np.float32), E.astype(np.float32), labels.astype(np.int32)

def compute_bounds(samples, pad):
    allN = np.concatenate([s[0] for s in samples])
    allE = np.concatenate([s[1] for s in samples])
    return (float(allN.min()-pad), float(allN.max()+pad), float(allE.min()-pad), float(allE.max()+pad))

def rasterize_two_label(samples, bounds, res_m, floor_id, box_id, remap=True):
    nmin, nmax, emin, emax = bounds
    H = int(math.ceil((nmax - nmin) / res_m))
    W = int(math.ceil((emax - emin) / res_m))

    occ = np.zeros((H, W), dtype=np.uint32)
    sem = np.zeros((H, W), dtype=np.uint8)  # only 0/1/2 if remap

    # simple priority: if any box hits cell -> box; else if any floor hits -> floor
    # (box overwrites floor)
    for N, E, L in samples:
        ix = ((E - emin) / res_m).astype(np.int32)
        iy = ((N - nmin) / res_m).astype(np.int32)
        inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ix = ix[inside]; iy = iy[inside]; L = L[inside]

        np.add.at(occ, (iy, ix), 1)

        if remap:
            # floor -> 1, box -> 2
            is_floor = (L == floor_id)
            is_box = (L == box_id)

            sem[iy[is_floor], ix[is_floor]] = 1
            sem[iy[is_box], ix[is_box]] = 2
        else:
            # keep original IDs modulo 256 in uint8 (only works if IDs < 256)
            sem[iy, ix] = np.clip(L, 0, 255).astype(np.uint8)

    return occ, sem


def colorize_bev_semantic(sem_u8):
    """Return RGB visualization for semantic BEV: 0=other, 1=floor, 2=box."""
    rgb = np.zeros((sem_u8.shape[0], sem_u8.shape[1], 3), dtype=np.uint8)
    rgb[sem_u8 == 1] = np.array([93, 220, 11], dtype=np.uint8)      # floor
    rgb[sem_u8 == 2] = np.array([243, 69, 141], dtype=np.uint8)     # nvidia_box
    return rgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--max_dist", type=float, default=60.0)
    ap.add_argument("--res", type=float, default=0.15)
    ap.add_argument("--padding", type=float, default=2.0)
    ap.add_argument("--cam_x_is_east", type=int, default=1)
    ap.add_argument("--cam_y_is_north", type=int, default=0)
    ap.add_argument("--keep_labels", type=str, default="floor,nvidia_box")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    frames, labels_json = build_frame_index(args.input)
    if args.limit and args.limit > 0:
        frames = frames[:args.limit]

    label_map = parse_label_ids(labels_json)
    keep = [s.strip() for s in args.keep_labels.split(",") if s.strip()]
    if len(keep) != 2:
        raise ValueError("--keep_labels must be exactly two labels, e.g. floor,nvidia_box")

    floor_name, box_name = keep[0], keep[1]
    if floor_name not in label_map or box_name not in label_map:
        print("[ERROR] Could not find label IDs in labels json.")
        print("Available keys (sample):", list(label_map.keys())[:30])
        raise KeyError(f"Missing one of: {floor_name}, {box_name}")

    floor_id = int(label_map[floor_name])
    box_id = int(label_map[box_name])
    keep_ids = np.array([1, 2], dtype=np.int32)
    floor_id, box_id = 1, 2


    meta0 = load_json(frames[0][3])
    K = np.array(meta0["K"], dtype=np.float32)
    W, H = int(meta0["resolution"][0]), int(meta0["resolution"][1])

    uv, rays = rays_from_pixels(K, H, W, args.stride)

    print(f"[INFO] Frames: {len(frames)} stride={args.stride} samples/frame={len(uv)}")
    print(f"[INFO] BEV res={args.res} m/cell")
    print(f"[INFO] Keeping labels: {floor_name}={floor_id}, {box_name}={box_id}")

    samples = []
    used = 0

    for (idx, dep_path, sem_path, meta_path) in frames:
        meta = load_json(meta_path)
        ned = meta.get("ned")
        att = meta.get("att_euler_deg")
        yaw = att.get("yaw") if att else None
        if ned is None or yaw is None:
            continue

        sem_ids = read_semantic_ids_png(sem_path)
        dist = np.load(dep_path)

        proj = project_frame_to_ground(
            dist, sem_ids, uv, rays,
            ned=ned, yaw_deg=yaw,
            max_dist=args.max_dist,
            cam_x_is_east=bool(args.cam_x_is_east),
            cam_y_is_north=bool(args.cam_y_is_north),
            keep_ids=keep_ids
        )
        if proj is None:
            continue

        N, E, L = proj
        if len(N) < 50:
            continue

        samples.append((N, E, L))
        used += 1
        if used % 50 == 0:
            print(f"[OK] used frames: {used} (latest idx={idx})")

    if not samples:
        raise RuntimeError("No usable samples. Try different cam axis flags or remove keep_labels filter.")

    bounds = compute_bounds(samples, args.padding)
    occ, sem = rasterize_two_label(samples, bounds, args.res, floor_id, box_id, remap=True)

    ensure_dir(args.out)
    np.save(os.path.join(args.out, "bev_occupancy.npy"), occ.astype(np.uint32))
    np.save(os.path.join(args.out, "bev_semantic_two_class.npy"), sem.astype(np.uint8))

    occ_img = (np.clip(occ.astype(np.float32) / max(1.0, occ.max()), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(occ_img).save(os.path.join(args.out, "bev_occupancy.png"))
    Image.fromarray(sem).save(os.path.join(args.out, "bev_semantic_two_class.png"))

    # ---- Colored semantic visualization (inflate box for visibility) ----
    box_mask = (sem == 2).astype(np.uint8)
    if box_mask.any():
        box_mask = cv2.dilate(box_mask, np.ones((3, 3), np.uint8), iterations=1)
        sem_vis = sem.copy()
        sem_vis[box_mask.astype(bool)] = 2
    else:
        sem_vis = sem

    sem_rgb = colorize_bev_semantic(sem_vis)
    Image.fromarray(sem_rgb).save(os.path.join(args.out, "bev_semantic_color.png"))


    with open(os.path.join(args.out, "bev_meta.json"), "w") as f:
        json.dump({
            "bounds": {"nmin": bounds[0], "nmax": bounds[1], "emin": bounds[2], "emax": bounds[3]},
            "resolution_m_per_cell": args.res,
            "labels": {floor_name: 1, box_name: 2},
            "source_label_ids": {floor_name: floor_id, box_name: box_id},
        }, f, indent=2)

    print("[DONE] Two-class BEV written to:", args.out)
    print(" - bev_semantic_two_class.png (0=other, 1=floor, 2=nvidia_box)")
    print("If it looks mirrored/rotated, toggle:")
    print("  --cam_x_is_east 0/1   --cam_y_is_north 0/1")

if __name__ == "__main__":
    main()
