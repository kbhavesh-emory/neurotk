#!/usr/bin/env python3
import argparse, json, os, time
from pathlib import Path
from typing import Tuple

try:
    import pyvips  # type: ignore
    _HAS_PYVIPS = True
    # Set pyvips temp directory to location with more space
    pyvips.cache_set_max(0)  # Disable operation cache to save memory
    pyvips.cache_set_max_mem(0)  # Disable memory cache
    pyvips.cache_set_max_files(0)  # Disable file descriptor cache
    # Use /nashome for temp files (more space available)
    import tempfile
    os.environ['TMPDIR'] = '/nashome/bhavesh/neurotk/cli/output/tmp'
    os.makedirs('/nashome/bhavesh/neurotk/cli/output/tmp', exist_ok=True)
except Exception:
    _HAS_PYVIPS = False

# Try to avoid loading huge TIFFs with Pillow by using tifffile metadata
try:
    import tifffile  # type: ignore
    _HAS_TIFFILE = True
except Exception:
    _HAS_TIFFILE = False

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# If we must open with PIL, disable DecompressionBomb limit for this script
Image.MAX_IMAGE_PIXELS = None  # <-- key change to avoid DecompressionBombError

import girder_client


def norm_api(url: str) -> str:
    url = url.rstrip("/")
    return url if url.endswith("/api/v1") else url + "/api/v1"


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def gc_login(cfg):
    api = norm_api(cfg["global"]["dsa_server_url"])
    token = cfg["global"]["dsa_api_key"]
    gc = girder_client.GirderClient(apiUrl=api)
    gc.authenticate(apiKey=token)
    return gc


def wsi_dims(gc, item_id: str) -> Tuple[float, float]:
    md = gc.get(f"/item/{item_id}/tiles")
    return float(md["sizeX"]), float(md["sizeY"])


def ensure_overlay_item(gc, folder_id: str, name: str):
    for it in gc.listItem(folder_id, name=name):
        return it
    return gc.createItem(folder_id, name=name, description="PPC overlay source")


def upload_file_to_item(gc, item_id: str, local_path: str, replace: bool = True):
    def _reset_tiles():
        try:
            gc.delete(f"/item/{item_id}/tiles")
        except girder_client.HttpError:
            # No existing tiles to delete.
            pass

    bn = os.path.basename(local_path)
    for f in gc.listFile(item_id):
        if f["name"] == bn:
            if replace:
                _reset_tiles()
                gc.delete(f"/file/{f['_id']}")
                break
            else:
                return f
    if replace:
        _reset_tiles()
    return gc.uploadFileToItem(item_id, local_path)


def ensure_tiles(gc, item_id: str, timeout: int = 240):
    try:
        gc.get(f"/item/{item_id}/tiles")
        return
    except girder_client.HttpError:
        gc.post(f"/item/{item_id}/tiles")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            gc.get(f"/item/{item_id}/tiles")
            return
        except girder_client.HttpError:
            time.sleep(1)
    raise RuntimeError("Timed out waiting for tiles on overlay item")


def delete_ann_by_name(gc, item_id: str, name: str):
    for a in gc.get(f"/annotation?itemId={item_id}"):
        nm = a.get("annotation", {}).get("name") or a.get("name")
        if nm == name:
            gc.delete(f"/annotation/{a['_id']}")


def post_ann(gc, item_id: str, annot: dict):
    last = None
    for path in (f"/annotation?itemId={item_id}", f"/annotation/item/{item_id}"):
        try:
            return gc.post(path, json=annot)
        except girder_client.HttpError as e:
            last = e
    raise last


def find_ppc_image_element(ppc: dict):
    best = None

    def walk(o):
        nonlocal best
        if best is not None:
            return
        if isinstance(o, dict):
            if o.get("type") == "image":
                if o.get("girderId") == "outputLabelImage":
                    best = o
                    return
                if best is None:
                    best = o
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(ppc)
    return best


def get_tiff_size_fast(tiff_path: str) -> Tuple[int, int]:
    """
    Return (width, height) without decoding full image.
    Prefer tifffile metadata; fallback to Pillow with MAX_IMAGE_PIXELS disabled.
    """
    if _HAS_TIFFILE:
        with tifffile.TiffFile(tiff_path) as tf:
            page = tf.pages[0]
            # tifffile reports shape as (rows, cols[, samples]); rows=height, cols=width
            shp = getattr(page, "shape", None)
            if shp is not None:
                if len(shp) >= 2:
                    h, w = int(shp[0]), int(shp[1])
                    return (w, h)
            # Fallback to tags
            w = int(page.tags["ImageWidth"].value)
            h = int(page.tags["ImageLength"].value)
            return (w, h)
    # Fallback: Pillow (safe because MAX_IMAGE_PIXELS=None above)
    with Image.open(tiff_path) as im:
        return im.size  # (w, h)


def _parse_hex_color(s: str) -> Tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError("--tint must be a 6-digit hex color, e.g. #FF0000")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return r, g, b


def _tint_tiff_with_alpha(src_path: str, hex_color: str, alpha: float) -> str:
    if not _HAS_PYVIPS:
        raise RuntimeError("pyvips not installed; cannot tint TIFF")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("--tint-alpha must be between 0 and 1")

    r, g, b = _parse_hex_color(hex_color)
    src = pyvips.Image.new_from_file(src_path, access="sequential")

    # Convert to grayscale for mask
    if src.bands >= 3:
        gray = src.colourspace("b-w")
    else:
        gray = src

    # Build alpha mask: any non-zero pixel gets alpha
    mask = gray > 0
    alpha_band = mask * int(round(alpha * 255))

    # Build solid color image and add alpha
    base = pyvips.Image.black(src.width, src.height).new_from_image([r, g, b])
    rgba = base.bandjoin(alpha_band)

    out_path = str(Path(src_path).with_suffix("")) + f"_tint_{hex_color.lstrip('#')}.tiff"
    rgba.write_to_file(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--item-id", required=True)
    ap.add_argument("--tiff", required=True)
    ap.add_argument("--ppc-anot", required=True)
    ap.add_argument("--name", default="PPC overlay (TIFF)")
    ap.add_argument("--opacity", type=float, default=0.35)
    ap.add_argument("--tint", default=None, help="Hex color to tint overlay TIFF, e.g. #FF0000")
    ap.add_argument("--tint-alpha", type=float, default=0.7, help="Alpha for tint mask (0-1)")
    ap.add_argument("--delete-existing", action="store_true")
    ap.add_argument("--debug-rect", action="store_true")
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_json(args.config)
    gc = gc_login(cfg)

    base_item = gc.get(f"/item/{args.item_id}")
    folder_id = base_item["folderId"]
    sizeX, sizeY = wsi_dims(gc, args.item_id)

    # Optionally tint the TIFF before upload
    tiff_path = args.tiff
    if args.tint:
        tiff_path = _tint_tiff_with_alpha(args.tiff, args.tint, args.tint_alpha)

    # Create an overlay item in the same folder, upload TIFF, and ensure tiling
    overlay_item = ensure_overlay_item(gc, folder_id, f"{Path(tiff_path).stem} (overlay source)")
    upload_file_to_item(gc, overlay_item["_id"], tiff_path, replace=True)
    ensure_tiles(gc, overlay_item["_id"])

    # Get TIFF dimensions WITHOUT fully decoding
    tW, tH = get_tiff_size_fast(tiff_path)

    # Load PPC anot to find image transform (if present)
    ppc = load_json(args.ppc_anot)
    img_el = find_ppc_image_element(ppc)
    ppc_params = ppc.get("attributes", {}).get("params", {}) if isinstance(ppc, dict) else {}
    ppc_region = str(ppc_params.get("region", "")).strip()
    has_region = bool(ppc_region)

    elements = []
    used_source = ""

    def is_identity(mat):
        return (
            isinstance(mat, list)
            and len(mat) == 2
            and len(mat[0]) == 2
            and len(mat[1]) == 2
            and float(mat[0][0]) == 1.0
            and float(mat[0][1]) == 0.0
            and float(mat[1][0]) == 0.0
            and float(mat[1][1]) == 1.0
        )

    if img_el and isinstance(img_el.get("transform"), dict):
        tf = img_el["transform"].copy()
        M = tf.get("matrix") or [[1.0, 0.0], [0.0, 1.0]]
        xoff = float(tf.get("xoffset", 0.0)) + args.dx
        yoff = float(tf.get("yoffset", 0.0)) + args.dy

        # If PPC transform lacks scale, infer from TIFF vs WSI size
        # Skip this when PPC was run on a region (crop), since scaling would misplace it.
        if (not has_region) and (tW and tH) and (sizeX and sizeY) and (
            abs(float(sizeX) - float(tW)) > 1 or abs(float(sizeY) - float(tH)) > 1
        ) and is_identity(M):
            sx = float(sizeX) / float(tW)
            sy = float(sizeY) / float(tH)
            M = [[sx, 0.0], [0.0, sy]]

        print("=== PLAN (PPC transform) ===")
        print(f"Xform : M={M} xoff={xoff} yoff={yoff}")
        print(f"Sizes : TIFF {tW}x{tH}  |  WSI {sizeX}x{sizeY}")

        if args.delete_existing:
            delete_ann_by_name(gc, args.item_id, args.name)

        elements.append({
            "type": "image",
            "girderId": overlay_item["_id"],
            "opacity": float(args.opacity),
            "hasAlpha": True,
            "transform": {
                "matrix": [[float(M[0][0]), float(M[0][1])], [float(M[1][0]), float(M[1][1])]],
                "xoffset": float(xoff), "yoffset": float(yoff)
            },
            "label": {"value": Path(tiff_path).name}
        })
        used_source = "ppc.anot image transform"

        if args.debug_rect:
            def map_pt(u, v):
                x = M[0][0]*u + M[0][1]*v + xoff
                y = M[1][0]*u + M[1][1]*v + yoff
                return [x, y, 0]
            elements.append({
                "type": "polyline", "closed": True,
                "points": [map_pt(0,0), map_pt(tW,0), map_pt(tW,tH), map_pt(0,tH), map_pt(0,0)],
                "lineColor": "rgba(255,0,0,1)", "lineWidth": 2,
                "label": {"value": "overlay extent (PPC xform)"}
            })
    else:
        # Fallback: identity + user offset
        M = [[1.0, 0.0], [0.0, 1.0]]
        if (not has_region) and (tW and tH) and (sizeX and sizeY) and (
            abs(float(sizeX) - float(tW)) > 1 or abs(float(sizeY) - float(tH)) > 1
        ):
            sx = float(sizeX) / float(tW)
            sy = float(sizeY) / float(tH)
            M = [[sx, 0.0], [0.0, sy]]
        if args.delete_existing:
            delete_ann_by_name(gc, args.item_id, args.name)
        elements.append({
            "type": "image",
            "girderId": overlay_item["_id"],
            "opacity": float(args.opacity),
            "hasAlpha": True,
            "transform": {
                "matrix": [[float(M[0][0]), float(M[0][1])], [float(M[1][0]), float(M[1][1])]],
                "xoffset": float(args.dx), "yoffset": float(args.dy)
            },
            "label": {"value": Path(tiff_path).name}
        })
        used_source = "fallback (identity+offset)"

    annot = {"name": args.name, "display": {"visible": "new"}, "elements": elements}
    res = post_ann(gc, args.item_id, annot)
    ann_id = res.get("_id") or res.get("annotation", {}).get("_id")
    print(f"[OK] posted annotation: {ann_id} (source: {used_source})")


if __name__ == "__main__":
    main()
