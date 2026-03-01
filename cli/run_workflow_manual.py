#!/usr/bin/env python3
"""
Run Steps 1-4 locally (no Docker) for a single image:
    Step 1: Fetch items from DSA server + verify local file matches  (requires --root-folder-id)
    Step 2: TissueSegmentation (GPU)
    Step 3: PositivePixelCountV2 analysis (CPU)
    Step 4: Upload PPC TIFF overlay to DSA (CPU)

Usage examples:
    # Single image — full pipeline (all steps)
    python run_workflow_manual.py --config dsa_workflow_config.json \\
        --root-folder-id 67ddb0782fc8ce397c5ef7fb --item-id 68cd7f2977dbcb39504b9681 --steps 1,2,3,4

    # Single image — step 1 only (fetch & cache folder list)
    python run_workflow_manual.py --config dsa_workflow_config.json \\
        --root-folder-id 67ddb0782fc8ce397c5ef7fb --steps 1

    # Single image — steps 2,3,4 (folder already fetched)
    python run_workflow_manual.py --config dsa_workflow_config.json \\
        --item-id 68cd7f2977dbcb39504b9681 --steps 2,3,4

    # Single image — step 3 only (segmentation already done)
    python run_workflow_manual.py --config dsa_workflow_config.json \\
        --item-id 68cd7f2977dbcb39504b9681 --steps 3

    # Single image — step 4 only (upload PPC overlay)
    python run_workflow_manual.py --config dsa_workflow_config.json \\
        --item-id 68cd7f2977dbcb39504b9681 --steps 4
"""
import json
import argparse
import subprocess
import sys
import os
from pathlib import Path

try:
    import girder_client
except ImportError:
    print("ERROR: girder_client not installed. Install with: pip install girder-client")
    sys.exit(1)

# User site-packages where torch/transformers live (cu118 build)
_USER_SITE = os.path.expanduser("~/.local/lib/python3.12/site-packages")

def _subprocess_env():
    """Return env with user site-packages in PYTHONPATH for GPU access."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(":") if p]
    if _USER_SITE not in parts:
        parts.insert(0, _USER_SITE)
    env["PYTHONPATH"] = ":".join(parts)
    return env

# Always use repo venv Python for subprocesses (has full large_image stack).
# VENV_PYTHON env var is only a fallback when the venv doesn't exist.
_venv_python = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
VENV_PYTHON = str(_venv_python) if _venv_python.exists() else (os.environ.get("VENV_PYTHON") or sys.executable)

# If the configured python path is missing, fall back to the current interpreter.
if not Path(VENV_PYTHON).exists():
    print(f"[WARN] VENV_PYTHON not found at {VENV_PYTHON}; falling back to {sys.executable}")
    VENV_PYTHON = sys.executable

# If not running inside the repo venv, re-exec with it when available.
_venv_exists = _venv_python.exists()
if _venv_exists and Path(sys.executable).resolve() != _venv_python.resolve():
    if os.environ.get("NEUROTK_VENV_REEXEC") != "1":
        os.environ["NEUROTK_VENV_REEXEC"] = "1"
        os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)


# ---------------------------
# Utilities
# ---------------------------
def _step1_output_file(config) -> str:
    return (
        config.get("step1", {}).get("output_file")
        or config.get("step2", {}).get("input_file")
        or "step2_images_output.json"
    )

def _match_item(rec: dict, wanted_item_id: str) -> bool:
    """Match record by itemId, _id, or id field."""
    wanted_item_id = str(wanted_item_id).strip()
    for k in ("itemId", "_id", "id"):
        v = str(rec.get(k, "")).strip()
        if v and v == wanted_item_id:
            return True
    dsa = rec.get("dsa") or {}
    for k in ("itemId", "_id", "id"):
        v = str(dsa.get(k, "")).strip()
        if v and v == wanted_item_id:
            return True
    return False


def _get_dsa_client(config):
    """Authenticate with DSA server."""
    dsa_url = config["global"]["dsa_server_url"]
    dsa_key = config["global"].get("dsa_api_key")

    gc = girder_client.GirderClient(apiUrl=dsa_url)
    if dsa_key:
        gc.authenticate(apiKey=dsa_key)
    else:
        raise ValueError("No DSA API key in config")
    return gc


def _get_all_items(gc, folder_id):
    """Recursively fetch all items from DSA folder."""
    all_items = []
    items = gc.get(f"/item?folderId={folder_id}")
    all_items.extend(items)

    subfolders = gc.get(f"/folder?parentType=folder&parentId={folder_id}")
    for f in subfolders:
        all_items.extend(_get_all_items(gc, f['_id']))

    return all_items


def _build_item_path(gc, item):
    """Build full folder path for DSA item."""
    path_parts = [item['name']]
    folder_id = item['folderId']
    while folder_id:
        folder = gc.get(f"/folder/{folder_id}")
        path_parts.insert(0, folder['name'])
        parent_type = folder.get('parentType')
        parent_id = folder.get('parentId')

        if parent_type == 'folder':
            folder_id = parent_id
        elif parent_type == 'collection':
            collection = gc.get(f"/collection/{parent_id}")
            path_parts.insert(0, collection['name'])
            break
        else:
            break
    return '/'.join(path_parts)


def _fetch_and_enrich_one_item(config, item_id: str):
    """
    Fetch a single item from DSA by ID and enrich with local file match.
    Returns a list of one record (same shape as step1 output), or None on failure.
    """
    try:
        gc = _get_dsa_client(config)
    except Exception as e:
        print(f"ERROR: DSA authentication failed: {e}")
        return None

    try:
        item = gc.get(f"/item/{item_id}")
    except Exception as e:
        print(f"ERROR: Item {item_id} not found on DSA: {e}")
        return None

    try:
        item_path = _build_item_path(gc, item)
        item['fullPath'] = item_path
    except Exception as e:
        item['fullPath'] = f"(error: {e})"

    store_path = config["global"].get("local_file_store", "")
    if not store_path or not os.path.exists(store_path):
        print(f"ERROR: Local file store not found: {store_path}")
        return None

    # Prefer meta.wsi_fp if present and file exists
    local_matches = []
    meta = item.get("meta", {})
    wsi_fp = meta.get("wsi_fp") or meta.get("wsi_fp_path")
    if wsi_fp and os.path.exists(wsi_fp):
        local_matches.append(wsi_fp)

    if not local_matches:
        # Fallback: scan store by filename
        img_name_lower = item["name"].lower()
        for root, _, files in os.walk(store_path):
            for fname in files:
                if fname.lower() == img_name_lower:
                    local_matches.append(os.path.join(root, fname))
                    break
            if local_matches:
                break

    item.update({
        "local_matches": local_matches,
        "has_local_match": bool(local_matches),
        "is_duplicate": len(local_matches) > 1,
        "match_count": len(local_matches),
    })

    if not item["has_local_match"]:
        print(f"ERROR: No local file found for item {item_id} ({item.get('name', '?')}). Check local_file_store and DSA meta.wsi_fp.")
        return None

    print(f"Fetched item from DSA: {item['name']} ({item_id}), local: {item['local_matches'][0]}")
    return [item]




def _run_step1_fetch_dsa(config, root_folder_id=None):
    """Step 1: Fetch items from DSA server."""
    print("\n" + "="*60)
    print("STEP 1: Fetch DSA ImageSet")
    print("="*60)
    
    try:
        gc = _get_dsa_client(config)
    except Exception as e:
        print(f"ERROR: DSA authentication failed: {e}")
        return None

    root_folder_id = root_folder_id or config["global"].get("root_folder_id", "")
    if not root_folder_id:
        print("ERROR: root_folder_id not provided. Use --root-folder-id or set it in config global.root_folder_id")
        return None
    stain_id_filter = config.get("step1", {}).get("stainID", "")
    max_images = config.get("step1", {}).get("max_images", 200)

    print(f"Fetching items from DSA folder: {root_folder_id}")
    if stain_id_filter:
        print(f"Filter: stainID = '{stain_id_filter}'")

    try:
        all_items = _get_all_items(gc, root_folder_id)
        print(f"Found {len(all_items)} items recursively")
    except Exception as e:
        print(f"ERROR fetching items: {e}")
        return None

    # Filter by stain ID if specified
    filtered = []
    for item in all_items:
        if stain_id_filter:
            meta = item.get('meta', {})
            np_schema = meta.get('npSchema', {})
            if np_schema.get('stainID') != stain_id_filter:
                continue
        filtered.append(item)
        if len(filtered) >= max_images:
            break

    print(f"After filtering: {len(filtered)} items")
    
    for i, item in enumerate(filtered[:5]):  # Show first 5
        try:
            item_path = _build_item_path(gc, item)
            item['fullPath'] = item_path
        except Exception as e:
            item['fullPath'] = f"(error: {e})"
        print(f"  {i+1}. {item['name']} ({item['_id']})")

    out_dir = Path(config["global"]["output_directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    step1_output = config.get("step1", {}).get("output_file", "step1_dsa_items.json")
    step1_path = out_dir / step1_output
    
    with open(step1_path, 'w') as f:
        json.dump(filtered, f, indent=2)
    
    print(f"✓ Step 1 complete: {len(filtered)} items saved to {step1_path}\n")
    return filtered


def _run_step2_verify_local(config, dsa_items):
    """Verify local file matches (merged into Step 1)."""
    print("="*60)
    print("STEP 2: Verify Local Files")
    print("="*60)
    
    out_dir = Path(config["global"]["output_directory"])
    store_path = config["global"].get("local_file_store", "")
    
    if not store_path or not os.path.exists(store_path):
        print(f"ERROR: Local file store not found: {store_path}")
        return None

    print(f"Scanning local file store: {store_path}")
    
    # Build local file index
    local_files = {}
    for root, _, files in os.walk(store_path):
        for fname in files:
            fname_lower = fname.lower()
            local_files.setdefault(fname_lower, []).append(os.path.join(root, fname))
    
    print(f"Found {len(local_files)} unique local files")

    # Match DSA items to local files
    enriched = []
    matched_count = 0
    for img in dsa_items:
        img_name_lower = img['name'].lower()
        matched_paths = local_files.get(img_name_lower, [])

        img.update({
            'local_matches': matched_paths,
            'has_local_match': bool(matched_paths),
            'is_duplicate': len(matched_paths) > 1,
            'match_count': len(matched_paths),
        })

        if matched_paths:
            matched_count += 1

        enriched.append(img)

    step1_output = _step1_output_file(config)
    step2_path = out_dir / step1_output
    
    with open(step2_path, 'w') as f:
        json.dump(enriched, f, indent=2)
    
    print(f"✓ Step 1 complete: {matched_count}/{len(enriched)} items have local matches")
    print(f"  Saved to {step2_path}\n")
    return enriched


def _run_step4_upload(config_path: str, config: dict, item_id: str, tiff_path: Path, ppc_anot_path: Path) -> bool:
    """Step 4: Upload PPC TIFF overlay to DSA."""
    s4 = config.get("step4", {})
    upload_script = Path(s4.get("script_path", "./cli/step4-Push.py"))

    if not upload_script.exists():
        print(f"[ERROR] Step 4 upload script not found: {upload_script}")
        return False

    cmd = [
        VENV_PYTHON, str(upload_script),
        "--config", config_path,
        "--item-id", str(item_id),
        "--tiff", str(tiff_path),
        "--ppc-anot", str(ppc_anot_path),
        "--name", str(s4.get("annotation_name", "PPC overlay (TIFF)")),
        "--opacity", str(s4.get("opacity", 0.35)),
        "--dx", str(s4.get("dx", 0.0)),
        "--dy", str(s4.get("dy", 0.0)),
    ]

    tint = s4.get("tint")
    if tint:
        cmd += ["--tint", str(tint), "--tint-alpha", str(s4.get("tint_alpha", 0.7))]
    if s4.get("delete_existing", False):
        cmd.append("--delete-existing")
    if s4.get("debug_rect", False):
        cmd.append("--debug-rect")

    print(f"Command: {' '.join(cmd)}")
    env = _subprocess_env()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode == 0:
        print("[OK] Step 4 upload completed")
        return True

    print("[ERROR] Step 4 upload failed:")
    print(result.stderr)
    return False


# ---------------------------
# Main
# ---------------------------
def main(config_path: str, item_id: str = None, force: bool = False, run_steps: str = "all", root_folder_id: str = None):
    """
    Execute workflow manually (Steps 1-4):
    Step 1: Fetch items from DSA + verify local files
    Step 2: GPU-based tissue segmentation
    Step 3: CPU-based PositivePixelCountV2 analysis
    Step 4: Upload PPC TIFF overlay to DSA
    """
    with open(config_path) as f:
        config = json.load(f)

    out_dir = Path(config["global"]["output_directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine which steps to run
    steps_to_run = set(run_steps.split(',')) if run_steps != "all" else {"1", "2", "3", "4"}

    # STEP 1: Fetch + verify local files
    all_images = None
    if "1" in steps_to_run:
        try:
            dsa_items = _run_step1_fetch_dsa(config, root_folder_id=root_folder_id)
            if not dsa_items:
                return
            all_images = _run_step2_verify_local(config, dsa_items)
            if not all_images:
                return
        except Exception as e:
            print(f"ERROR in Step 1: {e}")
            return
    else:
        # Load Step 1 output if not running Step 1
        step1_output = _step1_output_file(config)
        step1_path = out_dir / step1_output
        if step1_path.exists():
            with open(step1_path) as f:
                all_images = json.load(f)
            print(f"Loaded {len(all_images)} items from {step1_path}")
        else:
            print(f"ERROR: Step 1 output not found at {step1_path}")
            return

    # If only running Step 1, stop here
    if not (steps_to_run & {"2", "3", "4"}):
        print("\n✓ Step 1 complete!")
        return

    # STEP 2-4: Tissue Segmentation, PPC V2, and Upload
    if item_id is None:
        print("ERROR: --item-id required for Steps 2-4")
        return

    # Filter to requested item
    images = [r for r in all_images if _match_item(r, item_id)]
    if not images:
        # Item not in cached list: try fetching this single item from DSA
        print(f"Item {item_id} not in cached list. Fetching from DSA...")
        images = _fetch_and_enrich_one_item(config, item_id)
        if not images:
            step1_output = _step1_output_file(config)
            step1_path = out_dir / step1_output
            available = [r.get("_id", r.get("itemId", "N/A")) for r in all_images[:10]]
            raise SystemExit(
                f"No record for item_id {item_id} in {step1_path}. "
                f"Available item IDs: {available}"
            )

    # Build config
    s2 = config["step2"]
    s3 = config["step3"]
    s4 = config.get("step4", {})

    # === STEP 2: TissueSegmentation Parameters ===
    seg_docname = s2.get("docname", "GrayWhiteSegTest")
    seg_mag = float(s2.get("mag", 2.0))
    seg_tile_size = int(s2.get("tile_size", 512))
    seg_batch = int(s2.get("batch_size", 8))
    tissue_seg_script = Path(s2.get("script_path", "./cli/TissueSegmentation/TissueSegmentation.py"))

    # === STEP 3: PositivePixelCountV2 Parameters ===
    ppc = s3["ppc_parameters"]
    ppc_docname = ppc.get("docname", "Positive Pixel Count")
    hue_value = float(ppc.get("hue_value", 0.05))
    hue_width = float(ppc.get("hue_width", 0.15))
    saturation_minimum = float(ppc.get("saturation_minimum", 0.05))
    intensity_upper_limit = float(ppc.get("intensity_upper_limit", 0.95))
    intensity_weak_threshold = float(ppc.get("intensity_weak_threshold", 0.75))
    intensity_strong_threshold = float(ppc.get("intensity_strong_threshold", 0.45))
    intensity_lower_limit = float(ppc.get("intensity_lower_limit", 0.05))
    ppc_v2_script = Path(s3.get("script_path", "./cli/PositivePixelCount/PositivePixelCountV2.py"))

    results = []

    for rec in images:
        if not rec.get("has_local_match"):
            print(f"[SKIP] No local match for item {item_id}")
            continue

        local_path = Path(rec["local_matches"][0])
        if not local_path.exists():
            print(f"[ERROR] File not found: {local_path}")
            continue

        base = local_path.stem
        seg_anot = f"{base}.anot"
        ppc_anot = f"{base}-ppc.anot"
        ppc_tiff = f"{base}.tiff"

        seg_out_path = out_dir / seg_anot
        ppc_out_path = out_dir / ppc_tiff

        seg_out_exists = seg_out_path.exists()
        ppc_out_exists = ppc_out_path.exists()

        # Skip if BOTH outputs already exist (job already completed)
        if seg_out_exists and ppc_out_exists and not force and "4" not in steps_to_run:
            print(f"Skipping {base} (already processed)")
            results.append({
                "item_id": item_id,
                "image": rec.get("name", base),
                "status": "already_completed",
                "output_files": [str(seg_out_path), str(out_dir / ppc_anot), str(ppc_out_path)]
            })
            continue

        # ----- Step 2: GPU segmentation -----
        run_seg = True
        if not force and s2.get("check_output_exists", True) and seg_out_exists:
            print(f"[INFO] Seg output exists, skipping: {seg_out_path}")
            run_seg = False

        seg_status = "skipped"
        if "2" in steps_to_run and run_seg:
            print(f"\n[STEP 2] Running TissueSegmentation on {base}...")
            seg_cmd = [
                VENV_PYTHON, str(tissue_seg_script),
                seg_docname,
                str(local_path),
                str(seg_mag),
                str(seg_tile_size),
                str(seg_batch),
                "--image_annotation", str(seg_out_path)
            ]
            print(f"Command: {' '.join(seg_cmd)}")
            env = _subprocess_env()
            seg_result = subprocess.run(seg_cmd, capture_output=True, text=True, env=env)

            if seg_result.returncode == 0:
                print(f"[OK] Step 2 completed: {seg_anot}")
                seg_status = "completed"
            else:
                print(f"[ERROR] Step 2 failed:")
                print(seg_result.stderr)
                results.append({
                    "item_id": item_id,
                    "image": rec.get("name", base),
                    "status": "failed",
                    "step": 2,
                    "error": seg_result.stderr
                })
                continue

        # ----- Step 3: CPU PPC V2 -----
        run_ppc = True
        if not force and s3.get("check_output_exists", True) and ppc_out_exists:
            print(f"[INFO] PPC V2 TIFF exists, skipping: {ppc_out_path}")
            run_ppc = False

        ppc_status = "skipped"
        if "3" in steps_to_run and run_ppc:
            print(f"\n[STEP 3] Running PositivePixelCountV2 on {base}...")
            ppc_cmd = [
                VENV_PYTHON, str(ppc_v2_script),
                ppc_docname,
                str(local_path),
                str(hue_value),
                str(hue_width),
                str(saturation_minimum),
                str(intensity_upper_limit),
                str(intensity_weak_threshold),
                str(intensity_strong_threshold),
                str(intensity_lower_limit),
                "--image_annotation", str(out_dir / ppc_anot),
                "--outputLabelImage", str(ppc_out_path),
                #"--region=-1,-1,-1,-1"
                "--region=-1,-1,-1,-1"  # Process full image
            ]
            print(f"Command: {' '.join(ppc_cmd)}")
            env = _subprocess_env()
            ppc_result = subprocess.run(ppc_cmd, capture_output=True, text=True, env=env)

            if ppc_result.returncode == 0:
                print(f"[OK] Step 3 completed: {ppc_tiff}")
                ppc_status = "completed"
            else:
                print(f"[ERROR] Step 3 failed:")
                print(ppc_result.stderr)
                ppc_status = "failed"

        # ----- Step 4: Upload TIFF overlay -----
        upload_status = "skipped"
        if "4" in steps_to_run:
            if ppc_out_path.exists() and (out_dir / ppc_anot).exists():
                print(f"\n[STEP 4] Uploading PPC TIFF overlay for {base}...")
                ok = _run_step4_upload(
                    config_path,
                    config,
                    item_id,
                    ppc_out_path,
                    out_dir / ppc_anot,
                )
                upload_status = "completed" if ok else "failed"
            else:
                print("[WARN] Step 4 skipped: missing PPC outputs")
                upload_status = "missing_outputs"

        results.append({
            "item_id": item_id,
            "image": rec.get("name", base),
            "local_path": str(local_path),
            "seg_anot": str(seg_out_path),
            "ppc_anot": str(out_dir / ppc_anot),
            "ppc_tiff": str(ppc_out_path),
            "step2_status": seg_status,
            "step3_status": ppc_status,
            "step4_status": upload_status
        })

    # Save submission summary
    result_path = out_dir / "workflow_results.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {result_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run Steps 1-4 locally (no Docker)"
    )
    
  

    ap.add_argument("--config", default="dsa_workflow_config.json", help="Config file path")
    ap.add_argument("--root-folder-id", default=None, help="DSA folder ID to fetch all images from (Step 1)")
    ap.add_argument("--item-id", default=None, help="Target DSA item _id (required for Steps 2-4)")
    ap.add_argument("--force", action="store_true", help="Force reprocessing, ignore existing outputs")
    ap.add_argument("--steps", default="all", help="Steps to run: '1', '2', '3', '4', '1,2', '2,3', '3,4', or 'all' (default)")
    args = ap.parse_args()
    main(args.config, item_id=args.item_id, force=args.force, run_steps=args.steps, root_folder_id=args.root_folder_id)
