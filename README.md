# NeuroTK — Run Guide

Automated 4-step workflow for Whole Slide Image (WSI) analysis.

| Step | What it does | Where it runs |
|------|-------------|---------------|
| 1 | Fetch image list from DSA folder, match local `.svs` files | CPU |
| 2 | TissueSegmentation via SegFormer (HuggingFace) | **GPU** |
| 3 | PositivePixelCount (PPC) analysis | CPU (multi-core) |
| 4 | Upload PPC TIFF overlay back to DSA *(optional)* | CPU |

---

## 1. Setup

### Create virtual environment

```bash
cd /nashome/bhavesh/neurotk
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Install Python dependencies

```bash
pip install \
  histomicstk==1.4.0 \
  large-image==1.33.5 \
  large-image-source-openslide==1.33.5 \
  large-image-source-vips==1.33.5 \
  large-image-source-tiff==1.33.5 \
  large-image-source-multi==1.33.5 \
  large-image-converter==1.33.5 \
  opencv-python-headless \
  shapely scipy scikit-image scikit-learn pandas matplotlib \
  ctk-cli "dask[dataframe]<2024.11.0" distributed \
  girder-client girder-slicer-cli-web pyvips joblib

# GPU packages (torch + transformers) — installed to user site-packages
pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install --user transformers
```

> `torch` and `transformers` are installed to `~/.local/lib/python3.12/site-packages`.
> The workflow scripts automatically add this path to `PYTHONPATH` at runtime.

### Verify GPU

```bash
PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages" \
  .venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Expected: CUDA: True
```

---

## 2. Configuration

Edit `cli/dsa_workflow_config.json`:

```json
{
  "global": {
    "dsa_server_url":   "http://bdsa.pathology.emory.edu:8080/api/v1",
    "dsa_api_key":      "YOUR_API_KEY",
    "local_file_store": "/wsi_archive/APOLLO_NP",
    "output_directory": "/nashome/bhavesh/neurotk/cli/output"
  },
  "step1": {
    "stainID":    "",
    "max_images": 10,
    "output_file": "step2_images_output.json"
  }
}
```

| Field | Description |
|-------|-------------|
| `dsa_server_url` | DSA API base URL |
| `dsa_api_key` | Your DSA API key (must have `core.data.write` scope for Step 4) |
| `local_file_store` | Root path of WSI `.svs` files on disk |
| `output_directory` | Where outputs and logs are saved |
| `max_images` | Limit how many images Step 1 fetches (set to `1` for testing) |

### DSA API Key — required scopes

When creating your API key at `http://bdsa.pathology.emory.edu:8080` → My Account → API Keys, enable:

- `core.data.read`
- `core.data.write` ← required for Step 4 upload
- `core.data.own`
- `core.user_info.read`

### Finding your folder ID

Open the DSA folder in the browser. The URL looks like:
```
http://bdsa.pathology.emory.edu:8080/#collection/<collectionId>/folder/<folderId>
                                                                         ^^^^^^^^
                                                               use this as --root-folder-id
```

---

## 3. Running the Workflow

All commands run from `cli/`:

```bash
cd /nashome/bhavesh/neurotk/cli
source ../.venv/bin/activate
```

---

### Option A — Background (recommended for multiple images)

```bash
# Fetch folder + run steps 1,2,3 on all images
./run_workflow_background.sh --root-folder-id folderi-d--steps 1,2,3

# Reuse cached image list — run steps 2,3 only
./run_workflow_background.sh --steps 2,3

# PPC only (segmentation already done)
./run_workflow_background.sh --steps 3

# Force reprocess everything
./run_workflow_background.sh --root-folder-id folderi-d--steps 1,2,3 --force
```

**Monitor progress** (live, refreshes every 15 s):
```bash
./monitor_progress.sh          # live mode
./monitor_progress.sh --once   # print once and exit
```

```
  Image                               Seg    PPC     DSA Link
  -----                               ---    ---     --------
  E04-32_ABETA_2B                     [x]    [x]     http://bdsa.../item/68cd...
  E22-95_TAU_1                        [x]    [ ]     http://bdsa.../item/697f...
```

**Stop:**
```bash
./stop.sh
```

**Tail the log:**
```bash
tail -f output/logs/workflow_<TIMESTAMP>.log
```

---

### Option B — Foreground (all images, single terminal)

```bash
# Fetch folder + run steps 1,2,3 on all images
python -u run_workflow.py --config dsa_workflow_config.json \
  --root-folder-id folderi-d--steps 1,2,3

# Reuse cached list — run steps 2,3
python -u run_workflow.py --config dsa_workflow_config.json --steps 2,3 --all-items

# PPC only (step 3), uses cached item list
python -u run_workflow.py --config dsa_workflow_config.json --steps 3 --all-items
```

---

### Option C — Single image

```bash
# Steps 1,2,3 for one image
python -u run_workflow_manual.py \
  --config dsa_workflow_config.json \
  --root-folder-id folderi-d\
  --item-id <DSA_ITEM_ID> \
  --steps 1,2,3

# Step 1 only — fetch and cache folder list
python -u run_workflow_manual.py --config dsa_workflow_config.json \
  --root-folder-id folderi-d--steps 1

# Run individual steps (folder already cached from step 1)
python -u run_workflow_manual.py --config dsa_workflow_config.json --item-id <ID> --steps 2
python -u run_workflow_manual.py --config dsa_workflow_config.json --item-id <ID> --steps 3

# Steps 2,3 together for one image
python -u run_workflow_manual.py --config dsa_workflow_config.json --item-id <ID> --steps 2,3
```

---

## 4. Output Files

All outputs land in `cli/output/`:

| File | Description |
|------|-------------|
| `step2_images_output.json` | DSA item list with local file matches (Step 1) |
| `<IMAGE_NAME>.anot` | TissueSegmentation annotation (Step 2) |
| `<IMAGE_NAME>-ppc.anot` | PPC annotation (Step 3) |
| `<IMAGE_NAME>.tiff` | PPC label image (Step 3) |
| `workflow_results.json` | Final run summary for all images |
| `output/logs/` | Per-run log files |

---

## 5. PPC Parameters (Step 3)

Configured in `dsa_workflow_config.json` under `step3.ppc_parameters`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hue_value` | 0.83 | Target hue (0–1) |
| `hue_width` | 0.15 | Hue tolerance |
| `saturation_minimum` | 0.05 | Min saturation |
| `intensity_upper_limit` | 0.95 | Max intensity |
| `intensity_weak_threshold` | 0.65 | Weak positive threshold |
| `intensity_strong_threshold` | 0.35 | Strong positive threshold |
| `intensity_lower_limit` | 0.05 | Min intensity |

---

## 6. Troubleshooting

**`No module named 'torch'`**
```bash
echo $PYTHONPATH
# Should contain: /nashome/bhavesh/.local/lib/python3.12/site-packages
# Background scripts set this automatically.
```

**`No local file found for item`**
- Check `local_file_store` in config
- Verify file exists: `find /wsi_archive/APOLLO_NP -name "<IMAGE>.svs"`

**`CUDA: False`**
- Run `nvidia-smi` to confirm GPU is visible
- Verify PYTHONPATH includes the user site-packages (see Setup above)

**Step 4 gives `401 You must be logged in`**
- Regenerate your DSA API key with `core.data.write` scope enabled (see Configuration above)

**PPC shows `[ ]` after run**
- Check the log: `tail -50 output/logs/workflow_*.log | grep -E "ERROR|Step 3"`
- Rerun with `--force` flag

**Step 1 gives 0 items**
- Verify `root_folder_id` and `dsa_api_key` in the config
- Test DSA connectivity: `curl http://bdsa.pathology.emory.edu:8080/api/v1/system/version`

