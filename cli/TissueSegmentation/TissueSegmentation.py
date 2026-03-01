from histomicstk.cli.utils import CLIArgumentParser
from pprint import pprint
import large_image, json
import numpy as np
import cv2
from pathlib import Path

# Absolute path to the SegFormer model directory, anchored to this script file.
_MODEL_DIR = str(Path(__file__).resolve().parent / "model")

try:
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    _torch_available = True
except Exception:
    _torch_available = False
    SegformerForSemanticSegmentation = None

try:
    from dsa_helpers.ml.segformer_semantic_segmentation import inference as _seg_inference
except Exception:
    _seg_inference = None


def _gpu_inference(thumbnail_img, model, tile_size=512, batch_size=8):
    """
    Run SegFormer inference on a thumbnail using GPU (or CPU if unavailable).
    Returns (mask, contours) matching the expected output shape.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    model = model.to(device)
    model.eval()

    processor = SegformerImageProcessor.from_pretrained(_MODEL_DIR)

    h, w = thumbnail_img.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    # Build list of (y, x) tile origins
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tiles.append((y, x))

    # Process in batches
    for batch_start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[batch_start: batch_start + batch_size]
        batch_imgs = []
        for (ty, tx) in batch_tiles:
            patch = thumbnail_img[ty: ty + tile_size, tx: tx + tile_size]
            batch_imgs.append(patch)

        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Upsample logits to each tile's original size
        for i, (ty, tx) in enumerate(batch_tiles):
            patch_h = min(tile_size, h - ty)
            patch_w = min(tile_size, w - tx)
            logits = outputs.logits[i: i + 1]  # (1, num_labels, H', W')
            upsampled = torch.nn.functional.interpolate(
                logits, size=(patch_h, patch_w), mode="bilinear", align_corners=False
            )
            seg = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
            pred_mask[ty: ty + patch_h, tx: tx + patch_w] = seg

    # Tissue = label 1; convert to binary
    binary = (pred_mask == 1).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return binary, contours


def _fallback_inference(thumbnail_img, *args, **kwargs):
    """Simple tissue segmentation fallback using Otsu thresholding.

    Returns (mask, contours) to match expected output shape.
    """
    # Convert to grayscale and apply blur
    gray = cv2.cvtColor(thumbnail_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold to separate tissue/background
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background is white and tissue is dark
    if np.mean(gray[mask == 255]) > np.mean(gray[mask == 0]):
        mask = cv2.bitwise_not(mask)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours


def main(args):
    """Tissue segmentation at a low resolution scale using SegFormer for
    semantic segmentation (HuggingFace implementation). The output is scaled
    back to the original resolution and saved as DSA annotations.

    args:
        The CLI arguments.
        - docname: The name of the annotation document saved as output.
        - inputImageFile: Filepath to the WSI.
        - mag: Magnification level the thumbnail to perform segmentation.
        - tile_size: The size of the tiles to be used for segmentation.
        - batch_size: The batch size to be used for segmentation.
        - outputAnnotationFile: Filepath to save the output annotation.

    """
    # Print the CLI arguments.
    pprint(vars(args))

    # Get the tile source.
    ts = large_image.getTileSource(args.inputImageFile)

    kwargs = {
        "scale": {"magnification": 2.0},
        "format": large_image.tilesource.TILE_FORMAT_NUMPY,
    }

    print("Getting thumbnail image...")
    thumbnail_img = ts.getRegion(**kwargs)[0][:, :, :3]
    print(f"   Thumbnail image obtained of shape: {thumbnail_img.shape}.")

    # Create the map between of labels.
    id2label = {0: "background", 1: "tissue"}
    label2id = {v: k for k, v in id2label.items()}

    model = None
    use_gpu = _torch_available and SegformerForSemanticSegmentation is not None
    use_dsa_helpers = _seg_inference is not None

    if use_gpu or use_dsa_helpers:
        try:
            print("Loading SegFormer model...")
            model = SegformerForSemanticSegmentation.from_pretrained(
                _MODEL_DIR, id2label=id2label, label2id=label2id
            )
            import torch
            print(f"   CUDA available: {torch.cuda.is_available()}")
        except Exception as e:
            print(f"[WARN] SegFormer load failed: {e}. Using fallback segmentation.")
            model = None

    # Run inference: prefer dsa_helpers, then direct GPU, then CPU fallback.
    print("Running inference...")
    if model is not None and use_dsa_helpers:
        contours = _seg_inference(
            thumbnail_img, model, tile_size=args.tile_size, batch_size=args.batch_size
        )[1]
    elif model is not None and use_gpu:
        contours = _gpu_inference(
            thumbnail_img, model, tile_size=args.tile_size, batch_size=args.batch_size
        )[1]
    else:
        print("[INFO] Using CPU Otsu fallback segmentation.")
        contours = _fallback_inference(thumbnail_img, tile_size=args.tile_size, batch_size=args.batch_size)[1]

    # Convert the contours to DSA-style elements.
    print("Pushing results as annotations...")
    elements = []

    # Calculate the scale factor to go from low mag -> high mag coordinates.
    sf = ts.getMetadata()["magnification"] / args.mag

    for contour in contours:
        if contour is None:
            continue

        # Normalize contour shape to (N, 2)
        contour = np.asarray(contour)
        if contour.ndim == 3 and contour.shape[1] == 1:
            contour = contour[:, 0, :]
        contour = contour.squeeze()

        # Skip degenerate contours
        if contour.ndim != 2 or contour.shape[0] < 3 or contour.shape[1] != 2:
            continue

        # Convert to list of [x, y, 0] tuples.
        contour_tuples = [
            (int(point[0] * sf), int(point[1] * sf), 0) for point in contour
        ]

        elements.append(
            {
                "fillColor": "rgba(0, 255, 0, 0.25)",
                "lineColor": "rgb(0, 255, 0)",
                "lineWidth": 4,
                "type": "polyline",
                "closed": True,
                "points": contour_tuples,
                "label": {"value": "predicted tissue"},
                "group": "tissue",
            }
        )

    # Create an annotation dictionary
    ann_doc = {
        "name": args.docname,
        "elements": elements,
        "attributes": {
            "params": vars(args),
        },
    }

    pprint(ann_doc)

    # Save the annotation dictionary to the output annotation file
    with open(args.outputAnnotationFile, "w") as fh:
        json.dump(ann_doc, fh, separators=(",", ":"), sort_keys=False)

    print("Done!")


if __name__ == "__main__":
    # Run main function, passing CLI arguments.
    main(CLIArgumentParser().parse_args())
