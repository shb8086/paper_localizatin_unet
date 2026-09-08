# Document Corner Localization — Two-Stage CNN Pipeline

> **Paper:** "A New Image Dataset for Document Corner Localization"
> S. B. Dizaj, M. Soheili, and A. Mansouri
> *2020 International Conference on Machine Vision and Image Processing (MVIP), Iran*
> DOI: [10.1109/MVIP49855.2020.9116896](https://doi.org/10.1109/MVIP49855.2020.9116896)

---

## What This Repository Does

This project solves **document corner localization**: given a photo of a document (on a desk, floor, or other background), it automatically finds the exact pixel coordinates of all four corners of the document. Those corners can then be used to perform a perspective warp, producing a clean, flat, front-facing scan — the same outcome as a document scanner app.

```
Input photo                 →  Two-stage CNN pipeline  →  Four corners + warped output
(document on background)                                   (LeftUp, RightUp, RightDown, LeftDown)
```

The system is deliberately lightweight: both stages run inference on **32 × 32 thumbnails**, making it fast even on CPU.

---

## How It Works

### Stage 1 — Coarse 4-Corner Detection (`networks/train_4point.py`)

A small CNN sees the whole image (resized to 32 × 32) and predicts the rough location of all four corners at once, returning **8 normalized coordinates** (x, y per corner).

```
Input (32×32×3)
→ Conv 5×5 ×20  →  MaxPool
→ Conv 5×5 ×40  →  Conv 5×5 ×40  →  MaxPool
→ Conv 5×5 ×60  →  Conv 5×5 ×60  →  MaxPool
→ Conv 5×5 ×80  →  MaxPool
→ Conv 5×5 ×100 →  MaxPool
→ FC 500  →  FC 500  →  FC 8        ← (x,y) × 4 corners
```

This stage divides the image into four corner-region crops to feed into Stage 2.

### Stage 2 — Fine Localization via Iterative Zoom-In (`networks/train_corner.py`)

For each of the four corner crops, the pipeline zooms in repeatedly until it pinpoints the exact corner pixel:

1. Resize the current crop to 32 × 32 → feed to the per-corner CNN (`corner_locator.pb`)
2. CNN outputs an (x, y) prediction within the crop
3. Cut a **7 % window** centred on the prediction → use as the next crop
4. Repeat, accumulating pixel offsets, until the crop shrinks below 10 × 10 px
5. Sum all offsets → final sub-pixel corner coordinate

The per-corner CNN is a **U-Net** encoder–decoder (`unet/unet.py`):

```
Encoder: depth-3  (Conv 3×3 → ReLU → Conv 3×3 → ReLU → MaxPool) × 3
Decoder: Transposed Conv (upsample) + skip connection (crop-and-concat) × 3
Output:  1×1 Conv → pixel-wise softmax
```

---

## Repository Structure

```
.
├── inference.py          # Run the full 2-stage pipeline on one image
├── evaluate.py           # Compute IoU between predicted and ground-truth corners
├── train_deeplab.py      # DeepLab-based training script (segmentation backbone)
├── export_model.py       # Freeze a checkpoint to a .pb inference graph
├── dataset.py            # TFRecord dataset loader; registers the 'paper' dataset
├── data_utils.py         # Data loading, IoU helpers, ground-truth validation
├── requirements.txt      # Python dependencies
├── corner_locator.pb     # Pre-trained Stage 2 model (per-corner CNN, frozen graph)
│
├── run_xception.sh       # End-to-end train + eval + export (Xception backbone)
├── run_mobilenetv2.sh    # Same workflow with MobileNetV2 backbone
│
├── networks/             # Training scripts for both pipeline stages
│   ├── train_4point.py   # Train Stage 1: coarse 4-corner regression network
│   └── train_corner.py   # Train Stage 2: per-corner iterative zoom-in CNN
│
└── unet/                 # U-Net implementation (used for Stage 2 backbone)
    ├── unet.py           # Unet model and Trainer
    ├── layers.py         # Building blocks: conv, deconv, pooling, loss
    ├── image_util.py     # Data providers (SimpleDataProvider, ImageDataProvider)
    ├── image_gen.py      # Toy synthetic data generator for U-Net experiments
    └── util.py           # Image save/combine/crop helpers
```

---

## Installation

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
```

Requires **Python 3** and **TensorFlow 1.15**. The code uses `tf.compat.v1` automatically so it runs under both TF 1.x and TF 2.x.

For training with `train_deeplab.py` and `export_model.py` you also need the DeepLab research code on your `PYTHONPATH` — see the [DeepLab setup guide](https://github.com/tensorflow/models/tree/master/research/deeplab).

---

## Quick Start — Inference

Two pre-trained frozen graphs are needed:

| File | What it does | Included? |
|---|---|---|
| `corner_locator.pb` | Stage 2 per-corner CNN | ✅ Yes |
| `four_point.pb` | Stage 1 coarse 4-corner CNN | ❌ Train it (see below) |

```bash
# Basic — uses ./corner_locator.pb and ./four_point.pb by default
python inference.py -i photo.jpg

# Explicit model paths
python inference.py \
    -i photo.jpg \
    --segment_model ./corner_locator.pb \
    --find_model    ./four_point.pb

# Also compute IoU against ground-truth (needs a same-named .csv file)
python inference.py -i photo.jpg --evaluate
```

**Outputs** (written next to the input image):

| File | Content |
|---|---|
| `photo_4CornerXY.txt` | Four corner coordinates: LeftUp, RightUp, RightDown, LeftDown |
| `photo_Result.jpg` | Input image with the predicted quadrilateral drawn on it |

---

## Training

### Stage 1 — Coarse 4-corner network

```bash
python networks/train_4point.py \
    --train_image  /data/train_images.npy \
    --train_gt     /data/train_gt.npy \
    --val_image    /data/val_images.npy \
    --val_gt       /data/val_gt.npy \
    --checkpoint_dir ./checkpoints/4point \
    --steps 1000000 \
    --batch_size 10
```

Ground-truth `.npy` files contain arrays of shape `(N, 8)` — four (x, y) pairs per image, normalized to `[0, 1]`.

### Stage 2 — Per-corner CNN

```bash
python networks/train_corner.py \
    --train_image  /data/corner_train_images.npy \
    --train_gt     /data/corner_train_gt.npy \
    --val_image    /data/corner_val_images.npy \
    --val_gt       /data/corner_val_gt.npy \
    --checkpoint_dir ./checkpoints/corner \
    --steps 500000 \
    --batch_size 100
```

Ground-truth `.npy` files contain arrays of shape `(N, 2)` — one (x, y) corner location per crop, in pixel coordinates for a 32 × 32 image.

After training, freeze the checkpoint:

```bash
python export_model.py \
    --checkpoint_path ./checkpoints/corner/model.ckpt \
    --export_path     ./corner_locator.pb \
    --num_classes 2
```

---

## Evaluation

```bash
# IoU score for a single prediction vs. ground truth
python evaluate.py \
    --pred photo_4CornerXY.txt \
    --gt   photo.csv

# Or run evaluation automatically after inference
python inference.py -i photo.jpg --evaluate
```

The IoU is computed on the quadrilateral formed by the four predicted corners vs. the ground-truth bounding box from the CSV.

---

## Dataset

The accompanying dataset (document images with ground-truth 4-corner CSV annotations) is hosted on Mendeley:

**[Download — Mendeley Dataset](https://data.mendeley.com/datasets/x3nm4cxr83/1)**

Each CSV has one row per image with the pixel coordinates of the four corners. The `data_utils.py` helpers load CSVs, validate coordinates, and compute IoU.

---

## Citation

```
S. B. Dizaj, M. Soheili and A. Mansouri, "A New Image Dataset for Document Corner
Localization," 2020 International Conference on Machine Vision and Image Processing
(MVIP), Iran, 2020, pp. 1-4, doi: 10.1109/MVIP49855.2020.9116896.
```

```json
{
  "author": "S. B. Dizaj and M. Soheili and A. Mansouri",
  "title": "A New Image Dataset for Document Corner Localization",
  "conference": "2020 International Conference on Machine Vision and Image Processing (MVIP)",
  "location": "Iran",
  "year": 2020,
  "pages": "1-4",
  "doi": "10.1109/MVIP49855.2020.9116896"
}
```
