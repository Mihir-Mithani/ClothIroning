# Setup & Run Guide

## 1. PyCharm Setup

### Clone / open project
1. Open PyCharm → **File → Open** → select the `garment_ironing/` folder
2. PyCharm will detect `requirements.txt` automatically

### Create virtual environment
In PyCharm terminal:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Run configuration
1. **Run → Edit Configurations → + → Python**
2. Script: `app.py`
3. Working directory: project root
4. Click **Run** or press Shift+F10

The app opens at **http://localhost:7860**

---

## 2. Generate example images (first run only)
```bash
python create_example_assets.py
```

---

## 3. Train custom wrinkle model (optional)
```bash
# Prepare your dataset in data/train/images, data/train/masks, data/val/...
python train_wrinkle_model.py --data_dir data --epochs 50 --batch_size 8
```

---

## 4. Publish to HuggingFace Spaces

### Install HF CLI
```bash
pip install huggingface_hub
huggingface-cli login   # paste your HF token
```

### Create a new Space
```bash
huggingface-cli repo create garment-ironing --type space --space_sdk gradio
```

### Push all files
```bash
cd garment_ironing
git init
git add .
git commit -m "initial commit"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/garment-ironing
git push -u origin main
```

HuggingFace will automatically install `requirements.txt` and run `app.py`.

---

## 5. Environment variables (optional secrets)

If you want to use private HF model weights, set in HF Space secrets:
- `HF_TOKEN` — your HuggingFace token

---

## 6. GPU vs CPU behaviour

| Hardware | Ironing method | Speed |
|----------|---------------|-------|
| NVIDIA GPU (≥8 GB VRAM) | Stable Diffusion inpainting | ~10–30 s/image |
| CPU only | Classical bilateral + guided filter | ~3–8 s/image |

HuggingFace Spaces free tier runs on CPU. Upgrade to a GPU Space for SD inpainting.

---

## 7. File structure

```
garment_ironing/
├── app.py                      ← Gradio UI + entry point
├── requirements.txt
├── README.md                   ← HuggingFace Spaces config + docs
├── train_wrinkle_model.py      ← Training script (run locally)
├── create_example_assets.py    ← Generate placeholder example images
├── assets/                     ← Example images for the UI
├── models/
│   ├── __init__.py
│   ├── segmentation.py         ← SegFormer garment segmenter
│   ├── wrinkle_detector.py     ← Multi-scale wrinkle detection
│   └── ironing_pipeline.py     ← Classical + SD inpainting ironer
└── utils/
    ├── __init__.py
    ├── image_utils.py          ← Resize, compare, overlay helpers
    └── analysis.py             ← Wrinkle scoring + report generation
```
