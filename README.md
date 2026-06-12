---
title: AI Virtual Garment Ironing System
emoji: 👔
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 4.19.0
app_file: app.py
pinned: false
license: mit
short_description: Upload a photo — AI removes wrinkles from your clothing
tags:
  - computer-vision
  - image-segmentation
  - garment-processing
  - fashion-ai
  - stable-diffusion
  - wrinkle-removal
---

# 👔 AI Virtual Garment Ironing System

An AI-powered application that detects, segments, and virtually irons clothing in photographs.

## What it does

Upload a photo of yourself wearing clothes. The system will:

1. **Segment** the garment from the background using SegFormer
2. **Detect** wrinkles using multi-scale edge analysis + a CNN
3. **Iron** targeted wrinkle zones using fabric-aware, luminance-only smoothing
4. **Preserve** logos, patterns, face, background, lighting, and structural folds
5. **Show** a before/after comparison and detailed analysis report

## Features

- Adjustable ironing intensity (Light / Medium / Professional Press)
- Fabric-specific processing (Cotton, Linen, Silk, Denim, Polyester, Wool)
- Clothing type hints for better segmentation
- Wrinkle heatmap visualisation
- Batch processing
- Detailed analysis report per image

## Supported garments

Shirts, t-shirts, blouses, trousers, dresses, jackets, blazers, sarees, kurtas, and more.

## Running locally

```bash
git clone https://huggingface.co/spaces/your-username/garment-ironing
cd garment-ironing
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860 in your browser.

## Technical stack

| Component | Technology |
|-----------|-----------|
| Garment segmentation | SegFormer (mattmdjaga/segformer_b2_clothes) |
| Wrinkle detection | Multi-scale Canny + Gabor + custom CNN |
| Ironing | Targeted bilateral/median smoothing + luminance correction |
| UI | Gradio 4.x |

The deterministic ironing path is used by default on both CPU and GPU because
it preserves garment identity, logos, color, skin, and background. Diffusion
inpainting remains an explicit opt-in for experimentation.

## License

MIT — free to use and modify.
