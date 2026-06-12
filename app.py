"""
AI Virtual Garment Ironing System
HuggingFace Spaces Entry Point
"""

import gradio as gr
import numpy as np
from PIL import Image
import torch

from models.segmentation import GarmentSegmenter
from models.wrinkle_detector import WrinkleDetector
from models.ironing_pipeline import IroningPipeline
from utils.image_utils import (
    resize_for_processing,
    restore_original_size,
    create_comparison_image,
    overlay_wrinkle_mask,
)
from utils.analysis import analyze_wrinkles, generate_analysis_report

# ── Global model instances (loaded once at startup) ──────────────────────────
segmenter     = None
detector      = None
iron_pipeline = None


def load_models():
    global segmenter, detector, iron_pipeline
    print("Loading models…")
    segmenter     = GarmentSegmenter()
    detector      = WrinkleDetector()
    iron_pipeline = IroningPipeline()
    print("All models loaded.")


# ── Core processing function ──────────────────────────────────────────────────
def process_image(
    input_image: np.ndarray,
    intensity: str,
    fabric_type: str,
    clothing_type: str,
    preserve_folds: bool,
    show_mask: bool,
) -> tuple:
    if input_image is None:
        return None, None, None, "Please upload an image first."

    try:
        if segmenter is None or detector is None or iron_pipeline is None:
            load_models()

        pil_img = Image.fromarray(input_image.astype(np.uint8)).convert("RGB")

        proc_img, scale_info = resize_for_processing(pil_img, max_size=1024)

        seg_result   = segmenter.segment(proc_img, clothing_type=clothing_type)
        garment_mask = seg_result["garment_mask"]
        garment_labels = seg_result["labels"]
        if not garment_mask.any():
            return (
                None, None, None,
                "No garment was detected. Try a clearer photo or choose the clothing type.",
            )

        wrinkle_result = detector.detect(proc_img, garment_mask)
        wrinkle_map    = wrinkle_result["wrinkle_map"]
        wrinkle_score  = wrinkle_result["wrinkle_score"]
        wrinkle_zones  = wrinkle_result["zones"]

        intensity_map = {"Light": 0.45, "Medium": 0.8, "Professional Press": 1.0}
        intensity_val = intensity_map.get(intensity, 0.8)

        ironed_proc = iron_pipeline.iron(
            image=proc_img,
            garment_mask=garment_mask,
            wrinkle_map=wrinkle_map,
            intensity=intensity_val,
            fabric_type=fabric_type,
            preserve_structural_folds=preserve_folds,
        )

        ironed_full   = restore_original_size(ironed_proc, pil_img, scale_info)
        ironed_np     = np.array(ironed_full)
        comparison_np = create_comparison_image(pil_img, ironed_full)

        mask_vis_np = None
        if show_mask:
            mask_vis_np = overlay_wrinkle_mask(pil_img, wrinkle_map, garment_mask)

        analysis = analyze_wrinkles(
            wrinkle_score=wrinkle_score,
            zones=wrinkle_zones,
            labels=garment_labels,
            fabric=fabric_type,
            intensity=intensity,
        )
        report = generate_analysis_report(analysis)
        if seg_result["confidence"] < 0.45:
            report = (
                "Warning: garment segmentation confidence is low. "
                "Use a clearer, well-lit photo for a cleaner result.\n\n" + report
            )

        return ironed_np, comparison_np, mask_vis_np, report

    except Exception as exc:
        import traceback
        print(traceback.format_exc())
        return None, None, None, f"Error during processing:\n{str(exc)}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    css = """
    .title-row { text-align: center; padding: 1.5rem 0 0.5rem; }
    .title-row h1 { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .title-row p  { color: #555; font-size: 1rem; }
    .result-col { background: #f9f9f9; border-radius: 12px; padding: 1rem; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="AI Virtual Garment Ironing") as demo:
        with gr.Row(elem_classes="title-row"):
            gr.Markdown(
                """
                # 👔 AI Virtual Garment Ironing System
                Upload a photo of yourself wearing clothes — AI detects wrinkles,
                segments your garment, and generates a freshly ironed result.
                """
            )

        with gr.Row():

            # ── Left: inputs ──────────────────────────────────────────────────
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload photo",
                    type="numpy",
                    sources=["upload", "webcam"],
                    height=380,
                )

                with gr.Group():
                    gr.Markdown("### ⚙️ Ironing settings")

                    intensity = gr.Radio(
                        choices=["Light", "Medium", "Professional Press"],
                        value="Medium",
                        label="Ironing intensity",
                    )

                    with gr.Row():
                        fabric_type = gr.Dropdown(
                            choices=[
                                "Auto-detect", "Cotton", "Linen", "Silk",
                                "Denim", "Polyester", "Wool", "Synthetic blend",
                            ],
                            value="Auto-detect",
                            label="Fabric type",
                        )
                        clothing_type = gr.Dropdown(
                            choices=[
                                "Auto-detect", "Shirt / Blouse", "T-Shirt",
                                "Trousers", "Dress", "Jacket / Blazer",
                                "Traditional wear", "Saree / Kurta",
                            ],
                            value="Auto-detect",
                            label="Clothing type",
                        )

                    with gr.Row():
                        preserve_folds = gr.Checkbox(
                            value=True,
                            label="Preserve natural body folds",
                            info="Keep folds caused by posture & movement",
                        )
                        show_mask = gr.Checkbox(
                            value=False,
                            label="Show wrinkle heatmap",
                        )

                iron_btn = gr.Button(
                    "✨ Iron this garment",
                    variant="primary",
                    size="lg",
                )

                gr.Examples(
                    examples=[
                        ["assets/example_shirt.jpg",    "Medium",             "Cotton",    "Shirt / Blouse", True,  False],
                        ["assets/example_tshirt.jpg",   "Light",              "Polyester", "T-Shirt",        True,  False],
                        ["assets/example_dress.jpg",    "Professional Press", "Linen",     "Dress",          True,  True ],
                        ["assets/example_trousers.jpg", "Medium",             "Denim",     "Trousers",       False, False],
                    ],
                    inputs=[input_image, intensity, fabric_type, clothing_type, preserve_folds, show_mask],
                    label="Try an example",
                    cache_examples=False,
                )

            # ── Right: outputs ────────────────────────────────────────────────
            with gr.Column(scale=1, elem_classes="result-col"):
                gr.Markdown("### 📸 Results")

                with gr.Tabs():
                    with gr.TabItem("Ironed result"):
                        output_ironed = gr.Image(
                            label="Ironed garment",
                            type="numpy",
                            height=380,
                        )

                    with gr.TabItem("Before / After"):
                        output_comparison = gr.Image(
                            label="Side-by-side comparison",
                            type="numpy",
                            height=380,
                        )

                    with gr.TabItem("Wrinkle map"):
                        output_mask = gr.Image(
                            label="Wrinkle heatmap overlay",
                            type="numpy",
                            height=380,
                        )

                    with gr.TabItem("Analysis"):
                        output_analysis = gr.Textbox(
                            label="Garment analysis report",
                            lines=18,
                            max_lines=30,
                        )
        # ── Batch processing ──────────────────────────────────────────────────
        with gr.Accordion("📦 Batch processing (multiple images)", open=False):
            gr.Markdown(
                "Upload multiple images for batch ironing. "
                "All images will use the settings chosen above."
            )
            batch_files   = gr.File(file_count="multiple", label="Upload images")
            batch_btn     = gr.Button("Process batch", variant="secondary")
            batch_gallery = gr.Gallery(label="Batch results", columns=3, height=400)

        # ── How it works ──────────────────────────────────────────────────────
        with gr.Accordion("ℹ️ How it works", open=False):
            gr.Markdown(
                """
                **Processing pipeline**

                1. **Garment segmentation** — SegFormer isolates clothing pixels
                2. **Wrinkle detection** — multi-scale edge + CNN maps crease density
                3. **Structural fold preservation** — body-movement folds are kept
                4. **Fabric-aware smoothing** — targeted luminance correction
                5. **Texture reconstruction** — logos, patterns, buttons preserved
                6. **Lighting / shadow blending** — composited back onto original scene

                **Tips for best results**
                - Use a well-lit, clear photo
                - Ensure the full garment is visible
                - Higher resolution input → sharper output
                """
            )

        # ── Event handlers ────────────────────────────────────────────────────
        iron_btn.click(
            fn=process_image,
            inputs=[input_image, intensity, fabric_type, clothing_type, preserve_folds, show_mask],
            outputs=[output_ironed, output_comparison, output_mask, output_analysis],
        )

        def process_batch(files, intensity_val, fabric, clothing, preserve, mask):
            if not files:
                return []
            results = []
            for f in files:
                img = np.array(Image.open(f.name).convert("RGB"))
                ironed, _, _, _ = process_image(img, intensity_val, fabric, clothing, preserve, mask)
                if ironed is not None:
                    results.append(ironed)
            return results

        batch_btn.click(
            fn=process_batch,
            inputs=[batch_files, intensity, fabric_type, clothing_type, preserve_folds, show_mask],
            outputs=[batch_gallery],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_models()
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
    )
