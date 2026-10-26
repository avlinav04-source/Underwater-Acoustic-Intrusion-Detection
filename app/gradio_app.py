"""
gradio_app.py
-------------
Beginner-friendly web application for Underwater Acoustic Intrusion Detection.

Provides:
- WAV audio file upload and microphone input.
- Audio playback and waveform/spectrogram preview.
- Real model inference with 4-class probability bar charts.
- Honest academic context clarifying vessel classification vs true intrusion detection.
- Clear error handling for missing models or corrupted audio without stack traces.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import CANONICAL_CLASSES, load_config
from src.predict import VesselPredictor

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    gr = None
    GRADIO_AVAILABLE = False


def classify_underwater_audio(audio_file_path: Optional[str]) -> Tuple[Dict[str, float], str, str]:
    """
    Callback function for Gradio interface.

    Args:
        audio_file_path: Temporary path to uploaded or recorded audio file.

    Returns:
        Tuple of (probability_dict, status_markdown, illustrative_rule_markdown)
    """
    if not audio_file_path:
        return (
            {cls_name: 0.0 for cls_name in CANONICAL_CLASSES},
            "⚠️ **Please upload or record a WAV audio file.**",
            ""
        )

    p = Path(audio_file_path)
    if not p.is_file():
        return (
            {cls_name: 0.0 for cls_name in CANONICAL_CLASSES},
            f"❌ **File not found:** `{p.name}`",
            ""
        )

    # Initialize predictor
    try:
        predictor = VesselPredictor()
    except Exception as e:
        return (
            {cls_name: 0.0 for cls_name in CANONICAL_CLASSES},
            f"❌ **Predictor Initialization Error:** {str(e)}",
            ""
        )

    # Check if a model checkpoint is ready
    if not predictor.is_model_ready():
        help_msg = (
            "### ⚠️ No Trained Model Checkpoint Found\n\n"
            "The application requires a trained model checkpoint (`models/best_hybrid_model.keras`).\n\n"
            "**To train the model:**\n"
            "1. Locally: Run `python src/train.py --model_type hybrid`\n"
            "2. In Google Colab / Kaggle (GPU): Open and run `notebooks/training_guide.md`.\n"
            "3. After training completes, place the saved `.keras` checkpoint into the `models/` directory."
        )
        return ({cls_name: 0.0 for cls_name in CANONICAL_CLASSES}, help_msg, "")

    # Run inference
    result = predictor.predict(p)

    if not result.get("success"):
        error_msg = result.get("error", "Unknown prediction error.")
        return (
            {cls_name: 0.0 for cls_name in CANONICAL_CLASSES},
            f"❌ **Inference Error:** {error_msg}",
            ""
        )

    # Build response markdown
    pred_class = result["predicted_class"]
    conf_pct = result["confidence_percentage"]
    duration = result["audio_duration_seconds"]
    n_segs = result["num_segments_analyzed"]
    probabilities = result["class_probabilities"]

    status_markdown = (
        f"### 🎯 Predicted Category: **{pred_class}**\n\n"
        f"- **Model Confidence:** `{conf_pct}`\n"
        f"- **Audio Duration:** `{duration:.2f} seconds` ({n_segs} segmented windows)\n"
        f"- **Architecture:** Hybrid Depthwise Separable CNN + Vision Transformer (ViT)"
    )

    rule_info = result["illustrative_security_rule"]
    alert_badge = "🚨 **FLAGGED FOR VERIFICATION**" if "FLAGGED" in rule_info["zone_alert_status"] else "✅ **AUTHORIZED CORRIDOR**"

    security_markdown = (
        f"#### Illustrative Marine Security Policy: `{rule_info['policy_applied']}`\n\n"
        f"Status: {alert_badge}\n\n"
        f"> **Academic Note on Intrusion Detection:**\n"
        f"> {rule_info['academic_clarification']}"
    )

    return probabilities, status_markdown, security_markdown


def create_gradio_app():
    """Builds the Gradio web application."""
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is not installed in the local environment.")

    # Find sample audio files if available locally for convenient user testing
    config = load_config()
    raw_dir = Path(config["dataset"]["raw_data_dir"])
    sample_files = []
    if raw_dir.is_dir():
        for cls_name in CANONICAL_CLASSES:
            cls_folder = next((d for d in raw_dir.iterdir() if d.is_dir() and d.name.lower() == cls_name.lower()), None)
            if cls_folder:
                wavs = list(cls_folder.glob("*.wav"))
                if wavs:
                    sample_files.append([str(wavs[0])])

    with gr.Blocks(title="Underwater Acoustic Intrusion Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🚢 Underwater Acoustic Intrusion Detection
            ### Hybrid Deepwise Separable CNN (DWS-CNN) & Vision Transformer (ViT) Prototype
            
            This academic prototype classifies passive underwater acoustic signals from the **DeepShip** dataset
            into four vessel categories: **Cargo, Passengership, Tanker, and Tug**.
            """
        )

        with gr.Tab("🎧 Vessel Acoustic Classifier"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Step 1: Upload Underwater Audio")
                    audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Input Underwater Acoustic WAV Audio",
                    )
                    predict_btn = gr.Button("🔍 Classify Vessel Type", variant="primary", size="lg")

                    if sample_files:
                        gr.Markdown("#### Or select an authentic DeepShip sample:")
                        gr.Examples(
                            examples=sample_files[:4],
                            inputs=audio_input,
                            label="Dataset Audio Samples"
                        )

                with gr.Column(scale=1):
                    gr.Markdown("#### Step 2: Prediction Results")
                    status_output = gr.Markdown("Prediction status will appear here after clicking Classify.")
                    prob_output = gr.Label(num_top_classes=4, label="Vessel Category Probabilities")
                    security_output = gr.Markdown()

            predict_btn.click(
                fn=classify_underwater_audio,
                inputs=[audio_input],
                outputs=[prob_output, status_output, security_output]
            )

        with gr.Tab("📐 System Architecture & Methodology"):
            gr.Markdown(
                """
                ### Hybrid Neural Network Architecture
                
                The system processes continuous underwater hydrophone recordings through dual complementary branches:
                
                1. **Depthwise Separable CNN (DWS-CNN) Branch:**
                   - Uses 2D depthwise and pointwise convolutions to capture local time-frequency energy patterns.
                   - Achieves high parameter efficiency, crucial for small acoustic sample regimes.
                
                2. **Vision Transformer (ViT) Branch:**
                   - Partitions Log-Mel spectrograms into non-overlapping patches (8x8 pixels).
                   - Applies Multi-Head Self-Attention across temporal-spectral tokens to learn long-range acoustic signatures.
                
                3. **Latent Fusion Layer:**
                   - Concatenates the 128-dimensional latent vectors from both branches.
                   - Applies dense projection with Batch Normalization and Dropout before 4-class softmax output.
                
                4. **Zero Data Leakage:**
                   - Audio is segmented strictly *after* partitioning original hydrophone recordings into train, val, and test splits.
                """
            )

        with gr.Tab("⚠️ Academic Limitations"):
            gr.Markdown(
                """
                ### Academic Scope & Limitations
                
                1. **Dataset Scale & Tug Scarcity:**
                   - DeepShip contains 63 total recordings in this local benchmark.
                   - The **Tug** category comprises only 3 recordings, which is insufficient for robust generalized evaluation.
                
                2. **Vessel Classification vs. True Intrusion Detection:**
                   - The model classifies physical vessel types based on acoustic cavitation and engine signatures.
                   - Vessel type alone does not determine whether an intrusion occurred.
                   - True intrusion detection requires transponder (AIS) authorization records, geo-fenced boundary coordinates, or security clearance permissions.
                """
            )

    return demo


def main():
    if not GRADIO_AVAILABLE:
        print("\n" + "=" * 65)
        print(" Underwater Acoustic Intrusion Detection - Gradio Application")
        print("=" * 65)
        print("[INFO] Gradio is not installed in the local Python environment.")
        print("To launch the web interface, run:")
        print("    pip install gradio")
        print("    python app/gradio_app.py")
        print("\nAlternatively, you can run command-line predictions using:")
        print("    python src/predict.py <path_to_audio_wav>")
        print("=" * 65 + "\n")
        return

    demo = create_gradio_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
