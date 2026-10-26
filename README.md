<<<<<<< HEAD
# Underwater Acoustic Intrusion Detection

A Deep Learning system for passive underwater acoustic vessel classification and intrusion detection prototyping using the **DeepShip** dataset. Powered by a **Hybrid Depthwise Separable CNN (DWS-CNN) and Vision Transformer (ViT)** architecture.

---

## 🌊 1. Project Overview

Underwater acoustic signals provide vital intelligence for maritime security, port defense, marine sanctuary protection, and traffic monitoring. Passive hydrophones capture radiated underwater sound without revealing sensor locations.

This academic prototype provides an end-to-end, reproducible machine learning pipeline that transforms raw hydrophone audio into normalized **Log-Mel Spectrograms**, isolates recordings to guarantee **zero data leakage**, extracts multi-scale spatial and temporal attention features, and classifies vessel acoustics into four target categories:
* 🚢 **Cargo**
* 🛳️ **Passengership**
* 🛢️ **Tanker**
* ⚓ **Tug**

---

## 🚀 2. Key Features

* **Real Deep Learning Architectures:**
  * **DWS-CNN Branch:** True Depthwise Separable Convolutions (`SeparableConv2D`) for parameter-efficient spatial-frequency acoustic texture extraction.
  * **Vision Transformer (ViT) Branch:** Patch extraction ($8 \times 8$), learnable positional embeddings, and Multi-Head Self-Attention for modeling long-range engine cadences.
  * **Latent Fusion:** Concatenation and non-linear dense projection with Batch Normalization and Dropout.
* **Strict Zero-Leakage Guarantee:** Splitting is enforced at the **original recording level** *before* windowing. Segments from the same recording never cross splits.
* **Minority Class Loss Balancing:** Automatically computes and applies balanced class weights ($10.75\times$ on the scarce Tug category) to prevent majority bias.
* **Interactive Gradio Web Application:** Real-time audio upload, microphone input, audio playback, confidence distribution charts, and illustrative harbor security rules.
* **26 Automated Unit Tests:** Covering file integrity, corrupted audio handling, amplitude normalization, windowing math, split isolation, and model shapes.
* **Honest Academic Context:** Explicitly distinguishes physical vessel classification from operational intrusion detection.

---

## 🛠️ 3. Technology Stack

* **Programming Language:** Python 3.10 - 3.13
* **Audio Signal Processing:** `librosa`, `soundfile`, `scipy`
* **Deep Learning Framework:** `TensorFlow` / `Keras`
* **Data Processing & ML Metrics:** `numpy`, `pandas`, `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`
* **Web Interface:** `gradio`
* **Configuration:** `pyyaml`
* **Testing:** `unittest`, `pytest`

---

## 📁 4. Project Directory Structure

```text
Underwater_Acoustic_Intrusion_Detection/
│
├── README.md                          # Project overview and quickstart guide
├── requirements.txt                   # Project package dependencies
├── .gitignore                         # Excludes raw audio, model weights, and logs
├── .env.example                       # Environment variable path template
│
├── config/
│   └── config.yaml                    # Master audio, model, split, and training configuration
│
├── src/
│   ├── __init__.py                    # Source package initializer
│   ├── inspect_dataset.py             # Basic dataset scanner and category validator
│   ├── analyze_audio.py               # Comprehensive audio integrity & duration analysis
│   ├── audio_preprocessing.py         # Resampling, mono downmix, peak normalization, windowing
│   ├── feature_extraction.py          # Log-Mel spectrogram generation and split caching
│   ├── dataset_split.py               # Recording-level train/val/test splitting (zero leakage)
│   ├── model_dws_cnn.py               # Depthwise Separable CNN branch
│   ├── model_vit.py                   # Vision Transformer branch (patches + attention)
│   ├── model_hybrid.py                # Hybrid dual-branch architecture with latent fusion
│   ├── train.py                       # Training pipeline with class weights and callbacks
│   ├── evaluate.py                    # Independent test set evaluation and confusion matrices
│   ├── predict.py                     # Inference engine and illustrative policy evaluator
│   └── utils.py                       # Seed setting, config loading, and JSON serialization
│
├── app/
│   ├── __init__.py                    # App package initializer
│   └── gradio_app.py                  # Gradio interactive web browser interface
│
├── notebooks/
│   └── training_guide.md              # Google Colab / Kaggle free GPU training instructions
│
├── tests/
│   ├── test_dataset.py                # Audio integrity, scanner, and split leakage tests
│   ├── test_preprocessing.py          # Resampling, normalization, windowing, and Mel tests
│   ├── test_models.py                 # Neural network shapes, patch math, and forward passes
│   └── test_prediction.py             # Inference pipeline and short audio rejection tests
│
├── outputs/                           # Generated reports, confusion matrices, and split manifest
│   ├── audio_metadata.csv             # Verified technical metadata for all 63 recordings
│   ├── dataset_inspection_report.json # Duration statistics per class
│   └── dataset_split.json             # Leak-free split manifest with class weights
│
├── models/                            # Trained model checkpoints (.keras) and metadata
│   └── .gitkeep
│
└── docs/                              # Comprehensive technical documentation
    ├── system_architecture.md         # Architecture diagrams, data flow, and fusion logic
    ├── methodology.md                 # Signal processing, zero-leakage math, and training loss
    ├── installation_guide.md          # Setup guide for Windows PowerShell and Colab
    ├── user_guide.md                  # Complete CLI and web application user manual
    ├── limitations.md                 # DeepShip dataset limits, Tug scarcity, and domain shifts
    └── results_template.md            # Scientific experimental results template
```

---

## 📊 5. Verified Dataset Inspection Findings

Dataset inspection was performed across the local **DeepShip** dataset:

* **Total Recordings:** 63 intact `.wav` files (0 corrupted, 0 empty).
* **Sampling Rate:** 100% consistent at 32,000 Hz.
* **Channels:** 100% single-channel (mono).
* **Audio Encoding:** 32-bit floating-point (`FLOAT`).
* **Total Audio Duration:** 1.59 hours (5,723 seconds).

### Category Distribution & Duration Breakdown

| Vessel Category | Verified Files | Total Duration | Mean Duration | Min Duration | Max Duration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cargo** | 12 | 38.7 min | 193.7 s | 183.0 s | 202.0 s |
| **Passengership** | 20 | 19.1 min | 57.3 s | 6.0 s | 204.0 s |
| **Tanker** | 28 | 27.7 min | 59.3 s | 10.0 s | 174.0 s |
| **Tug** | **3** | 9.9 min | 197.7 s | 190.0 s | 203.0 s |
| **Total** | **63** | **1.59 hrs** | **90.8 s** | **6.0 s** | **204.0 s** |

---

## 🔒 6. Zero Data Leakage Partitioning

To avoid the segmented acoustic data leakage common in published literature, data is partitioned **strictly at the recording level**:

* **Training Split:** 43 recordings (8 Cargo, 14 Passengership, 20 Tanker, 1 Tug).
* **Validation Split:** 10 recordings (2 Cargo, 3 Passengership, 4 Tanker, 1 Tug).
* **Test Split:** 10 recordings (2 Cargo, 3 Passengership, 4 Tanker, 1 Tug).
* **Recording Overlap Check:** 100% disjoint. No hydrophone recording or segment appears in more than one partition.

---

## ⚡ 7. Quickstart Commands (Windows PowerShell)

### 7.1 Inspect Dataset & Verify Metadata
```powershell
python src/analyze_audio.py
```

### 7.2 Run Data Splitting & Leakage Audit
```powershell
python src/dataset_split.py
```

### 7.3 Run Automated Test Suites
```powershell
python -m unittest discover tests/
```

### 7.4 Train the Model
```powershell
# Train the primary Hybrid architecture
python src/train.py --model_type hybrid --epochs 35 --batch_size 16

# Or train on Google Colab (GPU): See notebooks/training_guide.md
```

### 7.5 Evaluate on Unseen Test Split
```powershell
python src/evaluate.py --model_path models/best_hybrid_model.keras
```

### 7.6 Classify Audio via Command-Line
```powershell
python src/predict.py "C:\path\to\sample.wav"
```

### 7.7 Launch the Interactive Gradio Web Interface
```powershell
python app/gradio_app.py
```
Open your browser at: `http://127.0.0.1:7860`.

---

## 🛡️ 8. Honest Academic Clarification on Intrusion Detection

1. **Vessel Classification vs. Intrusion Detection:**
   The neural network classifies acoustic vessel signatures (Cargo, Passengership, Tanker, Tug). It **does not** intrinsically detect illegal intrusions. An operational intrusion alert is a business/security policy rule applied on top of classifications (e.g., verifying whether an identified Tanker holds authorization to enter a Marine Protected Area).
2. **Tug Category Limitation:**
   With only 3 physical recordings available in DeepShip, the held-out test set contains segments from **a single Tug recording**. True generalization on Tugboats requires further acoustic data collection.
3. **No Fabricated Metrics:**
   In accordance with academic standards, results tables in `docs/results_template.md` are reserved exclusively for measured experimental runs.

---

## 📖 9. Documentation Index

* [System Architecture](docs/system_architecture.md)
* [Methodology & Signal Processing](docs/methodology.md)
* [Installation Guide](docs/installation_guide.md)
* [User Guide](docs/user_guide.md)
* [Limitations & Constraints](docs/limitations.md)
* [Results Template](docs/results_template.md)
* [Google Colab GPU Guide](notebooks/training_guide.md)
=======
# Underwater-Acoustic-Intrusion-Detection
Underwater vessel classification using a hybrid Depthwise Separable CNN (DWS-CNN) and Vision Transformer (ViT) architecture with acoustic signal processing and deep learning.
>>>>>>> f3f22fa129028bf8f584c5e58903675e77336ce2
