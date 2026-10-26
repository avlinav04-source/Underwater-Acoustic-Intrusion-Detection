# Installation & Setup Guide

This guide provides instructions for setting up the **Underwater Acoustic Intrusion Detection** environment on Windows, running tests, executing training locally or in Google Colab/Kaggle, and launching the Gradio web interface.

---

## 1. System Requirements

* **Operating System:** Windows 10/11 (or Linux/macOS)
* **Python Version:** Python 3.10 - 3.13
* **Recommended RAM:** 8 GB or higher
* **Storage:** ~500 MB for repository and dependencies; ~2 GB for DeepShip audio dataset.
* **GPU (Optional):** NVIDIA GPU with CUDA support for fast training. A GPU is **not required** for audio inspection, feature extraction, testing, or running inference.

---

## 2. Windows Local Setup

### Step 2.1: Clone or Open Project in PowerShell

Open Windows PowerShell and navigate to the project directory:

```powershell
cd c:\Users\avlin\OneDrive\Documents\Underwater_Acoustic_Intrusion_Detection
```

### Step 2.2: (Recommended) Create a Virtual Environment

Isolate the project packages in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 2.3: Install Core Audio & Scientific Dependencies

Install the core libraries for dataset inspection, audio processing, and tests:

```powershell
pip install -r requirements.txt
```

Core libraries installed include:
* `soundfile` & `librosa` (audio loading, STFT, Mel spectrograms)
* `numpy` & `scipy` (numerical operations)
* `pandas` (metadata & logging)
* `scikit-learn` (metrics & data splitting)
* `matplotlib` & `seaborn` (visualization)
* `pyyaml` (configuration)

---

## 3. Deep Learning & Web App Dependencies

### Step 3.1: Install TensorFlow / Keras

To train models or run inference with trained checkpoints:

```powershell
pip install tensorflow
```

### Step 3.2: Install Gradio & Pytest

To launch the web interface and run test suites with pytest:

```powershell
pip install gradio pytest
```

---

## 4. Configuring Dataset Path

By default, the project is configured to look for the local DeepShip dataset at:
`C:\Users\avlin\Downloads\UNDERWATER PROJECT\DeepShip-main`

To adjust this path, either:
1. Edit `config/config.yaml`:
   ```yaml
   dataset:
     raw_data_dir: "C:\\path\\to\\your\\DeepShip-main"
   ```
2. Or set the environment variable:
   ```powershell
   $env:DEEPSHIP_DATA_DIR = "C:\path\to\your\DeepShip-main"
   ```

---

## 5. Verification & Running Unit Tests

Run the full automated test suite using Python's built-in `unittest` runner:

```powershell
python -m unittest discover tests/
```

Or if `pytest` is installed:

```powershell
pytest -v tests/
```

Expected output:
* 26 test cases covering dataset inspection, corrupted file handling, normalization, windowing, log-Mel shapes, zero recording leakage, and edge cases.

---

## 6. GPU Training Options (Google Colab / Kaggle)

If your local computer does not have an NVIDIA GPU, train the models using Google Colab's free T4 GPU:

1. Open Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)
2. Follow the detailed step-by-step instructions in `notebooks/training_guide.md`.
3. Download the generated `best_hybrid_model.keras` checkpoint back to your local `models/` directory for local application deployment!
