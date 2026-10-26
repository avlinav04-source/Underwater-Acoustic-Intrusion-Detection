# Google Colab & Kaggle GPU Training Guide

This guide provides step-by-step instructions for executing the model training pipeline on **Google Colab** (Free NVIDIA T4 GPU) or **Kaggle**, generating the trained model checkpoint (`best_hybrid_model.keras`), and transferring it to your local Windows environment.

---

## 1. Why Train on Google Colab or Kaggle?

While feature extraction, dataset inspection, and inference run fast on a standard Windows CPU, training the **Vision Transformer** and **Hybrid DWS-CNN + ViT** over 35 epochs benefits dramatically from GPU tensor core acceleration:
* **CPU Training Time (Local):** ~15 to 30 minutes
* **GPU Training Time (Google Colab T4):** ~2 to 4 minutes

---

## 2. Google Colab Execution Workflow

### Step 1: Open Google Colab with GPU Runtime
1. Navigate to [https://colab.research.google.com/](https://colab.research.google.com/).
2. Create a **New Notebook**.
3. In the top menu, select: **Runtime** -> **Change runtime type** -> Select **T4 GPU** -> Click **Save**.

### Step 2: Mount Google Drive or Upload Project
You can either upload your project folder as a ZIP file or clone your Git repository:

```python
# In a Colab code cell:
from google.colab import files
import os

# Upload your project zip or DeepShip dataset
# Alternatively, clone your repository:
# !git clone https://github.com/your-username/Underwater_Acoustic_Intrusion_Detection.git
# %cd Underwater_Acoustic_Intrusion_Detection
```

### Step 3: Install Required Audio Packages
Colab comes with TensorFlow pre-installed! You only need to install audio processing libraries:

```python
!pip install librosa soundfile pyyaml
```

### Step 4: Run Audio Preprocessing and Zero-Leakage Splitting

```python
# 1. Verify dataset path and generate audio metadata
!python src/analyze_audio.py --data_dir "/path/to/DeepShip-main"

# 2. Perform zero-leakage recording-level split
!python src/dataset_split.py
```

### Step 5: Train the Models

Train the primary **Hybrid DWS-CNN + ViT** architecture:

```python
!python src/train.py --model_type hybrid --epochs 35 --batch_size 16
```

*(Optional: Train standalone baselines for comparison)*
```python
!python src/train.py --model_type dws_cnn --epochs 35
!python src/train.py --model_type vit --epochs 35
```

### Step 6: Evaluate on the Held-Out Test Set

```python
!python src/evaluate.py --model_path models/best_hybrid_model.keras
```

### Step 7: Download Checkpoint to Your Local Windows Machine

```python
from google.colab import files

# Download the trained model checkpoint
files.download("models/best_hybrid_model.keras")

# Download the evaluation report and confusion matrix
files.download("outputs/evaluation_report.json")
files.download("outputs/confusion_matrix_best_hybrid_model.png")
```

---

## 3. Integrating the Downloaded Model on Windows

Once downloaded to your computer:
1. Move `best_hybrid_model.keras` into your local project's `models/` directory:
   `c:\Users\avlin\OneDrive\Documents\Underwater_Acoustic_Intrusion_Detection\models\`
2. Launch the Gradio web application locally:
   ```powershell
   python app/gradio_app.py
   ```
3. Your local web application will automatically detect and load the trained model!
