# Experimental Results Template

> **Note on Academic Integrity:**  
> In strict accordance with scientific standards, this document contains an unpopulated results template. Metrics must only be entered after executing actual training and evaluation runs (`python src/evaluate.py`). No fabricated or estimated accuracy figures are included.

---

## 1. Overall Model Comparison on Independent Test Split

*All models must be evaluated on the identical recording-level test split (`outputs/features/test_features.npz`).*

| Architecture | Test Accuracy (%) | Macro F1-Score | Weighted F1-Score | Test Loss | Trainable Parameters | Training Time (min) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DWS-CNN Baseline** | *[Pending Run]* | *[Pending Run]* | *[Pending Run]* | *[Pending Run]* | ~110,000 | *[Pending]* |
| **ViT Baseline** | *[Pending Run]* | *[Pending Run]* | *[Pending Run]* | *[Pending Run]* | ~145,000 | *[Pending]* |
| **Hybrid DWS-CNN + ViT** | *[Pending Run]* | *[Pending Run]* | *[Pending Run]* | *[Pending Run]* | ~275,000 | *[Pending]* |

---

## 2. Per-Class Performance Breakdown (Hybrid Architecture)

| Vessel Category | Precision | Recall | F1-Score | Test Recording Support | Test Segment Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cargo** | *[Pending]* | *[Pending]* | *[Pending]* | 2 recordings | ~65 segments |
| **Passengership** | *[Pending]* | *[Pending]* | *[Pending]* | 3 recordings | ~35 segments |
| **Tanker** | *[Pending]* | *[Pending]* | *[Pending]* | 4 recordings | ~75 segments |
| **Tug** | *[Pending]* | *[Pending]* | *[Pending]* | 1 recording *(Caveat)* | ~66 segments |

*Caveat: Tug metrics represent segments from a single held-out test recording due to dataset availability (3 total recordings).*

---

## 3. Confusion Matrix (Counts)

```text
                  Predicted
             Cargo   Passenger   Tanker   Tug
True Cargo     [ ]      [ ]       [ ]     [ ]
     Pass.     [ ]      [ ]       [ ]     [ ]
     Tanker    [ ]      [ ]       [ ]     [ ]
     Tug       [ ]      [ ]       [ ]     [ ]
```

---

## 4. Key Experimental Observations

*To be recorded after completing training runs:*
* Impact of Depthwise Separable Convolutions on convergence speed and overfitting mitigation.
* Attention weight distribution across spectrogram patches in the Vision Transformer branch.
* Whether the feature fusion layer demonstrates a statistically significant improvement over standalone DWS-CNN and ViT baselines.
