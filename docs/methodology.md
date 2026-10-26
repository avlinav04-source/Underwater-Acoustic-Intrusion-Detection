# Methodology

## 1. Research Problem & Objective

Passive acoustic intrusion detection in littoral zones, marine protected areas, and harbor approaches requires automated recognition of surface vessels by their radiated underwater acoustic signatures. The objective of this work is to establish a rigorous, leakage-free benchmark prototype for classifying the four vessel categories in the DeepShip dataset (**Cargo, Passengership, Tanker, and Tug**) using a hybrid deep neural network combining Depthwise Separable Convolutions (DWS-CNN) and Vision Transformers (ViT).

---

## 2. Audio Preprocessing Pipeline

Underwater acoustic hydrophone data exhibits wide dynamic range and ambient oceanic background noise (wave turbulence, biological sound, wind-sea interactions). The preprocessing steps are designed to maximize signal stability and computational efficiency:

1. **Format Validation:** Audio files are inspected for container validity, channel layout, and frame consistency using `SoundFile`.
2. **Channel Downmixing:** Multi-channel recordings are downmixed to single-channel (mono) via mean averaging:
   $$x_{\text{mono}}(t) = \frac{1}{C}\sum_{c=1}^C x_c(t)$$
3. **Resampling:** DeepShip audio is natively sampled at $32{,}000\text{ Hz}$. Because vessel propeller cavitation and machinery noise predominantly concentrate below $8{,}000\text{ Hz}$, the audio is resampled to $f_s = 16{,}000\text{ Hz}$ using high-quality anti-aliasing interpolation (`librosa.resample`). This cuts downstream matrix sizes in half without sacrificing diagnostic acoustic frequencies.
4. **Safe Peak Normalization:** Audio amplitudes are scaled to $0.95$ peak amplitude while guarding against zero-division on silent segments:
   $$\hat{x}(t) = \begin{cases} \frac{x(t)}{\max_k |x(k)|} \times 0.95 & \text{if } \max_k |x(k)| > 10^{-7} \\ 0 & \text{otherwise} \end{cases}$$
5. **Fixed-Window Segmentation:** Hydrophone audio recordings are continuous (often 3 to 4 minutes in length). Slicing signals into fixed $3.0\text{-second}$ non-overlapping segments ($48{,}000$ samples) allows dense feature sampling while preserving stationary acoustic patterns.

---

## 3. Log-Mel Spectrogram Extraction

Raw waveforms are converted into 2D time-frequency representations via the Short-Time Fourier Transform (STFT):

* **FFT Size ($N_{\text{fft}}$):** $1{,}024$ samples ($64\text{ ms}$ analysis window).
* **Hop Length ($H$):** $512$ samples ($32\text{ ms}$ frame shift, $50\%$ overlap).
* **Mel Filterbank ($N_{\text{mel}}$):** $64$ triangular Mel frequency bands spanning $20\text{ Hz}$ to $8{,}000\text{ Hz}$.
* **Decibel Conversion:** Dynamic range is compressed to log scale relative to peak power with an $80\text{ dB}$ floor:
  $$S_{\text{dB}}(m, t) = \max\left(10 \log_{10}\frac{|X(m, t)|^2}{\max |X|^2}, -80.0\right)$$
* **Min-Max Normalization:** Scaled linearly to $[0.0, 1.0]$ for optimal gradient behavior during neural network backpropagation:
  $$\tilde{S}(m, t) = \frac{S_{\text{dB}}(m, t) - (-80.0)}{80.0}$$
* **Tensor Output:** Each $3.0\text{-second}$ audio window yields a fixed matrix of shape:
  $$\mathbf{X}_i \in \mathbb{R}^{64 \times 94 \times 1}$$

---

## 4. Prevention of Data Leakage (Recording-Level Splitting)

### 4.1 The Data Leakage Risk in Acoustic Classification

A common flaw in audio ML literature is random segment-level shuffling, where overlapping or adjacent 3-second segments from the *same* continuous recording appear in both training and test sets. Because oceanic background noise, water temperature profile, and hydrophone placement are static across a single recording, random segment splitting allows models to achieve artificially inflated test scores simply by recognizing the specific background recording environment rather than the vessel's acoustic signature.

### 4.2 Strict Recording-Level Isolation

To ensure strict zero-leakage:
1. Data splitting is performed **exclusively at the original recording level** *before* any window segmentation.
2. Let $\mathcal{R}_{\text{train}}$, $\mathcal{R}_{\text{val}}$, and $\mathcal{R}_{\text{test}}$ be the disjoint sets of original hydrophone recordings:
   $$\mathcal{R}_{\text{train}} \cap \mathcal{R}_{\text{val}} = \emptyset, \quad \mathcal{R}_{\text{train}} \cap \mathcal{R}_{\text{test}} = \emptyset, \quad \mathcal{R}_{\text{val}} \cap \mathcal{R}_{\text{test}} = \emptyset$$
3. All segments inherit their parent recording identifier. Segments in the test set belong solely to recordings never seen during training or hyperparameter tuning.

### 4.3 Handling the Tug Minority Class

The DeepShip dataset contains an extreme class imbalance:
* Cargo: 12 recordings
* Passengership: 20 recordings
* Tanker: 28 recordings
* **Tug: 3 recordings**

With only 3 total Tug recordings, conventional stratified 70/15/15 splitting would allocate zero recordings to either validation or test splits. To guarantee representation across all phases while preventing leakage:
* Exactly **1 Tug recording** is assigned to training.
* Exactly **1 Tug recording** is assigned to validation.
* Exactly **1 Tug recording** is assigned to the test set.

---

## 5. Class-Weighted Loss Balancing

Because Tanker ($20$ training recordings) and Passengership ($14$ training recordings) dominate the training pool relative to Tug ($1$ training recording), standard empirical risk minimization would bias the model toward majority classes. 

Balanced class weights $w_c$ are incorporated into the sparse categorical cross-entropy loss:

$$w_c = \frac{N_{\text{total}}}{C \cdot N_c}$$

For the verified training split:
* **Cargo ($c=0$):** $w_0 = 1.344$
* **Passengership ($c=1$):** $w_1 = 0.768$
* **Tanker ($c=2$):** $w_2 = 0.537$
* **Tug ($c=3$):** $w_3 = 10.750$

The minority Tug class receives a $10.75\times$ gradient weight, penalizing misclassifications on scarce classes during backpropagation.

---

## 6. Training & Evaluation Protocol

* **Loss Function:** Class-weighted Sparse Categorical Cross-Entropy.
* **Optimizer:** Adam with initial learning rate $\eta = 5 \times 10^{-4}$ and weight decay $10^{-4}$.
* **Learning Rate Scheduling:** `ReduceLROnPlateau` decays $\eta$ by factor $0.5$ if validation loss stagnates for 4 epochs (minimum $\eta = 10^{-5}$).
* **Early Stopping:** Monitored on validation loss with patience of 8 epochs.
* **Model Checkpointing:** Saves only the checkpoint with the lowest validation loss.
* **Test Set Evaluation:** Evaluated exactly once on the held-out test split, reporting Accuracy, Per-Class Precision, Recall, F1-Score, Macro-F1, Weighted-F1, and Confusion Matrices.
