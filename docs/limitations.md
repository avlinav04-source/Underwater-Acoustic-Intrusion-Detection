# Academic Limitations & Honest Technical Disclosure

This document outlines the operational constraints, dataset limitations, and academic boundaries of the **Underwater Acoustic Intrusion Detection** prototype.

---

## 1. Dataset Scale Constraints

The local benchmark utilizes the public **DeepShip** dataset:
* **Total Recordings:** 63 WAV audio files.
* **Total Audio Duration:** 1.59 hours (95.4 minutes).
* **Sample Rate:** 32 kHz (downsampled to 16 kHz during preprocessing).
* **Environment:** Recordings collected in real-world sea environments under variable sea-state conditions.

While 1.59 hours of audio yields approximately $1{,}900$ discrete 3.0-second segments, the effective statistical sample size is bounded by the **63 distinct physical recordings**. Training deep neural networks (particularly Vision Transformers) on limited distinct recording instances carries a significant risk of memorizing specific recording sessions if recording-level isolation is not strictly enforced.

---

## 2. Severe Class Imbalance & The Tug Minority Limitation

The distribution of recordings across the four classes is heavily skewed:

| Class | Total Recordings | Percentage of Dataset | Assigned Train | Assigned Val | Assigned Test |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tanker** | 28 | 44.4% | 20 | 4 | 4 |
| **Passengership** | 20 | 31.7% | 14 | 3 | 3 |
| **Cargo** | 12 | 19.0% | 8 | 2 | 2 |
| **Tug** | **3** | **4.8%** | **1** | **1** | **1** |

### Key Academic Observation on Tug:
With only **three total recordings** for Tug:
1. Conventional 70/15/15 stratified splitting cannot allocate proportional samples to validation and test sets without starving one set.
2. The prototype assigns exactly 1 Tug recording to Train, 1 to Val, and 1 to Test.
3. Consequently, **the test evaluation for Tug is based on segments derived from a single continuous recording**.
4. **Honest Academic Conclusion:** A single test recording cannot provide a statistically definitive measure of generalized real-world Tugboat acoustic performance across different engine models, propeller types, or sea states. Results for the Tug category must be reported with this explicit caveat.

---

## 3. Underwater Acoustic Propagation & Domain Shift

Passive acoustic detection underwater is governed by complex acoustic propagation phenomena:
* **Multipath Reflection:** Acoustic waves reflect off the sea surface and the seabed, creating constructive and destructive frequency interference (Lloyd's mirror effect).
* **Sound Speed Profile (SSP):** Variations in water temperature, salinity, and depth refract acoustic rays, creating acoustic shadow zones and surface ducting.
* **Background Ambient Noise:** Breaking waves, wind speed, rainfall, and biological sound (snapping shrimp, cetaceans) dynamically alter the signal-to-noise ratio (SNR).
* **Hydrophone Sensitivity:** Differences in hydrophone frequency response, depth placement, and orientation cause domain shifts between recording locations.

A model trained on DeepShip recordings captured in a specific coastal strait may experience performance degradation when deployed in a shallow harbor, arctic environment, or open ocean without domain adaptation.

---

## 4. Vessel Type Classification vs. Intrusion Detection

The machine learning model is an **acoustic vessel classifier** that identifies whether a hydrophone signal originates from a Cargo ship, Passengership, Tanker, or Tugboat.

### What the Model Does:
* Distinguishes acoustic characteristics such as propeller cavitation noise, shaft rotation rate, cylinder firing frequencies, and hull vibrations.

### What the Model Does NOT Do:
* **It does not identify whether a vessel is an "intruder".**
* A Cargo vessel transiting a designated international commercial shipping lane is **authorized**.
* That same Cargo vessel entering a restricted naval anchorage or shallow coral reef reserve without authorization is an **intruder**.
* A small unflagged Tug operating at night inside a port facility with its AIS transponder disabled is a **suspect intruder**.

### Operational Requirements for True Intrusion Detection:
To build a genuine maritime security intrusion detection system, the acoustic classifier must be coupled with:
1. **Automated Identification System (AIS) Transponder Correlation:** Cross-referencing detected acoustic signatures against broadcast AIS maritime identities.
2. **Geo-Fenced Geographic Boundaries:** Rules mapping recognized vessel types to maritime security zones (e.g., Marine Protected Areas, restricted military basins).
3. **Anomaly & Dark Vessel Detection:** Identifying acoustic contacts that emit strong propeller cavitation but broadcast no electronic transponder signals.
