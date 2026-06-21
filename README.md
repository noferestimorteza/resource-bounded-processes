# Resource-Bounded Processes

Early-stage prediction of resource-bound type (CPU-bound, I/O-bound, memory-bound, etc.) for threads using real-time Linux kernel event analysis. This repository contains the full pipeline, from raw kernel trace ingestion and feature extraction to labeling and model training accompanying the research paper cited below.

---

## Paper

If you use this code or data, please cite:

> Noferesti, M., Amiri Delouei, F., and Aryan, S. (2026). **Early-Stage Resource-Bound Prediction for Threads Using Real-Time Kernel Event Analysis.** *Journal of Electrical and Computer Engineering Innovations (JECEI)*. doi: [10.22061/jecei.2026.12900.914](https://jecei.sru.ac.ir/article_12573.html)

**BibTeX:**
```bibtex
@article{noferesti2026resource,
  author  = {Noferesti, Morteza and Amiri Delouei, Fatemeh and Aryan, Sina},
  title   = {Early-Stage Resource-Bound Prediction for Threads Using Real-Time Kernel Event Analysis},
  journal = {Journal of Electrical and Computer Engineering Innovations (JECEI)},
  year    = {2026},
  doi     = {10.22061/jecei.2026.12900.914},
  url     = {https://jecei.sru.ac.ir/article_12573.html}
}
```

---

## Overview

Modern operating systems schedule thousands of threads simultaneously. Knowing early — before a thread finishes — whether it is CPU-bound, I/O-bound, or memory-bound enables smarter scheduling, resource allocation, and performance tuning. This project captures low-level Linux kernel events (system calls) via tracing tools, extracts per-thread behavioral features, and trains several machine learning and deep learning classifiers to predict the resource-bound category of each thread.

The pipeline is:

```
Raw kernel trace (CTF / LTTng)
        │
        ▼
  getfeatures.py          ← parse syscall entry/exit events, compute durations
        │
        ▼
  PrepareTopFiveEvents.py ← select top-5 most discriminative syscall categories
        │
        ▼
  ExpandTop5ForML.py      ← expand/pivot features for ML consumption
        │
        ▼
  labelPIDTID.py          ← assign ground-truth resource-bound label per thread
        │
        ▼
  [Model scripts]         ← train & evaluate classifiers
```

---

## Repository Structure

```
resource-bounded-processes/
├── Data/                        # Sample or preprocessed datasets
├── getfeatures.py               # Step 1: Feature extraction from raw kernel traces
├── labelPIDTID.py               # Step 2: Labeling threads by dominant syscall category
├── PrepareTopFiveEvents.py      # Step 3: Select top-5 syscall event categories
├── ExpandTop5ForML.py           # Step 4: Expand features for ML models
├── RF.py                        # Random Forest classifier
├── LightGBM.py                  # LightGBM classifier
├── MLP.py                       # Multi-Layer Perceptron
├── TSF.py                       # Time Series Forest classifier
├── CNNBiLSTM.py                 # CNN + Bidirectional LSTM deep learning model
├── BertFor.py                   # BERT-based sequence classification model
├── Ensemble.py                  # Ensemble of multiple models
└── LICENSE                      # MIT License
```

---

## Getting Started

### Raw Dataset

The raw kernel trace data used in this project is hosted in a dedicated dataset repository:

**[https://github.com/mnoferestibrocku/dataset-repo](https://github.com/mnoferestibrocku/dataset-repo)**

Download or clone it and place the relevant CSV files (e.g., `holedata.csv`, `event_type_with_category.csv`) in the project root before running the pipeline.

### Running the Pipeline

**1. Extract features from the raw trace:**

```bash
python getfeatures.py
# Input:  holedata.csv, event_type_with_category.csv
# Output: system_call_analysis.csv
```

This script parses syscall entry/exit events, matches entry–exit pairs per thread, computes duration statistics (count, total time, average time, max time) grouped by PID, TID, and syscall category.

**2. Label each thread:**

```bash
python labelPIDTID.py
# Input:  system_call_analysis.csv
# Output: PIDTID_labeled.csv
```

Each thread (PID/TID pair) is assigned the syscall category with the highest combined score (weighted by call count and total time), serving as the ground-truth resource-bound label.

**3. Prepare features for ML:**

```bash
python PrepareTopFiveEvents.py
python ExpandTop5ForML.py
```

**4. Train and evaluate models:**

```bash
python RF.py          # Random Forest
python LightGBM.py    # LightGBM
python MLP.py         # Multi-Layer Perceptron
python TSF.py         # Time Series Forest
python CNNBiLSTM.py   # CNN-BiLSTM deep model
python BertFor.py     # BERT-based model
python Ensemble.py    # Ensemble all models
```

---

## Models

| Script | Model | Type |
|---|---|---|
| `RF.py` | Random Forest | Classical ML |
| `LightGBM.py` | LightGBM | Gradient Boosting |
| `MLP.py` | Multi-Layer Perceptron | Neural Network |
| `TSF.py` | Time Series Forest | Time-series ML |
| `CNNBiLSTM.py` | CNN + Bidirectional LSTM | Deep Learning |
| `BertFor.py` | BERT for Sequence Classification | Transformer |
| `Ensemble.py` | Ensemble | Combined |

---

## Input / Output Files

| File | Description |
|---|---|
| `holedata.csv` | Raw LTTng kernel trace exported as tab-separated CSV |
| `event_type_with_category.csv` | Mapping of syscall event names to resource categories |
| `system_call_analysis.csv` | Per-thread syscall statistics (output of `getfeatures.py`) |
| `PIDTID_labeled.csv` | Thread-level resource-bound labels (output of `labelPIDTID.py`) |

---

## License

This project is licensed under the [MIT License](LICENSE).

