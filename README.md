# NiChart_SPARE

Training and inference of SPARE (Spatial Patterns of Abnormality for Recognition of Early X) biomarker scores from brain ROI volumes ([NiChart_DLMUSE](https://github.com/CBICA/NiChart_DLMUSE)) and white matter lesion volumes ([NiChart_DLWMLS](https://github.com/CBICA/NiChart_DLWMLS)).

## Supported SPARE types

| Type | Task | Notes |
|---|---|---|
| `CL` | Classification | Generic binary/multi-class classifier |
| `RG` | Regression | Generic continuous target |
| `AD` | Classification | Alzheimer's disease |
| `BA` | Regression | Brain age |
| `CVM`, `HT`, `T2B`, `SM`, `BMI` | Classification | Cardiovascular & metabolic disease scores; Age/Sex/ICV confounds removed automatically during prep |

## Installation

```bash
# Create and activate a mamba environment (Python 3.10 recommended)
mamba create -n nichart_spare python=3.10 -y
mamba activate nichart_spare

# Clone and install
git clone https://github.com/CBICA/NiChart_SPARE.git
cd NiChart_SPARE
pip install -e .
```

To also install development dependencies (pytest, flake8, etc.):

```bash
pip install -e ".[dev]"
```

## Running the tests

The test suite uses synthetic data generated in memory — no external data files required.

```bash
pytest tests/
```

To run a specific test file:

```bash
pytest tests/test_pipeline.py -v
```

## Workflow

NiChart_SPARE uses a three-stage pipeline. Data is prepared once and can then be used for training or inference.

### Stage 1 — Prepare data

Applies task-specific transforms and produces a standardized CSV:
column 0 = `MRID`, column 1 = target (omit for inference inputs), remaining columns = features.

```bash
# Classification (e.g. AD diagnosis)
NiChart_SPARE prep \
    -t CL \
    -i raw_input.csv \
    -o prepped.csv \
    -tc DX \
    -ic Study,SITE,Sex \
    -kv MRID

# CVM types — Age, Sex, and ICV effects are removed automatically
NiChart_SPARE prep \
    -t CVM \
    -i raw_input.csv \
    -o prepped.csv \
    -tc Disease \
    -ic Study,SITE \
    -kv MRID

# Regression (e.g. brain age)
NiChart_SPARE prep \
    -t RG \
    -i raw_input.csv \
    -o prepped.csv \
    -tc Age \
    -ic Study,SITE,Sex \
    -kv MRID
```

Key options for `prep`:

| Flag | Description |
|---|---|
| `-t` | SPARE type (determines preprocessing applied) |
| `-tc` | Target column name; omit when preparing inference-only data |
| `-ic` | Comma-separated columns to drop (e.g. `Study,SITE,Sex`) |
| `-icv True` | Divide all MUSE ROI volumes by ICV before any other step |
| `--age_col`, `--sex_col` | Column names for Age and Sex (CVM types only, defaults: `Age`, `Sex`) |

### Stage 2 — Train

Reads a prepared CSV and trains an SVM model.

```bash
# Train a classifier
NiChart_SPARE train \
    -t CL \
    -i prepped.csv \
    -mo model.joblib \
    -sk linear \
    -ht True \
    -cf 5 \
    -cb True \
    -v 1

# Train a regressor with bias correction
NiChart_SPARE train \
    -t RG \
    -i prepped.csv \
    -mo model.joblib \
    -sk linear \
    -ht True \
    -cf 5 \
    -bc 1 \
    -v 1
```

Key options for `train`:

| Flag | Description |
|---|---|
| `-t` | SPARE type (determines CL vs RG pipeline) |
| `-sk` | SVM kernel: `linear_fast`, `linear`, `rbf`, `poly`, `sigmoid` |
| `-ht True/False` | Run GridSearchCV hyperparameter tuning (slow but recommended) |
| `-cf` | Cross-validation folds (default: 5; set 0 to skip CV) |
| `-cb True/False` | Class balancing via `class_weight="balanced"` (classification only) |
| `-bc` | Bias correction: `0` = none, `1` = Beheshti et al., `2` = Cole et al. (regression only) |
| `-tw True/False` | Train final model on full dataset after CV (default: True) |

### Stage 3 — Test (inference)

Applies a trained model to new prepared data.

```bash
# Prepare test data (same type as training, target column optional)
NiChart_SPARE prep \
    -t CL \
    -i raw_test.csv \
    -o prepped_test.csv \
    -ic Study,SITE,Sex

# Run inference
NiChart_SPARE test \
    -i prepped_test.csv \
    -m model.joblib \
    -o predictions.csv \
    -kv MRID
```

Output columns: `MRID`, `SPARE_<type>`, and (for classifiers) `SPARE_<type>_decision_function`. If the target column is present in the test data, `GT_<type>` is also written for evaluation.

## Worked example

The following runs a complete classifier pipeline on a hypothetical dataset.
Assume `data/raw.csv` has columns: `MRID`, `Study`, `SITE`, `Sex`, `DX` (0/1), and brain ROI volume columns (`DL_MUSE_Volume_*`).

```bash
# 1. Prepare training data — drop confound columns, standardize format
NiChart_SPARE prep \
    -t CL \
    -i data/raw_train.csv \
    -o data/prepped_train.csv \
    -tc DX \
    -ic Study,SITE,Sex \
    -kv MRID

# 2. Train with hyperparameter tuning and 5-fold CV
NiChart_SPARE train \
    -t CL \
    -i data/prepped_train.csv \
    -mo models/spare_cl.joblib \
    -sk linear \
    -ht True \
    -cf 5 \
    -cb True \
    -v 1

# 3. Prepare test data (same type; omit -tc if no ground truth available)
NiChart_SPARE prep \
    -t CL \
    -i data/raw_test.csv \
    -o data/prepped_test.csv \
    -tc DX \
    -ic Study,SITE,Sex \
    -kv MRID

# 4. Run inference — GT_CL column is written because DX was present in test data
NiChart_SPARE test \
    -i data/prepped_test.csv \
    -m models/spare_cl.joblib \
    -o results/predictions.csv \
    -kv MRID
```

For a **CVM** model the only difference is in the `prep` step: `Age`, `Sex`, and `DL_MUSE_Volume_702` (ICV) must be present in the raw CSV and must **not** be listed in `-ic`, as they are used for residualization and then removed automatically.

```bash
NiChart_SPARE prep \
    -t CVM \
    -i data/raw_cvm.csv \
    -o data/prepped_cvm.csv \
    -tc Disease \
    -ic Study,SITE \
    -kv MRID
```

## SVM kernels

| Kernel | Notes |
|---|---|
| `linear` | Recommended default |
| `linear_fast` | Uses `LinearSVC`/`LinearSVR`; faster but prone to bias |
| `rbf` | Good for non-linear boundaries |
| `poly` | Polynomial kernel |
| `sigmoid` | Sigmoid kernel |

## Publications

- **SPARE-BA**  
  Habes, M. et al. Advanced brain aging: relationship with epidemiologic and genetic risk factors, and overlap with Alzheimer disease atrophy patterns. *Transl Psychiatry* 6, e775 (2016). [doi:10.1038/tp.2016.39](https://doi.org/10.1038/tp.2016.39)

- **SPARE-AD**  
  Davatzikos, C. et al. Longitudinal progression of Alzheimer's-like patterns of atrophy in normal older adults: the SPARE-AD index. *Brain* 132, 2026–2035 (2009). [doi:10.1093/brain/awp091](https://doi.org/10.1093/brain/awp091)

- **diSPARE-AD**  
  Hwang, G. et al. Disentangling Alzheimer's disease neurodegeneration from typical brain ageing using machine learning. *Brain Commun* 4, fcac117 (2022). [doi:10.1093/braincomms/fcac117](https://doi.org/10.1093/braincomms/fcac117)

- **SPARE-CVMs** (HT, HL, T2D, SM, OB)  
  Govindarajan, S.T., Mamourian, E., Erus, G. et al. Machine learning reveals distinct neuroanatomical signatures of cardiovascular and metabolic diseases in cognitively unimpaired individuals. *Nat Commun* 16, 2724 (2025). [doi:10.1038/s41467-025-57867-7](https://doi.org/10.1038/s41467-025-57867-7)
