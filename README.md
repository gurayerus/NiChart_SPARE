# NiChart_SPARE

Training and inference of SPARE (Spatial Patterns of Abnormality for Recognition of Early X) biomarker scores from brain ROI volumes ([NiChart_DLMUSE](https://github.com/CBICA/NiChart_DLMUSE)) and white matter lesion volumes ([NiChart_DLWMLS](https://github.com/CBICA/NiChart_DLWMLS)).

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

---

## Workflow overview

```
NiChart_SPARE train -c train_config.json   # prep + train → model + metrics
NiChart_SPARE test  -m model.joblib -i raw.csv -o results/   # prep + inference → predictions
```

Preprocessing (column selection, value remapping, confound residualization) is configured in the JSON file and replayed automatically at inference time using parameters stored inside the model file.

---

## Quick usage — pre-trained models

Models are hosted on Hugging Face and downloaded automatically on first use (cached in `~/.cache/huggingface`).

```bash
# List all available pre-trained tasks
NiChart_SPARE test --list-tasks

# Run inference using a registered task (model downloads automatically)
NiChart_SPARE test -t AD -i raw.csv -o ./results/

# Use a specific model version
NiChart_SPARE test -t AD --version v3.0-raw -i raw.csv -o ./results/

# Use an explicit local model file
NiChart_SPARE test -m /path/to/model.joblib -i raw.csv -o ./results/
```

Output is written to `./results/` (filename controlled by the model's stored config, default: `predictions.csv`). A `prepped.csv` is also saved there showing the data as it was fed to the model.

### Show model info and download URL

```bash
NiChart_SPARE test -t AD --model-info
```

---

## Offline / firewall-restricted environments

1. Find the download URL:
   ```bash
   NiChart_SPARE test -t AD --model-info
   ```

2. On a machine with internet access, download the model file:
   ```bash
   wget <URL printed by --model-info>
   ```

3. Transfer the `.joblib` file to your target machine and run inference directly:
   ```bash
   NiChart_SPARE test -m /path/to/SPARE-AD.joblib -i raw.csv -o ./results/
   ```

---

## Training a custom model

### 1. Prepare a training config (JSON)

All training options live in a single JSON file. Example for SPARE-AD (classification):

```json
{
  "description": {
    "spare_type": "AD",
    "model_tag": "SPARE_AD",
    "model_version": "1.0",
    "run_tag": "SPARE_AD_v1.0"
  },

  "input": {
    "in_dir": ".",
    "in_csv": "raw_data.csv",
    "key_col": "MRID",
    "target_col": "disease",
    "data_cols": ["Age", "Sex", "DL_MUSE_Volume_*"],
    "mappings": {
      "Sex": {"M": 0, "F": 1}
    }
  },

  "pre-processing": {},

  "model": {
    "svm_type": "classification",
    "svm_kernel": "linear",
    "hyperparameter_tuning": true,
    "train_whole": true,
    "cv_fold": 5,
    "class_balancing": true,
    "age_bias_correction": 0,
    "verbose": 1
  },

  "post-processing": {},

  "output": {
    "out_dir": "../output",
    "out_csv": "pred_train.csv",
    "out_cols": ["MRID", "SPARE_AD"],
    "out_model_dir": "model"
  }
}
```

Example for SPARE-BA (regression with age-bias correction):

```json
{
  "description": {
    "spare_type": "BA",
    "model_tag": "SPARE_BA",
    "model_version": "1.0",
    "run_tag": "SPARE_BA_v1.0"
  },

  "input": {
    "in_dir": ".",
    "in_csv": "raw_data.csv",
    "key_col": "MRID",
    "target_col": "Age",
    "data_cols": ["Sex_M", "DL_MUSE_Volume_702", "H_DL_MUSE_Volume_*"]
  },

  "pre-processing": {},

  "model": {
    "svm_type": "regression",
    "svm_kernel": "linear",
    "hyperparameter_tuning": true,
    "train_whole": true,
    "cv_fold": 5,
    "class_balancing": false,
    "age_bias_correction": 1,
    "verbose": 1
  },

  "post-processing": {},

  "output": {
    "out_dir": "../output",
    "out_csv": "pred_train.csv",
    "out_cols": ["MRID", "SPARE_BA"],
    "out_model_dir": "model"
  }
}
```

Example with confound residualization in `pre-processing` (e.g. SPARE-DIABETES):

```json
{
  "pre-processing": {
    "residualization": {
      "age_col": "Age",
      "sex_col": "Sex_M",
      "icv_col": "DL_MUSE_Volume_702"
    }
  }
}
```

When `residualization` is specified, the listed columns must appear in `data_cols`; they are consumed by the residualization step and dropped from the feature set before training.

### 2. Run training

```bash
NiChart_SPARE train -c path/to/train_config.json
```

Add `-n` to attach a free-text note to the run:

```bash
NiChart_SPARE train -c train_config.json -n "baseline linear kernel"
```

Output directory (`out_dir/run_tag/`):

```
output/SPARE_AD_v1.0/
  config.json         ← copy of the config used
  meta.json           ← timestamp, git commit, notes
  prepped.csv         ← preprocessed training data
  train.log           ← timestamped log
  metrics.json        ← per-fold CV scores
  model/
    SPARE_AD_v1.0.joblib
```

### 3. Run inference on new data

```bash
NiChart_SPARE test \
    -m output/SPARE_AD_v1.0/model/SPARE_AD_v1.0.joblib \
    -i new_subjects.csv \
    -o results/
```

The prep config stored inside the model is replayed automatically on `new_subjects.csv`. Output:

```
results/
  prepped.csv         ← data as fed to the model
  pred_train.csv      ← predictions (filename set by out_csv in training config)
```

---

## Config reference

### `description`

| Field | Required | Description |
|---|---|---|
| `spare_type` | yes | Label used for the output score column (`SPARE_<spare_type>`) |
| `model_tag` | yes | Human-readable model name |
| `model_version` | no | Version string (default: `"1.0"`) |
| `run_tag` | yes | Output directory name; must be unique per run |

### `input`

| Field | Required | Description |
|---|---|---|
| `in_csv` | yes | Input CSV filename |
| `in_dir` | no | Directory containing `in_csv`, relative to the config file (default: `.`) |
| `key_col` | no | Unique ID column (default: `MRID`) |
| `target_col` | yes | Column to predict |
| `data_cols` | no | Feature column patterns; trailing `*` matches by prefix (e.g. `"DL_MUSE_Volume_*"`). Columns for preprocessing steps must be included here. |
| `mappings` | no | Value remapping applied before preprocessing, e.g. `{"Sex": {"M": 0, "F": 1}}` |

### `pre-processing`

Optional preprocessing steps applied in declaration order:

| Step | Parameters | Effect |
|---|---|---|
| `residualization` | `age_col`, `sex_col`, `icv_col` | Removes Age/Sex/ICV effects using pre-fitted coefficients; the three columns are dropped after |

### `model`

| Field | Default | Description |
|---|---|---|
| `svm_type` | — | **Required.** `"classification"` or `"regression"` |
| `svm_kernel` | `"linear"` | SVM kernel: `linear`, `linear_fast`, `rbf`, `poly`, `sigmoid` |
| `hyperparameter_tuning` | `true` | Run grid search before final training |
| `train_whole` | `true` | Train a final model on the full dataset after CV |
| `cv_fold` | `5` | Number of cross-validation folds |
| `class_balancing` | `true` | Apply class_weight='balanced' (classification only) |
| `age_bias_correction` | `0` | `0` = none, `1` = Beheshti et al. residual correction, `2` = Cole et al. rescaling (regression only) |
| `verbose` | `1` | Verbosity level (0–3) |

### `post-processing`

Reserved for future use. Leave as `{}`.

### `output`

| Field | Default | Description |
|---|---|---|
| `out_dir` | — | **Required.** Output directory, relative to the config file |
| `out_model_dir` | `""` | Subdirectory inside `out_dir/run_tag/` where the model is saved |
| `out_csv` | `"predictions.csv"` | Predictions filename (used at inference) |
| `out_cols` | all | Columns to include in the output CSV (e.g. `["MRID", "SPARE_AD"]`) |

---

## SVM kernels

| Kernel | Notes |
|---|---|
| `linear` | Recommended default |
| `linear_fast` | Uses `LinearSVC`/`LinearSVR`; faster but prone to bias |
| `rbf` | Good for non-linear boundaries |
| `poly` | Polynomial kernel |
| `sigmoid` | Sigmoid kernel |

---

## Running the tests

```bash
pytest tests/
```

---

## Example scripts

Working examples are in `examples/`:

```
examples/
  SPARE_AD/
    training/
      input/train_config.json
      run_training.sh         ← bash examples/SPARE_AD/training/run_training.sh
    testing/
      input/raw_data.csv
      run_testing.sh          ← bash examples/SPARE_AD/testing/run_testing.sh
  SPARE_BA/   …
  SPARE_DIABETES/   …
```

---

## Distribution options

### pip (recommended for most users)

```bash
pip install NiChart_SPARE
```

Models are **not** bundled in the pip package. They download lazily, per-task, on first use.

### Docker / Singularity

```bash
docker build -t nichart_spare .
```

All registered default-version models are downloaded and baked into the image at build time, so no internet access is needed at runtime.

```bash
docker run --rm -v $(pwd):/data nichart_spare \
    test -t AD -i /data/raw.csv -o /data/results/
```

For Singularity, build from the Docker image or supply a compatible definition file.

---

## Version alignment

GitHub release tags, Hugging Face repo revision/tags, and `task_registry.yaml` version keys are kept aligned (e.g. GitHub tag `v3.0` ↔ HF revision `v3.0` ↔ registry key `v3.0`) so any combination of code and model is traceable.

Model weights are hosted on Hugging Face; code is on GitHub. License terms may differ between the two — check the HF repo card for each model before use.

---

## Publications

- **SPARE-BA**
  Habes, M. et al. Advanced brain aging: relationship with epidemiologic and genetic risk factors, and overlap with Alzheimer disease atrophy patterns. *Transl Psychiatry* 6, e775 (2016). [doi:10.1038/tp.2016.39](https://doi.org/10.1038/tp.2016.39)

- **SPARE-AD**
  Davatzikos, C. et al. Longitudinal progression of Alzheimer's-like patterns of atrophy in normal older adults: the SPARE-AD index. *Brain* 132, 2026–2035 (2009). [doi:10.1093/brain/awp091](https://doi.org/10.1093/brain/awp091)

- **diSPARE-AD**
  Hwang, G. et al. Disentangling Alzheimer's disease neurodegeneration from typical brain ageing using machine learning. *Brain Commun* 4, fcac117 (2022). [doi:10.1093/braincomms/fcac117](https://doi.org/10.1093/braincomms/fcac117)

- **SPARE-CVMs** (HT, HL, T2D, SM, OB)
  Govindarajan, S.T., Mamourian, E., Erus, G. et al. Machine learning reveals distinct neuroanatomical signatures of cardiovascular and metabolic diseases in cognitively unimpaired individuals. *Nat Commun* 16, 2724 (2025). [doi:10.1038/s41467-025-57867-7](https://doi.org/10.1038/s41467-025-57867-7)
