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

## Quick usage — pre-trained models

Models are hosted on Hugging Face and downloaded automatically on first use (cached in `~/.cache/huggingface`).

```bash
# List all available pre-trained tasks
NiChart_SPARE test --list-tasks

# Run inference using a registered task (model downloads automatically)
NiChart_SPARE test -t AD -i prepped.csv -o ./results/

# Use a specific version
NiChart_SPARE test -t AD --version v3.0-raw -i prepped.csv -o ./results/

# Use an explicit local model file instead of the registry
NiChart_SPARE test -m /path/to/model.joblib -i prepped.csv -o ./results/
```

Output is written to `./results/predictions.csv`.

## Advanced usage — inference options

### Show model info and download URL

```bash
NiChart_SPARE test -t AD --model-info
```

Prints the Hugging Face repo, revision, direct download URL, and a manual invocation example.

### Listing available tasks

```bash
NiChart_SPARE test --list-tasks
```

## Offline / firewall-restricted environments

If the automatic download fails (e.g., firewall, air-gapped HPC):

1. Find the download URL:
   ```bash
   NiChart_SPARE test -t AD --model-info
   ```

2. On a machine with internet access, download the model file:
   ```bash
   wget <URL printed by --model-info>
   ```

3. Transfer the `.joblib` file to your target machine.

4. Run inference pointing directly at the local file:
   ```bash
   NiChart_SPARE test -m /path/to/SPARE-AD-Harmonized-ISTAGING_3_0.joblib \
       -i prepped.csv -o ./results/
   ```

## Full workflow — prepare → train → test

Prep, train, and test can each be run as separate explicit steps, or prep can be embedded
directly inside the train and test commands for a simpler one-step invocation.

### Option A — Explicit three-step workflow

#### Step 1 — Prepare data

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

#### Step 2 — Train

```bash
NiChart_SPARE train -c configs/train_ad.yaml
```

**Minimal config** (`configs/train_ad.yaml`) — assumes `input_csv` is already prepped:

```yaml
input_csv: ../data/prepped_train.csv   # already-prepped CSV
output_folder: ../models
model_tag: SPARE_AD
model_version: "1.0"

spare_type: AD
svm_kernel: linear
hyperparameter_tuning: true
cv_fold: 5
class_balancing: true
```

#### Step 3 — Test (inference)

```bash
NiChart_SPARE test \
    -m models/experiments/.../SPARE_AD_v1.0.joblib \
    -i data/prepped_test.csv \
    -o results/
```

---

### Option B — Inline prep (raw data → model / predictions in one command)

#### Training with inline prep

Add `target_column` (and any other prep fields) to the config. Prep runs automatically
and `prepped.csv` is saved alongside the model in the experiment directory.

```yaml
input_csv: ../data/raw_train.csv      # raw CSV
output_folder: ../models
model_tag: SPARE_AD
model_version: "1.0"

spare_type: AD

# Inline prep — runs automatically because target_column is specified
target_column: DX
ignore_columns: Study,SITE,Sex
# icv_correction: false
# age_col: Age                        # CVM residualization only
# sex_col: Sex

svm_kernel: linear
hyperparameter_tuning: true
cv_fold: 5
class_balancing: true
```

```bash
NiChart_SPARE train -c configs/train_ad.yaml
```

Experiment output:

```
models/experiments/<timestamp>_SPARE_AD/
  config.yaml        ← copy of the config used
  meta.json          ← run_id, timestamp, git commit, notes
  prepped.csv        ← prepped training data (saved for inspection/reuse)
  SPARE_AD_v1.0.joblib
  metrics.json       ← per-fold CV scores
```

#### Inference with inline prep

Use `--prep-type` to prep raw data on the fly. `prepped.csv` is saved in the output directory.

```bash
NiChart_SPARE test \
    -t AD \
    -i data/raw_test.csv \
    -o results/ \
    --prep-type CL \
    --ignore-columns Study,SITE,Sex
```

```
results/
  prepped.csv        ← prepped input (saved for inspection/reuse)
  predictions.csv
```

Add a `--notes` label to any training run for later reference:

```bash
NiChart_SPARE train -c configs/train_ad.yaml -n "baseline linear kernel"
```

## SVM kernels

| Kernel | Notes |
|---|---|
| `linear` | Recommended default |
| `linear_fast` | Uses `LinearSVC`/`LinearSVR`; faster but prone to bias |
| `rbf` | Good for non-linear boundaries |
| `poly` | Polynomial kernel |
| `sigmoid` | Sigmoid kernel |

## Running the tests

```bash
pytest tests/
```

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

All registered default-version models are downloaded and baked into the image at build time,
so no internet access is needed at runtime.

```bash
docker run --rm -v $(pwd):/data nichart_spare \
    test -t AD -i /data/prepped.csv -o /data/results/
```

For Singularity, build from the Docker image or supply a compatible definition file.

## Version alignment

GitHub release tags, Hugging Face repo revision/tags, and `task_registry.yaml` version keys are
kept aligned (e.g. GitHub tag `v3.0` ↔ HF revision `v3.0` ↔ registry key `v3.0`) so any
combination of code and model is traceable.

Model weights are hosted on Hugging Face; code is on GitHub. License terms may differ between
the two — check the HF repo card for each model before use.

## Publications

- **SPARE-BA**
  Habes, M. et al. Advanced brain aging: relationship with epidemiologic and genetic risk factors, and overlap with Alzheimer disease atrophy patterns. *Transl Psychiatry* 6, e775 (2016). [doi:10.1038/tp.2016.39](https://doi.org/10.1038/tp.2016.39)

- **SPARE-AD**
  Davatzikos, C. et al. Longitudinal progression of Alzheimer's-like patterns of atrophy in normal older adults: the SPARE-AD index. *Brain* 132, 2026–2035 (2009). [doi:10.1093/brain/awp091](https://doi.org/10.1093/brain/awp091)

- **diSPARE-AD**
  Hwang, G. et al. Disentangling Alzheimer's disease neurodegeneration from typical brain ageing using machine learning. *Brain Commun* 4, fcac117 (2022). [doi:10.1093/braincomms/fcac117](https://doi.org/10.1093/braincomms/fcac117)

- **SPARE-CVMs** (HT, HL, T2D, SM, OB)
  Govindarajan, S.T., Mamourian, E., Erus, G. et al. Machine learning reveals distinct neuroanatomical signatures of cardiovascular and metabolic diseases in cognitively unimpaired individuals. *Nat Commun* 16, 2724 (2025). [doi:10.1038/s41467-025-57867-7](https://doi.org/10.1038/s41467-025-57867-7)
