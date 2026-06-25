# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NiChart_SPARE computes SPARE (Spatial Patterns of Abnormality for Recognition) biomarker scores from brain ROI volumes ([NiChart_DLMUSE](https://github.com/CBICA/NiChart_DLMUSE)) and white matter lesion volumes ([NiChart_DLWMLS](https://github.com/CBICA/NiChart_DLWMLS)). It supports SVM-based model training and inference via a CLI.

## Installation & Setup

```bash
git clone https://github.com/CBICA/NiChart_SPARE.git
cd NiChart_SPARE
pip install -e .
```

Dev dependencies: `pip install -e ".[dev]"`

## Common Commands

**Prepare data (classification — drops confound columns, no other transforms):**
```bash
NiChart_SPARE prep -t CL -i raw.csv -o prepped.csv -tc DX -ic Study,SITE,Sex -kv MRID
```

**Prepare data (CVM — applies Age/Sex/ICV residualization, then drops those columns):**
```bash
NiChart_SPARE prep -t CVM -i raw.csv -o prepped.csv -tc Disease -ic Study,SITE -kv MRID
```

**Train a classifier:**
```bash
NiChart_SPARE train -t CL -i prepped.csv -mo model.joblib -sk linear -ht True -cf 5 -cb True -v 1
```

**Train a regressor (with bias correction):**
```bash
NiChart_SPARE train -t RG -i prepped.csv -mo model.joblib -sk linear -ht False -bc 1 -cf 5 -v 1
```

**Run inference:**
```bash
NiChart_SPARE test -i prepped_test.csv -m model.joblib -o predictions.csv -kv MRID
```

**Wrapper script** (merges demographics CSV and preprocesses `Sex`/ICV columns before invoking `NiChart_SPARE`):
```bash
python scripts/wrapper.py -i input.csv -demog demographics.csv [NiChart_SPARE args...]
```

**Lint:** `flake8 NiChart_SPARE/` (config in `setup.cfg`; ignores E203, E501, E722, W503, B950)

**Type check:** `mypy NiChart_SPARE/` (strict mode; test errors are ignored)

There is no test suite currently.

## Architecture

### Three-stage workflow

```
NiChart_SPARE prep  -t <TYPE> -i raw.csv      -o prepped.csv   [task-specific options]
NiChart_SPARE train -t <TYPE> -i prepped.csv  -mo model.joblib [training options]
NiChart_SPARE test            -i prepped.csv  -m model.joblib -o preds.csv
```

**Intermediate format** (`prepped.csv`): column 0 = MRID, column 1 = target (omitted for inference inputs), columns 2+ = features. This format is the contract between the three stages.

### Data flow

```
__main__.py
  ├── prep  → prep_data.prep_data()
  │              data_prep.load_csv_data()
  │              data_prep.correct_icv()          [if --icv_correction]
  │              data_prep.apply_cvm_residualization()  [CVM types only]
  │              → writes standardised CSV
  │
  ├── train → train.train_model()
  │              data_prep.preprocess_classification_data()  or  preprocess_regression_data()
  │              pipelines/spare_svm_classification.py  or  spare_svm_regression.py
  │              → saves .joblib (model + encoder + scaler + metadata + CV results)
  │
  └── test  → inference.infer_model()
                 data_prep.preprocess_classification_data()  or  preprocess_regression_data()
                 (uses encoder/scaler loaded from model)
                 → writes predictions CSV
```

### SPARE Types

| Type key(s) | Task | Prep transform | Training pipeline |
|---|---|---|---|
| `CL`, `AD` | Classification | none | `spare_svm_classification.py` |
| `RG`, `BA` | Regression | none | `spare_svm_regression.py` |
| `CVM`, `HT`, `T2B`, `SM`, `BMI` | Classification | Age/Sex/ICV residualization | `spare_svm_classification.py` |

`util.get_pipeline_module()` routes RG/BA → regression pipeline; everything else → classification pipeline.

### Kernel Options

`linear_fast` (uses `LinearSVC`/`LinearSVR`), `linear`, `rbf`, `poly`, `sigmoid`. `linear_fast` is prone to bias per the README.

### Saved Model Format

```python
{
  'model':                {'model': <sklearn estimator>, 'bias': <bias terms or None>},
  'meta_data':            {spare_type, package_version, model_description,
                           training_data_description, pipeline_description},
  'preprocessor':         {'feature_encoder': <LabelEncoder dict>,
                           'feature_scaler':  <StandardScaler dict>},
  'hyperparameter_tuning': {tuner, best_params, search_grid},
  'cross_validation':     {'Repeat_N': {'scores': {}, 'cv_results': {}}},
}
```

`meta_data['training_data_description']['feature_names']` is used at inference time to validate and subset input columns.

### Key Modules

- [NiChart_SPARE/\_\_main\_\_.py](NiChart_SPARE/__main__.py) — CLI entry point; subcommands `prep`, `train`, `test`
- [NiChart_SPARE/prep\_data.py](NiChart_SPARE/prep_data.py) — `prep_data()`: task-specific transforms (ICV correction, CVM residualization, column drops) → standardized CSV. No encoding/scaling.
- [NiChart_SPARE/train.py](NiChart_SPARE/train.py) — `train_model()`: encodes + scales features, dispatches to CL/RG pipeline, serializes model
- [NiChart_SPARE/inference.py](NiChart_SPARE/inference.py) — `infer_model()`: loads saved encoder/scaler, runs prediction, writes output CSV
- [NiChart_SPARE/svm.py](NiChart_SPARE/svm.py) — `get_svm_hyperparameter_grids()`: shared per-kernel C/gamma/epsilon search grids (imported by pipeline modules)
- [NiChart_SPARE/data\_prep.py](NiChart_SPARE/data_prep.py) — Low-level utilities: `load_csv_data`, `validate_dataframe`, `correct_icv`, `apply_cvm_residualization`, `preprocess_classification_data`, `preprocess_regression_data`
- [NiChart_SPARE/util.py](NiChart_SPARE/util.py) — `get_pipeline_module()`, metadata/preprocessor builders, `expspace()`
- [NiChart_SPARE/data\_analysis.py](NiChart_SPARE/data_analysis.py) — Regression/classification metrics, SVM feature importance, PDP/ICE plotting, ROC-AUC plotting, CV score extraction helpers
- [NiChart_SPARE/pipelines/](NiChart_SPARE/pipelines/) — `train_svc_model` / `train_svr_model`; return `(model, [bias,] ht, cv)`
- [NiChart_SPARE/reference/](NiChart_SPARE/reference/) — `covparams_scaler_sparecvms_dl2.csv`: pre-computed coefficients for CVM residualization
- [Models/](Models/) — Pre-trained `.joblib` model files
- [scripts/wrapper.py](scripts/wrapper.py) — Merges demographics CSV, adds `Sex_M` column, then calls `NiChart_SPARE`

### CVM Residualization

`apply_cvm_residualization()` in `data_prep.py` applies pre-fitted linear regression coefficients (from `reference/covparams_scaler_sparecvms_dl2.csv`) to remove Age, Sex, and ICV effects from ROI volumes. This happens in `prep_data` (not during training), so both train and test inputs must be prepped with the same CVM type. The Age, Sex, and ICV columns are dropped from the feature set after residualization.

### Bias Correction (Regression)

Controlled by `-bc` / `--bias_correction` in `train`:
- `0`: disabled
- `1`: Beheshti et al. — residual approach (fit `LinearRegression` on `y_train` → residuals)
- `2`: Cole et al. — rescale predictions by fitted slope/intercept
