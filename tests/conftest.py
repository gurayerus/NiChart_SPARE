"""
Shared pytest fixtures.

All synthetic data is generated in memory and written to pytest's tmp_path,
so no static data files are needed.  Three raw-data shapes are covered:

  CL  — binary classification, generic ROI features
  RG  — continuous regression (brain age), generic ROI features
  CVM — binary classification with Age / Sex / ICV columns required by
        the CVM residualization step (uses H_DL_MUSE_Volume_* columns
        that are present in the reference CSV)
"""

import numpy as np
import pandas as pd
import pytest

RNG = np.random.default_rng(0)
N = 60  # small enough to run fast

# Generic ROI features for CL / RG
ROI_GENERIC = [f'DL_MUSE_Volume_{i}' for i in range(1, 11)]

# A subset of ROI columns that exist in the CVM reference CSV
ROI_CVM = [
    'H_DL_MUSE_Volume_100',
    'H_DL_MUSE_Volume_101',
    'H_DL_MUSE_Volume_102',
    'H_DL_MUSE_Volume_103',
    'H_DL_MUSE_Volume_104',
]


# ── raw DataFrame builders ──────────────────────────────────────────────────

def make_cl_df() -> pd.DataFrame:
    return pd.DataFrame({
        'MRID':  [f'S{i:03d}' for i in range(N)],
        'Study': 'SiteA',
        'DX':    RNG.integers(0, 2, N),
        **{col: RNG.normal(5_000, 500, N) for col in ROI_GENERIC},
    })


def make_rg_df() -> pd.DataFrame:
    return pd.DataFrame({
        'MRID':  [f'S{i:03d}' for i in range(N)],
        'Study': 'SiteA',
        'Age':   RNG.uniform(50, 90, N),
        **{col: RNG.normal(5_000, 500, N) for col in ROI_GENERIC},
    })


def make_cvm_df() -> pd.DataFrame:
    return pd.DataFrame({
        'MRID':               [f'S{i:03d}' for i in range(N)],
        'Study':              'SiteA',
        'Disease':            RNG.integers(0, 2, N),
        'Age':                RNG.uniform(50, 90, N),
        'Sex':                RNG.choice(['M', 'F'], N),
        'DL_MUSE_Volume_702': RNG.normal(1_400_000, 100_000, N),
        **{col: RNG.normal(20_000, 3_000, N) for col in ROI_CVM},
    })


# ── raw CSV fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def raw_cl_csv(tmp_path):
    p = tmp_path / 'raw_cl.csv'
    make_cl_df().to_csv(p, index=False)
    return str(p)


@pytest.fixture
def raw_rg_csv(tmp_path):
    p = tmp_path / 'raw_rg.csv'
    make_rg_df().to_csv(p, index=False)
    return str(p)


@pytest.fixture
def raw_cvm_csv(tmp_path):
    p = tmp_path / 'raw_cvm.csv'
    make_cvm_df().to_csv(p, index=False)
    return str(p)


# ── prepped CSV fixtures ────────────────────────────────────────────────────

@pytest.fixture
def prepped_cl_csv(raw_cl_csv, tmp_path):
    from NiChart_SPARE.prep_data import prep_data
    out = str(tmp_path / 'prepped_cl.csv')
    prep_data(raw_cl_csv, key_col='MRID', target_col='DX',
              data_cols=['DL_MUSE_Volume_*'], output_file=out)
    return out


@pytest.fixture
def prepped_rg_csv(raw_rg_csv, tmp_path):
    from NiChart_SPARE.prep_data import prep_data
    out = str(tmp_path / 'prepped_rg.csv')
    prep_data(raw_rg_csv, key_col='MRID', target_col='Age',
              data_cols=['DL_MUSE_Volume_*'], output_file=out)
    return out


@pytest.fixture
def prepped_cvm_csv(raw_cvm_csv, tmp_path):
    from NiChart_SPARE.prep_data import prep_data
    out = str(tmp_path / 'prepped_cvm.csv')
    prep_data(raw_cvm_csv, key_col='MRID', target_col='Disease',
              data_cols=['Age', 'Sex', 'DL_MUSE_Volume_702', 'H_DL_MUSE_Volume_*'],
              preprocessing={'residualization': {
                  'age_col': 'Age', 'sex_col': 'Sex', 'icv_col': 'DL_MUSE_Volume_702',
              }},
              output_file=out)
    return out


# ── trained model fixtures ──────────────────────────────────────────────────

TRAIN_KWARGS = dict(
    kernel='linear',
    tune_hyperparameters=False,
    cv_fold=2,
    cross_validate=True,
    train_whole_set=True,
    verbose=0,
)


@pytest.fixture
def cl_model(prepped_cl_csv, tmp_path):
    from NiChart_SPARE.train import train_model
    out = str(tmp_path / 'model_cl.joblib')
    train_model(prepped_cl_csv, out, 'CL', **TRAIN_KWARGS)
    return out


@pytest.fixture
def rg_model(prepped_rg_csv, tmp_path):
    from NiChart_SPARE.train import train_model
    out = str(tmp_path / 'model_rg.joblib')
    train_model(prepped_rg_csv, out, 'RG', **TRAIN_KWARGS)
    return out


@pytest.fixture
def cvm_model(prepped_cvm_csv, tmp_path):
    from NiChart_SPARE.train import train_model
    out = str(tmp_path / 'model_cvm.joblib')
    train_model(prepped_cvm_csv, out, 'CVM', **TRAIN_KWARGS)
    return out
