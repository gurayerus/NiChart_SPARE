"""End-to-end pipeline tests: prep → train → test."""

import pandas as pd
import pytest

from NiChart_SPARE.prep_data import prep_data
from NiChart_SPARE.train import train_model
from NiChart_SPARE.inference import infer_model
from tests.conftest import TRAIN_KWARGS, N

RESID_PREPROCESSING = {
    'residualization': {
        'age_col': 'Age', 'sex_col': 'Sex', 'icv_col': 'DL_MUSE_Volume_702',
    }
}


class TestCLPipeline:
    def test_full_pipeline(self, raw_cl_csv, tmp_path):
        prepped = str(tmp_path / 'prepped.csv')
        model   = str(tmp_path / 'model.joblib')
        out_dir = str(tmp_path / 'preds')

        prep_data(raw_cl_csv, key_col='MRID', target_col='DX',
                  data_cols=['DL_MUSE_Volume_*'], output_file=prepped)
        train_model(prepped, model, 'CL', **TRAIN_KWARGS)
        df = infer_model(prepped, model, out_dir)

        assert len(df) == N
        assert 'SPARE_CL' in df.columns
        assert set(df['SPARE_CL'].unique()).issubset({0, 1})

    def test_inference_without_target(self, raw_cl_csv, tmp_path):
        """Inference should work when the test CSV has no target column."""
        prepped_train = str(tmp_path / 'prepped_train.csv')
        prepped_test  = str(tmp_path / 'prepped_test.csv')
        model         = str(tmp_path / 'model.joblib')
        out_dir       = str(tmp_path / 'preds')

        prep_data(raw_cl_csv, key_col='MRID', target_col='DX',
                  data_cols=['DL_MUSE_Volume_*'], output_file=prepped_train)
        train_model(prepped_train, model, 'CL', **TRAIN_KWARGS)

        # Test CSV: no target column — data_cols doesn't include DX, target_col=None
        prep_data(raw_cl_csv, key_col='MRID', target_col=None,
                  data_cols=['DL_MUSE_Volume_*'], output_file=prepped_test)
        df = infer_model(prepped_test, model, out_dir)

        assert 'SPARE_CL' in df.columns
        assert 'GT_CL' not in df.columns


class TestRGPipeline:
    def test_full_pipeline(self, raw_rg_csv, tmp_path):
        prepped = str(tmp_path / 'prepped.csv')
        model   = str(tmp_path / 'model.joblib')
        out_dir = str(tmp_path / 'preds')

        prep_data(raw_rg_csv, key_col='MRID', target_col='Age',
                  data_cols=['DL_MUSE_Volume_*'], output_file=prepped)
        train_model(prepped, model, 'RG', **TRAIN_KWARGS)
        df = infer_model(prepped, model, out_dir)

        assert len(df) == N
        assert 'SPARE_RG' in df.columns
        assert df['SPARE_RG'].dtype.kind == 'f'


class TestCVMPipeline:
    def test_full_pipeline(self, raw_cvm_csv, tmp_path):
        prepped = str(tmp_path / 'prepped.csv')
        model   = str(tmp_path / 'model.joblib')
        out_dir = str(tmp_path / 'preds')

        prep_data(raw_cvm_csv, key_col='MRID', target_col='Disease',
                  data_cols=['Age', 'Sex', 'DL_MUSE_Volume_702', 'H_DL_MUSE_Volume_*'],
                  preprocessing=RESID_PREPROCESSING,
                  output_file=prepped)

        prepped_df = pd.read_csv(prepped)
        assert 'Age' not in prepped_df.columns
        assert 'Sex' not in prepped_df.columns
        assert 'DL_MUSE_Volume_702' not in prepped_df.columns

        train_model(prepped, model, 'CVM', **TRAIN_KWARGS)
        df = infer_model(prepped, model, out_dir)

        assert len(df) == N
        assert 'SPARE_CVM' in df.columns
