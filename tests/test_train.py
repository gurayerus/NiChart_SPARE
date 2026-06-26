"""Tests for train.train_model()."""

import joblib
import os
import pytest

from NiChart_SPARE.train import train_model, CLASSIFICATION_TYPES, REGRESSION_TYPES
from tests.conftest import TRAIN_KWARGS


class TestModelFileCreated:
    def test_cl_model_saved(self, prepped_cl_csv, tmp_path):
        out = str(tmp_path / 'model.joblib')
        train_model(prepped_cl_csv, out, 'CL', **TRAIN_KWARGS)
        assert os.path.exists(out)

    def test_rg_model_saved(self, prepped_rg_csv, tmp_path):
        out = str(tmp_path / 'model.joblib')
        train_model(prepped_rg_csv, out, 'RG', **TRAIN_KWARGS)
        assert os.path.exists(out)

    def test_cvm_model_saved(self, prepped_cvm_csv, tmp_path):
        out = str(tmp_path / 'model.joblib')
        train_model(prepped_cvm_csv, out, 'CVM', **TRAIN_KWARGS)
        assert os.path.exists(out)


class TestModelMetadata:
    def test_spare_type_stored(self, cl_model):
        data = joblib.load(cl_model)
        assert data['meta_data']['spare_type'] == 'CL'

    def test_feature_names_stored(self, cl_model):
        data = joblib.load(cl_model)
        assert len(data['meta_data']['training_data_description']['feature_names']) > 0

    def test_target_column_stored(self, cl_model):
        data = joblib.load(cl_model)
        assert data['meta_data']['training_data_description']['target_column'] == 'DX'

    def test_preprocessor_keys_present(self, cl_model):
        data = joblib.load(cl_model)
        assert 'feature_encoder' in data['preprocessor']
        assert 'feature_scaler' in data['preprocessor']

    def test_cv_results_stored(self, cl_model):
        data = joblib.load(cl_model)
        assert 'Repeat_0' in data['cross_validation']

    def test_rg_target_column_stored(self, rg_model):
        data = joblib.load(rg_model)
        assert data['meta_data']['training_data_description']['target_column'] == 'Age'


class TestValidation:
    def test_unknown_spare_type_raises(self, prepped_cl_csv, tmp_path):
        with pytest.raises(ValueError, match='Unknown spare_type'):
            train_model(prepped_cl_csv, str(tmp_path / 'model.joblib'),
                        'INVALID', **TRAIN_KWARGS)

    def test_type_sets_are_disjoint(self):
        assert CLASSIFICATION_TYPES.isdisjoint(REGRESSION_TYPES)
