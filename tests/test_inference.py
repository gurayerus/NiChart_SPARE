"""Tests for inference.infer_model()."""

import pandas as pd
import pytest

from NiChart_SPARE.inference import infer_model


class TestOutputColumns:
    def test_cl_score_column_present(self, cl_model, prepped_cl_csv, tmp_path):
        df = infer_model(prepped_cl_csv, cl_model, str(tmp_path / 'out'))
        assert 'SPARE_CL' in df.columns

    def test_cl_decision_function_present(self, cl_model, prepped_cl_csv, tmp_path):
        df = infer_model(prepped_cl_csv, cl_model, str(tmp_path / 'out'))
        assert 'SPARE_CL_decision_function' in df.columns

    def test_rg_score_column_present(self, rg_model, prepped_rg_csv, tmp_path):
        df = infer_model(prepped_rg_csv, rg_model, str(tmp_path / 'out'))
        assert 'SPARE_RG' in df.columns

    def test_rg_no_decision_function(self, rg_model, prepped_rg_csv, tmp_path):
        df = infer_model(prepped_rg_csv, rg_model, str(tmp_path / 'out'))
        assert 'SPARE_RG_decision_function' not in df.columns

    def test_mrid_column_present(self, cl_model, prepped_cl_csv, tmp_path):
        df = infer_model(prepped_cl_csv, cl_model, str(tmp_path / 'out'))
        assert 'MRID' in df.columns

    def test_ground_truth_written_when_target_present(self, cl_model, prepped_cl_csv, tmp_path):
        df = infer_model(prepped_cl_csv, cl_model, str(tmp_path / 'out'))
        assert 'GT_CL' in df.columns


class TestOutputFile:
    def test_csv_written(self, cl_model, prepped_cl_csv, tmp_path):
        out_dir = tmp_path / 'out'
        infer_model(prepped_cl_csv, cl_model, str(out_dir))
        assert (out_dir / 'predictions.csv').exists()

    def test_row_count_matches_input(self, cl_model, prepped_cl_csv, tmp_path):
        df = infer_model(prepped_cl_csv, cl_model, str(tmp_path / 'out'))
        input_rows = len(pd.read_csv(prepped_cl_csv))
        assert len(df) == input_rows


class TestAppendSpareTag:
    def test_custom_tag_used(self, cl_model, prepped_cl_csv, tmp_path):
        df = infer_model(prepped_cl_csv, cl_model, str(tmp_path / 'out'),
                         append_spare_tag='MyBiomarker')
        assert 'SPARE_MyBiomarker' in df.columns
        assert 'SPARE_CL' not in df.columns


class TestValidation:
    def test_missing_feature_raises(self, cl_model, tmp_path):
        import pandas as pd, numpy as np
        bad = pd.DataFrame({'MRID': ['S001'], 'some_wrong_col': [1.0]})
        bad.to_csv(tmp_path / 'bad.csv', index=False)
        with pytest.raises(ValueError, match='missing columns'):
            infer_model(str(tmp_path / 'bad.csv'), cl_model, str(tmp_path / 'out'))

    def test_missing_key_variable_raises(self, cl_model, prepped_cl_csv, tmp_path):
        df = pd.read_csv(prepped_cl_csv).drop(columns=['MRID'])
        df.to_csv(tmp_path / 'no_mrid.csv', index=False)
        with pytest.raises(ValueError, match="Key variable 'MRID'"):
            infer_model(str(tmp_path / 'no_mrid.csv'), cl_model, str(tmp_path / 'out'))
