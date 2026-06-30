"""Tests for prep_data.prep_data()."""

import pandas as pd
import pytest

from NiChart_SPARE.prep_data import prep_data
from tests.conftest import N, ROI_GENERIC, ROI_CVM


class TestOutputFormat:
    def test_cl_col0_is_mrid(self, raw_cl_csv, tmp_path):
        df, _ = prep_data(raw_cl_csv, 'CL', target_column='DX',
                       ignore_columns=['Study'],
                       output_file=str(tmp_path / 'out.csv'))
        assert df.columns[0] == 'MRID'

    def test_cl_col1_is_target(self, raw_cl_csv, tmp_path):
        df, _ = prep_data(raw_cl_csv, 'CL', target_column='DX',
                       ignore_columns=['Study'],
                       output_file=str(tmp_path / 'out.csv'))
        assert df.columns[1] == 'DX'

    def test_row_count_preserved(self, raw_cl_csv, tmp_path):
        df, _ = prep_data(raw_cl_csv, 'CL', target_column='DX',
                       ignore_columns=['Study'],
                       output_file=str(tmp_path / 'out.csv'))
        assert len(df) == N

    def test_ignored_columns_dropped(self, raw_cl_csv, tmp_path):
        df, _ = prep_data(raw_cl_csv, 'CL', target_column='DX',
                       ignore_columns=['Study'],
                       output_file=str(tmp_path / 'out.csv'))
        assert 'Study' not in df.columns

    def test_feature_columns_present(self, raw_cl_csv, tmp_path):
        df, _ = prep_data(raw_cl_csv, 'CL', target_column='DX',
                       ignore_columns=['Study'],
                       output_file=str(tmp_path / 'out.csv'))
        for col in ROI_GENERIC:
            assert col in df.columns

    def test_rg_col1_is_target(self, raw_rg_csv, tmp_path):
        df, _ = prep_data(raw_rg_csv, 'RG', target_column='Age',
                       ignore_columns=['Study'],
                       output_file=str(tmp_path / 'out.csv'))
        assert df.columns[1] == 'Age'

    def test_output_csv_written(self, raw_cl_csv, tmp_path):
        out = tmp_path / 'out.csv'
        prep_data(raw_cl_csv, 'CL', target_column='DX',
                  ignore_columns=['Study'],
                  output_file=str(out))
        assert out.exists()
        assert len(pd.read_csv(out)) == N


class TestInferenceMode:
    """When target_column is absent from the input, output should have no target column."""

    def test_no_target_col_in_output(self, raw_cl_csv, tmp_path):
        df, _ = prep_data(raw_cl_csv, 'CL', target_column=None,
                       ignore_columns=['Study', 'DX'],
                       output_file=str(tmp_path / 'out.csv'))
        assert 'DX' not in df.columns
        assert df.columns[0] == 'MRID'


class TestCVMPrep:
    def test_confounds_removed_from_features(self, raw_cvm_csv, tmp_path):
        df, _ = prep_data(raw_cvm_csv, 'CVM', target_column='Disease',
                       ignore_columns=['Study'],
                       output_file=str(tmp_path / 'out.csv'))
        feature_cols = df.columns[2:].tolist()
        assert 'Age' not in feature_cols
        assert 'Sex' not in feature_cols
        assert 'DL_MUSE_Volume_702' not in feature_cols

    def test_roi_features_retained(self, raw_cvm_csv, tmp_path):
        df, _ = prep_data(raw_cvm_csv, 'CVM', target_column='Disease',
                       ignore_columns=['Study'],
                       output_file=str(tmp_path / 'out.csv'))
        for col in ROI_CVM:
            assert col in df.columns

    def test_cvm_missing_confound_raises(self, tmp_path):
        """Prep should raise if required CVM columns are absent."""
        import pandas as pd, numpy as np
        bad = pd.DataFrame({
            'MRID': [f'S{i}' for i in range(10)],
            'Disease': np.random.randint(0, 2, 10),
            'H_DL_MUSE_Volume_100': np.random.normal(20000, 3000, 10),
            # Age, Sex, DL_MUSE_Volume_702 intentionally absent
        })
        bad.to_csv(tmp_path / 'bad.csv', index=False)
        with pytest.raises(ValueError, match='residualization requires'):
            prep_data(str(tmp_path / 'bad.csv'), 'CVM', target_column='Disease')


class TestICVCorrection:
    def test_icv_correction_runs(self, tmp_path):
        import pandas as pd, numpy as np
        df = pd.DataFrame({
            'MRID': [f'S{i}' for i in range(20)],
            'DX': np.random.randint(0, 2, 20),
            'DL_MUSE_Volume_702': np.random.normal(1_400_000, 100_000, 20),
            'DL_MUSE_Volume_1': np.random.normal(5_000, 500, 20),
            'DL_MUSE_Volume_2': np.random.normal(5_000, 500, 20),
        })
        df.to_csv(tmp_path / 'icv.csv', index=False)
        out, _ = prep_data(str(tmp_path / 'icv.csv'), 'CL',
                        target_column='DX',
                        icv_correction=True,
                        icv_column='DL_MUSE_Volume_702',
                        output_file=str(tmp_path / 'out.csv'))
        # ICV-corrected values should be small fractions (volume / ICV)
        assert out['DL_MUSE_Volume_1'].mean() < 1.0
