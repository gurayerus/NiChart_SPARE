"""
Task-specific data preparation for SVM training and inference.

Produces a standardized CSV:
  Column 0 : key_variable (MRID)
  Column 1 : target_column  (omitted when not present in the input, e.g. inference)
  Column 2+: input features (ready to feed directly into train.py / inference.py)

What this module handles (no encoding or scaling — those live in train/inference):
  - Loading and column subsetting
  - Optional ICV correction (ROI volumes divided by ICV)
  - CVM-type residualization (Age / Sex / ICV confound removal)
  - Dropping user-specified ignore columns
"""

import pandas as pd
from .preprocessing import (
    load_csv_data,
    validate_dataframe,
    correct_icv,
    apply_cvm_residualization,
)

# Types that require Age/Sex/ICV residualization before classification
CVM_TYPES = frozenset({'CVM', 'HT', 'T2B', 'SM', 'BMI'})


def prep_data(
    input_file: str,
    spare_type: str,
    key_variable: str = 'MRID',
    target_column: str = None,
    ignore_columns: list = None,
    output_file: str = None,
    icv_correction: bool = False,
    icv_column: str = 'DL_MUSE_Volume_702',
    age_col: str = 'Age',
    sex_col: str = 'Sex',
) -> pd.DataFrame:
    """
    Prepare a raw input CSV for SVM training or inference.

    Parameters
    ----------
    input_file : str
        Path to the raw input CSV.
    spare_type : str
        SPARE task type: CL, RG, AD, BA, CVM, HT, T2B, SM, BMI.
        Determines which task-specific transforms are applied.
    key_variable : str
        Column that uniquely identifies each sample (default: MRID).
    target_column : str or None
        Column to predict.  May be None or absent for inference inputs.
    ignore_columns : list or None
        Columns to drop before writing output (e.g. ['Study', 'SITE', 'Sex']).
    output_file : str or None
        If provided, write the prepared DataFrame to this CSV path.
    icv_correction : bool
        Divide all DL_MUSE_Volume_* ROI columns by the ICV column.
    icv_column : str
        ICV column name used for correction (default: DL_MUSE_Volume_702).
    age_col : str
        Age column name, used only for CVM residualization.
    sex_col : str
        Sex column name, used only for CVM residualization.

    Returns
    -------
    pd.DataFrame
        Columns: [key_variable, target_column (if present), feature1, feature2, ...]
    """
    spare_type = spare_type.upper()

    # --- Load and drop ignored columns ---
    df = load_csv_data(input_file, drop_columns=ignore_columns or [])

    if key_variable not in df.columns:
        raise ValueError(f"Key variable '{key_variable}' not found in input.")

    has_target = target_column is not None and target_column in df.columns
    if has_target:
        validate_dataframe(df, target_column)

    # --- Task-specific transforms ---

    if icv_correction:
        df = correct_icv(df, icv_col=icv_column, roi_col_keyword='DL_MUSE_Volume_')

    if spare_type in CVM_TYPES:
        _check_cvm_columns(df, age_col, sex_col, icv_column)
        df = apply_cvm_residualization(df, age_col=age_col, sex_col=sex_col, dlicv_col=icv_column)
        # Confound columns have been absorbed into residuals; drop them from features
        for col in [age_col, sex_col, icv_column]:
            if col in df.columns:
                df = df.drop(columns=[col])

    # --- Standardize column order: [MRID, target?, features...] ---
    reserved = [key_variable]
    if has_target:
        reserved.append(target_column)
    feature_cols = [c for c in df.columns if c not in reserved]
    df = df[reserved + feature_cols]

    print(f"Prepared data: {len(df)} samples, {len(feature_cols)} features")

    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Saved prepared data to: {output_file}")

    return df


def _check_cvm_columns(df: pd.DataFrame, age_col: str, sex_col: str, icv_col: str) -> None:
    missing = [c for c in [age_col, sex_col, icv_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"CVM residualization requires columns {[age_col, sex_col, icv_col]}; "
            f"missing: {missing}"
        )
