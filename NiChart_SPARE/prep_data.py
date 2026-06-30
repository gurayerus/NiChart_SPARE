"""
Task-specific data preparation for SVM training and inference.

Produces a standardized CSV:
  Column 0 : key_variable (MRID)
  Column 1 : target_column  (omitted when not present in the input, e.g. inference)
  Column 2+: input features (ready to feed directly into train.py / inference.py)

What this module handles (no encoding or scaling — those live in train/inference):
  - Loading and column subsetting
  - Schema validation against an expected column list (with wildcard patterns)
  - Optional ICV correction (ROI volumes divided by ICV)
  - CVM-type residualization (Age / Sex / ICV confound removal)
  - Dropping user-specified ignore columns
"""

import sys
import pandas as pd
from .preprocessing import (
    load_csv_data,
    validate_dataframe,
    correct_icv,
    apply_cvm_residualization,
)

# Types that require Age/Sex/ICV residualization before classification
CVM_TYPES = frozenset({'CVM', 'HT', 'T2B', 'SM', 'BMI'})


def _validate_columns_schema(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Validate df against an expected column schema and filter to matched columns.

    Patterns ending with '*' are prefix-matched against actual column names.
    All other patterns require an exact match.  Mismatches produce warnings but
    do not abort — the function always returns a filtered DataFrame.
    """
    exact = [c for c in columns if not c.endswith('*')]
    prefixes = [c[:-1] for c in columns if c.endswith('*')]

    matched = {
        col for col in df.columns
        if col in exact or any(col.startswith(p) for p in prefixes)
    }

    missing_exact = [c for c in exact if c not in df.columns]
    if missing_exact:
        print(
            f"Warning: column(s) declared in schema not found in CSV: {missing_exact}",
            file=sys.stderr,
        )

    empty_prefixes = [p for p in prefixes if not any(c.startswith(p) for c in df.columns)]
    if empty_prefixes:
        print(
            f"Warning: no columns matched pattern(s): {[p + '*' for p in empty_prefixes]}",
            file=sys.stderr,
        )

    unmatched = [c for c in df.columns if c not in matched]
    if unmatched:
        print(
            f"Warning: column(s) not in schema, excluded from processing: {unmatched}",
            file=sys.stderr,
        )

    return df[[c for c in df.columns if c in matched]]


def prep_data(
    input_file: str,
    spare_type: str,
    key_variable: str = 'MRID',
    target_column: str = None,
    columns: list = None,
    ignore_columns: list = None,
    output_file: str = None,
    icv_correction: bool = False,
    icv_column: str = 'DL_MUSE_Volume_702',
    age_col: str = 'Age',
    sex_col: str = 'Sex',
    cvm_mean_age: float = None,
) -> tuple:
    """
    Prepare a raw input CSV for SVM training or inference.

    Parameters
    ----------
    columns : list[str] or None
        Expected column names/patterns for the input CSV.  A trailing '*' acts as a
        prefix wildcard (e.g. 'DL_MUSE_Volume_*' matches all columns starting with
        that string).  When provided, the CSV is validated against this schema and
        filtered to only the matched columns; mismatches produce warnings but do not
        abort.  When None, no schema validation is performed.

    Returns
    -------
    (df, cvm_mean_age) : tuple[pd.DataFrame, float | None]
        df            — prepared DataFrame [key_variable, target?, features...]
        cvm_mean_age  — mean age used for CVM centering (None for non-CVM types).
                        Save this value and pass it back at inference time via
                        the cvm_mean_age parameter so centering is identical.
    """
    spare_type = spare_type.upper()

    # --- Load and drop ignored columns ---
    df = load_csv_data(input_file, drop_columns=ignore_columns or [])

    # --- Validate and filter to declared column schema (if provided) ---
    if columns:
        df = _validate_columns_schema(df, columns)

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
        df, cvm_mean_age = apply_cvm_residualization(
            df, age_col=age_col, sex_col=sex_col, dlicv_col=icv_column,
            mean_age=cvm_mean_age,
        )
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

    return df, cvm_mean_age


def _check_cvm_columns(df: pd.DataFrame, age_col: str, sex_col: str, icv_col: str) -> None:
    missing = [c for c in [age_col, sex_col, icv_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"CVM residualization requires columns {[age_col, sex_col, icv_col]}; "
            f"missing: {missing}"
        )
