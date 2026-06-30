"""
Task-specific data preparation for SVM training and inference.

Produces a standardized DataFrame / CSV:
  Column 0 : key_col   (MRID)
  Column 1 : target_col  (omitted when not present in the input, e.g. inference)
  Column 2+: feature columns derived from data_cols patterns, after any pre-processing

What this module handles (no encoding or scaling — those live in train/inference):
  - Loading the raw CSV
  - Selecting and ordering feature columns by pattern list (data_cols)
  - Applying explicit value mappings (e.g. Sex: {M: 0, F: 1})
  - Pre-processing steps declared in config:
      residualization  — Age/Sex/ICV confound removal (pre-fitted CVM coefficients)
      icv_correction   — divide ROI volumes by ICV
"""

import sys
import pandas as pd
from .preprocessing import (
    load_csv_data,
    validate_dataframe,
    correct_icv,
    apply_cvm_residualization,
)


def _expand_data_cols(data_cols: list, available: list) -> tuple:
    """Expand wildcard patterns against available column names in declaration order.

    Returns (matched_cols, unmatched_available) where:
      matched_cols        — ordered list of columns that satisfy at least one pattern
      unmatched_available — CSV columns not matched by any pattern (will be excluded)
    """
    matched = []
    seen = set()
    for pattern in data_cols:
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            for c in available:
                if c.startswith(prefix) and c not in seen:
                    matched.append(c)
                    seen.add(c)
        else:
            if pattern in available and pattern not in seen:
                matched.append(pattern)
                seen.add(pattern)
            elif pattern not in available:
                print(f"Warning: data_cols entry '{pattern}' not found in input CSV.", file=sys.stderr)
    unmatched = [c for c in available if c not in seen]
    return matched, unmatched


def _check_residualization_columns(df: pd.DataFrame, age_col: str, sex_col: str, icv_col: str) -> None:
    missing = [c for c in [age_col, sex_col, icv_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"residualization requires columns [{age_col}, {sex_col}, {icv_col}]; "
            f"missing: {missing}"
        )


def prep_data(
    input_file: str,
    key_col: str = 'MRID',
    target_col: str = None,
    data_cols: list = None,
    mappings: dict = None,
    preprocessing: dict = None,
    output_file: str = None,
    cvm_mean_age: float = None,
) -> tuple:
    """
    Prepare a raw input CSV for SVM training or inference.

    Parameters
    ----------
    input_file : str
        Path to the raw input CSV.
    key_col : str
        Column that uniquely identifies each sample (default: MRID).
    target_col : str or None
        Column to predict.  When absent from the CSV the output has no target column
        (inference mode).
    data_cols : list[str] or None
        Feature column names / patterns (trailing '*' = prefix wildcard).
        Lists only the feature columns — key_col and target_col are always added
        automatically.  Columns in the CSV not matched by any pattern are excluded
        with a warning.  When None, all columns except key_col and target_col are used.
    mappings : dict or None
        Per-column value remapping applied before preprocessing.
        e.g. {"Sex": {"M": 0, "F": 1}}
    preprocessing : dict or None
        Pre-processing steps to apply.  Supported keys:
          "residualization": {"age_col", "sex_col", "icv_col"} — CVM residualization;
              the three confound columns are dropped from features after the step.
          "icv_correction":  {"icv_col"} — divide all DL_MUSE_Volume_ ROIs by ICV.
    output_file : str or None
        If given, write the prepared DataFrame to this CSV path.
    cvm_mean_age : float or None
        Mean age used for centering during training residualization.  Pass the saved
        value at inference time so centering is identical.  When None (training),
        the mean is computed from the current data and returned.

    Returns
    -------
    (df, cvm_mean_age) : tuple[pd.DataFrame, float | None]
    """
    # --- Load ---
    df = load_csv_data(input_file)

    if key_col not in df.columns:
        raise ValueError(f"Key column '{key_col}' not found in input.")

    has_target = target_col is not None and target_col in df.columns

    # --- Select feature columns ---
    # candidate pool excludes key and target so they are never accidentally treated as features
    candidates = [c for c in df.columns if c != key_col and c != target_col]

    if data_cols is not None:
        feature_cols, unmatched = _expand_data_cols(data_cols, candidates)
        if unmatched:
            print(
                f"Warning: CSV column(s) not covered by data_cols, excluded: {unmatched}",
                file=sys.stderr,
            )
    else:
        feature_cols = candidates  # use everything

    # Build working DataFrame: [key_col, (target_col?), feature_cols...]
    select = [key_col]
    if has_target:
        select.append(target_col)
    # Include columns needed for preprocessing even if not final features
    # (preprocessing steps drop them after use)
    select += feature_cols
    df = df[[c for c in select if c in df.columns]]

    if has_target:
        validate_dataframe(df, target_col)

    # --- Apply value mappings ---
    if mappings:
        for col, mapping in mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
            else:
                print(f"Warning: mapping declared for column '{col}' which is not in data.", file=sys.stderr)

    # --- Apply pre-processing steps ---
    if preprocessing:
        if 'residualization' in preprocessing:
            params = preprocessing['residualization']
            age_c = params.get('age_col', 'Age')
            sex_c = params.get('sex_col', 'Sex')
            icv_c = params.get('icv_col', 'DL_MUSE_Volume_702')
            _check_residualization_columns(df, age_c, sex_c, icv_c)
            df, cvm_mean_age = apply_cvm_residualization(
                df, age_col=age_c, sex_col=sex_c, dlicv_col=icv_c,
                mean_age=cvm_mean_age,
            )
            for col in [age_c, sex_c, icv_c]:
                if col in df.columns:
                    df = df.drop(columns=[col])

        if 'icv_correction' in preprocessing:
            params = preprocessing['icv_correction']
            icv_c = params.get('icv_col', 'DL_MUSE_Volume_702')
            df = correct_icv(df, icv_col=icv_c, roi_col_keyword='DL_MUSE_Volume_')

    feature_count = len(df.columns) - (1 + int(has_target))
    print(f"Prepared data: {len(df)} samples, {feature_count} features")

    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Saved prepared data to: {output_file}")

    return df, cvm_mean_age
