"""
SVM model inference on standardized prepared data.

Expected input CSV format (produced by prep_data.prep_data):
  Column containing key_variable (MRID)          — identifies each sample
  Feature columns matching training feature names — used for prediction
  Target column (optional)                        — when present, written as GT_* in output

The module loads the encoder and scaler saved during training and applies
them to the new data before predicting.  No task-specific data transforms
are applied here — those belong in prep_data.py and must be run first.
"""

import joblib
import numpy as np
import pandas as pd

from .preprocessing import (
    load_csv_data,
    preprocess_classification_data,
    preprocess_regression_data,
)

REGRESSION_TYPES = frozenset({'RG', 'BA'})


def infer_model(
    input_file: str,
    model_path: str,
    output_file: str,
    key_variable: str = 'MRID',
    append_spare_tag: str = '',
) -> pd.DataFrame:
    """
    Apply a trained model to a prepared input CSV and write predictions.

    Parameters
    ----------
    input_file : str
        Path to a CSV produced by prep_data.prep_data().
        Must contain all feature columns used during training.
        May optionally contain the target column (for evaluation output).
    model_path : str
        Path to a .joblib model file produced by train.train_model().
    output_file : str
        Destination path for the predictions CSV.
    key_variable : str
        Column that uniquely identifies each sample (default: MRID).
    append_spare_tag : str
        When non-empty, rename SPARE_<type> → SPARE_<tag> in output.

    Returns
    -------
    pd.DataFrame
        Predictions DataFrame (also written to output_file).
    """
    # --- Load model ---
    model_data   = joblib.load(model_path)
    model_info   = model_data['model']
    meta_data    = model_data['meta_data']
    preprocessor = model_data['preprocessor']

    model       = model_info['model']
    bias_terms  = model_info['bias']
    spare_type  = meta_data['spare_type']
    target_col  = meta_data['training_data_description']['target_column']
    feature_names = meta_data['training_data_description']['feature_names']

    # --- Load and validate input data ---
    df = load_csv_data(input_file)

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns required by the model: {missing}")

    if key_variable not in df.columns:
        raise ValueError(f"Key variable '{key_variable}' not found in input.")

    # Subset to key + (optional) target + features
    cols = [key_variable]
    has_target = target_col in df.columns
    if has_target:
        cols.append(target_col)
    cols += feature_names
    df = df[cols]

    # --- Apply saved encoder and scaler ---
    feature_df = df.drop(columns=[key_variable])

    if spare_type in REGRESSION_TYPES:
        X, y, _, _ = preprocess_regression_data(
            df=feature_df,
            target_column=target_col,
            feature_encoder=preprocessor['feature_encoder'],
            feature_scaler=preprocessor['feature_scaler'],
            for_training=False,
        )
    else:
        X, y, _, _ = preprocess_classification_data(
            df=feature_df,
            target_column=target_col,
            feature_encoder=preprocessor['feature_encoder'],
            feature_scaler=preprocessor['feature_scaler'],
            for_training=False,
        )

    # --- Predict ---
    predictions = model.predict(X)

    if bias_terms is not None:
        if bias_terms['method'] == 1:
            predictions = predictions - bias_terms['model'].predict(predictions.reshape(-1, 1))
        elif bias_terms['method'] == 2:
            predictions = (predictions - bias_terms['intercept']) / bias_terms['coef']

    # --- Build output DataFrame ---
    score_col = f"SPARE_{append_spare_tag if append_spare_tag else spare_type}"
    out = pd.DataFrame({key_variable: df[key_variable].values, score_col: predictions})

    if spare_type not in REGRESSION_TYPES:
        out[f"{score_col}_decision_function"] = model.decision_function(X)

    if has_target and y is not None:
        out[f"GT_{spare_type}"] = y.values

    out.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")

    return out
