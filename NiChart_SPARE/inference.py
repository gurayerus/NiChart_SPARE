"""
SVM model inference on raw or prepared data.

If the model was trained with inline prep (target_column set in config), the
prep configuration is stored inside the .joblib file and is applied automatically
before inference.  Otherwise the input must already be in the standard prepared
format (column 0 = key variable, remaining columns = features).
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

from .preprocessing import (
    load_csv_data,
    preprocess_classification_data,
    preprocess_regression_data,
)

def infer_model(
    input_file: str,
    model_path: str,
    output_dir: str,
    key_variable: str = 'MRID',
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
    output_dir : str
        Directory where predictions.csv (and any other outputs) are written.
        Created automatically if it does not exist.
    key_variable : str
        Column that uniquely identifies each sample (default: MRID).

    Returns
    -------
    pd.DataFrame
        Predictions DataFrame (also written to output_dir/predictions.csv).
    """
    # --- Load model ---
    model_data    = joblib.load(model_path)
    model_info    = model_data['model']
    meta_data     = model_data['meta_data']
    preprocessor  = model_data['preprocessor']
    prep_config   = model_data.get('prep_config')
    output_config = model_data.get('output_config')

    model       = model_info['model']
    bias_terms  = model_info['bias']
    spare_type  = meta_data['spare_type']
    svm_type    = meta_data.get('svm_type', 'regression' if spare_type in {'RG', 'BA'} else 'classification')
    target_col  = meta_data['training_data_description']['target_column']
    feature_names = meta_data['training_data_description']['feature_names']

    # Use key_col from stored prep config when available
    if prep_config and prep_config.get('key_col'):
        key_variable = prep_config['key_col']

    # --- Auto-apply prep when the model stores a prep config ---
    if prep_config:
        from .prep_data import prep_data
        os.makedirs(output_dir, exist_ok=True)
        prepped_path = os.path.join(output_dir, 'prepped.csv')
        prep_data(
            input_file=input_file,
            key_col=prep_config.get('key_col', 'MRID'),
            target_col=prep_config.get('target_col'),
            data_cols=prep_config.get('data_cols'),
            mappings=prep_config.get('mappings'),
            preprocessing=prep_config.get('preprocessing'),
            output_file=prepped_path,
            cvm_mean_age=prep_config.get('cvm_mean_age'),
        )
        input_file = prepped_path
        print(f"Prepped data saved to: {prepped_path}")

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

    if svm_type == 'regression':
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
    score_col = f"SPARE_{spare_type}"
    if svm_type != 'regression':
        score_values = model.decision_function(X)
    else:
        score_values = predictions
    out = pd.DataFrame({key_variable: df[key_variable].values, score_col: score_values})

    if has_target and y is not None:
        out[f"GT_{spare_type}"] = y.values

    # Apply output column filter and filename from saved output_config
    if output_config:
        out_cols = output_config.get('out_cols')
        if out_cols:
            missing = [c for c in out_cols if c not in out.columns]
            if missing:
                import sys as _sys
                print(f"Warning: out_cols references unknown column(s), skipped: {missing}",
                      file=_sys.stderr)
            out = out[[c for c in out_cols if c in out.columns]]
        out_csv = output_config.get('out_csv', 'predictions.csv')
    else:
        out_csv = 'predictions.csv'

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, out_csv)
    out.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")

    # --- Evaluation (only when ground truth is available) ---
    if has_target and y is not None:
        from .data_analysis import report_classification_metrics, report_regression_metrics
        if svm_type == 'regression':
            metrics = report_regression_metrics(y, score_values)
            summary = f"MAE: {metrics['MAE']:.3f}  RMSE: {metrics['RMSE']:.3f}  R2: {metrics['R2']:.3f}"
        else:
            binary_preds = (score_values > 0).astype(int)
            metrics = report_classification_metrics(y, binary_preds, score_values)
            summary = (f"ROC-AUC: {metrics['ROC-AUC']:.3f}  "
                       f"Accuracy: {metrics['Accuracy']:.3f}  "
                       f"Balanced Accuracy: {metrics['Balanced Accuracy']:.3f}")
        print(f"Evaluation  : {summary}")
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Metrics saved to: {metrics_path}")

    return out
