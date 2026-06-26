"""
SVM model training on standardized prepared data.

Expected input CSV format (produced by prep_data.prep_data):
  Column 0 : key_variable (MRID)       — dropped before training
  Column 1 : target column             — the variable to predict
  Column 2+: input features            — all used as-is

The module handles feature encoding, scaling, hyperparameter search,
cross-validation, and model serialization.  It does not perform any
task-specific data transforms — those belong in prep_data.py.
"""

import importlib
import joblib
import pandas as pd
from importlib.metadata import version

from .preprocessing import (
    load_csv_data,
    validate_dataframe,
    preprocess_classification_data,
    preprocess_regression_data,
)
from .util import get_metadata, get_preprocessors

VERSION = version("NiChart_SPARE")

REGRESSION_TYPES = frozenset({'RG', 'BA'})
CLASSIFICATION_TYPES = frozenset({'CL', 'AD', 'CVM', 'HT', 'T2B', 'SM', 'BMI'})


def train_model(
    input_file: str,
    model_path: str,
    spare_type: str,
    kernel: str = 'linear',
    tune_hyperparameters: bool = True,
    cv_fold: int = 5,
    class_balancing: bool = True,
    cross_validate: bool = True,
    train_whole_set: bool = True,
    bias_correction: int = 0,
    verbose: int = 1,
) -> None:
    """
    Train an SVM model on a prepared input CSV and save it to disk.

    Parameters
    ----------
    input_file : str
        Path to a CSV produced by prep_data.prep_data().
        Column 0 must be the key variable (MRID), column 1 the target,
        and all remaining columns are treated as input features.
    model_path : str
        Destination path for the saved model (.joblib).
    spare_type : str
        SPARE task type (CL, AD, CVM, HT, T2B, SM, BMI → classification;
        RG, BA → regression).  Stored in model metadata.
    kernel : str
        SVM kernel: linear_fast, linear, rbf, poly, sigmoid.
    tune_hyperparameters : bool
        Run GridSearchCV before final training.
    cv_fold : int
        Number of folds for cross-validation.
    class_balancing : bool
        Apply class_weight='balanced' (classification only).
    cross_validate : bool
        Run a separate CV pass to collect per-fold scores.
    train_whole_set : bool
        Train a final model on the full dataset after CV.
    bias_correction : int
        0 = none, 1 = Beheshti et al., 2 = Cole et al. (regression only).
    verbose : int
        Verbosity level (0–3).
    """
    spare_type = spare_type.upper()
    if spare_type not in REGRESSION_TYPES | CLASSIFICATION_TYPES:
        raise ValueError(
            f"Unknown spare_type '{spare_type}'. "
            f"Expected one of: {sorted(REGRESSION_TYPES | CLASSIFICATION_TYPES)}"
        )

    # --- Load prepared data ---
    df = load_csv_data(input_file)

    # Column roles are determined by position
    key_variable  = df.columns[0]
    target_column = df.columns[1]

    # Drop the key variable; keep [target, features...]
    df = df.drop(columns=[key_variable])
    validate_dataframe(df, target_column)

    # --- Encode, scale, and train ---
    if spare_type in REGRESSION_TYPES:
        X, y, feature_encoder, feature_scaler = preprocess_regression_data(
            df,
            target_column=target_column,
            encode_categorical_features=True,
            scale_features=True,
            for_training=True,
        )
        pipeline = importlib.import_module('NiChart_SPARE.pipelines.spare_svm_regression')
        model, bias, ht, cv = pipeline.train_svr_model(
            X, y,
            kernel=kernel,
            tune_hyperparameters=tune_hyperparameters,
            cv_fold=cv_fold,
            get_cv_scores=cross_validate,
            train_whole_set=train_whole_set,
            bias_correction=bias_correction,
            verbose=verbose,
        )

    else:  # classification
        X, y, feature_encoder, feature_scaler = preprocess_classification_data(
            df,
            target_column=target_column,
            encode_categorical_features=True,
            scale_features=True,
            for_training=True,
        )
        pipeline = importlib.import_module('NiChart_SPARE.pipelines.spare_svm_classification')
        model, ht, cv = pipeline.train_svc_model(
            X, y,
            kernel=kernel,
            tune_hyperparameters=tune_hyperparameters,
            cv_fold=cv_fold,
            class_balancing=class_balancing,
            get_cv_scores=cross_validate,
            train_whole_set=train_whole_set,
            verbose=verbose,
        )
        bias = None

    # --- Serialize ---
    meta_data   = get_metadata(
        spare_type, VERSION, 'SVM', kernel, target_column,
        df, tune_hyperparameters, cv_fold, class_balancing, train_whole_set,
    )
    preprocessor = get_preprocessors(feature_encoder, feature_scaler)

    joblib.dump(
        {
            'model':                {'model': model, 'bias': bias},
            'meta_data':            meta_data,
            'preprocessor':         preprocessor,
            'hyperparameter_tuning': ht,
            'cross_validation':     cv,
        },
        model_path,
    )
    print(f"Model saved to: {model_path}")
