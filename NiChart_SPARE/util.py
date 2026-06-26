"""
NiChart_SPARE shared utilities.
"""
import importlib

import numpy as np

REGRESSION_TYPES     = frozenset({'RG', 'BA'})
CLASSIFICATION_TYPES = frozenset({'CL', 'AD', 'CVM', 'HT', 'T2B', 'SM', 'BMI'})


def get_pipeline_module(spare_type: str):
    """Return the pipeline module for the given spare_type."""
    spare_type = spare_type.upper()
    if spare_type in REGRESSION_TYPES:
        return importlib.import_module('NiChart_SPARE.pipelines.spare_svm_regression')
    if spare_type in CLASSIFICATION_TYPES:
        return importlib.import_module('NiChart_SPARE.pipelines.spare_svm_classification')
    raise ValueError(
        f"Unsupported spare_type '{spare_type}'. "
        f"Expected one of: {sorted(REGRESSION_TYPES | CLASSIFICATION_TYPES)}"
    )


# ---- Model metadata helpers ------------------------------------------------

def get_metadata(
    spare_type, package_version, model_type, kernel, target_column,
    df, tune_hyperparameters, cv_fold, class_balancing, train_whole_set,
) -> dict:
    return {
        'spare_type': spare_type,
        'package_version': package_version,
        'model_description': {
            'model_type': model_type,
            'kernel': kernel,
        },
        'training_data_description': {
            'target_column': target_column,
            'feature_names': [f for f in df.columns if f != target_column],
            'feature_count': len(df.columns) - 1,
            'data_size': len(df),
        },
        'pipeline_description': {
            'hyperparameter_tuning': tune_hyperparameters,
            'cv_fold': cv_fold,
            'model_class_balancing': class_balancing,
            'trained_using_whole_set': train_whole_set,
        },
    }


def get_preprocessors(feature_encoder, feature_scaler) -> dict:
    return {'feature_encoder': feature_encoder, 'feature_scaler': feature_scaler}


def get_hyperparameter_tuning(tuner, best_params, paramgrid) -> dict:
    return {
        'hyperparameter_tuner': tuner,
        'best_params': best_params,
        'search_grid': paramgrid,
    }


# ---- Misc ------------------------------------------------------------------

def expspace(span: list) -> list:
    """Exponentially spaced values between 10^span[0] and 10^span[1]."""
    return np.exp(
        np.linspace(span[0], span[1], num=int(span[1]) - int(span[0]) + 1)
    ).tolist()


def get_svm_hyperparameter_grids() -> dict:
    """Per-kernel hyperparameter search grids for classification and regression."""
    classification_grids = {
        'linear_fast': {'C': expspace([-4, 1])},
        'linear':      {'C': expspace([-4, 1])},
        'rbf':         {'C': expspace([-4, 1]), 'gamma': ['scale', 'auto']},
        'poly':        {'C': expspace([-4, 1]), 'degree': [2, 3], 'gamma': ['scale', 'auto']},
        'sigmoid':     {'C': expspace([-4, 1]), 'gamma': ['scale', 'auto'], 'coef0': [-1, 0, 1]},
    }
    regression_grids = {
        'linear_fast': {'C': expspace([-4, 1]), 'epsilon': [0.01, 0.1, 0.2]},
        'linear':      {'C': expspace([-4, 1]), 'epsilon': [0.01, 0.1, 0.2]},
        'rbf':         {'C': expspace([-4, 1]), 'gamma': ['scale', 'auto'], 'epsilon': [0.01, 0.1, 0.2]},
        'poly':        {'C': expspace([-4, 1]), 'degree': [2, 3], 'gamma': ['scale', 'auto'], 'epsilon': [0.01, 0.1]},
        'sigmoid':     {'C': expspace([-4, 1]), 'gamma': ['scale', 'auto'], 'epsilon': [0.01, 0.1]},
    }
    return {'classification': classification_grids, 'regression': regression_grids}
