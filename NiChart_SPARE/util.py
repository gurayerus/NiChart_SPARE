"""
NiChart_SPARE shared utilities.
"""
import importlib

import numpy as np

SVM_TYPES = frozenset({'classification', 'regression'})


def get_pipeline_module(svm_type: str):
    """Return the pipeline module for the given svm_type ('classification' or 'regression')."""
    svm_type = svm_type.lower()
    if svm_type == 'regression':
        return importlib.import_module('NiChart_SPARE.pipelines.spare_svm_regression')
    if svm_type == 'classification':
        return importlib.import_module('NiChart_SPARE.pipelines.spare_svm_classification')
    raise ValueError(
        f"Unsupported svm_type '{svm_type}'. Expected 'classification' or 'regression'."
    )


# ---- Model metadata helpers ------------------------------------------------

def get_metadata(
    spare_type, svm_type, package_version, model_type, kernel, target_column,
    df, tune_hyperparameters, cv_fold, class_balancing, train_whole_set,
    model_tag=None, model_version=None,
) -> dict:
    return {
        'spare_type': spare_type,
        'svm_type':   svm_type,
        'package_version': package_version,
        'model_description': {
            'model_type': model_type,
            'kernel': kernel,
            'model_tag': model_tag,
            'model_version': model_version,
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
