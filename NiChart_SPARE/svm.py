"""
SVM hyperparameter grids shared by all pipeline modules.
"""
from .util import expspace


def get_svm_hyperparameter_grids() -> dict:
    """Return per-kernel hyperparameter search grids for classification and regression."""
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
