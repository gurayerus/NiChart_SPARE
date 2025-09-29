"""
NiChart_SPARE Utilities
Common functions used across different SPARE pipeline modules.
"""
# import sys
import pandas as pd
import numpy as np
# import joblib
# from typing import Tuple, Optional, Dict, Any, Union
# from sklearn.preprocessing import StandardScaler, LabelEncoder
import importlib

def get_pipeline_module(spare_type):
    spare_type = spare_type.upper()
    module_map = {
        'CL': 'NiChart_SPARE.pipelines.spare_svm_classification',
        'RG': 'NiChart_SPARE.pipelines.spare_svm_regression',
        'CVM' : 'NiChart_SPARE.pipelines.spare_svm_cvm'
        # 'AD': 'NiChart_SPARE.pipelines.spare_ad',
        # 'BA': 'NiChart_SPARE.pipelines.spare_ba',
        # 'HT': 'NiChart_SPARE.pipelines.spare_ht',
    }
    if spare_type not in module_map:
        raise ValueError(f"Unsupported SVM SPARE type: {spare_type}")
    return importlib.import_module(module_map[spare_type])

# ############# Model Saving ################

def get_metadata(spare_type, 
                 package_version,
                 model_type, 
                 kernel, 
                 target_column,
                 df, 
                 tune_hyperparameters, 
                 cv_fold, 
                 class_balancing, 
                 train_whole_set):
    return {
            "spare_type":spare_type,
            "package_version":package_version,
            "model_description":{
                "model_type":model_type,
                "kernel":kernel
                },
            "training_data_description":{
                "target_column":target_column,
                "feature_names":[f for f in df.columns if f != target_column],
                "feature_count":len(df.columns.tolist())-1,
                "data_size":len(df)
                },
            "pipeline_description":{
                "hyperparameter_tuning":tune_hyperparameters,
                "cv_fold":cv_fold,
                "model_class_balancing":class_balancing,
                "trained_using_whole_set":train_whole_set
                }
            }

def get_preprocessors(feature_encoder, 
                      feature_scaler):
    return {
                'feature_encoder':feature_encoder,
                'feature_scaler':feature_scaler
            }

def get_hyperparameter_tuning(tuner, 
                              best_params, 
                              paramgrid):
    return {
                "hyperparameter_tuner":tuner,
                "best_params":best_params,
                "search_grid":paramgrid
            }

# ############# MISC ################
def expspace(span: list) -> np.ndarray:
    return np.exp(np.linspace(span[0], span[1], num=int(span[1]) - int(span[0]) + 1)).tolist()


def is_regression_model(spare_type):
    """Check if the SPARE type uses regression (continuous target)"""
    regression_types = ['BA']  # Brain Age is continuous
    return spare_type.upper() in regression_types
