"""
Sklearn SVM specific functions
"""
import sys
import joblib
from importlib.metadata import version
# from typing import Tuple, Optional, Dict, Any
# from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

from .util import (
	expspace, 
    get_pipeline_module,
    get_metadata,
    get_preprocessors
)

from .data_prep import (
	load_csv_data, 
	validate_dataframe, 
	preprocess_classification_data, 
	preprocess_regression_data
)

VERSION = version("NiChart_SPARE")

def get_svm_hyperparameter_grids():
    """Get hyperparameter grids for different kernels and model types."""
    classification_grids = {
        'linear_fast': {
            "C": expspace([-4, 1])
        },
        'linear': {
            "C": expspace([-4, 1])
        },
        'rbf': {
            "C": expspace([-4, 1]),
            "gamma": ['scale', 'auto']
        },
        'poly': {
            "C": expspace([-4, 1]),
            'degree': [2, 3],
            'gamma': ['scale', 'auto']
        },
        'sigmoid': {
            "C": expspace([-4, 1]),
            'gamma': ['scale', 'auto'],
            'coef0': [-1, 0, 1]
        }
    }
    regression_grids = {
        'linear_fast': {
            'C':  expspace([-4, 1]),
            'epsilon': [0.01, 0.1, 0.2]
        },
        'linear': {
            'C':  expspace([-4, 1]),
            'epsilon': [0.01, 0.1, 0.2]
        },
        'rbf': {
            'C':  expspace([-4, 1]),
            'gamma': ['scale', 'auto'],
            'epsilon': [0.01, 0.1, 0.2]
        },
        'poly': {
            'C':  expspace([-4, 1]),
            'degree': [2, 3],
            'gamma': ['scale', 'auto'],
            'epsilon': [0.01, 0.1]
        },
        'sigmoid': {
            'C':  expspace([-4, 1]),
            'gamma': ['scale', 'auto'],
            'epsilon': [0.01, 0.1]
        }
    }
    
    return {
        'classification': classification_grids,
        'regression': regression_grids
    }



def train_svm_model(input_file, 
					model_path, 
					spare_type, 
					target_column, 
					kernel, 
					tune_hyperparameters, 
                    cross_validate,
                    cv_fold,
                    class_balancing,
					train_whole_set, 
                    bias_correction,
					drop_columns=None,
                    verbose=1):
    
    model = None
    bias = None
    meta_data={}
    preprocessor={}
    # hyperparameter_tuning={}
    # cross_validation={}
    
    """Train model using the pipeline functions"""
    # Load data
    print("Loading training data...")
    df = load_csv_data(input_file, drop_columns=drop_columns)
    
    # Input validation
    print(f"Validating input...")
    validate_dataframe(df, target_column)
    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")
    print(f"Success.")

    # Get pipeline module
    pipeline_module = get_pipeline_module(spare_type)
   
    # Initialize variables
    feature_encoder, feature_scaler = (None, None)

    # Regression tasks
    if spare_type in ['RG','BA']:
        # Preprocess data (no label encoding for regression)
        print(f"Preprocessing the input...{df.shape}")
        X, y, feature_encoder, feature_scaler = preprocess_regression_data( 
            df, 
            target_column = target_column, 
            encode_categorical_features=True,
            scale_features=True,
            # scale_target=False,
            for_training=True)
        print(f"Input preprocessing completed.")

        # Training
        if kernel.lower() in ['linear_fast','linear','poly', 'rbf', 'sigmoid']:
            model, bias, ht, cv = pipeline_module.train_svr_model(
                X,
                y,
                kernel=kernel,
                tune_hyperparameters=tune_hyperparameters,
                cv_fold=cv_fold,
                get_cv_scores=cross_validate,
                train_whole_set=train_whole_set,
                bias_correction=bias_correction,
                verbose=verbose
                )
        else:
            print(f"Unsupported SVM kernel entry. Please select among: linear, poly, rbf, sigmoid.")
    
    # Classification tasks
    elif spare_type in ['CL','AD']:
        
        # Input validation
        print(f"Validating input...")
        validate_dataframe(df, target_column)
        print(f"Success.")

        # Preprocess the input df, split into X, y
        print(f"Preprocessing the input...{df.shape}")
        X, y, feature_encoder, feature_scaler = preprocess_classification_data(
            df, 
            target_column = target_column, 
            encode_categorical_features=True,
            scale_features=True,
            # encode_categorical_target=False,
            for_training=True)
        print(f"Input preprocessing completed.")

        # Training
        if kernel.lower() in ['linear_fast','linear','poly', 'rbf', 'sigmoid']:
            model, ht, cv = pipeline_module.train_svc_model(
                X,
                y,
                kernel=kernel,
                tune_hyperparameters=tune_hyperparameters,
                cv_fold=cv_fold,
                class_balancing=class_balancing,
                get_cv_scores=cross_validate,
                train_whole_set=train_whole_set
                )
        else:
            print(f"Unsupported SVM kernel entry. Please select among: linear, poly, rbf, sigmoid.")

    else:
        print(f"{spare_type} is not supported.")
        sys.exit(1)

    if model != None and model_path != None:
            # Create Metadata for model saving
            meta_data = get_metadata(spare_type, 
                                    "0.1.0",
                                    "SVM", 
                                    kernel, 
                                    target_column,
                                    df, 
                                    tune_hyperparameters, 
                                    cv_fold, 
                                    class_balancing, 
                                    train_whole_set)
            # Create info
            preprocessor = get_preprocessors(feature_encoder, feature_scaler)
            # Save model
            save_svm_model({'model':model, 'bias':bias}, 
                           meta_data, 
                           preprocessor, 
                           ht, 
                           cv, 
                           model_path)
            print(f"Model saved to: {model_path}")
    else:
        raise("Error: Missing model or output path to save.")


# Save the trained model and components to a file.
def save_svm_model(
    model, # main model (best performing model)
    meta_data:dict, # spare_type, feature_names, data_description, 
    preprocessor: dict, # encoder & scaler
    hyperparameter_tuning:dict,  #  Search strategy, param grid, best params, scoring
    cross_validation:dict, # strategy, n_splits, scores (each fold)
    filepath: str # path to save the model
) -> None:
    model_data = {
        'model': model,
        'meta_data':meta_data,
        'preprocessor': preprocessor,
        'hyperparameter_tuning':hyperparameter_tuning,
        'cross_validation':cross_validation
    }
    joblib.dump(model_data, filepath)


# Load a trained model and components from a file
def load_svm_model(filepath: str):
    model_data = joblib.load(filepath)
    return model_data['model'], model_data['meta_data'], model_data['preprocessor'], model_data['hyperparameter_tuning'], model_data['cross_validation']


def infer_svm_model(input_file, 
                    model_path, 
                    spare_type, 
                    output_file, 
                    key_variable='MRID',
                    drop_columns=None):
    """Make predictions using trained model"""
    
    # Load model
    print("Loading trained model...")
    model_info, meta_data, preprocessor, _, _ = load_svm_model(model_path) # TBF

    model = model_info['model']
    bias_terms = model_info['bias']

    # Load data
    print("Loading prediction data...")
    df = load_csv_data(input_file, drop_columns=drop_columns)

    # # *(TBF) in case non-encoded Sex column in the input df
    # if "Sex" in df.columns.tolist() and "Sex_M" not in df.columns.tolist():
    #     if "M" in df['Sex'].unique():
    #         df['Sex_M'] = df['Sex'].apply(lambda x: 1 if x=='M' else 'F')

    # Check all columns exist in the input file
    for nf in meta_data['training_data_description']['feature_names']:
        if nf not in df.columns:
            raise("Missing columns:"+nf)
        else:
            print(f"Checked:\t{nf}")

    # subset for only needed columns
    if meta_data['training_data_description']['target_column'] in df.columns.tolist():
        print(f"DEBUG: target_column is in df columns.")
        print(f"DEBUG: key_variable: {key_variable}")
        print(f"DEBUG: meta_data['training_data_description']['feature_names']: {meta_data['training_data_description']['feature_names']}")
        print(f"DEBUG: meta_data['training_data_description']['target_column']: {meta_data['training_data_description']['target_column']}")
        df = df[[key_variable, meta_data['training_data_description']['target_column']] + meta_data['training_data_description']['feature_names']]
    else:
        print(f"DEBUG: target column is not in df columns.")
        print(f"DEBUG: key_variable: {key_variable}")
        print(f"DEBUG: meta_data['training_data_description']['feature_names']: {meta_data['training_data_description']['feature_names']}")
        print(f"DEBUG: meta_data['training_data_description']['target_column']: {meta_data['training_data_description']['target_column']}")
        df = df[[key_variable] + meta_data['training_data_description']['feature_names']]


    print(f"Preprocessing the input...{df.shape}")

    # Regression task
    if spare_type in ['RG','BA']:
        X, y, _, _ = preprocess_regression_data( 
            df = df.drop([key_variable],axis=1),
            target_column = meta_data['training_data_description']['target_column'],
            feature_encoder = preprocessor['feature_encoder'],
            feature_scaler= preprocessor['feature_scaler'],
            for_training=False
            )
        print(f"Input preprocessing completed. Feature shape: {X.shape}")
        pass        
    # Classification task 
    elif spare_type in ['CL','AD']:
        # Preprocess data
        
        X, y, _, _ = preprocess_classification_data( 
            df = df.drop([key_variable],axis=1),
            target_column = meta_data['training_data_description']['target_column'],
            feature_encoder = preprocessor['feature_encoder'],
            feature_scaler= preprocessor['feature_scaler'],
            for_training=False
            )
        print(f"Input preprocessing completed. Feature shape: {X.shape}")
    
    
    # Get prediction
    predictions = model.predict(X)
    # Correct for bias
    if bias_terms != None:
        if bias_terms['method'] == 1:
            print("Correcting bias (residual approach)")
            predictions = predictions - bias_terms['model'].predict(predictions.reshape(-1,1))
        elif bias_terms['method'] == 2:
            print("Correcting bias (Cole et al.)")
            predictions = (predictions - bias_terms['intercept']) / bias_terms['coef']
    
    # Create output dataframe
    output_df = pd.DataFrame()
    output_df[key_variable] = df[key_variable]
    output_df['SPARE_'+spare_type] = predictions
    
    if spare_type in ['CL','AD']:
        output_df['SPARE_'+spare_type+'_decision_function'] = model.decision_function(X)

    if meta_data['training_data_description']['target_column'] in df.columns:
        output_df['GT_'+spare_type] = y
    
    # Save predictions
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")
