# Functions for data preprocessing for training and inference.

# FEATURES ADDED

# Data selection for 95% confidence interval
# SVR Correction

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import column_or_1d


# ############# DATA CHECK ##############
def validate_dataframe(df: pd.DataFrame, target_column: str) -> None:
    """Validate input dataframe and target column."""
    if df.empty:
        raise ValueError("Dataframe is empty")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")


# ############# DATA PREPROCESSING ##############
def load_csv_data(file_path, drop_columns=None):
    """Load CSV data and return dataframe, optionally dropping specified columns"""
    df = pd.read_csv(file_path)
    print(f"Loaded data: {len(df)} samples, {len(df.columns)} columns")
    if drop_columns:
        for col in drop_columns:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"Dropped column: {col}")
            else:
                print(f"Warning: Column to drop not found: {col}")
    return df


# Encode features
def encode_feature_df(df: pd.DataFrame):
    if (df.dtypes == 'object').any():
        columns_in_order = df.columns
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(f"Encoding the following columns: {object_cols}")
        df_oc = df[object_cols]
        encoders = {}
        for oc in object_cols:
            encoder = LabelEncoder()
            df_oc.loc[:, oc] = encoder.fit_transform(df_oc[oc].values)
            encoders[oc] = encoder
        #df_encoded = pd.DataFrame(df_oc, columns=object_cols, index=df.index)

        return pd.concat([df[~df.columns.isin(object_cols)], df_oc], axis=1)[columns_in_order], encoders
    else:
        return df, None
    
# Scale features
def scale_feature_df(df: pd.DataFrame):
    X = df.copy()
    scalers={}
    for c in X.columns:
        if c != 'Sex_M':
            scaler = StandardScaler()
            scaler.fit(X[c].to_numpy().reshape(-1,1))
            scalers[c] = scaler
            X[c] = scaler.transform(X[c].to_numpy().reshape(-1,1))
        else:
            X[c] = X[c].astype(float)
    return X, scalers


# Prepare data for regressor training and testing
# df: dataframe containing only essential columns + target column
def preprocess_regression_data(
    df: pd.DataFrame, 
    target_column: str,
    encode_categorical_features: bool = True,
    scale_features: bool = True,
    for_training: bool = True,
    feature_encoder: LabelEncoder = None, # for inference
    feature_scaler: StandardScaler = None, # for inference
):
    X, y = (None, None)
    
    if for_training:
        """Preprocess data for training: handle missing values and encode categorical featurs & targets."""
        feature_encoder, feature_scaler = (None, None)
        df = df.dropna(subset=[target_column]) # Remove rows with missing target values
        
        # Separate features and target
        print(f"Dropped target column: {target_column}")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode feature labels if they're not numeric and encoding is requested
        if encode_categorical_features:
            print(f"Encoding the categorical features.")
            X, feature_encoder = encode_feature_df(X)
        # Scale features if requested
        if scale_features:
            print(f"Scaling the features.")
            # feature_scaler = StandardScaler()
            # X = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)
            X, feature_scaler = scale_feature_df(X)

    else:
        """Preprocess data for inference: handle missing values and encode categorical features."""
        # Check if ground truth is provided in the df, if so, drop it.
        if target_column in df.columns:
            print(f"Dropped target column: {target_column}")
            X = df.drop([target_column],axis=1)
            y = df[target_column]
        else:
            X = df
            y = None
        
        if feature_encoder != None:
            for ec in feature_encoder.keys():
                X[ec] = feature_encoder[ec].transform(X[ec])
        
        if feature_scaler != None:
            for fs in feature_scaler.keys():
                X[fs] = feature_scaler[fs].transform(X[fs])
    
    return X, y, feature_encoder, feature_scaler #, target_scaler


# Prepare data for classifier training and testing
# df: dataframe containing only essential columns + target column
def preprocess_classification_data(
    df: pd.DataFrame, 
    target_column: str,
    encode_categorical_features: bool = True,
    scale_features: bool = True,
    # encode_categorical_target: bool = False,
    for_training: bool = True,
    feature_encoder: LabelEncoder = None, # for inference
    feature_scaler: StandardScaler = None, # for inference
    # target_encoder: StandardScaler = None # for inference
):
    X, y = (None, None)

    if for_training == True:
         # Separate features and target
        X = df.drop([target_column],axis=1)
        y = df[target_column]
        feature_encoder, feature_scaler = (None, None)
        """Preprocess data for training: handle missing values and encode categorical featurs & targets."""
        df = df.dropna(subset=[target_column]) # Remove rows with missing target values
        
        # Encode feature labels if they're not numeric and encoding is requested
        if encode_categorical_features:
            print(f"Encoding the categorical features.")
            X, feature_encoder = encode_feature_df(X)
        # Scale features if requested
        if scale_features:
            print(f"Scaling the features.")
            feature_scaler = StandardScaler()
            X = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)

        # # Encode target labels if they're not numeric and encoding is requested
        # if encode_categorical_target and y.dtype == 'object':
        #     print(f"Encoding the categorical label")
        #     target_encoder = LabelEncoder()
        #     y = target_encoder.fit_transform(y)
    
    else:
        """Preprocess data for inference: handle missing values and encode categorical features."""
        # Check if ground truth is provided in the df, if so, drop it.
        if target_column in df.columns:
            X = df.drop([target_column],axis=1)
            y = df[target_column]
        else:
            X = df
            y = None
        
        if feature_encoder != None:
            for ec in feature_encoder.keys():
                X[ec] = feature_encoder[ec].fit_transform(X[ec])
        
        if feature_scaler != None:
            X = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X, y, feature_encoder, feature_scaler #, target_encoder


# def apply_cvm_residualization(df: pd.DataFrame, 
#                               age_col='Age', 
#                               sex_col='Sex', 
#                               dlicv_col='702'):
    
#     df_params = pd.read_csv(os.path.join('../Reference', 'covparams_scaler_sparecvms.csv'))
    
#     rois = df_params.loc[df_params['Features'].str.contains('Volume'),'Features']
    
#     meanage = df['Age_Original'].mean() # change to a fixed location in residualization map
    
#     df['Age_Original'] = df['Age'].copy(deep=True)
#     df['Mean_centered_age'] = df['Age_Original']-meanage
#     df['DLICV_Original'] = df[dlicv_col].copy(deep=True)
    
#     if 'Sex_M' not in df.columns and 0 not in df[sex_col].unique():
#         df['Sex_M'] = df[sex_col].map({'F':0,'M':1})
    
#     features = [age_col, dlicv_col] + [roi for roi in rois]
#     confounds = ['Sex_M', 'Mean_centered_age', 'DLICV']

#     for roi in rois:
#         df['Orig_' + roi] = df[roi].copy(deep = True)
#         df['Pred_' + roi] = df_params.loc[df_params['Features'] == roi, 'Intercept'].values + np.matmul(df[confounds], df_params.loc[df_params['Features'] == roi, confounds].to_numpy().reshape(-1))
#         df[roi] =  df['Orig_' + roi] - df['Pred_' + roi] 

#     for ft in features:
#         df[ft] = (df[ft] - df_params.loc[df_params['Features'] == ft, 'Scaler_Mean'].values) / np.sqrt(df_params.loc[df_params['Features'] == ft, 'Scaler_Var'].values)

#     df = df.drop(columns = df.columns[df.columns.str.startswith('Pred')])

#     features = features + [sex_col,]

#     return df, features


