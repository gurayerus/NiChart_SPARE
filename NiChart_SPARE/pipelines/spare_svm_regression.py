# SPARE module to train a misc biomarker using undefined set

"""
SPARE-CL Pipeline Module

This module contains functions for training and inference of SPARE-CL models.
"""

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.linear_model import LinearRegression

from ..data_analysis import (
	report_regression_metrics
)

from ..util import (
    get_hyperparameter_tuning
)

from ..svm import (
    get_svm_hyperparameter_grids,
)

# Accepts dataframe and target_column as input along with other parameters to perform an svc training
def train_svr_model(
    X,
    y,
    kernel: str = 'linear', # linear_fast, linear, rbf, poly, sigmoid 
    tune_hyperparameters: bool = False,
    cv_fold: int = 5,
    get_cv_scores: bool = True,
    train_whole_set: bool = True,
    random_state: int = 42, # for replication
    bias_correction: int = 2,
    verbose: int = 1,
    **svc_params
    ):

    svc_params = {'C':1.0,'epsilon':0.1} # for debugging

    # Items to return
    model = None
    bias_terms = None
    grid_search = None
    cv_scores = None
    best_cv_model = None
    best_cv_score = 0
    
    # Initialize base parameters
    if kernel == 'linear_fast':
        print(f"Training model with LinearSVR...")
        base_params = {'fit_intercept':True,
                       'max_iter': 1000000,
                       'verbose' : verbose > 1}
    else:
        print(f"Training model with default SVR with {kernel} kernel...")
        base_params = {'kernel': kernel, 
                       #'random_state': random_state,
                       'verbose' : verbose > 1}
    # Overwrite base parameters with svc_params
    base_params.update(svc_params)
        
    # Perform hyperparameter tuning when asked
    hyperparameter_tuning={}
    if tune_hyperparameters:
        print(f"Hyperparameter selection initated...")
        param_grids = get_svm_hyperparameter_grids()['regression'][kernel]
             
        # Create base model
        if kernel == 'linear_fast':
            base_model = LinearSVR(**base_params)
        else:
            base_model = SVR(**base_params)
    
        # Perform grid search with 5-fold CV
        cv = RepeatedKFold(n_splits=cv_fold,
                           n_repeats=1, 
                           random_state=random_state)
        
        grid_search = GridSearchCV(
            base_model,
            param_grids,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=verbose
        )
        
        grid_search.fit(X, y)
    
        # Get best parameters and CV score & Update the svc_params
        # cv_score = grid_search.best_score_
        base_params.update(grid_search.best_params_)

        print(f"Best parameters: {base_params}")
        print(f"Best CV {grid_search.scorer_}: {grid_search.best_score_:.3f}")

        hyperparameter_tuning = get_hyperparameter_tuning(grid_search, base_params, param_grids)

    else:
        print(f"Hyperparameter selection skipped...")

    # Perform another CV using the best parameter if get_cv_score parameter is True
    cv_info = {}
    cv_scores = {}
    cv_indexes = {}
    if get_cv_scores:
        print(f"Initiating {cv_fold}-fold CV")
        repeat=1
        for r in range(repeat):
            cv_scores["Repeat_%d"%r] = {'scores':{},
                                        'cv_results':{}}
        cv = RepeatedKFold(n_splits=cv_fold, 
                           n_repeats=repeat,
                           random_state=random_state)
        
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            print(f"Iteration {i+1} Repeat {(i+1)//cv_fold} Fold {i % cv.n_repeats}")
            # Save indexes per fold
            # cv_indexes["Fold_%d" % (i % cv.n_repeats)] = {'train_index':train_index,'test_index':test_index}
            df_cv_result_per_fold = pd.DataFrame()

            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            
            df_cv_result_per_fold['test_reference'] = y_test

            # Train model with current parameters
            if kernel == 'linear_fast':
                model = LinearSVR(**base_params)
                model.fit(X_train, y_train)
            else:
                model = SVR(**base_params)
                model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            df_cv_result_per_fold['test_prediction'] = y_pred_test

            cv_metric = report_regression_metrics(y_test, y_pred_test)
            print(f"Pre-correction: {cv_metric}")

            # correct for bias
            if bias_correction not in [None, 0]:
                if bias_correction==1:
                    # print("Correcting bias (residual approach)")
                    cb_method = 'residual approach'
                    residuals = y_train - y_pred_train
                    lin_fit = LinearRegression().fit(y_train.to_frame(), residuals)
                    m, c = lin_fit.coef_[0], lin_fit.intercept_
                    y_pred_test = y_pred_test - lin_fit.predict(y_pred_test.reshape(-1,1))

                elif bias_correction==2:
                    # print("Correcting bias (Cole et al.)")
                    cb_method = 'Cole et al.'
                    reg = LinearRegression().fit(y_train.values.reshape(-1, 1), y_pred_train)
                    m, c = reg.coef_[0], reg.intercept_
                    y_pred_test = (y_pred_test - c) / m

                df_cv_result_per_fold['test_prediction_BC'] = y_pred_test
                cv_metric = report_regression_metrics(y_test, y_pred_test)
                print(f"Post-correction {cb_method}: {cv_metric}\n")
            else:
                print("Skipping Bias Correction\n")    
            # # Get validation metrics
            # cv_metric = report_regression_metrics(y_test, y_pred_test)
            # print(f"Iteration {i} Repeat {(i)//cv_fold} Fold {i % cv_fold} metrics: {cv_metric}\n")
            
            # Save the scores
            # cv_scores["Fold_%d" % (i % cv_fold)] = cv_metric
            df_cv_result_per_fold['fold'] = i % cv_fold
            cv_scores['Repeat_%d' % ((i)//cv_fold)]['scores']["Fold_%d" % (i % cv_fold)] = cv_metric
            cv_scores['Repeat_%d' % ((i)//cv_fold)]['cv_results']["Fold_%d" % (i % cv_fold)] = df_cv_result_per_fold

            # Update the best performing model based off of ROC-AUC
            if cv_metric['MSE'] > best_cv_score:
                best_cv_model = model
                best_cv_score = cv_metric['MSE']

    # Train model using the best parameter and whole set
    if train_whole_set:
        print("Training the wholeset.")
        if kernel == 'linear_fast':
            model = LinearSVR(**base_params)
            model.fit(X, y)
        else:
            model = SVR(**base_params)
            model.fit(X, y)
        
        if bias_correction==1:
            print("Correcting bias (residual approach)")
            y_pred = model.predict(X)
            residuals = y - y_pred
            lin_fit = LinearRegression().fit(y.to_frame(), residuals)
            bias_terms = {'model':lin_fit,
                          'method':bias_correction,
                          'intercept':lin_fit.intercept_, 
                          'coef': lin_fit.coef_[0]}
        elif bias_correction==2:
            print("Correcting bias (Cole et al.)")
            reg = LinearRegression().fit(y_train.values.reshape(-1, 1), y_pred_train)
            bias_terms = {'model':reg,
                          'method':bias_correction,
                          'intercept':reg.intercept_, 
                          'coef': reg.coef_[0]}

    else:
        if tune_hyperparameters:
            model = grid_search.best_estimator_
        elif get_cv_scores:
            model = best_cv_model

    # cv_info['CV_Indexes'] = cv_indexes
    # cv_info['CV_Scores'] = cv_scores

    # Return model and the CV scores
    return model, bias_terms, hyperparameter_tuning, cv_scores