# SPARE module to train a misc biomarker using undefined set

"""
SPARE-CVM Pipeline Module

This module contains functions for training and inference of SPARE-CVM 
cardiovascular & metabolic diease models. According to the method proposed by
Govindarajan et. al.
"""

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from ..data_analysis import (
	report_classification_metrics
)

from ..util import (
    get_hyperparameter_tuning
)

from ..svm import (
    get_svm_hyperparameter_grids
)

# Accepts dataframe and target_column as input along with other parameters to perform an svc training
def train_svc_model(
    X,
    y,
    kernel: str = 'linear', # linear_fast, linear, rbf, poly, sigmoid 
    tune_hyperparameters: bool = False,
    cv_fold: int = 5,
    class_balancing: bool = True,
    get_cv_scores: bool = True,
    train_whole_set: bool = True,
    random_state: int = 42, # for replication
    verbose: int = 1,
    **svc_params
    ):
    # Items to return
    model = None
    grid_search = None
    cv_scores = None
    best_cv_model = None
    best_cv_score = 0
    
    # Initialize base parameters
    if kernel == 'linear_fast':
        print(f"Training model with LinearSVC...")
        base_params = {'fit_intercept':True,
                       'random_state': random_state,
                       'verbose' : verbose > 1
                       }
    else:
        print(f"Training model with default SVC with {kernel} kernel...")
        base_params = {'kernel': kernel,
                       'probability':True, 
                       'random_state': random_state,
                       'verbose' : verbose > 1}
    
    # Overwrite base parameters with svc_params
    base_params.update(svc_params)
    
    # Enable class_weight='balanced' if class_balancing parameter is passed and True
    if class_balancing:
        base_params.update({'class_weight':'balanced'})
    
    # Perform hyperparameter tuning when asked
    hyperparameter_tuning={}
    if tune_hyperparameters:
        print(f"Hyperparameter selection initated...")
        param_grids = get_svm_hyperparameter_grids()['classification'][kernel]
             
        # Create base model
        if kernel == 'linear_fast':
            base_model = LinearSVC(**base_params)
        else:
            base_model = SVC(**base_params)
    
        # Perform grid search with 5-fold CV
        cv = RepeatedStratifiedKFold(n_splits=cv_fold,
                                     n_repeats=1, 
                                     random_state=random_state)
        
        grid_search = GridSearchCV(
            base_model,
            param_grids,
            cv=cv,
            scoring='balanced_accuracy' if class_balancing == True else 'accuracy',
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
        # Use default parameters
        svc_params.setdefault('random_state', random_state)

    # Perform another CV using the best parameter if get_cv_score parameter is True
    cv_scores = {}
    if get_cv_scores:
        print(f"Initiating {cv_fold}-fold CV")
        repeat=3
        for r in range(repeat):
            cv_scores["Repeat_%d"%r] = {'scores':{},
                                        'cv_results':{}}
        cv = RepeatedStratifiedKFold(n_splits=cv_fold, 
                                     n_repeats=repeat, 
                                     random_state=random_state)

        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            df_cv_result_per_fold = pd.DataFrame()
            
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            df_cv_result_per_fold['test_reference'] = y_test

            # Train model with current parameters
            if kernel == 'linear_fast':
                model = LinearSVC(**base_params)
            else:
                model = SVC(**base_params)
            
            model.fit(X_train, y_train)
            
            mdf = model.decision_function(X_test)
            # Get decision function
            df_cv_result_per_fold['test_decision_function'] = mdf
            # Predict
            y_pred = model.predict(X_test)
            df_cv_result_per_fold['test_prediction'] = y_pred
            # Add fold info
            df_cv_result_per_fold['fold'] = i % cv_fold
            # Get validation metrics
            cv_metric = report_classification_metrics(y_test, y_pred, mdf)
            print(f"Iteration {i} Repeat {(i)//cv_fold} Fold {i % cv_fold} metrics: {cv_metric}")
            # Save the scores
            cv_scores['Repeat_%d' % ((i)//cv_fold)]['scores']["Fold_%d" % (i % cv_fold)] = cv_metric
            cv_scores['Repeat_%d' % ((i)//cv_fold)]['cv_results']["Fold_%d" % (i % cv_fold)] = df_cv_result_per_fold
            # cv_scores['Repeat_%d' % ((i)//cv_fold)]['cv_results']

            # Update the best performing model based off of ROC-AUC
            if cv_metric['ROC-AUC'] > best_cv_score:
                best_cv_model = model
                best_cv_score = cv_metric['ROC-AUC']
            

    # Train model using the best parameter and whole set
    if train_whole_set:
        print("Training the wholeset.")
        if kernel == 'linear_fast':
            model = LinearSVC(**base_params)
        else:
            model = SVC(**base_params)
        model.fit(X, y)
    
    else:
        if tune_hyperparameters:
            model = grid_search.best_estimator_
        elif get_cv_scores:
            model = best_cv_model
    
    # Return model and the CV scores
    return model, hyperparameter_tuning, cv_scores

