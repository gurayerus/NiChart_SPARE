import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def report_regression_metrics(y_true, y_pred):
    """Report regression metrics: MAE, MSE, RMSE, MAPE, sMAPE, R2, Adjusted R2."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    n = len(y_true)
    k = 1 if y_pred.ndim == 1 else y_pred.shape[1]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan

    return {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'sMAPE': float(smape),
        'R2': float(r2),
        'Adjusted R2': float(adj_r2)
    }


def report_classification_metrics(y_true, y_pred, y_decision_function):
    """Report classification metrics: ROC-AUC, Accuracy, Balanced Accuracy, Sensitivity, Specificity, Precision, Recall, F1."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    n_classes = len(np.unique(y_true))

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary' if n_classes == 2 else 'weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary' if n_classes == 2 else 'weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary' if n_classes == 2 else 'weighted', zero_division=0)

    # Sensitivity (Recall for positive class)
    if n_classes == 2:
        sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    else:
        sensitivity = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    # Specificity (Recall for negative class)
    if n_classes == 2:
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except Exception:
            specificity = None
    else:
        specificity = None  # Not well-defined for multiclass

    # ROC-AUC
    try:
        if n_classes == 2:
            roc_auc = roc_auc_score(y_true, y_decision_function)
        else:
            roc_auc = roc_auc_score(y_true, y_decision_function, multi_class='ovr')
    except Exception:
        roc_auc = None

    return {
        'Accuracy': float(accuracy),
        'Balanced Accuracy': float(balanced_accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1': float(f1),
        'Sensitivity': float(sensitivity),
        'Specificity': float(specificity),
        'ROC-AUC': float(roc_auc)
    }


###################################################################
########################  FEATURE ANALYSIS ########################
###################################################################

def get_linear_svm_feature_importance(model, 
                                      feature_names: list) -> pd.DataFrame:
    """Get feature importance from linear SVM model."""
    if model.kernel != 'linear':
        print("Feature importance is only meaningful for linear kernels")
        return pd.DataFrame()
    
    # Get feature importance from linear SVM
    importance = np.abs(model.coef_[0])
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


# Calculate the effect size of a disease for brain age model
# Optional figure generation for different disease classes
def ba_disease_effect_analysis(df, 
                               age_column = 'Age',
                               ba_columns = 'SPARE_BA',
                               disease_column='DX', 
                               export_figure=False):
    return 


# Partial Dependency Plot
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
# Generates and displays a Partial Dependence Plot (PDP) for specified features
# This function visualizes the marginal effect of one or two features on the predicted outcome of a trained machine learning model
def generate_pdp_plot(model, X_train, features_to_plot = ['Age','Sex']):
    """
    Args:
        model: A trained, scikit-learn compatible model (e.g., RandomForest, SVM).
        X_train (pd.DataFrame): The training data used to fit the model. It's
                                recommended to use the training set to create
                                the plot.
        features_to_plot (list or tuple): A list containing the name(s) of the
                                           feature(s) to plot.
                                           - For a one-way PDP, provide one
                                             feature name: ['FeatureName']
                                           - For a two-way PDP (heatmap),
                                             provide two: ['Feature1', 'Feature2']
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    print(f"Generating PDP for feature(s): {features_to_plot}...")

    # Create the PDP display object
    # The `from_estimator` method handles the calculations and plotting
    try:
        display = PartialDependenceDisplay.from_estimator(
            estimator=model,
            X=X_train,
            features=features_to_plot,
            kind='average',  # 'average' plots the mean response, 'individual' shows ICE plots
            grid_resolution=50 # Number of points in the grid for the feature
        )

        # Customize the plot appearance
        display.figure_.suptitle(
            f'Partial Dependence Plot for {", ".join(features_to_plot)}',
            fontsize=16
        )
        display.figure_.subplots_adjust(top=0.9) # Adjust title position
        plt.show()
    except Exception as e:
        print(f"An error occurred while generating the PDP plot: {e}")


# Generates and displays Individual Conditional Expectation (ICE) plots for a specified feature
# This function visualizes how the model's prediction for individual instances changes as a single feature's value changes.
def generate_ice_plot(model, X_train, feature_to_plot, n_ice_lines=50, kind='individual'):
    """
    Args:
        model: A trained, scikit-learn compatible model (e.g., RandomForest, SVM).
        X_train (pd.DataFrame): The training data used to fit the model.
        feature_to_plot (str): The name of the single feature to plot.
        n_ice_lines (int): The number of individual ICE lines to draw. A smaller
                           number is chosen to avoid cluttering the plot. The lines
                           are sampled randomly from X_train.
        kind (str): The type of plot to generate.
                    - 'individual': Shows only the ICE plots.
                    - 'average': Shows only the average (PDP).
                    - 'both': Shows ICE plots with the PDP overlaid in a
                              different color for context.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame to use feature names.")
    if not isinstance(feature_to_plot, str):
        raise TypeError("feature_to_plot must be a single string.")

    print(f"Generating ICE plots for feature: '{feature_to_plot}' (kind='{kind}')...")

    # Randomly sample instances from X_train to plot, to keep the plot readable
    ice_sample_indices = np.random.choice(X_train.index, size=n_ice_lines, replace=False)
    X_sample = X_train.loc[ice_sample_indices]

    try:
        # The from_estimator method handles the calculations and plotting.
        # We pass the full X_train for calculation but will only display
        # lines for our X_sample by using the `subsample` parameter.
        display = PartialDependenceDisplay.from_estimator(
            estimator=model,
            X=X_train,
            features=[feature_to_plot],
            kind=kind,
            subsample=X_sample, # Use our random sample for the lines
            grid_resolution=50,
            random_state=0,
            pd_line_kw={"color": "red", "linestyle": "--", "linewidth": 3} # Style for PDP line
        )

        # --- Customize the plot appearance for better readability ---
        title_map = {
            'individual': f'ICE Plots for {feature_to_plot}',
            'both': f'ICE (blue) and PDP (red) for {feature_to_plot}',
            'average': f'PDP for {feature_to_plot}'
        }
        fig = display.figure_
        fig.suptitle(title_map.get(kind, ''), fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.9)
        display.axes_[0,0].set_ylabel('Change in Prediction')

        plt.show()

    except Exception as e:
        print(f"An error occurred while generating the ICE plot: {e}")


##################################################################
######################## Effect ANALYSIS #########################
##################################################################
from scipy import stats

def t_test(sample1, sample2, equal_variance=True):
  """
  Performs an independent two-sample t-test.

  Args:
    sample1 (array-like): The first sample of data.
    sample2 (array-like): The second sample of data.
    equal_variance (bool): If True, performs a standard independent t-test that
      assumes equal population variances. If False, performs Welch's t-test,
      which does not assume equal population variance. Defaults to True.

  Returns:
    tuple: A tuple containing the t-statistic and the p-value.
  """
  t_statistic, p_value = stats.ttest_ind(a=sample1, b=sample2, equal_var=equal_variance)
  return t_statistic, p_value


def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

import os
import seaborn as sns
# For disease effect analysis. 
def ba_effect_analysis(df_ba = None, 
                       col_ba='SPARE_BA', 
                       df_disease = None,
                       col_disease='disease', 
                       df_covars = None,
                       key_variable='MRID',
                       col_ref_age='Age',
                       ax=None):
    
    df_merged = df_covars[[key_variable,col_ref_age]].merge(df_ba[[key_variable,col_ba]],on=key_variable,how='inner')
    df_merged = df_merged.merge(df_disease[[key_variable,col_disease]],on=key_variable,how='inner').dropna().reset_index(drop=True)
    df_merged['BA_Gap'] = df_merged[col_ba] - df_merged[col_ref_age]
    
    cohens_d = "%.3f" % cohen_d(df_merged['BA_Gap'],df_merged[col_disease])
    t,p = t_test(df_merged['BA_Gap'],df_merged[col_disease])

    sns.histplot(data=df_merged,x='BA_Gap',hue='disease',ax=ax)
    ax.set_title(f"Cohen's d: {cohens_d}, \nt: {t}, p: {p}")


####################################
###### Model level analysis ########
####################################


def get_cv_scores_from_model(model,
                             repeat_label=0):
    df_cv_scores = pd.DataFrame(model['cross_validation']['Repeat_%d'%repeat_label]['scores'])
    df_cv_scores['Average'] = df_cv_scores.mean(axis=1)
    return df_cv_scores


def get_cv_results_from_model(model,
                              repeat_label=0):
    return pd.concat(model['cross_validation']['Repeat_%d'%repeat_label]['cv_results'].values(),axis=0).sort_index()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_auc(model,
                 ax=None,
                 repeat_label=0,
                 title='Receiver Operating Characteristic (ROC) Curve'):
    """
    Generates and displays an ROC-AUC curve from a NiChart_SPARE CL model.

    Args:
        model: The NiChart_SPARE model file to fetch the scores.
    """
    df = pd.concat(model['cross_validation']['Repeat_%d'%repeat_label]['cv_results'].values(),axis=0).sort_index()

    # --- Input Validation ---
    required_columns = {'test_reference', 'test_prediction', 'test_decision_function'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain the columns: {', '.join(required_columns)}")

    # Extract the necessary columns from the DataFrame
    y_true = df['test_reference']
    y_score = df['test_decision_function']

    # --- ROC Curve Calculation ---
    # Compute the false positive rate (FPR), true positive rate (TPR), and thresholds
    # The roc_curve function returns these values which are essential for plotting.
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # --- AUC Calculation ---
    # Compute the Area Under the Curve (AUC) using the calculated FPR and TPR.
    # This value gives a single measure of the model's performance.
    roc_auc = auc(fpr, tpr)

    # --- Plotting the ROC Curve ---
    if ax == None:
        plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style for the plot
        plt.figure(figsize=(10, 8))
        # Plot the ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        # Plot the "chance" line, which represents a random classifier
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # --- Formatting the Plot ---
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR', fontsize=14)
        plt.ylabel('TPR', fontsize=14)
        plt.title(title +' Repeat %d' % repeat_label, fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True)
         # Display the plot
        plt.show()
    else:
        # Plot the ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        # Plot the "chance" line, which represents a random classifier
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # --- Formatting the Plot ---
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('FPR', fontsize=14)
        ax.set_ylabel('TPR', fontsize=14)
        ax.set_title(title +' Repeat %d' % repeat_label, fontsize=16)
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True)
    