import numpy as np


def binary_classification_metrics(y_pred, y_true, value=1):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    tp = len(y_pred[(y_pred == y_true) & (y_pred == value)])
    tn = len(y_pred[(y_pred == y_true) & (y_pred != value)])
    fp = len(y_pred[(y_pred != y_true) & (y_pred == value)])
    fn = len(y_pred[(y_pred != y_true) & (y_pred != value)])
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    recall = tp/(tp + fn)
    precision = tp/(tp + fp)
    f1_score = (2 * recall * precision)/(recall + precision)
    return accuracy, recall, precision, f1_score





def multiclass_f1_score(y_pred, y_true):
    """
    Computes macro_averaging f1_score for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    macro_averaging f1_score
    """

    f1_score_per_class = []
    for i in range(10):
        _, _, _, f1_score = binary_classification_metrics(y_pred, y_true, value=i)
        f1_score_per_class.append(f1_score)
    macroavg_f1_score = np.array(f1_score_per_class).mean()
    return macroavg_f1_score




def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    r2 = 1 - np.sum((y_pred-y_true)**2)/np.sum((y_true-y_true.mean())**2)
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    mse = np.mean(np.sum((y_pred - y_true)**2))
    return mse





def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    mae = np.mean(np.sum(abs(y_pred - y_true)))
    return mae

#%%
