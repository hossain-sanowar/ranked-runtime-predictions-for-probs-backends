import numpy as np
from sklearn.metrics import balanced_accuracy_score
from learning.metrics import rank_conversion


def balanced_accuracy_function(y_true_data, y_pred_data):
    """
    scikit-learn scoring function:
    Computes the balanced accuracy score for multiple predictions with respect to the ground truth data
    """

    # when performed for multioutput regression, y_true_data is a data frame and not a numpy array
    y_true_arr = np.asarray(y_true_data)
    y_pred_arr = np.asarray(y_pred_data)

    y_true_ranks = rank_conversion.convert_compl_arr_to_ranks(y_true_arr)
    y_pred_ranks = rank_conversion.convert_compl_arr_to_ranks(y_pred_arr)

    return balanced_accuracy_score(y_true_ranks, y_pred_ranks)
