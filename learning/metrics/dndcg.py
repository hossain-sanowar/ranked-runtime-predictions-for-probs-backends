import numpy as np
from learning.metrics import rank_conversion

"""
Contains all functions regarding the computation of the double normalized cumulative gain (dnDCG) value.
"""


def get_dndcg_array(y_true_ranks, y_pred_ranks):
    """
    Returns an array of dnDCG values for a prediction of rankings with regard to the ground truth
    """
    # get standard values
    idcg = get_idcg(y_true_ranks[0])
    worst_ndcg = get_worst_ndcg(y_true_ranks[0], idcg)

    dndcg_list = []
    for (y_true_rank, y_pred_rank) in zip(y_true_ranks, y_pred_ranks):
        dndcg_list.append(get_dndcg(y_true_rank, y_pred_rank, idcg, worst_ndcg))

    return np.asarray(dndcg_list)


def dndcg_function(y_true_data, y_pred_data):
    """
    scikit-learn scoring function:
    Computes the mean dnDCG value of multiple ground truth rankings and their predictions
    """
    # when performed for multioutput regression, y_true_data is a data frame and not a numpy array
    y_true_arr = np.asarray(y_true_data)
    y_pred_arr = np.asarray(y_pred_data)

    y_true_ranks = rank_conversion.convert_compl_arr_to_ranks(y_true_arr)
    y_pred_ranks = rank_conversion.convert_compl_arr_to_ranks(y_pred_arr)

    dndcg_mean = get_dndcg_array(y_true_ranks, y_pred_ranks).mean()

    return dndcg_mean


def get_dndcg(y_true_rank, y_pred_rank, idcg, worst_ndcg):
    ndcg_pred = get_ndcg(y_true_rank, y_pred_rank, idcg)
    dndcg_pred = (ndcg_pred - worst_ndcg) / (1 - worst_ndcg)

    return dndcg_pred


def get_ndcg(y_true_rank, y_pred_rank, idcg):
    true_len = len(y_true_rank)
    pred_dcg = get_dcg(y_true_rank, y_pred_rank, true_len, 0)
    return pred_dcg / idcg


def get_relevance(elm, y_true):
    return len(y_true)-y_true.index(elm)


def get_dcg(y_true_rank, y_pred_rank, p, i):
    if i < p:
        quotient = (2 ** get_relevance(y_pred_rank[i], y_true_rank) - 1) / np.log2(i + 2)
        return quotient + get_dcg(y_true_rank, y_pred_rank, p, i + 1)
    else:
        return 0


def get_idcg(y_true_rank):
    true_len = len(y_true_rank)
    return get_dcg(y_true_rank, y_true_rank, true_len, 0)


def get_worst_ndcg(y_true_rank, idcg):
    true_len = len(y_true_rank)
    return get_dcg(y_true_rank, y_true_rank[::-1], true_len, 0) / idcg

