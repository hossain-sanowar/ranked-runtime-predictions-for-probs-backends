import numpy as np

"""
Contains all functions regarding the conversion of three numeric values to rankings.
"""

RANK_LIST = ("kpz", "kzp", "pkz", "pzk", "zkp", "zpk")
number_of_ties = 0


def convert_compl_arr_to_ranks(y_arr):
    """
    Converts an array of three dimensional numeric predictions to an
    array of ranks in the form of strings
    """
    if not isinstance(y_arr[0], str):
        y_ranks = []
        for y in y_arr:
            rank = convert_to_rank(y)
            y_ranks.append(rank)
        return np.asarray(y_ranks)
    else:
        # already a rank_list
        return y_arr


def convert_to_rank(y_value):
    """
    Converts one three dimensional numeric prediction to a ranking in the form of a string
    """
    if isinstance(y_value, np.ndarray):
        rank = ""
        k = y_value[0]
        p = y_value[1]
        z = y_value[2]

        if k < p:
            if k < z:
                if p < z:
                    rank = "kpz"
                elif z < p:
                    rank = "kzp"
            elif z < k:
                rank = "zkp"
        elif p < k:
            if p < z:
                if k < z:
                    rank = "pkz"
                elif z < k:
                    rank = "pzk"
            elif z < p:
                rank = "zpk"

        # tie breaking with bias towards "pzk"
        if rank == "":
            print("TIE in: " + str(y_value)+"!")
            if p == z:
                if p <= k:
                    rank = "pzk"
                else:
                    rank = "kpz"
            elif p == k:            # p!=z and p==k
                if p < z:
                    rank = "pkz"
                elif p > z:
                    rank = "zpk"
            elif k == z:
                if p < k:
                    rank = "pzk"
                elif p > k:
                    rank = "zkp"
    else:
        rank = RANK_LIST[y_value]

    return rank
