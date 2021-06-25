import numpy as np

"""
This module contains the necessary tools to compute the resulting time to receive the best answer possible
on the decidability of a list of predicates when following a certain ranking.
Forms the foundation of the computation of the UDC value.
"""

TIMEOUT = 25e9
BACKEND_TO_INDEX_DICT = {"k": 0, "p": 1, "z": 2}


def get_times_as_array(y_ranks, y_true_numeric):
    times = []
    for (rank, number_arr) in zip(y_ranks, y_true_numeric):
        time = 0
        for backend_letter in rank:
            index = BACKEND_TO_INDEX_DICT[backend_letter]
            costs = number_arr[index]
            if 0 <= costs < TIMEOUT:
                time += costs
                break

            if TIMEOUT <= costs < 2*TIMEOUT:
                time += costs - TIMEOUT

            elif 2*TIMEOUT <= costs:        # technically <= 3*TIMEOUT
                time += costs - 2*TIMEOUT

        times.append(time)

    return np.asarray(times)












