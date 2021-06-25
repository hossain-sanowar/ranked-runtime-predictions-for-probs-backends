import unittest
import numpy as np

from learning.metrics.resulting_execution_time_conversion import TIMEOUT as TO, get_times_as_array


class TestResultingExecutionTimeConversion(unittest.TestCase):

    def test_get_times_as_array_only_VALID_or_INVALID_responses(self):
        # perfect prediction
        y_pred_ranks = ["kpz", "zpk"]
        y_true_numeric = [[1, 2, 3], [3, 2, 1]]

        arr = get_times_as_array(y_pred_ranks, y_true_numeric)

        self.assertEqual(np.array_equal(arr, [1, 1]), True)

        # worst prediction
        y_pred_ranks = ["zpk", "pzk"]
        y_true_numeric = [[3, 4, 5], [1, 3, 2]]

        arr = get_times_as_array(y_pred_ranks, y_true_numeric)

        self.assertEqual(np.array_equal(arr, [5, 3]), True)

        # medium prediction
        y_pred_ranks = ["pkz", "kpz", "zkp"]
        y_true_numeric = [[1, 2, 3], [1, 3, 2], [3, 9, 3]]

        arr = get_times_as_array(y_pred_ranks, y_true_numeric)

        self.assertEqual(np.array_equal(arr, [2, 1, 3]), True)

    def test_get_times_as_array_ONLY_UNKNOWN_or_TIMEOUT_or_ERROR(self):
        # Rankings dont matter
        y_pred_ranks1 = ["kpz", "zpk"]
        y_pred_ranks2 = ["zkp", "pkz"]
        y_pred_ranks3 = ["pzk", "pzk"]
        y_true_numeric = [[TO+1, 2*TO+2, 2*TO+3], [2*TO+3, TO+2, TO+1]]

        arr1 = get_times_as_array(y_pred_ranks1, y_true_numeric)
        arr2 = get_times_as_array(y_pred_ranks2, y_true_numeric)
        arr3 = get_times_as_array(y_pred_ranks3, y_true_numeric)

        self.assertEqual(np.array_equal(arr1, [6, 6]), True)
        self.assertEqual(np.array_equal(arr2, [6, 6]), True)
        self.assertEqual(np.array_equal(arr3, [6, 6]), True)

    def test_get_times_as_array_MIXED(self):
        y_pred_ranks = ["zkp", "kpz", "kzp"]
        y_true_numeric = [[TO+1, 2*TO+4, 3], [TO+7, TO+3, 5], [TO+3, 9, 1]]

        arr = get_times_as_array(y_pred_ranks, y_true_numeric)
        self.assertEqual(np.array_equal(arr, [3, 15, 4]), True)

        y_pred_ranks = ["pkz", "zpk", "kzp", "pkz"]
        y_true_numeric = [[3, 2*TO+4, 2*TO+3], [TO+7, TO+3, TO+5], [TO+3, 9, TO+1], [1, 9, 3]]

        arr = get_times_as_array(y_pred_ranks, y_true_numeric)
        # (7+15+13+9)/4 = 11
        self.assertEqual(np.array_equal(arr, [7, 15, 13, 9]), True)





