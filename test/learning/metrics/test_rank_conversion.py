import unittest
import pandas as pd
import numpy as np

from learning.metrics.rank_conversion import convert_to_rank, convert_compl_arr_to_ranks


class TestRankConversion(unittest.TestCase):

    def test_convert_to_rank_numeric_no_ties(self):
        self.assertEqual(convert_to_rank(np.asarray([1, 2, 3])), "kpz")
        self.assertEqual(convert_to_rank(np.asarray([1, 3, 2])), "kzp")
        self.assertEqual(convert_to_rank(np.asarray([2, 1, 3])), "pkz")
        self.assertEqual(convert_to_rank(np.asarray([3, 1, 2])), "pzk")
        self.assertEqual(convert_to_rank(np.asarray([2, 3, 1])), "zkp")
        self.assertEqual(convert_to_rank(np.asarray([3, 2, 1])), "zpk")

    def test_convert_to_rank_numeric_ties(self):
        self.assertEqual(convert_to_rank(np.asarray([1, 1, 1])), "pzk")
        self.assertEqual(convert_to_rank(np.asarray([1, 1, 2])), "pkz")
        self.assertEqual(convert_to_rank(np.asarray([2, 2, 1])), "zpk")
        self.assertEqual(convert_to_rank(np.asarray([1, 2, 2])), "kpz")
        self.assertEqual(convert_to_rank(np.asarray([2, 1, 1])), "pzk")
        self.assertEqual(convert_to_rank(np.asarray([1, 2, 1])), "zkp")
        self.assertEqual(convert_to_rank(np.asarray([2, 1, 2])), "pzk")

    def test_convert_to_rank_classification(self):
        self.assertEqual(convert_to_rank(0), "kpz")
        self.assertEqual(convert_to_rank(1), "kzp")
        self.assertEqual(convert_to_rank(2), "pkz")
        self.assertEqual(convert_to_rank(3), "pzk")
        self.assertEqual(convert_to_rank(4), "zkp")
        self.assertEqual(convert_to_rank(5), "zpk")

    def test_convert_compl_arr_to_ranks_numeric(self):
        y_arr = np.asarray([[1, 2, 3], [3, 1, 2]])
        out = convert_compl_arr_to_ranks(y_arr)
        self.assertEqual(np.array_equal(out, np.asarray(["kpz", "pzk"])), True)

        y_arr = np.asarray([[2, 3, 1], [1, 3, 2], [1, 2, 3]])
        out = convert_compl_arr_to_ranks(y_arr)
        self.assertEqual(np.array_equal(out, np.asarray(["zkp", "kzp", "kpz"])), True)

    def test_convert_compl_list_to_ranks_Already_Ranks(self):
        y_list = ["kzp", "zkp", "pzk"]
        y_arr = np.asarray(y_list)
        out = convert_compl_arr_to_ranks(y_list)
        self.assertEqual(np.array_equal(out, y_arr), True)

    def test_convert_compl_list_to_ranks_classification(self):
        y_list = [0, 1, 4]
        y_arr = np.asarray(y_list)
        out = convert_compl_arr_to_ranks(y_arr)
        y_ranks_arr = np.asarray(["kpz", "kzp", "zkp"])
        self.assertEqual(np.array_equal(out, y_ranks_arr), True)
