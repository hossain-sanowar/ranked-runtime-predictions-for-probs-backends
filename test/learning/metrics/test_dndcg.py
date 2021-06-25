import unittest

import numpy as np

from learning.metrics.dndcg import get_relevance, get_worst_ndcg, get_idcg, get_dndcg, get_dndcg_array
from learning.metrics.rank_conversion import convert_compl_arr_to_ranks


class TestDndcg(unittest.TestCase):

    def test_get_relevance(self):
        self.assertEqual(get_relevance("k", "kpz"), 3)
        self.assertEqual(get_relevance("p", "kpz"), 2)
        self.assertEqual(get_relevance("z", "kpz"), 1)
        self.assertEqual(get_relevance("g", "abcdefghi"), 3)

    def test_get_dndcg(self):

        idcg = get_idcg("kpz")
        worst_ndcg = get_worst_ndcg("kpz", idcg)

        self.assertAlmostEqual(get_dndcg("kpz", "kpz", idcg, worst_ndcg), 1.0)
        self.assertAlmostEqual(get_dndcg("kpz", "kzp", idcg, worst_ndcg), 0.9127, places=4)
        self.assertAlmostEqual(get_dndcg("kpz", "pkz", idcg, worst_ndcg), 0.5079, places=4)
        self.assertAlmostEqual(get_dndcg("kpz", "pzk", idcg, worst_ndcg), 0.2460, places=4)
        self.assertAlmostEqual(get_dndcg("kpz", "zkp", idcg, worst_ndcg), 0.1746, places=4)
        self.assertAlmostEqual(get_dndcg("kpz", "zpk", idcg, worst_ndcg), 0)

        # values from Healy's paper
        idcg = get_idcg("ABCDEFGH")
        worst_ndcg =get_worst_ndcg("ABCDEFGH", idcg)

        self.assertAlmostEqual(get_dndcg("ABCDEFGH", "ABCDEFGH", idcg, worst_ndcg), 1.0)

        self.assertAlmostEqual(get_dndcg("BACFDEGH", "ABCDEFGH", idcg, worst_ndcg), 0.778, places=3)

        self.assertAlmostEqual(get_dndcg("ABCDEFHG", "ABCDEFGH", idcg, worst_ndcg), 0.9998, places=4)
        self.assertAlmostEqual(get_dndcg("ABCDFEGH", "ABCDEFGH", idcg, worst_ndcg), 0.9989, places=4)
        self.assertAlmostEqual(get_dndcg("ABDCEFGH", "ABCDEFGH", idcg, worst_ndcg), 0.9899, places=4)
        self.assertAlmostEqual(get_dndcg("BACDEFGH", "ABCDEFGH", idcg, worst_ndcg), 0.7848, places=4)
        self.assertAlmostEqual(get_dndcg("ABCDHGFE", "ABCDEFGH", idcg, worst_ndcg), 0.9950, places=4)
        self.assertAlmostEqual(get_dndcg("DCBAEFGH", "ABCDEFGH", idcg, worst_ndcg), 0.3809, places=4)
        self.assertAlmostEqual(get_dndcg("HGFEDCBA", "ABCDEFGH", idcg, worst_ndcg), 0.0, places=4)

    def test_convert_to_dndcg_array(self):

        # perfect scoring on both predictions
        y_true_tuple_array = np.asarray([(1, 2, 3), (3, 2, 1)])
        y_pred_tuple_array = np.asarray([(1, 2, 3), (3, 2, 1)])

        y_true_ranks = convert_compl_arr_to_ranks(y_true_tuple_array)
        y_pred_ranks = convert_compl_arr_to_ranks(y_true_tuple_array)

        dndcg_array = get_dndcg_array(y_true_ranks, y_pred_ranks)
        self.assertAlmostEqual(np.asarray(dndcg_array).mean(), 1.0)

        # one perfect, other one worst
        y_true_tuple_array = np.asarray([(1, 2, 3), (3, 2, 1)])
        y_pred_tuple_array = np.asarray([(3, 2, 1), (3, 2, 1)])

        y_true_ranks = convert_compl_arr_to_ranks(y_true_tuple_array)
        y_pred_ranks = convert_compl_arr_to_ranks(y_pred_tuple_array)

        dndcg_list = get_dndcg_array(y_true_ranks, y_pred_ranks)
        self.assertAlmostEqual(np.asarray(dndcg_list).mean(), 0.5)
