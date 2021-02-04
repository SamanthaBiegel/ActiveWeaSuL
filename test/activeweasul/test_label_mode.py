import unittest
from activeweasul.label_model2 import LabelModel
import numpy as np


class LabelModelTest(unittest.TestCase):
    def test_get_psi(self):
        L = np.array([[0, 1, 0], [2, 2, 0], [1, 1, 0], [-1, -1, 0]])
        expected = {
            2: (
                np.array([[1, 0, 1, 1, 0], [0, 0, 0, 0, 0],
                          [0, 1, 1, 0, 1], [0, 0, 0, 0, 0]]),
                {'0': [0, 1], '1': [2], '0_1': [3, 4]}
            ),
            3: (
                np.array([[1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0, 1],
                          [0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
                {'0': [0, 1, 2], '1': [3, 4], '0_1': [5, 6, 7]}
            )
        }
        for card, exp in expected.items():
            lm = LabelModel(cardinality=card)
            res = lm.get_psi(L, cliques=[[0], [1], [0, 1]], nr_wl=0)
            np.testing.assert_array_equal(res[0], exp[0])
            assert res[1] == exp[1]
