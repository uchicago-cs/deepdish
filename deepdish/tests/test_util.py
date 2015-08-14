from __future__ import division, print_function, absolute_import
import unittest
import os
import numpy as np
import deepdish as dd


class TestUtil(unittest.TestCase):
    def test_pad(self):
        x = np.ones((2, 2))
        y = np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        y1 = dd.util.pad(x, (1, 2), value=0.0)
        np.testing.assert_array_equal(y, y1)

        x = np.ones((2, 2))
        y = np.array([[2, 2, 2, 2],
                      [2, 1, 1, 2],
                      [2, 1, 1, 2],
                      [2, 2, 2, 2]])
        y1 = dd.util.pad(x, 1, value=2.0)
        np.testing.assert_array_equal(y, y1)

    def test_pad_to_size(self):
        x = np.ones((2, 2))
        y = np.array([[1, 1, 0],
                      [1, 1, 0],
                      [0, 0, 0]])
        y1 = dd.util.pad_to_size(x, (3, 3), value=0.0)
        np.testing.assert_array_equal(y, y1)

    def test_pad_repeat_border(self):
        x = np.array([[1.0, 2.0],
                      [3.0, 4.0]])

        y = np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                      [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                      [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                      [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
                      [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
                      [3.0, 3.0, 3.0, 4.0, 4.0, 4.0]])
        y1 = dd.util.pad_repeat_border(x, 2)
        np.testing.assert_array_equal(y, y1)

        y = np.array([[1.0, 2.0],
                      [1.0, 2.0],
                      [1.0, 2.0],
                      [3.0, 4.0],
                      [3.0, 4.0],
                      [3.0, 4.0]])
        y1 = dd.util.pad_repeat_border(x, (2, 0))
        np.testing.assert_array_equal(y, y1)

    def test_pad_repeat_border_corner(self):
        x = np.array([[1.0, 2.0],
                      [3.0, 4.0]])

        y = np.array([[1.0, 2.0, 2.0, 2.0],
                      [3.0, 4.0, 4.0, 4.0],
                      [3.0, 4.0, 4.0, 4.0],
                      [3.0, 4.0, 4.0, 4.0]])
        y1 = dd.util.pad_repeat_border_corner(x, (4, 4))
        np.testing.assert_array_equal(y, y1)




if __name__ == '__main__':
    unittest.main()
