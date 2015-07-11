from __future__ import division, print_function, absolute_import
import unittest
from tempfile import NamedTemporaryFile
import os
import numpy as np
import deepdish as dd
from contextlib import contextmanager


class TestCore(unittest.TestCase):
    def test_multi_range(self):
        x0 = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        x1 = list(dd.multi_range(2, 3))
        assert x0 == x1

    def test_bytesize(self):
        assert dd.humanize_bytesize(1) == '1 B'
        assert dd.humanize_bytesize(2 * 1024) == '2 KB'
        assert dd.humanize_bytesize(3 * 1024**2) == '3 MB'
        assert dd.humanize_bytesize(4 * 1024**3) == '4 GB'
        assert dd.humanize_bytesize(5 * 1024**4) == '5 TB'

        assert dd.bytesize(np.ones((5, 2), dtype=np.int16)) == 20

        assert dd.memsize(np.ones((5, 2), dtype=np.int16)) == '20 B'

    def test_span(self):
        assert dd.span(np.array([0, -10, 20])) == (-10, 20)

    def test_apply_once(self):
        x = np.arange(3 * 4 * 5).reshape((3, 4, 5))

        np.testing.assert_array_almost_equal(dd.apply_once(np.std, x, [0, -1]),
                                             16.39105447 * np.ones((1, 4, 1)))

        x = np.arange(2 * 3).reshape((2, 3))
        np.testing.assert_array_equal(dd.apply_once(np.sum, x, 1, keepdims=False),
                                      np.array([3, 12]))

    def test_tupled_argmax(self):
        x = np.zeros((3, 4, 5))
        x[1, 2, 3] = 10

        assert dd.tupled_argmax(x) == (1, 2, 3)

    def test_slice(self):
        s = [slice(None, 3), slice(None), slice(2, None), slice(3, 4), Ellipsis, [1, 2, 3]]
        assert dd.aslice[:3, :, 2:, 3:4, ..., [1, 2, 3]]

    def test_timed(self):
        # These tests only make sure it does not cause errors
        with dd.timed():
            pass

        times = []
        with dd.timed(callback=times.append):
            pass
        assert len(times) == 1

        x = np.zeros(1)
        x[:] = np.nan
        with dd.timed(file=x):
            pass
        assert not np.isnan(x[0])


if __name__ == '__main__':
    unittest.main()
