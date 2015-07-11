from __future__ import division, print_function, absolute_import
import unittest
from tempfile import NamedTemporaryFile
import os
import numpy as np
import deepdish as dd
from contextlib import contextmanager


@contextmanager
def tmp_filename():
    f = NamedTemporaryFile(delete=False)
    yield f.name
    os.unlink(f.name)


def reconstruct(fn, x):
    dd.io.save(fn, x)
    return dd.io.load(fn)


def assert_array(fn, x):
    dd.io.save(fn, x)
    x1 = dd.io.load(fn)
    np.testing.assert_array_equal(x, x1)


class TestIO(unittest.TestCase):
    def test_basic_data_types(self):
        with tmp_filename() as fn:
            x = 100
            x1 = reconstruct(fn, x)
            assert x == x1

            x = 1.23
            x1 = reconstruct(fn, x)
            assert x == x1

            # This doesn't work - complex numpy arrays work however
            #x = 1.23 + 2.3j
            #x1 = reconstruct(fn, x)
            #assert x == x1

            x = 'this is a string'
            x1 = reconstruct(fn, x)
            assert x == x1

            x = None
            x1 = reconstruct(fn, x)
            assert x1 is None

    def test_numpy_array(self):
        with tmp_filename() as fn:
            x0 = np.arange(3 * 4 * 5, dtype=np.int64).reshape((3, 4, 5))
            assert_array(fn, x0)

            x0 = x0.astype(np.float32)
            assert_array(fn, x0)

            x0 = x0.astype(np.uint8)
            assert_array(fn, x0)

            x0 = x0.astype(np.complex128)
            x0[0] = 1 + 2j
            assert_array(fn, x0)

    def test_dictionary(self):
        with tmp_filename() as fn:
            d = dict(a=100, b='this is a string', c=np.ones(5))

            d1 = reconstruct(fn, d)

            assert d['a'] == d1['a']
            assert d['b'] == d1['b']
            np.testing.assert_array_equal(d['c'], d1['c'])

    def test_list(self):
        with tmp_filename() as fn:
            x = [100, 'this is a string', np.ones(3), dict(foo=100)]

            x1 = reconstruct(fn, x)

            assert isinstance(x1, list)
            assert x[0] == x1[0]
            assert x[1] == x1[1]
            np.testing.assert_array_equal(x[2], x1[2])
            assert x[3]['foo'] == x1[3]['foo']

    def test_tuple(self):
        with tmp_filename() as fn:
            x = (100, 'this is a string', np.ones(3), dict(foo=100))

            x1 = reconstruct(fn, x)

            assert isinstance(x1, tuple)
            assert x[0] == x1[0]
            assert x[1] == x1[1]
            np.testing.assert_array_equal(x[2], x1[2])
            assert x[3]['foo'] == x1[3]['foo']

    def test_sparse_matrices(self):
        import scipy.sparse as S
        with tmp_filename() as fn:
            x = S.lil_matrix((50, 70))
            x[34, 37] = 1
            x[34, 39] = 2.5
            x[34, 41] = -2
            x[38, 41] = -1

            x1 = reconstruct(fn, x.tocsr())
            assert x.shape == x1.shape
            np.testing.assert_array_equal(x.todense(), x1.todense())

            x1 = reconstruct(fn, x.tocsc())
            assert x.shape == x1.shape
            np.testing.assert_array_equal(x.todense(), x1.todense())

            x1 = reconstruct(fn, x.tocoo())
            assert x.shape == x1.shape
            np.testing.assert_array_equal(x.todense(), x1.todense())

            x1 = reconstruct(fn, x.todia())
            assert x.shape == x1.shape
            np.testing.assert_array_equal(x.todense(), x1.todense())

            x1 = reconstruct(fn, x.tobsr())
            assert x.shape == x1.shape
            np.testing.assert_array_equal(x.todense(), x1.todense())


if __name__ == '__main__':
    unittest.main()
