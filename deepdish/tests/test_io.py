from __future__ import division, print_function, absolute_import
import unittest
from tempfile import NamedTemporaryFile
import os
import numpy as np
import deepdish as dd
import pandas as pd
from contextlib import contextmanager

try:
    from types import SimpleNamespace
    _sns = True
except ImportError:
    _sns = False


@contextmanager
def tmp_filename():
    f = NamedTemporaryFile(delete=False)
    yield f.name
    f.close()
    os.unlink(f.name)


@contextmanager
def tmp_file():
    f = NamedTemporaryFile(delete=False)
    yield f
    f.close()
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

            x = u'this is a string'
            x1 = reconstruct(fn, x)
            assert x == x1

            x = b'this is a bytearray'
            x1 = reconstruct(fn, x)
            assert x == x1

            x = None
            x1 = reconstruct(fn, x)
            assert x1 is None

    def test_big_integers(self):
        with tmp_filename() as fn:
            x = 1239487239847234982392837423874
            x1 = reconstruct(fn, x)
            assert x == x1

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

    def test_numpy_array_zero_size(self):
        # Arrays where one of the axes is length 0. These zero-length arrays cannot
        # be stored natively in HDF5, so we'll have to store only the shape
        with tmp_filename() as fn:
            x0 = np.arange(0, dtype=np.int64)
            assert_array(fn, x0)

            x0 = np.arange(0, dtype=np.float32).reshape((10, 20, 0))
            assert_array(fn, x0)

            x0 = np.arange(0, dtype=np.complex128).reshape((0, 5, 0))
            assert_array(fn, x0)

    def test_numpy_string_array(self):
        with tmp_filename() as fn:
            x0 = np.array([[b'this', b'string'], [b'foo', b'bar']])
            assert_array(fn, x0)

            x0 = np.array([[u'this', u'string'], [u'foo', u'bar']])
            assert_array(fn, x0)

    def test_dictionary(self):
        with tmp_filename() as fn:
            d = dict(a=100, b='this is a string', c=np.ones(5),
                     sub=dict(a=200, b='another string',
                              c=np.random.randn(3, 4)))

            d1 = reconstruct(fn, d)

            assert d['a'] == d1['a']
            assert d['b'] == d1['b']
            np.testing.assert_array_equal(d['c'], d1['c'])
            assert d['sub']['a'] == d1['sub']['a']
            assert d['sub']['b'] == d1['sub']['b']
            np.testing.assert_array_equal(d['sub']['c'], d1['sub']['c'])

    def test_simplenamespace(self):
        if _sns:
            with tmp_filename() as fn:
                d = SimpleNamespace(
                    a=100, b='this is a string', c=np.ones(5),
                    sub=SimpleNamespace(a=200, b='another string',
                                        c=np.random.randn(3, 4)))

                d1 = reconstruct(fn, d)

                assert d.a == d1.a
                assert d.b == d1.b
                np.testing.assert_array_equal(d.c, d1.c)
                assert d.sub.a == d1.sub.a
                assert d.sub.b == d1.sub.b
                np.testing.assert_array_equal(d.sub.c, d1.sub.c)

    def test_softlinks_recursion(self):
        with tmp_filename() as fn:
            A = np.random.randn(3, 3)
            df = pd.DataFrame({'int': np.arange(3),
                               'name': ['zero', 'one', 'two']})
            AA = 4
            s = dict(A=A, B=A, c=A, d=A, f=A, g=[A, A, A], AA=AA, h=AA,
                     df=df, df2=df)
            s['g'].append(s)
            n = reconstruct(fn, s)
            assert n['g'][0] is n['A']
            assert (n['A'] is n['B'] is n['c'] is n['d'] is n['f'] is
                    n['g'][0] is n['g'][1] is n['g'][2])
            assert n['g'][3] is n
            assert n['AA'] == AA == n['h']
            assert n['df'] is n['df2']
            assert (n['df'] == df).all().all()

            # test 'sel' option on link ... need to read two vars
            # to ensure at least one is a link:
            col1 = dd.io.load(fn, '/A', dd.aslice[:, 1])
            assert np.all(A[:, 1] == col1)
            col1 = dd.io.load(fn, '/B', dd.aslice[:, 1])
            assert np.all(A[:, 1] == col1)

    def test_softlinks_recursion_sns(self):
        if _sns:
            with tmp_filename() as fn:
                A = np.random.randn(3, 3)
                AA = 4
                s = SimpleNamespace(A=A, B=A, c=A, d=A, f=A,
                                    g=[A, A, A], AA=AA, h=AA)
                s.g.append(s)
                n = reconstruct(fn, s)
                assert n.g[0] is n.A
                assert (n.A is n.B is n.c is n.d is n.f is
                        n.g[0] is n.g[1] is n.g[2])
                assert n.g[3] is n
                assert n.AA == AA == n.h

    def test_pickle_recursion(self):
        with tmp_filename() as fn:
            f = {4: 78}
            f['rec'] = f
            g = [23.4, f]
            h = dict(f=f, g=g)
            h2 = reconstruct(fn, h)
            assert h2['g'][0] == 23.4
            assert h2['g'][1] is h2['f']['rec'] is h2['f']
            assert h2['f'][4] == 78

    def test_list_recursion(self):
        with tmp_filename() as fn:
            lst = [1, 3]
            inlst = ['inside', 'list', lst]
            inlst.append(inlst)
            lst.append(lst)
            lst.append(inlst)
            lst2 = reconstruct(fn, lst)
            assert lst2[2] is lst2
            assert lst2[3][2] is lst2
            assert lst[3][2] is lst
            assert lst2[3][3] is lst2[3]
            assert lst[3][3] is lst[3]

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

    def test_array_scalar(self):
        with tmp_filename() as fn:
            v = np.array(12.3)
            v1 = reconstruct(fn, v)
            assert v1[()] == v and isinstance(v1[()], np.float64)

            v = np.array(40, dtype=np.int8)
            v1 = reconstruct(fn, v)
            assert v1[()] == v and isinstance(v1[()], np.int8)

    def test_load_group(self):
        with tmp_filename() as fn:
            x = dict(one=np.ones(10), two='string')
            dd.io.save(fn, x)

            one = dd.io.load(fn, '/one')
            np.testing.assert_array_equal(one, x['one'])
            two = dd.io.load(fn, '/two')
            assert two == x['two']

            full = dd.io.load(fn, '/')
            np.testing.assert_array_equal(x['one'], full['one'])
            assert x['two'] == full['two']

    def test_load_multiple_groups(self):
        with tmp_filename() as fn:
            x = dict(one=np.ones(10), two='string', three=200)
            dd.io.save(fn, x)

            one, three = dd.io.load(fn, ['/one', '/three'])
            np.testing.assert_array_equal(one, x['one'])
            assert three == x['three']

            three, two = dd.io.load(fn, ['/three', '/two'])
            assert three == x['three']
            assert two == x['two']

    def test_load_slice(self):
        with tmp_filename() as fn:
            x = np.arange(3 * 4 * 5).reshape((3, 4, 5))
            dd.io.save(fn, dict(x=x))

            s = dd.aslice[:2]
            xs = dd.io.load(fn, '/x', sel=s)
            np.testing.assert_array_equal(xs, x[s])

            s = dd.aslice[:, 1:3]
            xs = dd.io.load(fn, '/x', sel=s)
            np.testing.assert_array_equal(xs, x[s])

            xs = dd.io.load(fn, sel=s, unpack=True)
            np.testing.assert_array_equal(xs, x[s])

            dd.io.save(fn, x)
            xs = dd.io.load(fn, sel=s)
            np.testing.assert_array_equal(xs, x[s])

    def test_force_pickle(self):
        with tmp_filename() as fn:
            x = dict(one=dict(two=np.arange(10)),
                     three='string')
            xf = dict(one=dict(two=x['one']['two']),
                      three=x['three'])

            dd.io.save(fn, xf)
            xs = dd.io.load(fn)

            np.testing.assert_array_equal(x['one']['two'], xs['one']['two'])
            assert x['three'] == xs['three']

            # Try direct loading one
            two = dd.io.load(fn, '/one/two')
            np.testing.assert_array_equal(x['one']['two'], two)

    def test_non_string_key_dict(self):
        with tmp_filename() as fn:
            # These will be pickled, but it should still work
            x = {0: 'zero', 1: 'one', 2: 'two'}
            x1 = reconstruct(fn, x)
            assert x == x1

            x = {1+1j: 'zero', b'test': 'one', (1, 2): 'two'}
            x1 = reconstruct(fn, x)
            assert x == x1

    def test_force_pickle(self):
        with tmp_filename() as fn:
            x = {0: 'zero', 1: 'one', 2: 'two'}
            fx = dd.io.ForcePickle(x)
            d = dict(foo=x, bar=100)
            fd = dict(foo=fx, bar=100)
            d1 = reconstruct(fn, fd)
            assert d == d1

    def test_pandas_dataframe(self):
        with tmp_filename() as fn:
            # These will be pickled, but it should still work
            df = pd.DataFrame({'int': np.arange(3), 'name': ['zero', 'one', 'two']})
            df1 = reconstruct(fn, df)
            assert (df == df1).all().all()

    def test_pandas_series(self):
        rs = np.random.RandomState(1234)
        with tmp_filename() as fn:
            s = pd.Series(rs.randn(5), index=['a', 'b', 'c', 'd', 'e'])
            s1 = reconstruct(fn, s)
            assert (s == s1).all()

    def test_compression_true(self):
        rs = np.random.RandomState(1234)
        with tmp_filename() as fn:
            x = rs.normal(size=(1000, 5))
            for comp in [None, True, 'blosc', 'zlib', ('zlib', 5)]:
                dd.io.save(fn, x, compression=comp)
                x1 = dd.io.load(fn)
                assert (x == x1).all()

if __name__ == '__main__':
    unittest.main()
