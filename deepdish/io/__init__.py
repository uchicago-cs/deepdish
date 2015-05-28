"""
Reading and writing to file.

The main features of this package is `deepdish.io.save` and `deepdish.io.load`.
They can save most Python data structures into an HDF5 file.

>>> import deepdish as dd
>>> d = {'foo': np.arange(10), 'bar': np.ones((5, 4, 3))}
>>> dd.io.save('test.h5', d)

It will try its hardest to save it in a native way to HDF5::

    $ h5ls test.h5
    bar                      Dataset {5, 4, 3}
    foo                      Dataset {10}

We can also reconstruct the dictionary from file:

>>> d = dd.io.load('test.h5')

See the `deepdish.io.save` for what data structures it supports.
"""
from __future__ import division, print_function, absolute_import
from .mnist import load_mnist
from .norb import load_small_norb
from .casia import load_casia
from .cifar import load_cifar_10

try:
    import tables
    _pytables_ok = True
    del tables
except ImportError:
    _pytables_ok = False

if _pytables_ok:
    from .hdf5io import load, save
else:
    def _f(*args, **kwargs):
        raise ImportError("You need PyTables for this function")
    load = save = _f

__all__ = ['load', 'save', 'load_mnist', 'load_small_norb', 'load_casia', 'load_cifar_10']
