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
