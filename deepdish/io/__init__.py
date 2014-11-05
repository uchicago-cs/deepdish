from __future__ import division, print_function, absolute_import
from .mnist import load_mnist
from .norb import load_small_norb
from .casia import load_casia

try:
    import tables
    _pytables_ok = True
except ImportError:
    _pytables_ok = False
del tables

if _pytables_ok:
    from .hdf5io import load, save
else:
    def _f(*args, **kwargs):
        raise ImportError("You need PyTables for this function")
    load = save = _f

__all__ = ['load_mnist', 'load_small_norb', 'load_casia']
