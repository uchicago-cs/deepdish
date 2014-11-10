"""
.. todo::

    Based on code from pylearn2.datasets.hdf5

"""
__authors__ = "Mark Stoehr"
__copyright__ = "Copyright 2014, Mark Stoehr"
__credits__ = ["Mark Stoehr"]
__license__ = "MIT"
__maintainer__ = "deepdish.io"
__email__ = "mark@deepdish.io"

import numpy as np
import warnings
from pylearn2.datasets import (dense_design_matrix,
                                 control,
                                 cache)
from pylearn2.datasets.hdf5 import (HDF5Dataset,
                                    DenseDesignMatrix,
                                    HDF5DatasetIterator,
                                    HDF5ViewConverter,
                                    HDF5TopoViewConverter)
from pylearn2.utils import serial
try:
    import h5py
except ImportError:
    h5py = None
from pylearn2.utils.rng import make_np_rng

class MediaEval(DenseDesignMatrix):
    """
    .. todo::
    
        WRITEME

    Parameters
    ----------
    filename
    X
    y
    start
    stop
    """
    def __init__(self,filename,X,y,start,stop**kwargs):
        self.load_all = False
        if h5py is None:
            raise RuntimeError("Could not import h5py")
        self.h5py.File(filename)
        X = self.get_dataset(X)
        y = self.get_dataset(y)

        super(MediaEval,self).__init__(X=X,y=y,**kwargs)

    def _check_labels(self):
        """
        Sanity checks for X_labels and y_labels.

        Since the np.all test used for these labels does not work with HDF5
        datasets, we issue a warning that those values are not checked.
        """
        if self.X_labels is not None:
            assert self.X is not None
            assert self.view_converter is None
            assert self.X.ndim <= 2
            if self.load_all:
                assert np.all(self.X < self.X_labels)
            else:
                warnings.warn("HDF5Dataset cannot perform test np.all(X < " +
                              "X_labels). Use X_labels at your own risk.")

        if self.y_labels is not None:
            assert self.y is not None
            assert self.y.ndim <= 2
            if self.load_all:
                assert np.all(self.y < self.y_labels)
            else:
                warnings.warn("HDF5Dataset cannot perform test np.all(y < " +
                              "y_labels). Use y_labels at your own risk.")



    def get_dataset(self, dataset, load_all=False):
        """
        Get a handle for an HDF5 dataset, or load the entire dataset into
        memory.

        Parameters
        ----------
        dataset : str
            Name or path of HDF5 dataset.
        load_all : bool, optional (default False)
            If true, load dataset into memory.
        """
        if load_all:
            data = self._file[dataset][:]
        else:
            data = self._file[dataset]
            data.ndim = len(data.shape)  # hdf5 handle has no ndim
        return data

        
    def iterator(self, *args, **kwargs):
        """
        Get an iterator for this dataset.

        The FiniteDatasetIterator uses indexing that is not supported by
        HDF5 datasets, so we change the class to HDF5DatasetIterator to
        override the iterator.next method used in dataset iteration.

        Parameters
        ----------
        WRITEME
        """
        iterator = super(MediaEval, self).iterator(*args, **kwargs)
        iterator.__class__ = HDF5DatasetIterator
        return iterator
