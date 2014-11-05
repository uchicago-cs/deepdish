from __future__ import division, print_function, absolute_import
import os
import struct
import numpy as np

def load_small_norb(section, path=None, selection=None):
    """
    Loads the small NORB dataset [NORB]_.

    Parameters
    ---------- 
    section : ('training', 'testing')
        Select between the training and testing data.
    path : str
        Specifies the path in which you have your NORB files. Default is None, in which case the environment variable ``NORB_DIR`` is used.
    selection : slice
        Using a `slice` object, specify what subset of the dataset to load. An
        example is ``slice(50, 150)``, which would load 100 samples starting
        with the 50th. Currently does not support strides other than 1 and
        since the dataset come in twin pairs, your offsets and counts should be
        multiples of 2.

    Returns
    -------
    X : ndarray
        The images
    y : ndarray
        The labels
    """
    assert section in ('training', 'testing')

    if selection is not None:
        if selection.start is not None:
            assert selection.start % 2 == 0
        if selection.stop is not None:
            assert selection.stop % 2 == 0

    if path is None:
        path = os.environ['NORB_DIR']

    name = {
        'training': 'smallnorb-5x46789x9x18x6x2x96x96-training',
        'testing': 'smallnorb-5x01235x9x18x6x2x96x96-testing',
    }[section]

    dat_fn = os.path.join(path, name + '-dat.mat')
            
    with open(dat_fn, 'rb') as f:
        byte_matrix = struct.unpack('<I', f.read(4))[0]
        assert byte_matrix == 0x1e3d4c55, "Only supports byte matrix for now"

        ndim = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack('<IIII', f.read(4 * 4))

        if selection is not None and selection.start is not None:
            f.seek(selection.start * dims[2] * dims[3] * np.dtype(np.uint8).itemsize, 1)

        N = np.prod(dims[:2])
        if selection is not None and selection.stop is not None:
            N = min(N, selection.stop)
        if selection is not None and selection.start is not None:
            N -= selection.start

        count = N * dims[2] * dims[3]

        if count < 0:
            return np.empty((0, dims[2], dims[3])), np.empty((0,))

        X = np.fromfile(f, dtype=np.uint8, count=count)\
              .reshape((N, dims[2], dims[3]))

    cat_fn = os.path.join(path, name + '-cat.mat')

    with open(cat_fn, 'rb') as f:
        byte_matrix = struct.unpack('<I', f.read(4))[0]
        assert byte_matrix == 0x1e3d4c54, "Only supports intger matrix for now"

        ndim = struct.unpack('<I', f.read(4))[0]
        # The two latter values are ignored as per specifications
        dim0 = struct.unpack('<III', f.read(4 * 3))[0]

        if selection is not None and selection.start is not None:
            f.seek(selection.start * np.dtype(np.int32).itemsize // 2, 1)

        count = np.prod(dims)
        if selection is not None and selection.stop is not None:
            count = selection.stop 

        if selection is not None and selection.start is not None:
            count -= selection.start

        count //= 2

        y0 = np.fromfile(f, dtype=np.int32, count=count)

    y = np.empty(X.shape[0], dtype=np.int32)
    y[0::2] = y0
    y[1::2] = y0

    return X, y
    #return np.concatenate([X[:,0], X[:,1]]), np.concatenate([y, y])

