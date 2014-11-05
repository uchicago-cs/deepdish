from __future__ import division, print_function, absolute_import
import os
import struct
import glob
import numpy as np


def _load_gnt_file(fn, Xs, ys):
    with open(fn, 'rb') as f:
        while True:
            try:
                struct.unpack('<I', f.read(4))[0]
                tag_code = struct.unpack('<H', f.read(2))[0]
                # Reverse order of dimensions to get it right for numpy
                dims = struct.unpack('<HH', f.read(2 * 2))[::-1]

                X = np.fromfile(f, dtype=np.uint8, count=np.prod(dims))
                X = X.reshape(dims)
                Xs.append(X)

                ys.append(tag_code)

            except struct.error:
                break


def load_casia(section, dataset='HWDB1.1', path=None):
    """
    Load dataset CASIA with Chinese characters [CASIA]_.

    The `gnt` files should be located inside the following subdirectories:

    * HWDB1.1trn
    * HWDB1.1tst

    Parameters
    ----------
    section : ('training', 'testing')
        Section of the dataset
    dataset : ('HWDB1.1',)
        Which dataset to use.
    path : str
        Path of root CASIA directory. Default is None, in which case the
        environment variable ``CASIA_DIR`` is used.

    Returns
    -------
    X : list
        List of differently sized Chinese characters.
    y : ndarray
        Numpy array of integers with corresponding labels.

    """
    assert section in ('training', 'testing')

    if path is None:
        path = os.environ['CASIA_DIR']

    # TODO: Not sure if we should put this in a subfolder.
    if section == 'testing':
        sub_path = 'HWDB1.1tst'
    elif section == 'training':
        sub_path = 'HWDB1.1trn'
    else:
        raise ValueError("Section not valid option")

    fns = sorted(glob.glob(os.path.join(path, sub_path, '*.gnt')))

    Xs = []
    ys = []
    for fn in fns:
        # Fills Xs and ys in place
        _load_gnt_file(fn, Xs, ys)

    return Xs, np.asarray(ys)
