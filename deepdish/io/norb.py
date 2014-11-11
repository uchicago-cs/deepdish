from __future__ import division, print_function, absolute_import
import os
import struct
import numpy as np


def _load_norb_setup(section, path=None):
    assert section in ('training', 'testing')
    if path is None:
        path = os.environ['NORB_DIR']

    name = {
        'training': 'smallnorb-5x46789x9x18x6x2x96x96-training',
        'testing': 'smallnorb-5x01235x9x18x6x2x96x96-testing',
    }[section]

    return path, name


def _load_small_norb_labels(section, path=None, offset=0, count=None):
    path, name = _load_norb_setup(section, path)
    cat_fn = os.path.join(path, name + '-cat.mat')

    with open(cat_fn, 'rb') as f:
        byte_matrix = struct.unpack('<I', f.read(4))[0]
        assert byte_matrix == 0x1e3d4c54, "Only supports int matrix for now"

        struct.unpack('<I', f.read(4))[0]
        # The two latter values are ignored as per specifications
        dim0 = struct.unpack('<III', f.read(4 * 3))[0]

        f.seek(offset * np.dtype(np.int32).itemsize, 1)

        max_count = dim0 - offset
        if count is None:
            count = max_count
        else:
            count = min(count, max_count)

        if count > 0:
            y0 = np.fromfile(f, dtype=np.int32, count=count)
        else:
            count = 0
            y0 = np.empty(0)

        return y0


def _load_small_norb_info(section, path=None, offset=0, count=None):
    path, name = _load_norb_setup(section, path)
    dat_fn = os.path.join(path, name + '-info.mat')

    with open(dat_fn, 'rb') as f:
        byte_matrix = struct.unpack('<I', f.read(4))[0]
        assert byte_matrix == 0x1e3d4c54, "Only supports int matrix for now"

        struct.unpack('<I', f.read(4))[0]
        # The two latter values are ignored as per specifications
        dim0, dim1 = struct.unpack('<III', f.read(4 * 3))[:2]

        f.seek(offset * dim1 * np.dtype(np.int32).itemsize, 1)

        max_count = dim0 - offset
        if count is None:
            count = max_count
        else:
            count = min(count, max_count)

        if count > 0:
            y0 = np.fromfile(f, dtype=np.int32, count=count * dim1)
        else:
            count = 0
            y0 = np.empty(0)

        return y0.reshape((count, dim1))


def _load_small_norb_data(section, path=None, offset=0, count=None,
                          dtype=np.float64):
    path, name = _load_norb_setup(section, path)
    dat_fn = os.path.join(path, name + '-dat.mat')

    with open(dat_fn, 'rb') as f:
        byte_matrix = struct.unpack('<I', f.read(4))[0]
        assert byte_matrix == 0x1e3d4c55, "Only supports byte matrix for now"

        struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack('<IIII', f.read(4 * 4))

        f.seek(offset * dims[2] * dims[3] * np.dtype(np.uint8).itemsize, 1)

        max_count = dims[0] - offset
        if count is None:
            count = max_count
        else:
            count = min(count, max_count)

        if count > 0:
            X = np.fromfile(f, dtype=np.uint8, count=count * np.prod(dims[1:]))
        else:
            count = 0
            X = np.empty(0)

        X = X.reshape((count,) + dims[1:])

        if dtype in [np.float16, np.float32, np.float64]:
            X = X.astype(dtype) / 255
        elif dtype == np.uint8:
            pass  # Do nothing
        else:
            raise ValueError('Unsupported value for x_dtype in NORB loader')

        return X


def load_small_norb(section, ret='xy', offset=0, count=None, path=None,
                    x_dtype=np.float64):
    """
    Loads the small NORB dataset [NORB]_.

    Parameters
    ----------
    section : ('training', 'testing')
        Select between the training and testing data.
    offset : int
        Samples to skip.
    count : int
        Samples to load. Default is `None`, which loads until the end.
    ret : str
        What to return, `'x'` for data, `'y'` for labels and `'i'` for info.
        For instance, `'xy'` returns data and labels. If only one thing is
        selected, it is returned directly, instead of in a tuple of one.
    path : str
        Specifies the path in which you have your NORB files. Default is None,
        in which case the environment variable ``NORB_DIR`` is used.

    Returns
    -------
    X : ndarray
        The images (if `ret` specifies data)
    y : ndarray
        The labels (if `ret` specifies labels)
    info : ndarray (int32)
        Array of shape ``(N, 4)``, where `N` is the number of samples and 4 are
        the four pieces of information for each sample:
        * the instance in the category (0 to 9)
        * the elevation (0 to 8, which mean cameras are 30, 35, 40, 45, 50, 55,
          60, 65, 70 degrees from the horizontal respectively)
        * the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in
          degrees)
        * the lighting condition (0 to 5)

    """
    returns = []
    if 'x' in ret:
        X = _load_small_norb_data(section, path, offset=offset, count=count,
                                  dtype=x_dtype)
        returns.append(X)
    if 'y' in ret:
        y = _load_small_norb_labels(section, path, offset=offset, count=count)
        returns.append(y)
    if 'i' in ret:
        info = _load_small_norb_info(section, path, offset=offset, count=count)
        returns.append(info)

    if len(returns) == 1:
        return returns[0]
    else:
        return tuple(returns)
