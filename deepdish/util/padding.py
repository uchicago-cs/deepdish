from __future__ import division, print_function, absolute_import
import numpy as np


def pad(data, padwidth, value=0.0):
    """
    Pad an array with a specific value.

    Parameters
    ----------
    data : ndarray
        Numpy array of any dimension and type.
    padwidth : int or tuple
        If int, it will pad using this amount at the beginning and end of all
        dimensions. If it is a tuple (of same length as `ndim`), then the
        padding amount will be specified per axis.
    value : data.dtype
        The value with which to pad. Default is ``0.0``.

    See also
    --------
    pad_to_size, pad_repeat_border, pad_repeat_border_corner

    Examples
    --------
    >>> import deepdish as dd
    >>> import numpy as np

    Pad an array with zeros.

    >>> x = np.ones((3, 3))
    >>> dd.util.pad(x, (1, 2), value=0.0)
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    """
    data = np.asarray(data)
    shape = data.shape
    if isinstance(padwidth, int):
        padwidth = (padwidth,)*len(shape)

    padded_shape = tuple(map(lambda ix: ix[1]+padwidth[ix[0]]*2,
                             enumerate(shape)))
    new_data = np.empty(padded_shape, dtype=data.dtype)
    new_data[..., :] = value
    new_data[[slice(w, -w) if w > 0 else slice(None) for w in padwidth]] = data
    return new_data


def pad_to_size(data, shape, value=0.0):
    """
    This is similar to `pad`, except you specify the final shape of the array.

    Parameters
    ----------
    data : ndarray
        Numpy array of any dimension and type.
    shape : tuple
        Final shape of padded array. Should be tuple of length ``data.ndim``.
        If it has to pad unevenly, it will pad one more at the end of the axis
        than at the beginning. If a dimension is specified as ``-1``, then it
        will remain its current size along that dimension.
    value : data.dtype
        The value with which to pad. Default is ``0.0``. This can even be an
        array, as long as ``pdata[:] = value`` is valid, where ``pdata`` is the
        size of the padded array.

    Examples
    --------
    >>> import deepdish as dd
    >>> import numpy as np

    Pad an array with zeros.

    >>> x = np.ones((4, 2))
    >>> dd.util.pad_to_size(x, (5, 5))
    array([[ 0.,  1.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])


    """
    shape = [data.shape[i] if shape[i] == -1 else shape[i]
             for i in range(len(shape))]
    new_data = np.empty(shape)
    new_data[:] = value
    II = [slice((shape[i] - data.shape[i])//2,
                (shape[i] - data.shape[i])//2 + data.shape[i])
          for i in range(len(shape))]
    new_data[II] = data
    return new_data


def pad_repeat_border(data, padwidth):
    """
    Similar to `pad`, except the border value from ``data`` is used to pad.

    Parameters
    ----------
    data : ndarray
        Numpy array of any dimension and type.
    padwidth : int or tuple
        If int, it will pad using this amount at the beginning and end of all
        dimensions. If it is a tuple (of same length as `ndim`), then the
        padding amount will be specified per axis.

    Examples
    --------
    >>> import deepdish as dd
    >>> import numpy as np

    Pad an array by repeating its borders:

    >>> shape = (3, 4)
    >>> x = np.arange(np.prod(shape)).reshape(shape)
    >>> dd.util.pad_repeat_border(x, 2)
    array([[ 0,  0,  0,  1,  2,  3,  3,  3],
           [ 0,  0,  0,  1,  2,  3,  3,  3],
           [ 0,  0,  0,  1,  2,  3,  3,  3],
           [ 4,  4,  4,  5,  6,  7,  7,  7],
           [ 8,  8,  8,  9, 10, 11, 11, 11],
           [ 8,  8,  8,  9, 10, 11, 11, 11],
           [ 8,  8,  8,  9, 10, 11, 11, 11]])

    """

    data = np.asarray(data)
    shape = data.shape
    if isinstance(padwidth, int):
        padwidth = (padwidth,)*len(shape)

    padded_shape = tuple(map(lambda ix: ix[1]+padwidth[ix[0]]*2,
                             enumerate(shape)))
    new_data = np.empty(padded_shape, dtype=data.dtype)
    new_data[[slice(w, -w) if w > 0 else slice(None) for w in padwidth]] = data

    for i, pw in enumerate(padwidth):
        if pw > 0:
            selection = [slice(None)] * data.ndim
            selection2 = [slice(None)] * data.ndim

            # Lower boundary
            selection[i] = slice(0, pw)
            selection2[i] = slice(pw, pw+1)
            new_data[tuple(selection)] = new_data[tuple(selection2)]

            # Upper boundary
            selection[i] = slice(-pw, None)
            selection2[i] = slice(-pw-1, -pw)
            new_data[tuple(selection)] = new_data[tuple(selection2)]

    return new_data


def pad_repeat_border_corner(data, shape):
    """
    Similar to `pad_repeat_border`, except the padding is always done on the
    upper end of each axis and the target size is specified.

    Parameters
    ----------
    data : ndarray
        Numpy array of any dimension and type.
    shape : tuple
        Final shape of padded array. Should be tuple of length ``data.ndim``.
        If it has to pad unevenly, it will pad one more at the end of the axis
        than at the beginning.

    Examples
    --------
    >>> import deepdish as dd
    >>> import numpy as np

    Pad an array by repeating its upper borders.

    >>> shape = (3, 4)
    >>> x = np.arange(np.prod(shape)).reshape(shape)
    >>> dd.util.pad_repeat_border_corner(x, (5, 5))
    array([[  0.,   1.,   2.,   3.,   3.],
           [  4.,   5.,   6.,   7.,   7.],
           [  8.,   9.,  10.,  11.,  11.],
           [  8.,   9.,  10.,  11.,  11.],
           [  8.,   9.,  10.,  11.,  11.]])

    """
    new_data = np.empty(shape)
    new_data[[slice(upper) for upper in data.shape]] = data
    for i in range(len(shape)):
        selection = [slice(None)]*i + [slice(data.shape[i], None)]
        selection2 = [slice(None)]*i + [slice(data.shape[i]-1, data.shape[i])]
        new_data[selection] = new_data[selection2]
    return new_data
