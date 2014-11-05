from __future__ import division, print_function, absolute_import
_is_verbose = False
_is_silent = False

import warnings
import numpy as np
warnings.simplefilter("ignore", np.ComplexWarning)


def set_verbose(is_verbose):
    """
    Choose whether or not to display output from calls to ``amitgroup.info``.

    Parameters
    ----------
    is_verbose : bool
        If set to ``True``, info messages will be printed when running certain
        functions. Default is ``False``.
    """
    global _is_verbose
    _is_verbose = is_verbose


def info(*args, **kwargs):
    """
    Output info about the status of running functions. If you are writing a
    function that might take minutes to complete, periodic calls to the
    function with a description of the status is recommended.

    This function takes the same arguments as Python 3's ``print`` function.
    The only difference is that if ``amitgroup.set_verbose(True)`` has not be
    called, it will suppress any output.
    """
    if _is_verbose:
        print(*args, **kwargs)


def warning(*args, **kwargs):
    """
    Output warning, such as numerical instabilities.
    """
    if not _is_silent:
        print("WARNING: " + args[0], *args[1:], **kwargs)


class AbortException(Exception):
    """
    This exception is used for when the user wants to quit algorithms mid-way.
    The `AbortException` can for instance be sent by pygame input, and caught
    by whatever is running the algorithm.
    """
    pass


def bytesize(arr):
    """
    Returns the memory byte size of a Numpy array as an integer.
    """
    byte_size = np.prod(arr.shape) * np.dtype(arr.dtype).itemsize
    return byte_size


def humanize_bytesize(byte_size):
    order = np.log(byte_size) / np.log(1024)
    orders = [
        (5, 'PB'),
        (4, 'TB'),
        (3, 'GB'),
        (2, 'MB'),
        (1, 'KB'),
        (0, 'B')
    ]

    for ex, name in orders:
        if order >= ex:
            return '{:.4g} {}'.format(byte_size / 1024**ex, name)


def memsize(arr):
    """
    Returns the required memory of a Numpy array as a humanly readable string.
    """
    return humanize_bytesize(bytesize(arr))


def apply_once_over_axes(func, arr, axes, remove_axes=False):
    """
    Similar to `numpy.apply_over_axes`, except this performs the operation over
    a flattened version of all the axes, meaning the function will only be
    called once. This only makes a difference for non-linear functions.

    Parameters
    ----------
    func : callback
        Function that operates well on Numpy arrays and returns a single value
        of compatible dtype.
    arr : ndarray
        Array to do operation over.
    axes : int or iterable
        Specifies the axes to perform the operation. Only one call will be made
        to `func`, with all values flattened.
    remove_axes : bool
        By default, this is False, so the collapsed dimensions remain with
        length 1. This is simlar to `numpy.apply_over_axes`.  If this is set to
        True, the dimensions are removed, just like when using for instance
        `numpy.sum` over a single axis. Note that this is safer than
        subsequently calling squeeze, since this option will preserve length-1
        dimensions that were not operated on.
    """

    all_axes = np.arange(arr.ndim)
    if isinstance(axes, int):
        axes = set(axes)
    else:
        axes = set(axis % arr.ndim for axis in axes)

    principal_axis = min(axes)

    for i, axis in enumerate(axes):
        axis0 = principal_axis + i
        if axis != axis0:
            print('swapping', axis, axis0)
            all_axes[axis0], all_axes[axis] = all_axes[axis], all_axes[axis0]

    transposed_arr = arr.transpose(all_axes)

    new_shape = []
    new_shape2 = []
    for axis, dim in enumerate(arr.shape):
        if axis == principal_axis:
            new_shape.append(-1)
        elif axis not in axes:
            new_shape.append(dim)

        if axis in axes:
            new_shape2.append(1)
        else:
            new_shape2.append(dim)

    collapsed = np.apply_along_axis(func,
                                    principal_axis,
                                    transposed_arr.reshape(new_shape))

    if not remove_axes:
        return collapsed.reshape(new_shape2)
    else:
        return collapsed
