from __future__ import division, print_function, absolute_import
import time
import warnings
import numpy as np
import itertools as itr
import sys
from contextlib import contextmanager
warnings.simplefilter("ignore", np.ComplexWarning)

_is_verbose = False
_is_silent = False


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


def span(arr):
    """
    Calculate and return the mininum and maximum of an array.

    Parameters
    ----------
    arr : ndarray
        Numpy array.

    Returns
    -------
    min : dtype
        Minimum of array.
    max : dtype
        Maximum of array.
    """
    # TODO: This could be made faster with a custom ufunc
    return (np.min(arr), np.max(arr))


def apply_once(func, arr, axes, keepdims=True):
    """
    Similar to `numpy.apply_over_axes`, except this performs the operation over
    a flattened version of all the axes, meaning that the function will only be
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
    keepdims : bool
        By default, this is True, so the collapsed dimensions remain with
        length 1. This is simlar to `numpy.apply_over_axes` in that regard.  If
        this is set to False, the dimensions are removed, just like when using
        for instance `numpy.sum` over a single axis. Note that this is safer
        than subsequently calling squeeze, since this option will preserve
        length-1 dimensions that were not operated on.

    Examples
    --------
    >>> import deepdish as dd
    >>> import numpy as np
    >>> rs = np.random.RandomState(0)
    >>> x = rs.uniform(size=(10, 3, 3))

    Image that you have ten 3x3 images and you want to calculate each image's
    intensity standard deviation:

    >>> np.apply_over_axes(np.std, x, [1, 2]).ravel()
    array([ 0.06056838,  0.08230712,  0.08135083,  0.09938963,  0.08533604,
            0.07830725,  0.066148  ,  0.07983019,  0.08134123,  0.01839635])

    This is the same as ``x.std(1).std(1)``, which is not the standard
    deviation of all 9 pixels together. To fix this we can flatten the pixels
    and try again:

    >>> x.reshape(10, 9).std(axis=1)
    array([ 0.17648981,  0.32849108,  0.29409526,  0.25547501,  0.23649064,
            0.26928468,  0.20081239,  0.33052397,  0.29950855,  0.26535717])

    This is exactly what this function does for you:

    >>> dd.apply_once(np.std, x, [1, 2], keepdims=False)
    array([ 0.17648981,  0.32849108,  0.29409526,  0.25547501,  0.23649064,
            0.26928468,  0.20081239,  0.33052397,  0.29950855,  0.26535717])
    """

    all_axes = np.arange(arr.ndim)
    if isinstance(axes, int):
        axes = {axes}
    else:
        axes = set(axis % arr.ndim for axis in axes)

    principal_axis = min(axes)

    for i, axis in enumerate(axes):
        axis0 = principal_axis + i
        if axis != axis0:
            all_axes[axis0], all_axes[axis] = all_axes[axis], all_axes[axis0]

    transposed_arr = arr.transpose(all_axes)

    new_shape = []
    new_shape_keepdims = []
    for axis, dim in enumerate(arr.shape):
        if axis == principal_axis:
            new_shape.append(-1)
        elif axis not in axes:
            new_shape.append(dim)

        if axis in axes:
            new_shape_keepdims.append(1)
        else:
            new_shape_keepdims.append(dim)

    collapsed = np.apply_along_axis(func,
                                    principal_axis,
                                    transposed_arr.reshape(new_shape))

    if keepdims:
        return collapsed.reshape(new_shape_keepdims)
    else:
        return collapsed


def tupled_argmax(a):
    """
    Argmax that returns an index tuple. Note that `numpy.argmax` will return a
    scalar index as if you had flattened the array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    index : tuple
        Tuple of index, even if `a` is one-dimensional. Note that this can
        immediately be used to index `a` as in ``a[index]``.

    Examples
    --------
    >>> import numpy as np
    >>> import deepdish as dd
    >>> a = np.arange(6).reshape(2,3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> dd.tupled_argmax(a)
    (1, 2)
    """
    return np.unravel_index(np.argmax(a), np.shape(a))


def multi_range(*args):
    return itr.product(*[range(a) for a in args])


@contextmanager
def timed(name=None, file=sys.stdout, callback=None, wall_clock=True):
    """
    Context manager to make it easy to time the execution of a piece of code.
    This timer will never run your code several times and is meant more for
    simple in-production timing, instead of benchmarking. Reports the
    wall-clock time (using `time.time`) and not the processor time.

    Parameters
    ----------
    name : str
        Name of the timing block, to identify it.
    file : file handler
        Which file handler to print the results to. Default is standard output.
        If a numpy array and size 1 is given, the time in seconds will be
        stored inside it. Ignored if `callback` is set.
    callback : callable
        This offer even more flexibility than `file`. The callable will be
        called at the end of the execution with a single floating point
        argument with the elapsed time in seconds.

    Examples
    --------
    >>> import deepdish as dd
    >>> import time

    The `timed` function is a context manager, so everything inside the
    ``with`` block will be timed. The results will be printed by default to
    standard output:

    >>> with dd.timed('Sleep'):  # doctest: +SKIP
    ...     time.sleep(1)
    [timed] Sleep: 1.001035451889038 s

    Using the `callback` parameter, we can accumulate multiple runs into a
    list:

    >>> times = []
    >>> for i in range(3):  # doctest: +SKIP
    ...     with dd.timed(callback=times.append):
    ...         time.sleep(1)
    >>> times  # doctest: +SKIP
    [1.0035350322723389, 1.0035550594329834, 1.0039470195770264]
    """
    start = time.time()
    yield
    end = time.time()
    delta = end - start
    if callback is not None:
        callback(delta)
    elif isinstance(file, np.ndarray) and len(file) == 1:
        file[0] = delta
    else:
        name_str = ' {}'.format(name) if name is not None else ''
        print(("[timed]{0}: {1} s".format(name_str, delta)), file=file)


class SliceClass(object):
    def __getitem__(self, index):
        return index
aslice = SliceClass()
