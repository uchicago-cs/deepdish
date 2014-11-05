from __future__ import division, print_function, absolute_import
import os
import struct
from array import array
import numpy as np


def load_mnist(section="training", offset=0, count=None, ret='xy',
               x_dtype=np.float64, y_dtype=np.int64, path=None):
    """
    Loads MNIST files into a 3D numpy array.

    You have to download the data separately from [MNIST]_. It is recommended
    to set the environment variable ``MNIST_DIR`` to point to the folder where
    you put the data, so that you don't have to select path. On a Linux+bash
    setup, this is done by adding the following to your ``.bashrc``::

        export MNIST_DIR=/path/to/mnist

    Parameters
    ----------
    section : str
        Either "training" or "testing", depending on which section you want to
        load.
    offset : int
        Skip this many samples.
    count : int or None
        Try to load this many samples. Default is None, which loads until the
        end.
    ret : str
        What information to return. See return values.
    x_dtype : dtype
        Type of samples. If ``np.uint8``, intensities lie in {0, 1, ..., 255}.
        If a float type, then intensities lie in [0.0, 1.0].
    y_dtype : dtype
        Integer type to store labels.
    path : str
        Path to your MNIST datafiles. The default is ``None``, which will try
        to take the path from your environment variable ``MNIST_DIR``. The data
        can be downloaded from http://yann.lecun.com/exdb/mnist/.

    Returns
    -------
    images : ndarray
        Image data of shape ``(N, 28, 28)``, where ``N`` is the number of
        images. Returned if ``ret`` contains ``'x'``.
    labels : ndarray
        Array of size ``N`` describing the labels. Returned if ``ret``
        contains ``'y'``.

    Examples
    --------
    Assuming that you have downloaded the MNIST database and set the
    environment variable ``$MNIST_DIR`` point to the folder, this will load all
    images and labels from the training set:

    >>> images, labels = ag.io.load_mnist('training')  # doctest: +SKIP

    Load 100 samples from the testing set:

    >>> sevens = ag.io.load_mnist('testing', offset=200, count=100,
                                  ret='x') # doctest: +SKIP

    """

    # The files are assumed to have these names and should be found in 'path'
    files = {
        'training': ('train-images-idx3-ubyte',
                     'train-labels-idx1-ubyte',
                     60000),
        'testing': ('t10k-images-idx3-ubyte',
                    't10k-labels-idx1-ubyte',
                    10000),
    }

    if count is None:
        count = files[section][2] - offset

    if path is None:
        try:
            path = os.environ['MNIST_DIR']
        except KeyError:
            raise ValueError("Unspecified path requires the environment"
                             "variable $MNIST_DIR to be set")

    try:
        images_fname = os.path.join(path, files[section][0])
        labels_fname = os.path.join(path, files[section][1])
    except KeyError:
        raise ValueError("Data set must be 'testing' or 'training'")

    returns = ()
    if 'x' in ret:
        with open(images_fname, 'rb') as fimg:
            magic_nr, size, d0, d1 = struct.unpack(">IIII", fimg.read(16))
            fimg.seek(offset * d0 * d1, 1)
            images_raw = array("B", fimg.read(count * d0 * d1))

        images = np.asarray(images_raw, dtype=x_dtype).reshape(-1, d0, d1)
        if x_dtype == np.uint8:
            pass  # already this type
        elif x_dtype in (np.float16, np.float32, np.float64):
            images /= 255.0
        else:
            raise ValueError("Unsupported value for x_dtype")

        returns += (images,)

    if 'y' in ret:
        with open(labels_fname, 'rb') as flbl:
            magic_nr, size = struct.unpack(">II", flbl.read(8))
            flbl.seek(offset, 1)
            labels_raw = array("b", flbl.read(count))

        labels = np.asarray(labels_raw)

        returns += (labels,)

    if len(returns) == 1:
        return returns[0]  # Don't return a tuple of one
    else:
        return returns
