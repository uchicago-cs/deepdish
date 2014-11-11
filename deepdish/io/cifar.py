from __future__ import division, print_function, absolute_import
import deepdish as dd
import numpy as np
import os


def load_cifar_10(section, offset=0, count=10000, x_dtype=np.float32,
                  ret='xy'):
    assert section in ['training', 'testing']

    # TODO: This loads it from hdf5 files that I have prepared.

    # For now, only batches that won't be from several batches
    assert count <= 10000
    assert offset % count == 0
    assert 10000 % count == 0

    batch_offset = 0

    if section == 'training':
        batch_number = offset // 10000 + 1
        if batch_number > 5:
            name = None
            batch_offset = 0
            count = 0
        else:
            name = 'cifar_{}.h5'.format(batch_number)
            batch_offset = offset % 10000
    else:
        name = 'cifar_test.h5'
        batch_offset = offset

    if name is not None:
        data = dd.io.load(os.path.join(os.environ['CIFAR10_DIR'], name))
    else:
        data = dict(data=np.empty(0), labels=np.empty(0))

    X = data['data']
    y = data['labels']

    X0 = X[batch_offset:batch_offset+count].reshape(-1, 3, 32, 32)
    y0 = y[batch_offset:batch_offset+count]

    if x_dtype in [np.float16, np.float32, np.float64]:
        X0 = X0.astype(x_dtype) / 255
    elif x_dtype == np.uint8:
        pass  # Do nothing
    else:
        raise ValueError('load_cifar_10: Unsupported value for x_dtype')

    returns = []
    if 'x' in ret:
        returns.append(X0)
    if 'y' in ret:
        returns.append(y0)

    if len(returns) == 1:
        return returns[0]
    else:
        return tuple(returns)
