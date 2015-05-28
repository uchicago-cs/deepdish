import deepdish as dd
import numpy as np

def compute(indices, x):
    return indices, 2 * x

if dd.parallel.main(__name__):
    x = np.arange(100) * 10
    index_batches = np.array_split(np.arange(len(x)), 10)
    args = ((indices, x[indices]) for indices in index_batches)

    y = np.zeros_like(x)
    for indices, batch_y in dd.parallel.starmap_unordered(compute, args):
        print('Finished indices', indices)
        y[indices] = batch_y

    print(y)
