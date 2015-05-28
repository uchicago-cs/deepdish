import deepdish as dd
import numpy as np

def compute(batch, x):
    print('Processing batch', batch, 'on node', dd.parallel.rank())
    return 2 * x

if dd.parallel.main(__name__):
    x = np.arange(100)
    batches = np.array_split(x, 10)
    args = ((batch, x) for batch, x in enumerate(batches))

    # Execute and combine results
    y = np.concatenate(list(dd.parallel.starmap(compute, args)))
    print(y)
