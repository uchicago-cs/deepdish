---
layout: post
title: Creating an LMDB database in Python
author: gustav
---
LMDB is the database of choice when using [Caffe](http://caffe.berkeleyvision.org/) with large datasets. This is a tutorial of how to create an LMDB database from Python. First, let's look at the pros and cons of using LMDB over HDF5.

Reasons to use HDF5:

* Simple format to read/write.

Reasons to use LMDB:

* LMDB uses [memory-mapped files](http://en.wikipedia.org/wiki/Memory-mapped_file), giving much better I/O performance.
* Works well with really large datasets. The HDF5 files are always read entirely into memory, so you can't have any HDF5 file exceed your memory capacity. You can easily split your data into several HDF5 files though (just put several paths to `h5` files in your text file). Then again, compared to LMDB's page caching the I/O performance won't be nearly as good.

## LMDB from Python

You will need the Python package [lmdb](https://lmdb.readthedocs.org/en/release/) as well as Caffe's python package (`make pycaffe` in Caffe). LMDB provides key-value storage, where each \<key, value\> pair will be a sample in our dataset. The key will simply be a string version of an ID value, and the value will be a serialized version of the `Datum` class in Caffe (which are built using [protobuf](https://github.com/google/protobuf)).

```python
import numpy as np
import deepdish as dd
import lmdb
import caffe

N = 1000

# Let's pretend this is interesting data
X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
y = np.zeros(N, dtype=np.int64)

# We need to prepare the database for the size. If you don't have 
# deepdish installed, just set this to something comfortably big 
# (there is little drawback to settings this comfortably big).
map_size = dd.bytesize(X) * 2

env = lmdb.open('mylmdb', map_size=map_size)

for i in range(N):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = X.shape[1]
    datum.height = X.shape[2]
    datum.width = X.shape[3]
    datum.data = X[i].tobytes()
    datum.label = int(y[i])
    str_id = '{:08}'.format(i)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        txn.put(str_id, datum.SerializeToString())
```

You can also open up and inspect an extisting LMDB database from Python:

```python
import numpy as np
import lmdb
import caffe

env = lmdb.open('mylmdb', readonly=True)
with env.begin() as txn:
    raw_datum = env.get('00000000')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flat_x = np.fromstring(datum.data, dtype=np.uint8)
x = flat_x.reshape(datum.channels, datum.height, datum.width)
y = datum.label
```
