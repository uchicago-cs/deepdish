---
layout: post
title: Python dictionary to HDF5
author: gustav
---

I used to be a big fan of Numpy's
[savez](http://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html)
and
[load](http://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html),
since you can throw any Python structure in there that you want to save.

Since these files are not compatible between Python 2 and 3, they did not fit
my needs anymore. I took the matter to
[Stackoverflow](http://stackoverflow.com/questions/18071075/saving-dictionaries-to-file-numpy-and-python-2-3-friendly),
but a clear winner did not emerge.

Finally, I decided to write my own alternative to `savez`, basing it on
HDF5. The result can be found in our
[deepdish](https://github.com/uchicago-cs/deepdish) project and it even seconds
as a general-purpose HDF5 saver/loader. First, an example of how to write a
[Caffe](http://caffe.berkeleyvision.org)-compatible data file:

```python
import deepdish as dd
import numpy as np

X = np.zeros((100, 3, 32, 32))
y = np.zeros(100)

dd.io.save('test.h5', {'data': X, 'label': y})
```

Let's take a look at it:

```bash
$ h5ls test.h5
data                     Dataset {100, 3, 32, 32}
label                    Dataset {100}
```

It will load into a dictionary with `dd.io.load`:

```python
dd.io.load('test.h5')
```

Now, it does much more than that. It can save numbers, lists, strings,
dictionaries and numpy arrays. It will try to store things natively to HDF5, so
that it can be read by other programs as well. Another example:

```python
>>> x = [np.arange(3), {'d': 100, 'e': 'hello'}]
>>> dd.io.save('test.h5', x)
>>> dd.io.load('test.h5')
[array([0, 1, 2]), {'e': 'hello', 'd': 100}]
```

If it doesn't know how to save a particular data type, it will fall-back and
use pickling. This means it will still work, but you will lose the
compatibility across Python 2 and 3.
