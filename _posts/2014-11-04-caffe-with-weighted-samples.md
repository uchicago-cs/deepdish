---
layout: post
title: Caffe with weighted samples
author: gustav
---

[Caffe](http://caffe.berkeleyvision.org/) is a great framework for training and
running deep learning networks. However, it does not support weighted samples,
which is when you assign an importance for each sample. A weight (importance)
of 2 should have the same semantics for a sample as if you made a duplicate of
it.

I created an experimental fork of Caffe that supports this:

* [gustavla/caffe-weighted-samples](https://github.com/gustavla/caffe-weighted-samples)

This modification is so far rough around the edges and likely easy to break. I
have also not implemented support for it in all the loss layers, but only a
select few.

It works by adding the blob `sample_weight` to the dataset, alongside `data`
and `label`. The easiest way is to save the data as HDF5, which can easily be
done through Python:

```python
import h5py
import os

# X should have shape (samples, color channel, width, height)
# y should have shape (samples,)
# w should have shape (samples,)
# They should have dtype np.float32, even label

# DIR is an absolute path (important!)

h5_fn = os.path.join(DIR, 'data.h5')

with h5py.File(h5_fn, 'w') as f:
    f['data'] = X
    f['label'] = y
    f['sample_weight'] = w

text_fn = os.path.join(DIR, 'data.txt')
with open(text_fn, 'w'): as f:
    print(h5_fn, file=f)
```

Or, if you have our [deepdish](https://github.com/uchicago-cs/deepdish) package
installed, saving the HDF5 can be done as follows (also see [this post]({% post_url 2014-11-11-python-dictionary-to-hdf5 %})):

```python
dd.io.save(h5_fn, dict(data=X, label=y, sample_weight=w))
```

Now, load the `sample_weight` in your data layer:

```
layers {
    name: "example"
    type: HDF5_DATA
    top: "data"
    top: "label"
    top: "sample_weight"  # <-- add this
    hdf5_data_param {
        source: "/path/to/data.txt"
        batch_size: 100
    }
}
```

The file `data.txt` should contain a single line with the absolute path to
`h5_fn`, for instance `/path/to/data.h5`. Next, hook it up to the softmax layer
as:

```
layers {
    name: "loss"
    type: SOFTMAX_LOSS
    bottom: "last_layer"
    bottom: "label"
    bottom: "sample_weight"  # <-- add this
    top: "loss"
}
```

The layer `SOFTMAX_LOSS` is one of the few layers that have been
adapted to use `sample_weight`. If you want to use one that has not been
implemented yet, take inspiration from
[src/caffe/softmax\_loss\_layer.cpp](https://github.com/gustavla/caffe-weighted-samples/blob/master/src/caffe/layers/softmax_loss_layer.cpp).
Remember to also update `hpp` and `cu` files where needed. If you end up doing this, pull requests are welcome.
