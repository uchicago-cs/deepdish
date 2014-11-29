---
layout: post
title: DBM Posteriors from Pylearn2
author: mark
---

As a part of a recent project that I am working on I am using [Deep Boltzmann Machines
(DBMs)](www.cs.toronto.edu/~fritz/absps/dbm.pdf)
 posteriors
 as input features to a query-by-audio speech recognition
system.  Query-by-audio media retrieval means searching for audio content within a digital
collection of audio using an audio query. A prominent example of such as system is 
 [Shazam](http://www.shazam.com/) which finds music based on users' humming pattern.
In my work I am focusing on a speech version of the task where we search a collection
of speech recordings using a spoken query.  If the target language is English then the most straightforward
approach to this task is to use a speech recognizer on the database and use [lattice indexing](https://wiki.inf.ed.ac.uk/twiki/pub/CSTR/ListenSemester2201314/taslp_2011.pdf) or some similar method.  However, if the target language is unknown and potentially
has limited or zero annotated data then you can't build a speech recognizer and the lattice indexing approach fails.
My work is focusing on this latter scenario where there is little annotation.

[In this paper](http://groups.csail.mit.edu/sls/publications/2012/Zhang2_ICASSP_2012.pdf) the authors
explore this task by performing [dynamic time warping](https://www.ll.mit.edu/mission/communications/ist/publications/2009_12_13_Hazen_ASRU_FP.pdf) on the posteriors of a DBM with a soft-max output layer at the top.  As a baseline
in my own work I am beginning by attempting to reproduce their results.

In this post I will explain how I trained a DBM and computed posteriors over a speech dataset using [`pylearn2`](https://github.com/lisa-lab/pylearn2).  The steps are as follows:

- Assemble the data into an HDF5 file using `h5py`
- Write a training script to use a DBM
- Compute the posteriors
- Perform DTW over the computed features

# Saving the features

# `pylearn2` DBM training script

# `pylearn2` DBM posteriors

# DBM

I will write this up in a future post since I am using.



 The main task to accomplish is to train a 
and then use the posteriors from such a system as input features
to a dynamic time warping-based detection system.  The setting we consider
is we have MFCC features (which include a frame along with its immediate
context) and a posteriorgram generated from Gaussian Mixture Models. I used
the multiprediction training algorithm from `pylearn2` to learn a DBM
model. My intention is to extract the posteriors over novel data and then
to test the output.

I am working with deep Boltzmann machines (DBMs) in this post
that have been trained with the multiprediction training algorithm. 

# Using Theano

The fundamental structure for using Theano to process input is `function`
an example of how to use this is given in the Theano tutorial
```python
>>> x = T.dmatrix('x')
>>> s = 1/(1+T.exp(-x))
>>> logistic = function([x],s)
>>> logistic([[0, 1], [-1, -2]])
array([[ 0.5       ,  0.73105858],
       [ 0.26894142,  0.11920292]])
```
that pattern forms the basis for writing functions:

- declare a variable
- construct some modifications to it
- compile these into a function

Thus, we need to express the input as Theano variables
and get the posterior computation using a Theano function.

# Pylearn2 Mean Field Inference

The DBM in Pylearn2 does not directly interface with the input data but rather makes use
of a vector space abstraction which allows them to compile expressions for the mean field
inference.  To actually get posteriors we need to put our data into the vector space form
and then call the mean field inference parameters

## HDF5 dataset

Since my dataset is too large to fit into RAM I use an `pylearn2.datasets.hdf5.HDF5Dataset` instance
to access my data: this allows me to access chunks of data at a time as though I have an `numpy.ndarray`
in RAM without having the whole array loaded.

## `pylearn2.space`

The inputs in `pylearn2` are compiled using `pylearn2.space.VectorSpace` instances.  These
are derived from a class `pylearn2.space.Space` which makes the library more generic.  I have
only begun to understand the implications of the chosen internal representation and the rationale
goes beyond the scope of this post.  Instead, I am just trying to explicate how exactly
things are working.


The main thing to be aware of is the `pylearn2.space.VectorSpace` has a method
`make_theano_batch` which compiles the vector space object into an instance
of `theano.tensor.var.TensorVariable` which means that it can be used to construct
Theano expressions.  In particular, since we are just running the mean field inference
method that is baked into the `pylearn2.models.dbm.DBM` class all we need is the 
`VectorSpace` instance associated with the dimensions of the input space.
Then we can compile the expression and run it over the data.

To code to get the `pylearn2.space.VectorSpace` can be gotten from a `pylearn2.models.dbm.DBM`
instance with the method `get_input_space`

In order to compile our posteriors we need to have the correct input format for the inputs to the
function.

## Computing the posteriors

The posteriors are computed using mean-field inference.  The chunk of code relevant is in `pylearn2`.
In `pylearn2.costs.dbm.BaseCD.get_monitoring_channels` we have this chunk of code

```python
history = model.mf(X, return_history = True)
q = history[-1
Y_hat = q[-1]
```

and from t

