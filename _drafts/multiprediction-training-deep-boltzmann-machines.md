---
layout: post
title: Multiprediction Training for Deep Boltzmann Machines
author: mark
---

# Introduction

[Deep Boltzmann machines](http://www.cs.toronto.edu/~fritz/absps/dbm.pdf) (DBMs) are
a prominent example of a generative deep learning algorithm.  
I have begun using them in my research because they provide a natural way to
do deep learning in the unsupervised setting. In this post I will explain some of the motivation
behind them as well as give a discussion of 
[multiprediction training](http://papers.nips.cc/paper/5024-multi-prediction-deep-boltzmann-machines)
for DBMs that was introduced about a year ago.  I will also write up how I am
using them to do experiments in pylearn2.

# The Theory

## The Energy Functional

The DBM is an undirected graphical model that describes distributions over binary vectors.  The DBM
incorporates "layers" of latent binary vectors with bipartite connections between adjacent levels. Consider the case of a 2-layer network:
one visible layer of units which is a binary vector \\( \mathbf{v} \\), a hidden layer above that
\\( \mathbf{h}_1 \\), and a hidden layer above that \\( \mathbf{h}_2 \\).  Each layer is a vector
of binary random variables and the distribution is written as:

$$ P(\mathbf{v},\mathbf{h}) = \frac{1}{\mathcal{Z}(\mathbf{v},\mathbf{h})}e^{-E(\mathbf{v},\mathbf{h})} $$

where we abbreviate  
$$ \mathbf{h}=(\mathbf{h}_1,\mathbf{h}_2) $$
\\( E \\) is an energy function, and \\( \mathcal{Z} \\) is the
partition function.  The energy function expresses the bipartite
relationship between the layers:

<div> $$ E(\mathbf{v},\mathbf{h})=\mathbf{v}^{\top} W_1 \mathbf{h}_1 + \mathbf{h}_1 W_2 \mathbf{h}_2 $$ </div>

where \\( W_1,W_2 \\) are parameter matrices (here we are ignoring the bias terms).

Training these is difficult because the partition function is
intractable and requires summing over exponentially many configurations for the several layers.

## A Variational Lower Bound

One route to making the training tractable

# Implementing DBMs in pylearn2

The `pylearn2` library makes extensive use of Python's object facilities to define experiments.  Before we can describe the code
we will first review how training these algorithms works in general:

- Model parameters are initialized 
- The dataset is arranged into a sequence of small mini-batches
- We use each of these mini-batches, one at a time, to update the model parameters using a cost function
- After going through all the mini-batches we have completed one epoch
- Based on a termination criterion we either perform another training epoch or we return the current model estimate

The overall architecture of these steps is handled by a `pylearn2.train.Train`
which is initialized with:

- `dataset` which will be an instance whose class is inherited from `pylearn2.datasets.datasets.Dataset`
- `model` which will be an instance of `pylearn2.models.model.Model`
- `algorithm` which is an instance of `pylearn2.training_algorithms.train_algorithm.TrainingAlgorithm`
- `extensions` a list of objects which help monitor the termination criteria
- an optional `save_path` which is a string that contains a file path to where you want intermediate estimates of the model to be saved
- an optional `save_freq` which indicates how often to save the intermediate model as a function of epoch number

We will look at experiments that specify all of these objects.

## Dataset

The dataset controls how data gets into the model and converted into the mini-batches. The particular data that I am working with
is a moderate-sized speech dataset with 1.3 million examples so I cannot hold it in RAM.  Thus, I used a `pylearn2.datasets.hdf5.HDF5Dataset`
which makes use of `h5py` so that I can read chunks of the dataset on the fly from disk without having to load 
all of the data.  If you are working with larger signal processing datasets HDF5 is a great choice.

I had written another script prior to running the experiment to construct the HDF5 dataset with the training data and the labels.

## Model

I used the `pylearn2.models.dbm.DBM` class for my model.  To specify the model we need

- `batch_size` which indicates how large the mini-batches should be
- `visible_layer` which is the layer of the DBM that interfaces with the observed data
- `hidden_layers`: a list of hidden layers for the DBM
- `inference_procedure`: a class that represents the procedure for performing mean field inference in a DBM


According to the [pylearn2 tutorial](http://deeplearning.net/software/pylearn2/theano_to_pylearn2_tutorial.html)
there are two main things that need to be set in your deep learning model.  They are:

- A subclass of `pylearn2.costs.cost.Cost`
- A subclass of `pylearn2.models.model.Model`

these are specified in our experiment though a `dbm.yaml` file.  We are going to focus on multiprediction 
Deep Boltzmann Machines due to the strong results they get on a variety of tasks.  We will build
off of the `yaml` file included in the [multiprediction Deep Boltzmann Machines paper](http://www-etud.iro.umontreal.ca/~goodfeli/mp_dbm.pdf).
The basic file is a `pylearn2.train.Train` object whose initialization has fields:

