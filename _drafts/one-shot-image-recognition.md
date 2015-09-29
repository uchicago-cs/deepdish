---
layout: post
title: One Shot Image Recognition
author: mark
---

One-shot and zero-shot learning is a favorite topic of mine particularly in the
context of deep learning.  The literature on so called "fast" learning (learning
with very few examples) is as old as the literature on neural networks the
earliest example of the term "one-shot" being used to my knowledge was
[twenty-eight years ago](http://iopscience.iop.org/article/10.1209/0295-5075/4/8/001/meta). The
topic appeared in a
[paper](http://people.cs.uchicago.edu/~stoehr/papers/koch2015siamese.pdf) on
one-shot learning of handwritten characters from
the
[deep learning workshop](https://sites.google.com/site/deeplearning2015/accepted-papers)
in [ICML 2015](http://icml.cc/2015/).

When the deep learning revolution was just beginning in the late 2000s a
researcher who I deeply respect predicted that deep learning would only ever
work on large datasets.  He thought that in order to learn from a small number
of examples we would have to use very cleverly-engineered shallow methods.
Recent work suggests that this prediction is wrong, and this paper on one-shot
learning paper over handwritten characters is part of a growing literature
demonstrating the capability of deep learning algorithms to efficiently learn
from a small number of examples.

was primarily interested in methods that replicated the human ability to learn
quickly and efficiently from a small number of examples.  My guess is that deep
learning will actually give rise to methods with greater generalization ability
than previous machine learning methods.

The setting is that we train a learning system that can successfully
perform.  This is useful when retraining a machine learnign system is expensive
of unfeasible as in the case of the online learning setting such as web
retrieval.  The task considered in the paper is learning new alphabets of
characters from the Omniglot dataset:

Another paper of interest is a recurrent neural network that
[learns to converse](http://people.cs.uchicago.edu/~stoehr/papers/vinyals2015neural.pdf).
The network builds off of the 

