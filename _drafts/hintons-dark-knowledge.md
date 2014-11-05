------------------------
layout: post
title: Hinton's Dark Knowledge
author: gustav, mark
------------------------

On Thursday, October 2, 2014 [Geoffrey Hinton](http://www.cs.toronto.edu/~hinton/) gave a [talk](http://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/geoff_hinton_dark14.pdf) on what he calls "dark knowledge" which
he claims is most of what [deep learning methods](http://deeplearning.net/)
actually learn.  The talk presented an idea that had been introduced in
[(Caruana, 2006)](http://www.cs.cornell.edu/~caruana/compression.kdd06.pdf)
where a more complex model is used to train a simpler, compressed model. 
The main point of the talk introduces the idea that classifiers built from
a [softmax function](http://en.wikipedia.org/wiki/Softmax_function) have
a great deal more information contained in them than just a classifier, the
correlations in the softmax outputs are very informative. For example, when 
building a computer vision system to detect `cats`,`dogs`, and `boats` the output
entries for `cat` and `dog` in a softmax classifier will always have more
correlation than `cat` and `boat` since `cats` look similar to `dogs`.


The most interesting part of the talk
was the analysis of why the model compression worked and the reports of how it
could be used to improve large scale image classification at Google.

  There were three main points I understood from the talk:
1. A smaller network with fewer parameters can match the performance of a larger deep network with many more parameters
2. The smaller network can be trained with fewer examples and possibly without seeing all of the classes
3. Smaller specialized networks that classify among a small set of classes can improve the performance of large general networks that can classify between many classes

These three things come from using a technique similar to one discussed in [this paper](http://www.cs.cornell.edu/~caruana/compression.kdd06.pdf) that appeared in 2006.  
The technique is based on a temperature-type trick to use for training.  A deep neural network generally maps an input vector \\( \mathbf{x}\in\mathbb{R}^D_{in} \\) to an output feature vector 
    \\( \mathbf{h}\in\mathbb{R}^{D_{out}} \\) and a logistic regression is estimated to map the output
features \\( \mathbf{h} \\) to the classes \\( 1,\ldots, C \\).  The logistic regressor estimates a matrix of coefficients \\( (\boldsymbol{\beta}_1,\ldots,\boldsymbol{\beta}_C) \in \mathbb{R}^{D\times C} \\) so that
given an output vector \\( \mathbf{h} \\) we may compute a vector of \\( C \\) scores \\(\mathbf{z}\in\mathbb{R}^C \\):
$$ \mathbf{z}=\mathbf{h}^{\top}(\boldsymbol{\beta}_1,\ldots,\boldsymbol{\beta}_C) $$
We may use this vector of scores \\( \mathbf{z} \\) to estimate a class posterior given the output features $\mathbf{h}$ namely we set
$$ \hat{\mathbb{P}}(c\mid \mathbf{h}) = \frac{e^{\boldsymbol{\beta}^\top_c \mathbf{h}}}{\sum_{c'=1}^C e^{\boldsymbol{\beta}^\top_c \mathbf{h}} } $$
where \\( \hat{\mathbb{P}}(c\mid \mathbf{h}) \\) denotes the estimate of the posterior probability that our data point belongs to class
\\( c \\).  

The goal of the learning algorithm is to estimate each of the parameter vectors \\( \boldsymbol{\beta}_c \)) for all of the classes.
Usually, the parameters learned to minimize the log loss:
$$ \sum_{n=1}^NL(\mathbf{h}_n,y_n;\boldsymbol{\beta}) = -\sum_{n=1}^N\sum_{c=1}^C 1\{y_n=c\}\log\frac{e^{\boldsymbol{\beta}^\top_c \mathbf{h}}}{\sum_{c'=1}^C e^{\boldsymbol{\beta}^\top_c \mathbf{h}} } $$
which is the log-likelihood of the data under the logistic regression model.  \\( \boldsymbol{\beta} \\) is usually estimated with iterative algorithms
since there is no closed-form solution.

The loss function may be viewed as a cross-entropy between 
an empirical conditional distribution \\( y\mid x \\) and
a predicted conditional distribution given by the logistic model.
This cross-entropy view motivates the dark knowledge training
paradigm.  Instead of training the cross entropy against the data
one could train it against a prediction by another model.  In particular,
for each data point \\( \mathbf{x}_n \\) another neural network
may make a prediction
$$\mathbf{y} = (y_{n,1},y_{n,2},\ldots,y_{n,C})^\top $$
such that $$\sum_{c=1}^C y_{n,c} = 1$$ and \\( y_{n,c} > 0 \\)
for every class \\( c \\): i.e. the network outputs a distribution
over labels.  The idea is to train the smaller network using
this output distribution rather than the true labels.
The loss function then becomes
$$ \sum_{n=1}^NL(\mathbf{h}_n,y_n;\boldsymbol{\beta}) = -\sum_{n=1}^N\sum_{c=1}^C y_{n,c}\log\frac{e^{\boldsymbol{\beta}^\top_c \mathbf{h}}}{\sum_{c'=1}^C e^{\boldsymbol{\beta}^\top_c \mathbf{h}} }
$$
and solving for \\( \boldsymbol{\beta} \\) may be performed in the usual manner.


The first point suggests some efficiency gains.  The classifiers used for speech recognition and computer vision
at Google need to be efficient.

  The central observation is that ensemble deep learning methods
perform better than other algorithms, but that extra performance comes at great cost the information per
parameter in an ensemble method seems to be very low since an ensemble may
have 10 times as many parameters but be only a few more percent accurate
than a non-ensemble method.

| Competition   |  Team  | # Predictors  |
|---|---|---|
|  | BellKor's Pragmatic Chaos |  454 |
| [ILSVRC](http://www.image-net.org/challenges/LSVRC/2014/results) | [VGG](http://arxiv.org/pdf/1409.1556) | 7 |
| 									       
| [Extended Yale B](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html)  | Face Recognition  | 99.58%  |
| [AR](http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html) | Face Recognition  | 95.00%   |
| [FERET](http://www.itl.nist.gov/iad/humanid/feret/feret_master.html) (average)   | Face Recognition  | 97.25%  |
| [MNIST](http://yann.lecun.com/exdb/mnist/) | Digit Recognition  | 99.38% |
| [CUReT](http://www1.cs.columbia.edu/CAVE//exclude/curet/.index.html) | Texture Recognition |  99.61% |
| [CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html) | Object Recognition  | 78.67% |


  He draws an analogy to cosmology which posits that most
of the mass of the universe is in so-called [dark matter](http://en.wikipedia.org/wiki/Dark_matter). Unlike physics, however, he posits