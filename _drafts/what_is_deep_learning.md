---
layout: post
title: What Is Deep Learning
author: mark
---

We are planning a number of posts on "deep learning" and it is useful to take
some time now and again to provide a definition of deep learning.  A good
definition should give insight into the problems people are working on and should
hopefully clarify the approaches people use to solve these problems.  The central
motivating problem is we want computers to see, hear, read, etc. as well
as humans do.  Although, it is difficult to say exactly what is meant by
"seeing", for example, there are a large number of things software should
be able to do if computers could see e.g.: detect objects within images, generate descriptions, and organize images for a search engine.  We want to
write computer programs that can do all the tasks a person can do automatically
 (and hopefully surpass human performance in some areas).  No matter what
the case we can view software capable of such feats as being able to take
some digital input (such as an image or audio recording) we call \\( X \\)
and then produce a response \\( \hat{y} \\) that is judged against some
ideal response \\( y \\) of what the computer should be doing.  In speech
recognition the computer should be able to automatically produce a transcript
\\( \hat{y} \\) that closely matches the words spoken \\( y \\) in a particular
audio recording \\( X \\).

Finding the relationship between the desired outputs and high-dimensional inputs like speech, audio, and text
is difficult.
Deep learning is a family of algorithms that use
 multilayer architectures to learn a feature map \\( \phi \\)
so that \\( \phi(X) \\) has a simpler relationship to the desired response \\( y \\)
than the original input form \\( X \\).  By features, I mean an alternative encoding
of the input data.  Note that __learning__ is emphasized in the definition: 
as far as we know the feature maps
\\( \phi \\) learned by deep learning can be emulated with "shallow" architectures.  What is special
about deep learning is that they seem to learn useful encodings \\( \phi \\) in a purely data-driven
manner with limited human intervention compared to alternative algorithms such as shallow neural networks.


 is not the only approach to such problems, and this post will
clarify how deep learning is the next step in a natural progression of
learning methods.  Indeed we can write

Deep learning is generally concerned with predicting a
response \\( y \\) from input data \\( X \\): e.g. predicting the class
label of an image.  More sophisticated tasks such as automated image captioning
can be fit within this framework.

  When 
people discuss and work on deep learning they talk about the algorithms, hardware,
and massive datasets that are used with deep learning so a good definition
should make it easy to talk about those topics.  The definition that I am writing out here is not
new and can be found in many places such as [work](http://arxiv.org/abs/1503.00036) with Behnam Neyshabur exploring
this idea.  that I believe
does so can be found in [Shakir Mohamed's post](http://blog.shakirm.com/2015/01/a-statistical-view-of-deep-learning-i-recursive-glms/)

A deep network is composed of \\( L \\) layers where each layer \\( l \\) has a 
parameter weight matrix \\( W^{(l)} \\), an input vector \\( \mathbf{x}^{(l)} \\),
and an activation function \\( \boldsymbol{\sigma}^{(l)} \\).  The output of a deep network is
a response \\( \mathbf{y} \\) computed from the final activation 
function \\( \boldsymbol{\sigma}^{(L)} \\):
$$
\mathbf{y} = \boldsymbol{\sigma}^{(L)} \left(W^{(L)}\boldsymbol{\sigma}^{(L-1)}\left(W^{(L-1)}\cdots \boldsymbol{\sigma}^{(1)}\left(W^{(1)}\mathbf{x}^{(1)} \right)\cdots \right) \right).
$$
The equation above means you take the input \\( \mathbf{x}^{(1)} \\), multiply it by \\( W^{(1)} \\),
then pass it through activation \\( \boldsymbol{\sigma}^{(1)} \\) and then
$$ \mathbf{h}^{(1)} = \boldsymbol{\sigma}^{(1)}\left(W^{(1)} \mathbf{x}^{(1)}\right) $$
is the hidden response for the first layer.  \\( \mathbf{h}^{(1)} \\)
is passed to the next layer with matrix \\( W^{(2)} \\) and activation \\( \boldsymbol{\sigma}^{(2)} \\)
so that you compute hidden response \\( \mathbf{h}^{(2)} \\).  This procedure is continued
so that you produce a sequence of hidden responses  \\( \mathbf{h}^{(1)},\mathbf{h}^{(2)},\ldots, \mathbf{h}^{(L)} \\)
where \\(  \mathbf{h}^{(L)} = \mathbf{y} \\).

There are a couple of things to note.  Each activation function is, in general, a vector function: i.e. if layer \\( l \\)
as dimension \\( D _ l \\) and layer \\( l+1 \\) has dimension \\( D _ {l+1} \\) then
\\( \boldsymbol{\sigma}^{(l)} \\) is a function from \\( \mathbb{R}^{ D _ l } \\)
to \\( \mathbb{R}^{D _ {l+1} } \\).  An identity transform is the simplest example for a given
activation function \\( \boldsymbol{\sigma} \\), indeed, if all the activations are identity transforms then
a deep network is just a matrix factorized into \\( L \\) factors:
$$
\mathbf{y} = W^{(L)}W^{(L-1)}\cdots W^{(1)}\mathbf{x}^{(1)}.
$$
Often \\( \boldsymbol{\sigma} \\) will be a coordinate-wise non-linear function such as a
sigmoid or linear rectifier.  

A blog post earlier this year correctly pointed out that this is a generalization of generalized linear models (GLMs).
I would point out further that is a generalization of Vector Generalized linear models since 
GLMs generally have a univariate response.  The connection between vector generalized linear models
and neural networks has been known for some time with Michael I. Jordan having worked on these
ideas in the (early Nineties)[http://www.mitpressjournals.org/doi/abs/10.1162/neco.1994.6.2.181#.VV39m-RY6PQ]
( ungated version (here)[http://18.7.29.232/bitstream/handle/1721.1/7206/AIM-1440.pdf?sequence=2]).  However,
it wasn't until last year that the connection to generalized linear models was profitably used for unsupervised
training of general
deep neural networks.  In some very interesting work from [Ranganath et al.](http://arxiv.org/abs/1411.2581)
networks are trained where the hidden responses are probabilistically modeled using vector GLMs.  

The general form given earlier reduces to many different special-case variations.  One particularly important
example is the case of a convolutional neural network since these have produced such promising results in the
area of computer vision.  CNNs correspond to the case were the weight matrix \\( W \\) is a block circulant
with circulant blocks (BCCB) matrix and, thus, can be represented as 
$$ W = \left(F_D \otimes F_D\right)^* \Lambda \left(F_D \otimes F_D\right)
$$
where \\( F_D \\) is the Fourier transform matrix and \\( \Lambda \\) is a diagonal matrix whose diagonal entries
represent the frequency-domain representation of the convolution.  Usually \\( W \\) will be sparse (since we are often
convolving a small filter with the whole image) so the convolution is often represented in the spatial domain
rather than in the frequency domain.

Linear layers where \\( \boldsymbol{\sigma}^{(l)} \\) is the identity transform may seem somewhat redundant they
can be particularly powerful for forming "bottleneck" layers where the dimension of \\( \mathbf{h}^{(l)} \\)
is much less than either \\( \mathbf{h}^{(l-1)} \\) and \\( \mathbf{h}^{(l+1)} \\).  A bottleneck layer where
\\( \boldsymbol{\sigma}^{(l)} \\) is the identity but has low dimension is an example of low-rank matrix
factorization.  Low-rank factorizations have been shown to be useful for reducing the number of free parameters,
and, thus, speeding up training of the network without negatively impacting performance.

# Divide and Conquer - Twice over

As Michael Jordan in the other article points out (and again emphasized during recent talks as well as in
a private conversation) divide and conquer is an extremely powerful tool for solving mathematical problems.  He
advocates that we should take a "divide-and-conquer" view point to tackle statistical problems.  A data set in
machine learning is usually represented as a matrix \\( X \\) where there are \\( N \\) rows representing
the different data points and there are \\( D \\) columns representing the features.  Hence, there are naturally
two different ways to perform divide-and-conquer: 

1. Split the data along the rows so that we work with subgroups of the data separately (clustering)
2. Split the features into different groups

  I claim that
deep learning makes use of both types simultaneously.

1. clustering data into separate groups where each group is similar
2. handling separate dimensions of the data separately

Lets analyze what is going on to get some sense of what the definition means for understanding deep-learning.
The first thing to notice are the weight matrices \\( W^{(l)} \\).  Since we are viewing this theory as a way
of generalizing matrix factorizations and generalized linear models it is natural to think of the weight matrices
as dictionaries.  The term comes from a type of matrix factorization often used in signal processing.