------------------------
layout: post
title: Hinton's Dark Knowledge
author: gustav, mark
------------------------

On Thursday, October 2, 2014 [Geoffrey Hinton](http://www.cs.toronto.edu/~hinton/) gave a [talk](http://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/geoff_hinton_dark14.pdf) on what he calls "dark knowledge" which
he claims is most of what [deep learning methods](http://deeplearning.net/)
actually learn.  The talk presented an idea that had been introduced in
[this paper](http://www.cs.cornell.edu/~caruana/compression.kdd06.pdf)
although with a very different implementation.  The most interesting part of the talk
was the analysis of why the model compression worked and the reports of how it
could be used to improve large scale image classification at Google.

  There were three main points I understood from the talk:
1. A smaller network with fewer parameters can match the performance of a larger deep network with many more parameters
2. The smaller network can be trained with fewer examples and possibly without seeing all of the classes
3. Smaller specialized networks that classify among a small set of classes can improve the performance of large general networks that can classify between many classes

These three things come from using a technique similar to one discussed in [this paper](http://www.cs.cornell.edu/~caruana/compression.kdd06.pdf)

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