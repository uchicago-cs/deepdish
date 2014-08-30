---
layout: post
title: Deep PCA Nets
---

Tsung-Han Chan and colleagues recently uploaded to [ArXiv](arxiv.org) an [interesting paper](http://arxiv.org/abs/1404.3606) proposing a simple but effective baseline for deep learning.  They propose a novel two-layer architecture where
each layer convolves the image with a filterbank, followed by binary hasing, and finally block histogramming for indexing and pooling.  The filters in the filterbank are learned using simple algorithms such as random projections (RandNet),
principal component analysis (PCANet), and linear discriminant analysis (LDANet).  They find their network to be competitive with other deep learning methods
and [scattering networks](www.di.ens.fr/data/scattering) (introduced by Stéphane Mallat) on a variety of task: face recognition, face verification, hand-written digits recognition, texture discrimination, and object recognition.

The authors extensively test the technique on a variety of datasets and a strong performance on many datasets:

|Dataset   | Task  | Accuracy  |
|---|---|---|
| [Extended Yale B](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html)  | Face Recognition  | 99.58%  |
| [AR](http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html) | Face Recognition  | 95.00%   |
| [FERET](http://www.itl.nist.gov/iad/humanid/feret/feret_master.html) (average)   | Face Recognition  | 97.25%  |
| [MNIST](http://yann.lecun.com/exdb/mnist/) | Digit Recognition  | 99.38% |
| [CUReT](http://www1.cs.columbia.edu/CAVE//exclude/curet/.index.html) | Texture Recognition |  99.61% |
| [CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html) | Object Recognition  | 78.67% |
and they achieve state-of-the-art results on several of the [MNIST Variations](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations) tasks. The method compares favorably to hand-designed features, wavelet-derived featues, and deep-network learned features.



 Extended Yale B, [AR] http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html, FERET


