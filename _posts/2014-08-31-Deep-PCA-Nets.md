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


The authors achieve state-of-the-art results on several of the [MNIST Variations](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations) tasks. The method compares favorably to hand-designed features, wavelet-derived featues, and deep-network learned features.

## The algorithm

The main algorithm is cascading two filterbank convolutions
with a mean normalization step,
 followed by 
a binary hashing step and a final histogramming step.  Training 
involves estimating the filterbanks used for the convolutions,
and estimating the classifier to be used on the features.

### Filterbank Convolutions

The filterbanks are estimated by performing principal components
analysis (PCA) over patches. We extract all the \\( 7\times 7 \\)
patches from all the images and vectorize them so that each patch
is a flat 49-entry vector.  For each patch vector we take the mean
of the entries (the DC-component) and then subtract that mean
from each entry of the vector so that all of our patches
are now zero mean.  We perform PCA over these zero-mean
patch vectors and retain
the top eight components \\( W\in\mathbb{R}^{49\times 8} \\). Each
principle component (a column of \\( W \\)) is a filter and may be
converted into a \\( 7\times 7 \\) kernel which is convolved with
the input images.  The input images are zero-padded for the
convolution so that the output has the same dimension as the 
image itself.  So, using the eight columns of \\( W \\)
we take each input image \\( \mathcal{I}\\) and convert it
into eight output images \\( \mathcal{I} _ l \\)  where \\( 1\leq l\leq 8 \\). 

#### Second Layer

The second layer is constructed by iterating the algorithm from
the first layer over each of the eight output images.  For each
output image \\( \mathcal{I} _ l \\) we take the dense set
of flattened patch vectors, remove the DC-component, and then
estimate a PCA filterbank (again with eight filters).  Each filter
\\( w _ {l,k} \\) from the filterbank is convolved with \\( \mathcal{I} _ l \\) to produce a new image \\( \mathcal{I} _ {l,k} \\).  Repeating
this process for each filter in the filterbanks produces 64
images.

### Hashing and Histogramming

The 64 images have the same size as the original image thus we 
may view the filter outputs as producing a three-dimensional
array \\( \mathcal{J}\in\mathbb{R}^{H\times W\times 64} \\)
where \\( H\times W \\) are the dimensions of the input image. Each
of the 64 images is produced from a layer one filter \\( l _ 1 \\)
and a layer two filter \\( l _ 2 \\) so we denote the associated
image as \\( \mathcal{J} _ {l _ 1,l _ 2} \\).  Each
pixel \\( (x,y) \\) from the image has an associated
8-dimensional feature vector \\( \mathcal{J}(x,y)\in\mathbb{R}^{64} \\).  These feature vectors are converted into integers by using a
[Heaviside step function](http://en.wikipedia.org/wiki/Heaviside_step_function) \\( H \\) sum:
$$ \mathcal{K} _ {l _ 1}(x,y)=\sum _ {z=1}^{8} 2^{z-1}\cdot H(\mathcal{J} _ {l _ 1,z}(x,y,z)).   $$

We note that we produce a hashed image such as \\( \mathcal{K}_l \\)
for each filter \\( l \\) in the layer one filterbank so this means
that we have eight images after the hashing operation and the images
are all integers.

#### Histogramming

We then take \\( 7\times 7 \\) blocks of the hashed images
\\( \mathcal{K} \\) and compute a histogram with \\(2^{64} \\)
bins over the values observed.  These blocks can be disjoint
(used for face recognition) or they can be overlapping (useful
for digit recognition).  The histograms formed from these blocks
and from the several images are all concatenated into a feature 
vector.  Classification is then performed using this feature
vector.

### Classification

In the paper they estimate a multiclass linear SVM to operate
on the estimated feature vector for each image.  The same
setup was used for all input data.

## Author's Implementation

Code for the paper is [here](http://mx.nthu.edu.tw/~tsunghan/download/PCANet_demo.zip)
and it has implementations for cifar10 and MNIST basic (a subset of MNIST).  With a little extra
work one can also make it suitable for testing on the whole MNIST data set.

I tested this implementation on the MNIST dataset and received the following results:


## My Implementation

I was intrigued by the simplicity of the network and the strong
results so I implemented the network myself

  Each
filter is then convolved with zero-padded images so that the 
resultant convolution




 Extended Yale B, [AR] http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html, FERET


