---
layout: post
title: Initialization of deep networks
author: gustav
---

As we all know, the solution to a non-convex optimization algorithm (like
stochastic gradient descent) depends on the initial values of the parameters.
This post is about choosing initialization parameters for deep networks and how
it affects the convergence. We will also discuss the related topic of vanishing
gradients.

First, let's go back to the time of sigmoidal activation functions and
initialization of parameters using IID Gaussian or uniform distributions with fairly
arbitrarily set variances. Building deep networks was difficult because of
exploding or vanishing activations and gradients. Let's take activations first:
If all your parameters are too small, the variance of your activations will
drop in each layer. This is a problem if your activation function is sigmoidal,
since it is approximately linear close to 0. That is, you gradually lose your
non-linearity, which means there is no benefit to having multiple layers. If,
on the other hand, your activations become larger and larger, then your
activations will saturate and become meaningless, with gradients approaching 0.

![Activation functions]({{ site.baseurl }}/public/images/activation-functions.svg)

Let us consider one layer and forget about the bias. Note that the following analysis
and conclussion is taken from Glorot and Bengio[1]. Consider a weight matrix
\\( W \in \mathbf{R}^{m \times n} \\), where each element was drawn from an IID
Guassian with variance \\( \mathrm{Var}(W) \\). Note that we are a bit abusive with notation 
letting \\( W \\) denote both a matrix and a univariate random variable. We
also assume there is no correlation between our input and our weights and both
are zero-mean. If we consider one filter (row) in \\( W \\), say \\( \mathbf{w}
\\) (a random vector), then the variance of the output signal over the input signal is:

$$  
    \frac{ \mathrm{Var}(\mathbf{w}^T \mathbf{x}) }{ \mathrm{Var}(X) } = \frac{\sum _ n^N \mathrm{Var}(w _ n x _ n)}{\mathrm{Var}(X)} = \frac{n \mathrm{Var}(W) \mathrm{Var}(X)}{\mathrm{Var}(X)}= n\mathrm{Var}(W)
$$

As we build a deep network, we want the variance of the signal going forward in
the network to remain the same, thus it would be advantageous if \\( n \mathrm{Var}(W)
= 1. \\) The same argument can be made for the gradients, the signal going
backward in the network, and the conclusion is that we would also like \\( m
\mathrm{Var}(W) = 1. \\) Unless \\( n = m, \\) it is impossible to sastify both
of these conditions. In practice, it works well if both are approximately
satisfied. One thing that has never been clear to me is why it is only
necessary to satisfy these conditions when picking the initialization values of
\\( W. \\) It would seem that we have no guarantee that the conditions will
remain true as the network is trained.

Nevertheless, this *Xavier initialization* (after Glorot's first name) is a neat
trick that works well in practice. However, along came rectified linear units
(ReLU), a non-linearity that is scale-invariant around 0 *and* does not
saturate at large input values. This seemingly solved both of the problems the
sigmoid function had; or were they just alleviated? I am unsure of how widely
used Xavier initialization is, but if it is not, perhaps it is because ReLU
seemingly eliminated this problem.

However, take the most competative network as of recently, VGG[2]. They do not
use this kind of initialization, although they report that it was tricky to get
their networks to converge. They say that they first trained their most shallow
architecture and then used that to help initialize the second one, and so
forth. They presented 6 networks, so it seems like an awfully complicated
training process to get to the deepest one.

A recent paper by He et al.[3] presents a pretty straightforward generalization
of ReLU and Leaky ReLU. What is more interesting is their emphasis on the
benefits of Xavier initialization even for ReLU. They re-did the derivations
for ReLUs and discovered that the conditions were the same up to a factor 2.
The difficulty Simonyan and Zisserman had training VGG is apparently avoidable,
simply by using Xavier intialization (or better yet the ReLU adjusted version).
Using this technique, He et al. reportedly trained a whopping 30-layer deep
network to convergence in one go.

Another recent paper tackling the signal scaling problem is by Ioffe and
Szegedy[4]. They call the change in scale *internal covariate shift* and claim
this forces learning rates to be unnecessarily small. They suggest that if all
layers have the same scale and remain so throughout training, a much higher
learning rate becomes practically viable. You cannot just standardize the
signals, since you would lose expressive power (the bias disappears and in the
case of sigmoids we would be constrained to the linear regime). They solve this
by re-introducing two parameters per layer, scaling and bias, added again after
standardization. The training reportedly becomes about 6 times faster and they
present state-of-the-art results on ImageNet.

I reckon we will see a lot more work on this frontier in the next few years.
Especially since it also relates to the -- right now wildly popular --
Recurrent Neural Network (RNN), which connects output signals back as inputs.
The way you train such network is that you unroll the time axis, treating it as
an extremely deep feedforward network. This greatly exacerbates the vanishing
gradient problem. A popular solution, called Long-Term Short Memory (LTSM) is
to introduce a memory cell, a type of teleport that allows a signal to jump
ahead many many time steps. This means that the gradient is retained for all
those time steps and can be propagated back to an earlier time without
vanishing.

Until those advancements have been made, I will be sticking to Xavier
initialization. If you are using Caffe, the one take-away of this post is to
use the following on all your layers:

```
weight_filler { type: "xavier" }
```

### References

 1. X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,” in International conference on artificial intelligence and statistics, 2010, pp. 249–256.

 2. K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014. [[pdf](http://arxiv.org/pdf/1409.1556v5)]

 3. K. He, X. Zhang, S. Ren, and J. Sun, “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,” arXiv:1502.01852 [cs], Feb. 2015. [[pdf](http://arxiv.org/pdf/1502.01852v1)]

 4. S. Ioffe and C. Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,” arXiv:1502.03167 [cs], Feb. 2015. [[pdf](http://arxiv.org/pdf/1502.03167v2)]




