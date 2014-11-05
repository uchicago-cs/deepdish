---
layout: post
author: markgustav
title: Hinton's Dark Knowledge
---

On Thursday, October 2, 2014 [Geoffrey Hinton](http://www.cs.toronto.edu/~hinton/) gave a [talk](http://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/geoff_hinton_dark14.pdf) on what he calls "dark knowledge" which
he claims is most of what [deep learning methods](http://deeplearning.net/)
actually learn.  The talk presented an idea that had been introduced in
[(Caruana, 2006)](http://www.cs.cornell.edu/~caruana/compression.kdd06.pdf)
where a more complex model is used to train a simpler, compressed model. 
The main point of the talk introduces the idea that classifiers built from
a [softmax function](http://en.wikipedia.org/wiki/Softmax_function) have
a great deal more information contained in them than just a classifier; the
correlations in the softmax outputs are very informative. For example, when 
building a computer vision system to detect `cats`,`dogs`, and `boats` the output
entries for `cat` and `dog` in a softmax classifier will always have more
correlation than `cat` and `boat` since `cats` look similar to `dogs`.


Dark knowledge was used by Hinton in two different contexts:

- Model compression: using a simpler model with fewer parameters to match the performance of a larger model.
- Specialist Networks: training models specialized to disambiguate between a small number of easily confuseable classes.

## Preliminaries

A deep neural network typically maps an input vector \\( \mathbf{x}\in\mathbb{R}^{D _ {in}} \\) to a set of scores \\(f(\mathbf{x}) \in \mathbb{R}^C \\) for each of \\( C \\) classes. These scores are then interpreted as a posterior distribution over the labels using a [softmax function](http://en.wikipedia.org/wiki/Softmax_function)

$$ \hat{\mathbb{P}}(\mathbf{y} \mid \mathbf{x}; \boldsymbol{\Theta}) = \mathrm{softmax}(f(\mathbf{x}; \boldsymbol{\Theta})). $$

The parameters of the entire network are collected in \\( \boldsymbol{\Theta}
\\).  The goal of the learning algorithm is to estimate \\( \boldsymbol{\Theta}
\\).  Usually, the parameters are learned by minimizing the log loss for all
training samples

$$ L ^ \mathrm{(hard)} = \sum _ {n=1}^NL(\mathbf{x} _ n,y _ n;\boldsymbol{\Theta}) 
 = -\sum _ {n=1}^N\sum _ {c=1}^C 1 _ \{\\{y _ n=c\\}\} \log \hat{\mathbb{P}}(y _ c \mid \mathbf{x} _ n ; \boldsymbol{\Theta} ), $$

which is the negative of the log-likelihood of the data under the logistic
regression model. The parameters \\( \boldsymbol{\Theta} \\) are estimated with
iterative algorithms since there is no closed-form solution. 

This loss function may be viewed as a cross entropy between an empirical
posterior distribution and a predicted posterior distribution given by the
model. In the case above, the empirical posterior distribution is simply a
1-hot distribution that puts all its mass at the ground truth label.  This
cross-entropy view motivates the dark knowledge training paradigm, which can be
used to do model compression.

## Model compression

Instead of training the cross entropy against the labeled data one could train
it against the posteriors of a previously trained model. In Hinton's narrative,
this previous model is an ensemble method, which may contain many large deep
networks of similar or various architectures.  Ensemble methods have been
shown to consistently achieve strong performance on a variety of tasks for deep
neural networks. However, these networks have a large number of parameters,
which makes it computationally demanding to do inference on new samples. To
alleviate this, after training the ensemble and the error rate is sufficiently
low, we use the softmax outputs from the ensemble method to construct training
targets for the smaller, simpler model.

In particular, for each data point \\( \mathbf{x} _ n \\), our first bigger
ensemble network may make the prediction

$$ \mathbf{\hat y} _ n ^ \mathrm{(big)} = \mathbb{P} (\mathbf{y} \mid \mathbf{x}; \boldsymbol{\Theta} ^ \mathrm{(big)}). $$

The idea is to train the smaller network using this output distribution rather
than the true labels.  However, since the posterior estimates are typically low
entropy, the dark knowledge is largely indiscernible without a log transform.
To get around this, Hinton increases the entropy of the posteriors by using a
transform that "raises the temperature" as

$$ [g(\mathbf{y}; T)] _ k = \frac{y _ k^{1/T} }{\sum _ {k'} y _ {k'} ^ {1/T}}, $$

where \\( T \\) is a temperature parameter that when raised increases the entropy.
We now set our target distributions as

$$ \mathbf{y} ^ \mathrm{(target)} _ n = g(\mathbf{y} ^ \mathrm{(big)} _ n; T). $$

The loss function becomes

$$ L ^ \mathrm{(soft)} = \sum _ {n=1}^NL(\mathbf{x} _ n,y _ n;\boldsymbol{\Theta}^\mathrm{(small)}) = -\sum _ {n=1}^N \sum _ {c=1}^C y ^ \mathrm{(target)} _ {n, c} \log \hat{\mathbb{P}}(y _ c \mid \mathbf{x} _ n ; \boldsymbol{\Theta} ^ \mathrm{(small)}).
$$

Hinton mentioned that the best results by combining the two loss functions. At
first, we thought he meant alternating between them, as in train one batch with
\\( L ^ \mathrm{(hard)} \\) and the other with \\( L ^ \mathrm{(soft)} \\).
However, after a discussion with a professor that also attended the talk, it
seems as though Hinton took the a convex combination of the two loss functions

$$ L = \alpha L ^ \mathrm{(soft)} + (1 - \alpha) L ^ \mathrm{(hard)}, $$

where \\( \alpha \\) is a parameter. This professor had the impression
that an appropriate value was \\( \alpha = 0.9 \\) after asking Hinton about it.

One of the main settings for where this is useful is in the context of
speech recognition.  Here an ensemble phone recognizer may achieve a low phone
error rate, but it may be too slow to process user input on the fly.  A simpler
model replicated the ensemble method, however, can bring some of the
classification gains of large-scale ensemble deep network models to practical
speech systems.

## Specialist networks

Specialist networks are a way of using dark knowledge to improve the performance of deep network models regardless of their underlying
complexity.  They are used in the setting where there are many different classes.  As before, deep network is trained over the data
and each data point is assigned a target that corresponds to the temperature adjusted softmax output.  These softmax outputs
are then clustered multiple times using k-means and the resultant clusters indicate easily confuseable data points that
come from a subset of classes.  Specialist networks
are then trained only on the data in these clusters using a restricted number of classes.  They treat all classes not contained in
the cluster as coming from a single "other" class.  These specialist networks are then trained using alternating one-hot, temperature-adjusted technique.
The ensemble method constructed by combining the various specialist networks creates benefits for the overall network.

One technical hiccup created by the specialist network is that the specialist networks are trained using different classes than the full
network so combining the softmax outputs from multiple networks requires a combination trick.  Essentially there is an optimization problem
to solve: ensure that the catchall "dustbin" classes for the specialist networks match a sum of your softmax outputs. So that if you have
cars and cheetahs grouped together in one class for your dog detector you combine that network with your cars versus cheetahs network by ensuring
the output probabilities for cars and cheetahs sum to a probability similar to the catch-all output of the dog detector.



