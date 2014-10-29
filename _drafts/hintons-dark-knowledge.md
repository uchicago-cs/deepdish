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


Dark knowledge was used by Hinton in two different contexts:

- Model compression: using a simpler model with fewer parameters to match the performance of a larger model.
- Specialist Networks: training models specialized to disambiguate between a small number of easily confuseable classses.


To perform model compression we first train the a powerful ensemble method on a dataset.  Ensemble methods have been shown
to consistently achieve strong performance on a variety of tasks for deep neural networks but they have many parameters
that are likely redundant since there are multiple copies of similar models often.  After training the ensemble
and the error rate is sufficiently
low we then use the softmax outputs from the ensemble method to construct training targets for the smaller, simpler model.
The training targets are computed by applying the temperature transform
$$ Z(k;\alpha) = \frac{z_k^{\alpha} }{\sum_{k'}z_{k'}^\alpha} $$
where \\( (z_1,\ldots,z_K ) \\) are the softmax outputs from the larger complicated network.

The simpler model is then trained by alternating between using the standard 1-hot training vectors (where each of the training labels
are treated as having a single one in the entry corresponding to the true class and zeros everywhere else) and using the 
temperature-adjusted softmax outputs from the complicated model.  One of the main settings for where this is useful
is in the context of speech recognition.  Here an ensemble phone recognizer may achieve a low phone error rate, but 
it may be too slow to process user input on the fly.  A simpler model replicated the ensemble method, however, can bring
some of the classification gains of large-scale ensemble deep network models to practical speech systems.

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
the output probabilities for cars and cheetahs sums to a probability similar to the catch-all output of the dog detector.



