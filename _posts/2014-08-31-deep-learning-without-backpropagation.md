Most deep neural network training algorithms simultaneously train all layers often using backpropagation.
However, greedy layer-wise training of deep networks appears from time
to time.  Greedy training is attractive because of its computational
efficiency and simplicity. Hinton and Bengio investigated layer-wise training 
for deep belief networks.  More recently, greedy layer-wise training
has been applied to convolutional neural networks by Coates and Chan, untrained networks have also been considered by Mallat--all of which produced competitive results.  We will review these several architectures and 
discuss their features.

Some things to consider:
* what is back-propagation
* when can back-propagation be used
* what are examples of networks that don't use back-propagation
* what are the best results that such networks are able to achieve
* how does removing back-propagation increase simplicity
* greedy training is often used as pre-training (however this reached its popularity peak a few years ago and most competition winners today do not use unsupervised pre-training [1])
* how good can greedy training be? Its inherent limitations are not entirely clear, which is why it's interesting to explore, considering its benefits
* find papers that compare for instance a Deep Belief Network's performance with and without fine-tuning.
* Results on with/without fine-tuning: [2]

[1] Schmidhuber, J., Deep Learning in Neural Networks: An Overview
[2] Lamblin, P., Bengio, Y., Important Gains from Supervised Fine-Tuning of Deep Architectures on Large Labeled Sets. *(NIPS Deep Learning and Unsupervised Feature Learning Workshop, 2010*

