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
