---
layout: post
title: DBM Posteriors from Pylearn2
author: mark
---

As a part of a recent project that I am working on I am using Deep Boltzmann
Machine posteriors as input features to a query-by-audio speech recognition
system.  The main task to accomplish is to train a Deep Boltzmann Machine
and then use the posteriors from such a system as input features
to a dynamic-time-warping based detection system.  The setting we consider
is we have MFCC features (which include a frame along with its immediate
context) and a posteriorgram generated from Gaussian Mixture Models. I used
the multiprediction training algorith from `pylearn2` to learn a DBM
model. My intention is to extract the posteriors over novel data and then
to test the output.