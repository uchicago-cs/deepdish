---
layout: post
title: Convolutional Deep Maxout Networks for Phone Recognition -- Overview (Part 0)
author: mark
---

László Tóth has [recently](http://www.inf.u-szeged.hu/~tothl/pubs/ICASSP2013.pdf) [written](http://www.inf.u-szeged.hu/~tothl/pubs/IS2014-proof.pdf) a [series](http://www.inf.u-szeged.hu/~tothl/pubs/ICASSP2014.pdf)
of [papers](http://www.inf.u-szeged.hu/~tothl/pubs/IS2013.pdf) documenting
progress on phonetic recognition in the [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) database using deep networks.  His most recent paper "Convolutional Deep Maxout Networks for Phone Recognitione" (Interspeech, 2014) obtained a \\( 16.5\% \\) error rate on the core test set--a common TIMIT benchmark (details of which can be found in [this linked thesis](http://groups.csail.mit.edu/sls//publications/1998/phdthesis-drew.pdf))--which is state-of-the-art on the dataset.
This post is the beginning of a series of posts describing my explanation
and implementation
of his ideas.  This post is primarily concerned with explaining what is written in the papers.

# High Level Overview

These networks operate on log mel filterbank features--a filtered [spectrogram](http://en.wikipedia.org/wiki/Spectrogram) transformed to match a [mel scale](http://en.wikipedia.org/wiki/Mel_scale).  These filterbank features are a time-frequency representation 
which may serve as a visual representation of speech utterances, e.g.:

<center><img src="images/greasy.png" alt="Greasy Filterbank"></center>

which is a representation of someone saying "greasy".
These filterbank features are then processed by a special formulation of a [convolutional neural network](http://en.wikipedia.org/wiki/Convolutional_neural_network) (CNN)
 combined with down-sampling and max-pooling. These steps produce local time-frequency translation invariance and allow the network
to use many fewer parameters.  	 These convolutions are performed using a [maxout network](arxiv.org/pdf/1302.4389) which takes
the maximum of several input neurons to compute the hidden-layer response.  A multilayer maxout network processes
these inputs and dropout was used for regularization.

Training is bootstrapped from a [context-dependent](http://dx.doi.org/10.1109/ICASSP.1985.1168283) (CD) Gaussian mixture model-hidden Markov model (GMM-HMM) system which is forced-aligned to the speech data and gives a much richer annotation of the utterances.  
The force-aligned labels give a context-dependent phonetic label out of a set of 858 possible labels every 10 milliseconds
which is a much finer-grained annotation than the 61 labels over larger segments of speech given by the TIMIT transcriptions. The 
CNN is trained to match these state labels and it is used to produce posteriors over the state labels that is combined
with a simple bigram phonetic language model
to produce a sequence label output.

## Filterbank Features

Filterbank features are a representation of the speech signal using log mel-filtered spectrograms.  A speech signal is
a [pressure wave](http://en.wikipedia.org/wiki/Sound) produced by a vocal apparatus (such as a person talking) and we wish to
encode these using features amenable to a deep network.  Usually, the sound is saved onto a computer hard-drive as a
digitized waveform using a [Pulse-code modulation stream](http://en.wikipedia.org/wiki/Pulse-code_modulation#Modulation) (PCM)--this is the basis for [WAV](http://en.wikipedia.org/wiki/WAV) files (other formats such as [MP3](http://en.wikipedia.org/wiki/MP3) and [Ogg](http://en.wikipedia.org/wiki/Ogg) generally use lossy compression).  The PCM format captures the [amplitude](http://en.wikipedia.org/wiki/Amplitude)
 of the 
speech signal at regular intervals of time: e.g. in TIMIT there are 16 amplitude samples per millisecond so 25 milliseconds of
speech looks like:

<center><img src="images/greasy_wave.png" alt="Greasy Waveform" width="512"></center>


whose waveform structure is inferred from the list of amplitude
samples and sampling frequency.  Note that 
only the relative scale of the amplitude is important and the absolute
scale is determined entirely for efficiency in the audio coding.  We take the [Fourier transform](http://en.wikipedia.org/wiki/Fourier_transform) of the waveform, 
take the squared modulus, and
use a filterbank to convert the waveform to a [Mel scale](http://en.wikipedia.org/wiki/Mel_scale):

<center><img src="images/greasy_fbank.png" alt="Greasy Filterbank" width="512"></center>

The plot above corresponds to the ninth column, which we call the *frame*, of the visual representation of "greasy" shown in the first image:

<center><img src="images/greasy_fbank_column.png" alt="Greasy Filterbank"></center>

## Time-Frequency Convolutional Neural Networks 

The axes of the image correspond (approximately) to group operations: time translation and frequency transposition (or scaling) hence
we may meaningfully apply [convolutional filters](http://en.wikipedia.org/wiki/Convolution#Convolutions_on_groups) and hence
convolutional neural nets (CNNs) are a good choice for the
bottom layer of the neural network.