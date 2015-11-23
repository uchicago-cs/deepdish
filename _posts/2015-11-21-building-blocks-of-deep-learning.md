---
layout: post
title: The building blocks of Deep Learning
author: gustav
---

A feed-forward network is built up of nodes that make a directed ascyclic graph (DAG). This post will focus on how a single node works and what we need to implement if we want to define one. It is aimed at people who generally know how deep networks work, but can still be confused about exactly what gradients need to be computed for each node (i.e. myself, all the time).

# Network

In our network, we will have three different types of nodes (often called *layers* as well):

* Static data (data and labels)
* Dynamic data (parameters)
* Functions

This is a bit different from the traditional take on nodes, since we are not allowing nodes to have any
internal parameters. Instead, parameters will be fed into function nodes as dynamic data. A network with
two fully connected layers may look like this:

![node]({{ site.baseurl }}/public/images/building-block/example.svg)

The static data nodes are light blue and the dynamic data nodes (parameter nodes) are orange.

To train this network, all we need is the derivative of the loss, \\( L \\),
with respect to each of the parameter nodes. For this, we need to consider the
canonical building block, the function node:

![node]({{ site.baseurl }}/public/images/building-block/block.svg)

It takes any number of inputs and produces an output (that eventually leads to the loss). In the eyes of the node, it makes no distinction between static and dynamic data, which makes things both simpler and more flexible. What we need from this building block is a way to compute \\(\\mathbf{z}\\) and the derivative of \\( L \\) with respect to each of the inputs. First of all, we need a function that computes

\\begin{equation}
\\mathbf{z} = \\mathrm{forward}((\\mathbf{x}^1, \\dots, \\mathbf{x}^n)).
\\end{equation}

This is the simple part and should be trivial once you have decided what you want the node to do.

Next, computing the derivative of a single element of one of the inputs may look like (superscript omitted):

\\begin{equation}
\\frac{ \\partial L }{ \\partial x _ i  } = \\sum _ j \\frac{ \\partial L }{ \\partial  z _ j } \\frac{ \\partial z _ j }{\\partial x _ i }
 \\end{equation}

We broke the derivative up using the multivariable chain rule (also known as the [total derivative](https://en.wikipedia.org/wiki/Total_derivative)). It can also be written as

\\begin{equation}
\\frac{ \\partial L }{ \\partial \\mathbf{x} } = \\left(\\frac{ \\mathrm{d} \\mathbf{z} }{ \\mathrm{d} \\mathbf{x} }\\right) ^ \\intercal  \\frac{ \\partial L }{ \\partial \\mathbf{z} } \\quad\\quad \\left[ \\mathbb{R}^ { A \\times 1} = \\mathbb{R} ^ {A \\times B} \\mathbb{R} ^ {B \\times 1} \\right]
\\end{equation}

This assumes that the input size is \\( A \\) and the output size is \\( B \\). The derivative \\(  \\frac{ \\partial L }{ \\partial \\mathbf{z} } \\in \\mathbb{R} ^ {B} \\) is something that needs to be given to the building block from the outside (this is the gradient being back-propagated). The Jacobian \\( \\frac{ \\mathrm{d} \\mathbf{z} }{ \\mathrm{d} \\mathbf{x} } \\in \\mathbb{R} ^ {B \times A} \\) on the other hand needs to be defined by the node. However, we do not necessarily need to explicitly compute it or store it. All we need is to define the function

\\begin{equation}
\\frac{ \\partial L }{ \\partial \\mathbf{x}  } = \\mathrm{backward}\\left(\\mathbf{x}, \\mathbf{z}, \\frac{ \\partial L }{ \\partial \\mathbf{z} }\\right)
\\end{equation}

This would need to be done for each input separately. Since they sometimes share computations, frameworks like Caffe use a single function for the entire node's backward computation. In our code examples, we will adopt this as well, meaning we will be defining:


\\begin{equation}
\\left(\\frac{ \\partial L }{ \\partial \\mathbf{x}^1  }, \\dots, \\frac{ \\partial L }{ \\partial \\mathbf{x}^n  }\\right) = \\mathrm{backward}\\left((\\mathbf{x}^1, \\dots, \\mathbf{x}^n), \\mathbf{z}, \\frac{ \\partial L }{ \\partial \\mathbf{z} }\\right)
\\end{equation}

It is also common to support multiple outputs, however for simplicity (and without loss of generality) we will assume there is only one.

## Functions

So, the functions that we need to define for a single node is first the forward pass:

![forward]({{ site.baseurl }}/public/images/building-block/forward.svg)

The input data refers to all the inputs, so it will for instance be a list of arrays.

Next, the backward pass:

![backward]({{ site.baseurl }}/public/images/building-block/backward.svg)

It takes three inputs as described above and returns the gradient of the loss with respect to the input. It does not need to take the output data, since it can be computed from the input data. However, if it is needed, we might as well pass it in since we will have computed it already.

## Forward / Backward pass

A DAG describes a [partial ordering](https://en.wikipedia.org/wiki/Partially_ordered_set). First, we need to sort our nodes so that they do not violate the partial order. There will probably be several solutions to this, but we can pick one arbitrarily.

Once we have this ordering, we call forward on the list from the first node to the last. The order will guarantee that the dependencies of a node have been computed when we get to it. The Loss should be the last node. This is called a *forward pass*.

Then, we call backward on this list in reverse. This means that we start with the Loss node. Since we do not have any output diff at this point, we simply set it to an array of all ones. We proceed until we are done with the first in the list. This is called a *backward pass*.

Once the forward and the backward pass have been performed, we take the gradients that have arrived at each parameter node and perform a gradient descent update in the opposite direction.

## Weight sharing

By externalizing the parameters, it makes parameter sharing conceptually easy to deal with. For instance, if we wanted to share weights (but not biases), we could do:

![shared]({{ site.baseurl }}/public/images/building-block/shared.svg)

In this case, \\( \\mathbf{W} \\) would receive *two* gradient arrays, in which case the sum is taken before performing the update step.

## ReLU

As an example, the ReLU has a single input of the same size as the output, so \\( A = B \\). The output is computed elementwise as

\\begin{equation}
z _ i = \\max(0, x _ i)
\\end{equation}

which could translate to something like this in Python (who uses pseudo-code anymore?):

    def forward(inputs):
        return np.maximum(inputs[0], 0)

For the backward pass, the Jacobian will be a diagonal matrix, with entries

\\begin{equation}
\\frac{\\partial z _ i}{\\partial x _ i} = 1 _ {\\{ x _ i > 0  \\}},
\\end{equation}

where \\( 1 _ {\\{P\\}} \\) is 1 if the predicate \\( P\\) is true, and zero otherwise (see [Iverson bracket](https://en.wikipedia.org/wiki/Iverson_bracket)). We can now write the gradient of the loss as

\\begin{equation}
\\frac{ \\partial L }{ \\partial \\mathbf{x} } = \\left(\\frac{ \\mathrm{d} \\mathbf{z} }{ \\mathrm{d} \\mathbf{x} }\\right) ^ \\intercal  \\frac{ \\partial L }{ \\partial \\mathbf{z} } = \\mathbf{1} _ {\\{ \\mathbf{x} > \\mathbf{0} \\} } \\odot  \\frac{ \\partial L }{ \\partial \\mathbf{z} },
\\end{equation}

where \\( \odot \\) denotes an [elementwise product](https://en.wikipedia.org/wiki/Hadamard_product_%28matrices%29).

    def backward(inputs, output, output_diff):
        return [(inputs[0] > 0) * output_diff]

Note that we have to return a list, since we could have multiple inputs.

## Dense

Moving on to the dense (fully connected) layer where

\\begin{equation}
\\mathbf{z} = \\mathbf{W} ^ \\intercal \\mathbf{x} + \\mathbf{b} \\quad\\quad (\\mathbb{R}^{B \times 1} = \\mathbb{R}^{B \times A} \\mathbb{R}^{A \times 1} + \mathbb{R}^{B \times 1})
\\end{equation}

However, remember that we make no distinction between static and dynamic input, and from the point of view of our Dense node it simply looks like:

\\begin{equation}
\\mathbf{z} = \\mathbf{x} ^ 2 \\mathbf{x} ^ 1 + \\mathbf{x} ^ 3
\\end{equation}

Which might translate to:

    def forward(inputs):
        x, W, b = inputs
        return W.T @ x + b

For the backward pass, we need to compute all three Jacobians and multiply them by the gradient coming in from above. Let's start with \\( \\mathbf{x} \\):

\\begin{equation}
\\frac{\\mathrm{d} \\mathbf{z} }{\\mathrm{d} \\mathbf{x}} = \\mathbf{W} ^ \\intercal \\in \\mathbb{R} ^ {B \\times A}
\\end{equation}

which gives us

\\begin{equation}
\\text{Gradient #1 }\\rightarrow \\quad\\quad\\
\\frac{\\partial L}{\\partial \\mathbf{x}} = \\left(\\frac{ \\mathrm{d} \\mathbf{z} }{ \\mathrm{d} \\mathbf{x} }\\right) ^ \\intercal  \\frac{ \\partial L }{ \\partial \\mathbf{z} }= \\mathbf{W} \\frac{ \\partial L }{ \\partial \\mathbf{z} }
\\quad\\quad \\leftarrow\\text{ Gradient #1}
\\end{equation}

Moving on. Since \\( \mathbf{W} \\in \\mathbb{R} ^ {A \times B} \\), it means its Jacobian should have the dimensions \\( B \\times (A \\times B) \\). We know the bias will drop off, so we can write the output that we will be taking the Jacobian of as:

\\begin{equation}
\\mathbf{z}' = \\left(  \\sum _ {j = 1} ^ A W _ {j, 1} x _ j, \\dots, \\sum _ {j = 1} ^ A W _ {j, B} x _ j \\right)
\\end{equation}

Now, let's compute the derivative of \\( z' _ i \\) (and thus \\(z _ i \\)) with respect to \\( W _ {j, k} \\):

\\begin{equation}
\\frac{\\partial z _ i}{\\partial W _ {j, k}} =
\\left\\{
    \\begin{array}{ll}
        x _ j  & \\mbox{if } i = k \\\\
        0 & \\mbox{otherwise}
    \\end{array}
\\right.
\\end{equation}

With a bit of collapsing things together (Einstein notation is great for this, but the steps are omitted here), we get an outer product of two vectors
\\begin{equation}
\\text{Gradient #2 }\\rightarrow \\quad\\quad\\
\\frac{\\partial L}{\\partial \\mathbf{W}} = \\left(\\frac{ \\mathrm{d} \\mathbf{z} }{ \\mathrm{d} \\mathbf{W} }\\right) ^ \\intercal  \\frac{ \\partial L }{ \\partial \\mathbf{z} } = \mathbf{x} \\left( \\frac{ \\partial L }{ \\partial \\mathbf{z} } \\right)^\\intercal
\\quad\\quad \\leftarrow\\text{ Gradient #2}
\\end{equation}

The final Jacobian is simply an identity matrix

\\begin{equation}
\\frac{\\mathrm{d} \\mathbf{z} }{\\mathrm{d} \\mathbf{b}} = I \\in \\mathbb{R} ^ {B \\times B}
\\end{equation}

so the Loss derivative with respect to the bias is just the gradients coming in from above unchanged

\\begin{equation}
\\text{Gradient #3}\\rightarrow \\quad\\quad\\
\\frac{\\partial L}{\\partial \\mathbf{b}} = \\left(\\frac{ \\mathrm{d} \\mathbf{z} }{ \\mathrm{d} \\mathbf{b} }\\right) ^ \\intercal  \\frac{ \\partial L }{ \\partial \\mathbf{z} }= \\frac{ \\partial L }{ \\partial \\mathbf{z} }
\\quad\\quad \\leftarrow\\text{ Gradient #3}
\\end{equation}

We thus have all three gradients (with no regard as to which ones are parameters). This might translate in code to:

    def backward(inputs, output, output_diff):
        x, W, b = inputs
        return [
            W.T @ output_diff,
            np.outer(x, output_diff),
            output_diff,
        ]

Now, the frameworks that I know do not externalize the parameters, so instead
of returning the two last gradients, they would be applied to the interal
parameters through some other means. However, the main ideas and certainly the
math will be exactly the same.

## Loss

You should get the idea by now. The final note is that when we do this for the
Loss layer, we still need to pretend the node has been placed in the middle of
a network with an actual loss at the end of it. The Loss node should not be
different in any way, except that its output size is scalar. However, a good
loss node should theoretically be able to be used in the middle of a network, so
it should still query `output_diff` and use it correctly (even though it will be all ones when
used in the final position).

## Summary

In summary, the usual steps when constructing a new node/layer is:

* Compute the forward pass
* Calculate the Jacobian for all your inputs (static and dynamic alike)
* Multiply them with the gradient coming in from above. At this point, we will often realize that we do not have to ever store the entire Jacobian.

