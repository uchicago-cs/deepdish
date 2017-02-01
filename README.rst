.. image:: https://readthedocs.org/projects/deepdish/badge/?version=latest
    :target: https://readthedocs.org/projects/deepdish/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/uchicago-cs/deepdish.svg?branch=master
    :target: https://travis-ci.org/uchicago-cs/deepdish/

.. image:: https://img.shields.io/pypi/v/deepdish.svg
    :target: https://pypi.python.org/pypi/deepdish

.. image:: https://coveralls.io/repos/uchicago-cs/deepdish/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/uchicago-cs/deepdish?branch=master
   
.. image:: https://img.shields.io/badge/license-BSD%203--Clause-blue.svg?style=flat
    :target: http://opensource.org/licenses/BSD-3-Clause 

deepdish
========

Flexible HDF5 saving/loading and other data science tools from the University of Chicago. This repository also host a Deep Learning blog:

* http://deepdish.io

Installation
------------
::

    pip install deepdish

Alternatively (if you have conda with the `conda-forge <https://conda-forge.github.io/>`__ channel)::

    conda install deepdish


Main feature
------------
The primary feature of deepdish is its ability to save and load all kinds of
data as HDF5. It can save any Python data structure, offering the same ease of
use as pickling or `numpy.save <http://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html>`__. However, it improves by also offering:

- Interoperability between languages (HDF5 is a popular standard)
- Easy to inspect the content from the command line (using ``h5ls`` or our
  specialized tool ``ddls``)
- Highly compressed storage (thanks to a PyTables backend)
- Native support for scipy sparse matrices and pandas ``DataFrame``, ``Series``
  and ``Panel``
- Ability to partially read files, even slices of arrays

An example:

.. code:: python

    import deepdish as dd

    d = {
        'foo': np.ones((10, 20)),
        'sub': {
            'bar': 'a string',
            'baz': 1.23,
        },
    }
    dd.io.save('test.h5', d)

This can be reconstructed using ``dd.io.load('test.h5')``, or inspected through
the command line using either a standard tool::

    $ h5ls test.h5
    foo                      Dataset {10, 20}
    sub                      Group

Or, better yet, our custom tool ``ddls`` (or ``python -m deepdish.io.ls``)::

    $ ddls test.h5
    /foo                       array (10, 20) [float64]
    /sub                       dict
    /sub/bar                   'a string' (8) [unicode]
    /sub/baz                   1.23 [float64]

Read more at `Saving and loading data <http://deepdish.readthedocs.io/en/latest/io.html>`__.

Documentation
-------------

* http://deepdish.readthedocs.io/
