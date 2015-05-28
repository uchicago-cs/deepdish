#!/usr/bin/env python
from __future__ import division, print_function, absolute_import 

from setuptools import setup
import os.path

try:
    # This makes it installable without cython/numpy
    # (useful for building the documentation)
    import numpy as np
    from Cython.Build import cythonize
    with open('requirements.txt') as f:
        required = f.read().splitlines()

    compile_ext = True
except ImportError:
    with open('requirements_docs.txt') as f:
        required = f.read().splitlines()

    compile_ext = False

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
]

args = dict(
    name='deepdish',
    version='0.1.5',
    url="https://github.com/uchicago-cs/deepdish",
    description="Deep Learning experiments from University of Chicago.",
    maintainer='Gustav Larsson',
    maintainer_email='gustav.m.larsson@gmail.com',
    install_requires=required,
    packages=[
        'deepdish',
        'deepdish.parallel',
        'deepdish.plot',
        'deepdish.io',
        'deepdish.util',
        'deepdish.tools',
    ],
    license='BSD',
    classifiers=CLASSIFIERS,
)

if compile_ext:
    setup_requires=['numpy', 'cython'],
    args['ext_modules'] = cythonize("deepdish/plot/resample.pyx")
    args['include_dirs'] = [np.get_include()]

setup(**args)
