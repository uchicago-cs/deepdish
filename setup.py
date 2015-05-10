#!/usr/bin/env python
from __future__ import division, print_function, absolute_import 

from setuptools import setup
import numpy as np
import os.path

from Cython.Build import cythonize

CLASSIFIERS = [
'Development Status :: 3 - Alpha',
'Intended Audience :: Science/Research',
'License :: OSI Approved :: BSD License',
'Programming Language :: Python',
'Programming Language :: Python :: 3',
]

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='deepdish',
    version='0.1.4',
    url="https://github.com/uchicago-cs/deepdish",
    description="Deep Learning experiments from University of Chicago.",
    maintainer='Gustav Larsson',
    maintainer_email='gustav.m.larsson@gmail.com',
    setup_requires=['numpy', 'cython'],
    install_requires=required,
    packages=[
        'deepdish',
        'deepdish.io',
        'deepdish.util',
        'deepdish.plot',
        'deepdish.tools',
    ],
    ext_modules=cythonize("deepdish/plot/resample.pyx"),
    include_dirs=[np.get_include()],
    license='BSD',
    classifiers=CLASSIFIERS,
)
