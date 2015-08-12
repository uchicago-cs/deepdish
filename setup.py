#!/usr/bin/env python
from __future__ import division, print_function, absolute_import 

from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Topic :: Scientific/Engineering',
]

args = dict(
    name='deepdish',
    version='0.2.1',
    url="https://github.com/uchicago-cs/deepdish",
    description="Deep Learning experiments from University of Chicago.",
    maintainer='Gustav Larsson',
    maintainer_email='gustav.m.larsson@gmail.com',
    install_requires=required,
    scripts=['scripts/ddls'],
    packages=[
        'deepdish',
        'deepdish.parallel',
        'deepdish.io',
        'deepdish.util',
    ],
    license='BSD',
    classifiers=CLASSIFIERS,
)

setup(**args)
