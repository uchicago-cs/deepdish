#!/usr/bin/env python
from __future__ import division, print_function, absolute_import 

from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os.path

from Cython.Distutils import build_ext

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
"""


def cython_extension(modpath, mp=False):
    extra_compile_args = ["-O3"]
    extra_link_args = []
    if mp:
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-fopenmp')
    filepath = os.path.join(*modpath.split('.')) + ".pyx"
    return Extension(modpath, [filepath],
                     extra_compile_args=extra_compile_args,
                     extra_link_args=extra_link_args)

setup(
    name='deepdish',
    cmdclass={'build_ext': build_ext},
    version='0.9.1',
    url="https://github.com/uchicago-cs/deepdish",
    description="Deep Learning experiments from University of Chicago.",
    maintainer='Gustav Larsson',
    maintainer_email='gustav.m.larsson@gmail.com',
    packages=[
        'deepdish',
        'deepdish.io',
        'deepdish.util',
        'deepdish.plot',
    ],
    ext_modules=[
        cython_extension("deepdish.plot.resample"),
    ],
    include_dirs=[np.get_include()],
    license='BSD',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
)
