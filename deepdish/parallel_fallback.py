from __future__ import division, print_function, absolute_import

import sys
import itertools as itr
import numpy as np
from . import six


imap_unordered = map
imap = map
starmap_unordered = itr.starmap
starmap = itr.starmap


def rank():
    return 0


def main(name=None):
    return name == '__main__'
