from __future__ import print_function, division, absolute_import

# Load the following modules by default
from deepdish.core import (set_verbose,
                   info,
                   warning,
                   bytesize,
                   humanize_bytesize,
                   memsize,
                   span,
                   apply_once,
                   tupled_argmax,
                   multi_range,
                   timed,
                   aslice,
                   )

from deepdish import io
from deepdish import util
from deepdish import plot
from deepdish import image
from deepdish import parallel

__all__ = ['deepdish',
           'set_verbose',
           'info',
           'warning',
           'bytesize',
           'humanize_bytesize',
           'memsize',
           'span',
           'apply_once',
           'tupled_argmax',
           'multi_range',
           'io',
           'util',
           'plot',
           'image',
           'parallel',
           'timed',
           'aslice',
           ]

VERSION = (0, 1, 5)
ISRELEASED = True
__version__ = '{0}.{1}.{2}'.format(*VERSION)
if not ISRELEASED:
    __version__ += '.git'
