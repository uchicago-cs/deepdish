from __future__ import print_function, division, absolute_import

# Load the following modules by default
from .core import (set_verbose,
                   info,
                   warning,
                   bytesize,
                   humanize_bytesize,
                   memsize,
                   span,
                   apply_once,
                   tupled_argmax,
                   multi_range,
                   Timer,
                   )

import deepdish.io
import deepdish.util
import deepdish.plot
from . import image

try:
    import mpi4py
    from . import parallel
except ImportError:
    from . import parallel_fallback as parallel

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
           'Timer',
           ]

VERSION = (0, 1, 3)
ISRELEASED = True
__version__ = '{0}.{1}.{2}'.format(*VERSION)
if not ISRELEASED:
    __version__ += '.dev'
