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
                   )


# Lazy load these?
import deepdish.io
import deepdish.util
import deepdish.plot
from . import image

__all__ = ['deepdish',
           'set_verbose',
           'info',
           'warning',
           'bytesize',
           'humanize_bytesize',
           'memsize',
           'span',
           'apply_once',
           'io',
           ]

VERSION = (0, 0, 0)
ISRELEASED = False
__version__ = '{0}.{1}.{2}'.format(*VERSION)
if not ISRELEASED:
    __version__ += '.dev'
