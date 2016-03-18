from __future__ import print_function, division, absolute_import

# Load the following modules by default
from deepdish.core import (
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
from deepdish import image
from deepdish import parallel

class MovedPackage(object):
    def __init__(self, old_loc, new_loc):
        self.old_loc = old_loc
        self.new_loc = new_loc

    def __getattr__(self, name):
        raise ImportError('The package {} has been moved to {}'.format(
            self.old_loc, self.new_loc))

# This is temporary: remove after a few minor releases
plot = MovedPackage('deepdish.plot', 'vzlog.image')

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
           'image',
           'plot',
           'parallel',
           'timed',
           'aslice',
           ]

VERSION = (0, 3, 2)
ISRELEASED = True
__version__ = '{0}.{1}.{2}'.format(*VERSION)
if not ISRELEASED:
    __version__ += '.git'
