from __future__ import division, print_function, absolute_import
import os
import sys
if sys.version_info >= (3,):
    from configparser import ConfigParser
else:
    from ConfigParser import ConfigParser


def config():
    """
    Loads and returns a ConfigParser from ``~/.deepdish.conf``.
    """
    conf = ConfigParser()
    # Set up defaults
    conf.add_section('io')
    conf.set('io', 'compression', 'zlib')

    conf.read(os.path.expanduser('~/.deepdish.conf'))
    return conf
