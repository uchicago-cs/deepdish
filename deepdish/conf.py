from __future__ import division, print_function, absolute_import
import os
import sys
if sys.version_info >= (3,):
    from configparser import ConfigParser
else:
    from ConfigParser import ConfigParser

# Handles .deepdish.conf files

DEFAULTS = {
    'io': dict(compression='zlib'),
}


def config():
    """
    Returns the value for a specific option.
    """
    conf = ConfigParser()
    conf.read_dict(DEFAULTS)
    try:
        with open(os.path.expanduser('~/.deepdish.conf')) as f:
            conf.read_file(f)
    except FileNotFoundError:
        pass

    return conf
