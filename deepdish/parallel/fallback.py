from __future__ import division, print_function, absolute_import

import itertools as itr

__all__ = ['rank', 'imap_unordered', 'imap',
           'starmap_unordered', 'starmap', 'main']


def rank():
    """
    Returns MPI rank. If the MPI backend is not used, it will always return 0.
    """
    return 0


def imap_unordered(f, params):
    """
    This can return the elements in any particular order. This has a lower
    memory footprint than the ordered version and will be more responsive in
    terms of printing the results. For instance, if you run the ordered
    version, and the first batch is particularly slow, you won't see any
    feedback for a long time.
    """
    return map(f, params)


def imap(f, params):
    """
    Analogous to `itertools.imap` (Python 2) and `map` (Python 3), but run in
    parallel.
    """
    return map(f, params)


def starmap_unordered(f, params):
    """
    Similar to `imap_unordered`, but it will unpack the parameters. That is, it
    will call ``f(*p)``, for each `p` in `params`.
    """
    return itr.starmap(f, params)


def starmap(f, params):
    """
    Analogous to `itertools.starmap`, but run in parallel.
    """
    return itr.starmap(f, params)


def main(name=None):
    """
    Main function.

    Example use:

    >>> if gv.parallel.main(__name__):
    ...     res = gv.parallel.imap_unordered(f, params)

    """
    return name == '__main__'
