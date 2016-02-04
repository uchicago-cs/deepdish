from __future__ import division, print_function, absolute_import

import numpy as np
import tables
import warnings
from scipy import sparse
try:
    import pandas as pd
    _pandas = True
except ImportError:
    _pandas = False

try:
    from types import SimpleNamespace
    _sns = True
except ImportError:
    _sns = False

from deepdish import six

IO_VERSION = 9
DEEPDISH_IO_PREFIX = 'DEEPDISH_IO'
DEEPDISH_IO_VERSION_STR = DEEPDISH_IO_PREFIX + '_VERSION'
DEEPDISH_IO_UNPACK = DEEPDISH_IO_PREFIX + '_DEEPDISH_IO_UNPACK'
DEEPDISH_IO_ROOT_IS_SNS = DEEPDISH_IO_PREFIX + '_ROOT_IS_SNS'

# Types that should be saved as pytables attribute
ATTR_TYPES = (int, float, bool, six.string_types, six.binary_type,
              np.int8, np.int16, np.int32, np.int64, np.uint8,
              np.uint16, np.uint32, np.uint64, np.float16, np.float32,
              np.float64, np.bool_, np.complex64, np.complex128)

if _pandas:
    class _HDFStoreWithHandle(pd.io.pytables.HDFStore):
        def __init__(self, handle):
            self._path = None
            self._complevel = None
            self._complib = None
            self._fletcher32 = False
            self._filters = None

            self._handle = handle


def is_pandas_dataframe(level):
    return ('pandas_version' in level._v_attrs and
            'pandas_type' in level._v_attrs)


class ForcePickle(object):
    """
    When saving an object with `deepdish.io.save`, you can wrap objects in this
    class to force them to be pickled. They will automatically be unpacked at
    load time.
    """
    def __init__(self, obj):
        self.obj = obj


def _dict_native_ok(d):
    """
    This checks if a dictionary can be saved natively as HDF5 groups.

    If it can't, it will be pickled.
    """
    if len(d) >= 256:
        return False

    # All keys must be strings
    for k in d:
        if not isinstance(k, six.string_types):
            return False

    return True


def _get_compression_filters(compression=True):
    if compression is True:
        compression = 'blosc'

    if compression is False or compression is None:
        ff = None
    else:
        if isinstance(compression, (tuple, list)):
            compression, level = compression
        else:
            level = 9

        try:
            ff = tables.Filters(complevel=level, complib=compression,
                                shuffle=True)
        except Exception:
            warnings.warn(("(deepdish.io.save) Missing compression method {}: "
                           "no compression will be used.").format(compression))
            ff = None
    return ff


def _save_ndarray(handler, group, name, x, filters=None):
    if np.issubdtype(x.dtype, np.unicode_):
        # Convert unicode strings to pure byte arrays
        strtype = b'unicode'
        itemsize = x.itemsize // 4
        atom = tables.UInt8Atom()
        x = x.view(dtype=np.uint8)
    elif np.issubdtype(x.dtype, np.string_):
        strtype = b'ascii'
        itemsize = x.itemsize
        atom = tables.StringAtom(itemsize)
    else:
        atom = tables.Atom.from_dtype(x.dtype)
        strtype = None
        itemsize = None
    assert np.min(x.shape) > 0, ("deepdish.io.save does not support saving "
                                 "numpy arrays with a zero-length axis")
    # For small arrays, compression actually leads to larger files, so we are
    # settings a threshold here. The threshold has been set through
    # experimentation.
    if filters is not None and x.size > 300:
        node = handler.create_carray(group, name, atom=atom,
                                     shape=x.shape,
                                     chunkshape=None,
                                     filters=filters)
    else:
        node = handler.create_array(group, name, atom=atom,
                                    shape=x.shape)
    if strtype is not None:
        node._v_attrs.strtype = strtype
        node._v_attrs.itemsize = itemsize
    node[:] = x


def _save_pickled(handler, group, level, name=None):
    warnings.warn(('(deepdish.io.save) Pickling {}: This may cause '
                   'incompatibities (for instance between Python 2 and '
                   '3) and should ideally be avoided').format(level),
                  DeprecationWarning)
    node = handler.create_vlarray(group, name, tables.ObjectAtom())
    node.append(level)


def _save_level(handler, group, level, name=None, filters=None):
    if isinstance(level, ForcePickle):
        _save_pickled(handler, group, level, name=name)

    elif isinstance(level, dict) and _dict_native_ok(level):
        # First create a new group
        new_group = handler.create_group(group, name,
                                         "dict:{}".format(len(level)))
        for k, v in level.items():
            if isinstance(k, six.string_types):
                _save_level(handler, new_group, v, name=k)

    elif (_sns and isinstance(level, SimpleNamespace) and
          _dict_native_ok(level.__dict__)):
        # Create a new group in same manner as for dict
        new_group = handler.create_group(
            group, name, "SimpleNamespace:{}".format(len(level.__dict__)))
        for k, v in level.__dict__.items():
            if isinstance(k, six.string_types):
                _save_level(handler, new_group, v, name=k)

    elif isinstance(level, list) and len(level) < 256:
        # Lists can contain other dictionaries and numpy arrays, so we don't
        # want to serialize them. Instead, we will store each entry as i0, i1,
        # etc.
        new_group = handler.create_group(group, name,
                                         "list:{}".format(len(level)))

        for i, entry in enumerate(level):
            level_name = 'i{}'.format(i)
            _save_level(handler, new_group, entry, name=level_name)

    elif isinstance(level, tuple) and len(level) < 256:
        # Lists can contain other dictionaries and numpy arrays, so we don't
        # want to serialize them. Instead, we will store each entry as i0, i1,
        # etc.
        new_group = handler.create_group(group, name,
                                         "tuple:{}".format(len(level)))

        for i, entry in enumerate(level):
            level_name = 'i{}'.format(i)
            _save_level(handler, new_group, entry, name=level_name)

    elif isinstance(level, np.ndarray):
        _save_ndarray(handler, group, name, level, filters=filters)

    elif _pandas and isinstance(level, (pd.DataFrame, pd.Series, pd.Panel)):
        store = _HDFStoreWithHandle(handler)
        store.put(group._v_pathname + '/' + name, level)

    elif isinstance(level, (sparse.dok_matrix,
                            sparse.lil_matrix)):
        raise NotImplementedError(
            'deepdish.io.save does not support DOK or LIL matrices; '
            'please convert before saving to one of the following supported '
            'types: BSR, COO, CSR, CSC, DIA')

    elif isinstance(level, (sparse.csr_matrix,
                            sparse.csc_matrix,
                            sparse.bsr_matrix)):
        new_group = handler.create_group(group, name, "sparse:")

        _save_ndarray(handler, new_group, 'data', level.data, filters=filters)
        _save_ndarray(handler, new_group, 'indices', level.indices, filters=filters)
        _save_ndarray(handler, new_group, 'indptr', level.indptr, filters=filters)
        _save_ndarray(handler, new_group, 'shape', np.asarray(level.shape))
        new_group._v_attrs.format = level.format
        new_group._v_attrs.maxprint = level.maxprint

    elif isinstance(level, sparse.dia_matrix):
        new_group = handler.create_group(group, name, "sparse:")

        _save_ndarray(handler, new_group, 'data', level.data, filters=filters)
        _save_ndarray(handler, new_group, 'offsets', level.offsets, filters=filters)
        _save_ndarray(handler, new_group, 'shape', np.asarray(level.shape))
        new_group._v_attrs.format = level.format
        new_group._v_attrs.maxprint = level.maxprint

    elif isinstance(level, sparse.coo_matrix):
        new_group = handler.create_group(group, name, "sparse:")

        _save_ndarray(handler, new_group, 'data', level.data, filters=filters)
        _save_ndarray(handler, new_group, 'col', level.col, filters=filters)
        _save_ndarray(handler, new_group, 'row', level.row, filters=filters)
        _save_ndarray(handler, new_group, 'shape', np.asarray(level.shape))
        new_group._v_attrs.format = level.format
        new_group._v_attrs.maxprint = level.maxprint

    elif isinstance(level, ATTR_TYPES):
        setattr(group._v_attrs, name, level)

    elif level is None:
        # Store a None as an empty group
        new_group = handler.create_group(group, name, "nonetype:")

    else:
        _save_pickled(handler, group, level, name=name)


def _load_specific_level(handler, grp, path, sel=None):
    if path == '':
        if sel is not None:
            return _load_sliced_level(handler, grp, sel)
        else:
            return _load_level(handler, grp)
    vv = path.split('/', 1)
    if len(vv) == 1:
        if hasattr(grp, vv[0]):
            if sel is not None:
                return _load_sliced_level(handler, getattr(grp, vv[0]), sel)
            else:
                return _load_level(handler, getattr(grp, vv[0]))
        elif hasattr(grp, '_v_attrs') and vv[0] in grp._v_attrs:
            if sel is not None:
                raise ValueError("Cannot slice this type")
            v = grp._v_attrs[vv[0]]
            if isinstance(v, np.string_):
                v = v.decode('utf-8')
            return v
        else:
            raise ValueError('Undefined entry "{}"'.format(vv[0]))
    else:
        level, rest = vv
        if level == '':
            return _load_specific_level(handler, grp.root, rest, sel=sel)
        else:
            if hasattr(grp, level):
                return _load_specific_level(handler, getattr(grp, level),
                                            rest, sel=sel)
            else:
                raise ValueError('Undefined group "{}"'.format(level))


def _load_pickled(level):
    if isinstance(level[0], ForcePickle):
        return level[0].obj
    else:
        return level[0]


def _load_level(handler, level):
    if isinstance(level, tables.Group):
        if _sns and (level._v_title.startswith('SimpleNamespace:') or
                     DEEPDISH_IO_ROOT_IS_SNS in level._v_attrs):
            val = SimpleNamespace()
            dct = val.__dict__
        else:
            dct = {}
            val = dct

        # Load sub-groups
        for grp in level:
            lev = _load_level(handler, grp)
            n = grp._v_name
            # Check if it's a complicated pair or a string-value pair
            if n.startswith('__pair'):
                dct[lev['key']] = lev['value']
            else:
                dct[n] = lev

        # Load attributes
        for name in level._v_attrs._f_list():
            if name.startswith(DEEPDISH_IO_PREFIX):
                continue
            v = level._v_attrs[name]
            dct[name] = v

        if level._v_title.startswith('list:'):
            N = int(level._v_title[len('list:'):])
            lst = []
            for i in range(N):
                lst.append(dct['i{}'.format(i)])
            return lst
        elif level._v_title.startswith('tuple:'):
            N = int(level._v_title[len('tuple:'):])
            lst = []
            for i in range(N):
                lst.append(dct['i{}'.format(i)])
            return tuple(lst)
        elif level._v_title.startswith('nonetype:'):
            return None
        elif is_pandas_dataframe(level):
            assert _pandas, "pandas is required to read this file"
            store = _HDFStoreWithHandle(handler)
            return store.get(level._v_pathname)
        elif level._v_title.startswith('sparse:'):
            frm = level._v_attrs.format
            if frm in ('csr', 'csc', 'bsr'):
                shape = tuple(level.shape[:])
                cls = {'csr': sparse.csr_matrix,
                       'csc': sparse.csc_matrix,
                       'bsr': sparse.bsr_matrix}
                matrix = cls[frm](shape)
                matrix.data = level.data[:]
                matrix.indices = level.indices[:]
                matrix.indptr = level.indptr[:]
                matrix.maxprint = level._v_attrs.maxprint
                return matrix
            elif frm == 'dia':
                shape = tuple(level.shape[:])
                matrix = sparse.dia_matrix(shape)
                matrix.data = level.data[:]
                matrix.offsets = level.offsets[:]
                matrix.maxprint = level._v_attrs.maxprint
                return matrix
            elif frm == 'coo':
                shape = tuple(level.shape[:])
                matrix = sparse.coo_matrix(shape)
                matrix.data = level.data[:]
                matrix.col = level.col[:]
                matrix.row = level.row[:]
                matrix.maxprint = level._v_attrs.maxprint
                return matrix
            else:
                raise ValueError('Unknown sparse matrix type: {}'.format(frm))
        else:
            return val

    elif isinstance(level, tables.VLArray):
        if level.shape == (1,):
            return _load_pickled(level)
        else:
            return level[:]

    elif isinstance(level, tables.Array):
        if 'strtype' in level._v_attrs:
            strtype = level._v_attrs.strtype
            itemsize = level._v_attrs.itemsize
            if strtype == b'unicode':
                return level[:].view(dtype=(np.unicode_, itemsize))
            elif strtype == b'ascii':
                return level[:].view(dtype=(np.string_, itemsize))

        return level[:]


def _load_sliced_level(handler, level, sel):
    if isinstance(level, tables.VLArray):
        if level.shape == (1,):
            return _load_pickled(level)
        else:
            return level[sel]

    elif isinstance(level, tables.Array):
        return level[sel]

    else:
        raise ValueError('Cannot partially load this data type using `sel`')


def save(path, data, mode = 'w', compression='blosc'):
    """
    Save any Python structure to an HDF5 file. It is particularly suited for
    Numpy arrays. This function works similar to ``numpy.save``, except if you
    save a Python object at the top level, you do not need to issue
    ``data.flat[1]`` to retrieve it from inside a Numpy array of type
    ``object``.

    Five types of objects get saved natively in HDF5, the rest get serialized
    automatically.  For most needs, you should be able to stick to the five,
    which are:

    * Dictionaries
    * Lists and tuples
    * Basic data types (including strings and None)
    * Numpy arrays
    * SimpleNamespaces (for Python >= 3.3, but see note below)

    A recommendation is to always convert your data to using only these five
    ingredients. That way your data will always be retrievable by any HDF5
    reader. A class that helps you with this is `deepdish.util.Saveable`.

    Note that the SimpleNamespace type will be read in as dictionaries for
    earlier versions of Python.

    This function requires the `PyTables <http://www.pytables.org/>`_ module to
    be installed.

    Parameters
    ----------
    path : string
        Filename to which the data is saved.
    data : anything
        Data to be saved. This can be anything from a Numpy array, a string, an
        object, or a dictionary containing all of them including more
        dictionaries.
    mode: string
        The mode to open the file. It can be one of the following:
        *'w': Write; a new file is created (an existing file with the same
         name would be deleted).
        *'a': Append; an existing file is opened for reading and writing,
         and if the file does not exist it is created.
        The default is 'w'.
    compression : string or tuple
        Set compression method, choosing from `blosc`, `zlib`, `lzo`, `bzip2`
        and more (see PyTables documentation). It can also be specified as a
        tuple (e.g. ``('blosc', 5)``), with the latter value specifying the
        level of compression, choosing from 0 (no compression) to 9 (maximum
        compression).  Set to `None` to turn off compression. The default is
        `blosc` since it is fast; for greater compression rate, try for
        instance `zlib`.

    See also
    --------
    load
    """
    filters = _get_compression_filters(compression)

    with tables.open_file(path, mode=mode) as h5file:
        # If the data is a dictionary, put it flatly in the root
        group = h5file.root
        group._v_attrs[DEEPDISH_IO_VERSION_STR] = IO_VERSION
        # Sparse matrices match isinstance(data, dict), so we'll have to be
        # more strict with the type checking
        if type(data) == type({}) and _dict_native_ok(data):
            for key, value in data.items():
                _save_level(h5file, group, value, name=key, filters=filters)

        elif (_sns and isinstance(data, SimpleNamespace) and
              _dict_native_ok(data.__dict__)):
            group._v_attrs[DEEPDISH_IO_ROOT_IS_SNS] = True
            for key, value in data.__dict__.items():
                _save_level(h5file, group, value, name=key, filters=filters)

        else:
            _save_level(h5file, group, data, name='data', filters=filters)
            # Mark this to automatically unpack when loaded
            group._v_attrs[DEEPDISH_IO_UNPACK] = True


def load(path, group=None, sel=None, unpack=False):
    """
    Loads an HDF5 saved with `save`.

    This function requires the `PyTables <http://www.pytables.org/>`_ module to
    be installed.

    Parameters
    ----------
    path : string
        Filename from which to load the data.
    group : string
        Load a specific group in the HDF5 hierarchy.
    sel : slice or tuple of slices
        If you specify `group` and the target is a numpy array, then you can
        use this to slice it. This is useful for opening subsets of large HDF5
        files. To compose the selection, you can use `deepdish.aslice`.
    unpack : bool
        If True, a single-entry dictionaries will be unpacked and the value
        will be returned directly. That is, if you save ``dict(a=100)``, only
        ``100`` will be loaded.

    Returns
    -------
    data : anything
        Hopefully an identical reconstruction of the data that was saved.

    See also
    --------
    save

    """
    with tables.open_file(path, mode='r') as h5file:
        if group is not None:
            data = _load_specific_level(h5file, h5file, group, sel=sel)
        else:
            grp = h5file.root
            auto_unpack = (DEEPDISH_IO_UNPACK in grp._v_attrs and
                           grp._v_attrs[DEEPDISH_IO_UNPACK])
            do_unpack = unpack or auto_unpack
            if do_unpack and len(grp._v_children) == 1:
                name = next(iter(grp._v_children))
                data = _load_specific_level(h5file, grp, name, sel=sel)
                do_unpack = False
            elif sel is not None:
                raise ValueError("Must specify group with `sel` unless it "
                                 "automatically unpacks")
            else:
                data = _load_level(h5file, grp)

            if DEEPDISH_IO_VERSION_STR in grp._v_attrs:
                v = grp._v_attrs[DEEPDISH_IO_VERSION_STR]
            else:
                v = 0

            if v > IO_VERSION:
                warnings.warn('This file was saved with a newer version of '
                              'deepdish. Please upgrade to make sure it loads '
                              'correctly.')

            # Attributes can't be unpacked with the method above, so fall back
            # to this
            if do_unpack and isinstance(data, dict) and len(data) == 1:
                data = next(iter(data.values()))

    return data
