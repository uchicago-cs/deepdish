"""
Look inside HDF5 files from the terminal, especially those created by deepdish.
"""
from __future__ import division, print_function, absolute_import
from .hdf5io import (DEEPDISH_IO_VERSION_STR, DEEPDISH_IO_PREFIX,
                     DEEPDISH_IO_UNPACK, DEEPDISH_IO_ROOT_IS_SNS,
                     IO_VERSION, _sns, is_pandas_dataframe)
import tables
import numpy as np
import sys
import os
import re
from deepdish import io, six, __version__

COLORS = dict(
    black='30',
    darkgray='2;39',
    red='0;31',
    green='0;32',
    brown='0;33',
    yellow='0;33',
    blue='0;34',
    purple='0;35',
    cyan='0;36',
    white='0;39',
    reset='0'
)

MIN_COLUMN_WIDTH = 5
MIN_AUTOMATIC_COLUMN_WIDTH = 20
MAX_AUTOMATIC_COLUMN_WIDTH = 80

ABRIDGE_OVER_N_CHILDREN = 50
ABRIDGE_SHOW_EACH_SIDE = 5


def _format_dtype(dtype):
    dtype = np.dtype(dtype)
    dtype_str = dtype.name
    if dtype.byteorder == '<':
        dtype_str += ' little-endian'
    elif dtype.byteorder == '>':
        dtype_str += ' big-endian'
    return dtype_str


def _pandas_shape(level):
    if 'ndim' in level._v_attrs:
        ndim = level._v_attrs['ndim']
        shape = []
        for i in range(ndim):
            axis_name = 'axis{}'.format(i)
            if axis_name in level._v_children:
                axis = len(level._v_children[axis_name])
                shape.append(axis)
            elif axis_name + '_label0' in level._v_children:
                axis = len(level._v_children[axis_name + '_label0'])
                shape.append(axis)
            else:
                return None
        return tuple(shape)


def sorted_maybe_numeric(x):
    """
    Sorts x with numeric semantics if all keys are nonnegative integers.
    Otherwise uses standard string sorting.
    """
    all_numeric = all(map(str.isdigit, x))
    if all_numeric:
        return sorted(x, key=int)
    else:
        return sorted(x)


def paint(s, color, colorize=True):
    if colorize:
        if color in COLORS:
            return '\033[{}m{}\033[0m'.format(COLORS[color], s)
        else:
            raise ValueError('Invalid color')
    else:
        return s


def type_string(typename, dtype=None, extra=None,
                type_color='red', colorize=True):
    ll = [paint(typename, type_color, colorize=colorize)]
    if extra:
        ll += [extra]
    if dtype:
        ll += [paint('[' + dtype + ']', 'darkgray', colorize=colorize)]

    return ' '.join(ll)


def container_info(name, size=None, colorize=True, type_color=None,
                   final_level=False):
    if final_level:
        d = {}
        if size is not None:
            d['extra'] = '(' + str(size) + ')'
        if type_color is not None:
            d['type_color'] = type_color

        s = type_string(name, colorize=colorize, **d)
        # Mark that it's abbreviated
        s += ' ' + paint('[...]', 'darkgray', colorize=colorize)
        return s

    else:
        # If not abbreviated, then display the type in dark gray, since
        # the information is already conveyed through the children
        return type_string(name, colorize=colorize, type_color='darkgray')


def abbreviate(s, maxlength=25):
    """Color-aware abbreviator"""
    assert maxlength >= 4
    skip = False
    abbrv = None
    i = 0
    for j, c in enumerate(s):
        if c == '\033':
            skip = True
        elif skip:
            if c == 'm':
                skip = False
        else:
            i += 1

        if i == maxlength - 1:
            abbrv = s[:j] + '\033[0m...'
        elif i > maxlength:
            break

    if i <= maxlength:
        return s
    else:
        return abbrv


def print_row(key, value, level=0, parent='/', colorize=True,
              file=sys.stdout, unpack=False, settings={},
              parent_color='darkgray',
              key_color='white'):

    s = '{}{}'.format(paint(parent, parent_color, colorize=colorize),
                      paint(key, key_color, colorize=colorize))
    s_raw = '{}{}'.format(parent, key)
    if 'filter' in settings:
        if not re.search(settings['filter'], s_raw):
            settings['filtered_count'] += 1
            return

    if unpack:
        extra_str = '*'
        s_raw += extra_str
        s += paint(extra_str, 'purple', colorize=colorize)
    print('{}{} {}'.format(abbreviate(s, settings['left-column-width']),
                           ' '*max(0, (settings['left-column-width'] + 1 - len(s_raw))),
                           value))


class Node(object):
    def __repr__(self):
        return 'Node'

    def print(self, level=0, parent='/', colorize=True, max_level=None,
              file=sys.stdout, settings={}):
        pass

    def info(self, colorize=True, final_level=False):
        return paint('Node', 'red', colorize=colorize)


class FileNotFoundNode(Node):
    def __init__(self, filename):
        self.filename = filename

    def __repr__(self):
        return 'FileNotFoundNode'

    def print(self, level=0, parent='/', colorize=True, max_level=None,
              file=sys.stdout, settings={}):
        print(paint('File not found', 'red', colorize=colorize),
              file=file)

    def info(self, colorize=True, final_level=False):
        return paint('FileNotFoundNode', 'red', colorize=colorize)


class InvalidFileNode(Node):
    def __init__(self, filename):
        self.filename = filename

    def __repr__(self):
        return 'InvalidFileNode'

    def print(self, level=0, parent='/', colorize=True, max_level=None,
              file=sys.stdout, settings={}):
        print(paint('Invalid HDF5 file', 'red', colorize=colorize),
              file=file)

    def info(self, colorize=True, final_level=False):
        return paint('InvalidFileNode', 'red', colorize=colorize)


class DictNode(Node):
    def __init__(self):
        self.children = {}
        self.header = {}

    def add(self, k, v):
        self.children[k] = v

    def print(self, level=0, parent='/', colorize=True, max_level=None,
              file=sys.stdout, settings={}):
        if level < max_level:
            ch = sorted_maybe_numeric(self.children)
            N = len(ch)
            if N > ABRIDGE_OVER_N_CHILDREN and not settings.get('all'):
                ch = ch[:ABRIDGE_SHOW_EACH_SIDE] + [None] + ch[-ABRIDGE_SHOW_EACH_SIDE:]

            for k in ch:#sorted(self.children):
                if k is None:
                    #print(paint('... ({} omitted)'.format(N-20), 'darkgray', colorize=colorize))
                    omitted = N-2 * ABRIDGE_SHOW_EACH_SIDE
                    info = paint('{} omitted ({} in total)'.format(omitted, N),
                                 'darkgray', colorize=colorize)
                    print_row('...', info,
                              level=level,
                              parent=parent,
                              unpack=self.header.get('dd_io_unpack'),
                              colorize=colorize, file=file,
                              key_color='darkgray',
                              settings=settings)
                    continue
                v = self.children[k]
                final = level+1 == max_level

                if (not settings.get('leaves-only') or
                        not isinstance(v, DictNode)):
                    print_row(k,
                              v.info(colorize=colorize, final_level=final),
                              level=level,
                              parent=parent,
                              unpack=self.header.get('dd_io_unpack'),
                              colorize=colorize, file=file,
                              settings=settings)
                v.print(level=level+1, parent='{}{}/'.format(parent, k),
                        colorize=colorize, max_level=max_level, file=file,
                        settings=settings)

    def info(self, colorize=True, final_level=False):
        return container_info('dict', size=len(self.children),
                              colorize=colorize,
                              type_color='purple',
                              final_level=final_level)

    def __repr__(self):
        s = ['{}={}'.format(k, repr(v)) for k, v in self.children.items()]
        return 'DictNode({})'.format(', '.join(s))


class SimpleNamespaceNode(DictNode):
    def info(self, colorize=True, final_level=False):
        return container_info('SimpleNamespace', size=len(self.children),
                              colorize=colorize,
                              type_color='purple',
                              final_level=final_level)

    def print(self, level=0, parent='/', colorize=True, max_level=None,
              file=sys.stdout):
        if level == 0 and not self.header.get('dd_io_unpack'):
            print_row('', self.info(colorize=colorize,
                                    final_level=(0 == max_level)),
                      level=level, parent=parent, unpack=False,
                      colorize=colorize, file=file)
        DictNode.print(self, level, parent, colorize, max_level, file)

    def __repr__(self):
        s = ['{}={}'.format(k, repr(v)) for k, v in self.children.items()]
        return 'SimpleNamespaceNode({})'.format(', '.join(s))


class PandasDataFrameNode(Node):
    def __init__(self, shape):
        self.shape = shape

    def info(self, colorize=True, final_level=False):
        d = {}
        if self.shape is not None:
            d['extra'] = repr(self.shape)

        return type_string('DataFrame',
                           type_color='red',
                           colorize=colorize, **d)

    def __repr__(self):
        return 'PandasDataFrameNode({})'.format(self.shape)


class PandasPanelNode(Node):
    def __init__(self, shape):
        self.shape = shape

    def info(self, colorize=True, final_level=False):
        d = {}
        if self.shape is not None:
            d['extra'] = repr(self.shape)

        return type_string('Panel',
                           type_color='red',
                           colorize=colorize, **d)

    def __repr__(self):
        return 'PandasPanelNode({})'.format(self.shape)


class PandasSeriesNode(Node):
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype

    def info(self, colorize=True, final_level=False):
        d = {}
        if self.size is not None:
            d['extra'] = repr((self.size,))
        if self.dtype is not None:
            d['dtype'] = str(self.dtype)

        return type_string('Series',
                           type_color='red',
                           colorize=colorize, **d)

    def __repr__(self):
        return 'SeriesNode()'


class ListNode(Node):
    def __init__(self, typename='list'):
        self.children = []
        self.typename = typename

    def append(self, v):
        self.children.append(v)

    def __repr__(self):
        s = [repr(v) for v in self.children]
        return 'ListNode({})'.format(', '.join(s))

    def print(self, level=0, parent='/', colorize=True,
              max_level=None, file=sys.stdout, settings={}):

        if level < max_level:
            for i, v in enumerate(self.children):
                k = str(i)
                final = level + 1 == max_level
                print_row(k, v.info(colorize=colorize,
                                    final_level=final),
                          level=level, parent=parent + 'i',
                          colorize=colorize, file=file,
                          settings=settings)
                v.print(level=level+1, parent='{}{}/'.format(parent + 'i', k),
                        colorize=colorize, max_level=max_level, file=file,
                        settings=settings)

    def info(self, colorize=True, final_level=False):
        return container_info(self.typename, size=len(self.children),
                              colorize=colorize,
                              type_color='purple',
                              final_level=final_level)


class NumpyArrayNode(Node):
    def __init__(self, shape, dtype, statistics=None, compression=None):
        self.shape = shape
        self.dtype = dtype
        self.statistics = statistics
        self.compression = compression

    def info(self, colorize=True, final_level=False):
        if not self.statistics:
            s = type_string('array', extra=repr(self.shape),
                            dtype=str(self.dtype),
                            type_color='red',
                            colorize=colorize)

            if self.compression:
                if self.compression['complib'] is not None:
                    compstr = '{} lvl{}'.format(self.compression['complib'],
                                                self.compression['complevel'])
                else:
                    compstr = 'none'
                s += ' ' + paint(compstr, 'yellow', colorize=colorize)

        else:
            s = type_string('array', extra=repr(self.shape),
                            type_color='red',
                            colorize=colorize)
            raw_s = type_string('array', extra=repr(self.shape),
                                type_color='red',
                                colorize=False)

            if len(raw_s) < 25:
                s += ' ' * (25 - len(raw_s))
            s += paint(' {:14.2g}'.format(self.statistics.get('mean')),
                       'white', colorize=colorize)
            s += paint(u' \u00b1 ', 'darkgray', colorize=colorize)
            s += paint('{:.2g}'.format(self.statistics.get('std')),
                       'reset', colorize=colorize)

        return s

    def __repr__(self):
        return ('NumpyArrayNode(shape={}, dtype={})'
                .format(self.shape, self.dtype))


class SparseMatrixNode(Node):
    def __init__(self, fmt, shape, dtype):
        self.sparse_format = fmt
        self.shape = shape
        self.dtype = dtype

    def info(self, colorize=True, final_level=False):
        return type_string('sparse {}'.format(self.sparse_format),
                           extra=repr(self.shape),
                           dtype=str(self.dtype),
                           type_color='red',
                           colorize=colorize)

    def __repr__(self):
        return ('NumpyArrayNode(shape={}, dtype={})'
                .format(self.shape, self.dtype))


class ValueNode(Node):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'ValueNode(type={})'.format(type(self.value))

    def info(self, colorize=True, final_level=False):
        if isinstance(self.value, six.text_type):
            if len(self.value) > 25:
                s = repr(self.value[:22] + '...')
            else:
                s = repr(self.value)

            return type_string(s, dtype='unicode',
                               type_color='green',
                               extra='({})'.format(len(self.value)),
                               colorize=colorize)
        elif isinstance(self.value, six.binary_type):
            if len(self.value) > 25:
                s = repr(self.value[:22] + b'...')
            else:
                s = repr(self.value)

            return type_string(s, dtype='ascii',
                               type_color='green',
                               extra='({})'.format(len(self.value)),
                               colorize=colorize)
        elif self.value is None:
            return type_string('None', dtype='python',
                               type_color='blue',
                               colorize=colorize)
        else:
            return type_string(repr(self.value)[:20],
                               dtype=str(np.dtype(type(self.value))),
                               type_color='blue',
                               colorize=colorize)


class ObjectNode(Node):
    def __init__(self):
        pass

    def __repr__(self):
        return 'ObjectNode'

    def info(self, colorize=True, final_level=False):
        return type_string('pickled', dtype='object', type_color='yellow',
                           colorize=colorize)


class SoftLinkNode(Node):
    def __init__(self, target):
        self.target = target

    def info(self, colorize=True, final_level=False):
        return type_string('link -> {}'.format(self.target),
                           dtype='SoftLink',
                           type_color='cyan',
                           colorize=colorize)

    def __repr__(self):
        return ('SoftLinkNode(target={})'
                .format(self.target))


def _tree_level(level, raw=False, settings={}):
    if isinstance(level, tables.Group):
        if _sns and (level._v_title.startswith('SimpleNamespace:') or
                     DEEPDISH_IO_ROOT_IS_SNS in level._v_attrs):
            node = SimpleNamespaceNode()
        else:
            node = DictNode()

        for grp in level:
            node.add(grp._v_name, _tree_level(grp, raw=raw, settings=settings))

        for name in level._v_attrs._f_list():
            v = level._v_attrs[name]
            if name == DEEPDISH_IO_VERSION_STR:
                node.header['dd_io_version'] = v

            if name == DEEPDISH_IO_UNPACK:
                node.header['dd_io_unpack'] = v

            if name.startswith(DEEPDISH_IO_PREFIX):
                continue

            if isinstance(v, np.ndarray):
                node.add(name, NumpyArrayNode(v.shape, _format_dtype(v.dtype)))
            else:
                node.add(name, ValueNode(v))

        if (level._v_title.startswith('list:') or
                level._v_title.startswith('tuple:')):
            s = level._v_title.split(':', 1)[1]
            N = int(s)
            lst = ListNode(typename=level._v_title.split(':')[0])
            for i in range(N):
                t = node.children['i{}'.format(i)]
                lst.append(t)
            return lst
        elif level._v_title.startswith('nonetype:'):
            return ValueNode(None)
        elif is_pandas_dataframe(level):
            pandas_type = level._v_attrs['pandas_type']
            if raw:
                # Treat as regular dictionary
                pass
            elif pandas_type == 'frame':
                shape = _pandas_shape(level)
                new_node = PandasDataFrameNode(shape)
                return new_node
            elif pandas_type == 'series':
                try:
                    values = level._v_children['values']
                    size = len(values)
                    dtype = values.dtype
                except:
                    size = None
                    dtype = None
                new_node = PandasSeriesNode(size, dtype)
                return new_node
            elif pandas_type == 'wide':
                shape = _pandas_shape(level)
                new_node = PandasPanelNode(shape)
                return new_node
            # else: it will simply be treated as a dict

        elif level._v_title.startswith('sparse:') and not raw:
            frm = level._v_attrs.format
            dtype = level.data.dtype
            shape = tuple(level.shape[:])
            node = SparseMatrixNode(frm, shape, dtype)
            return node

        return node
    elif isinstance(level, tables.VLArray):
        if level.shape == (1,):
            return ObjectNode()
        node = NumpyArrayNode(level.shape, 'unknown')
        return node
    elif isinstance(level, tables.Array):
        stats = {}
        if settings.get('summarize'):
            stats['mean'] = level[:].mean()
            stats['std'] = level[:].std()

        compression = {}
        if settings.get('compression'):
            compression['complib'] = level.filters.complib
            compression['shuffle'] = level.filters.shuffle
            compression['complevel'] = level.filters.complevel

        node = NumpyArrayNode(level.shape, _format_dtype(level.dtype),
                              statistics=stats, compression=compression)

        if hasattr(level._v_attrs, 'zeroarray_dtype'):
            dtype = level._v_attrs.zeroarray_dtype
            node = NumpyArrayNode(tuple(level), _format_dtype(dtype))

        elif hasattr(level._v_attrs, 'strtype'):
            strtype = level._v_attrs.strtype
            itemsize = level._v_attrs.itemsize
            if strtype == b'unicode':
                shape = level.shape[:-1] + (level.shape[-1] // itemsize // 4,)
            elif strtype == b'ascii':
                shape = level.shape

            node = NumpyArrayNode(shape, strtype.decode('ascii'))

        return node
    elif isinstance(level, tables.link.SoftLink):
        node = SoftLinkNode(level.target)
        return node
    else:
        return Node()


def get_tree(path, raw=False, settings={}):
    fn = os.path.basename(path)
    try:
        with tables.open_file(path, mode='r') as h5file:
            grp = h5file.root
            s = _tree_level(grp, raw=raw, settings=settings)
            s.header['filename'] = fn
            return s
    except OSError:
        return FileNotFoundNode(fn)
    except IOError:
        return FileNotFoundNode(fn)
    except tables.exceptions.HDF5ExtError:
        return InvalidFileNode(fn)


def _column_width(level):
    if isinstance(level, tables.Group):
        max_w = 0
        for grp in level:
            max_w = max(max_w, _column_width(grp))
        for name in level._v_attrs._f_list():
            if name.startswith(DEEPDISH_IO_PREFIX):
                continue
            max_w = max(max_w, len(level._v_pathname) + 1 + len(name))
        return max_w
    else:
        return len(level._v_pathname)


def _discover_column_width(path):
    if not os.path.isfile(path):
        return MIN_AUTOMATIC_COLUMN_WIDTH

    with tables.open_file(path, mode='r') as h5file:
        return _column_width(h5file.root)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description=("Look inside HDF5 files. Works particularly well "
                     "for HDF5 files saved with deepdish.io.save()."),
        prog='ddls',
        epilog='example: ddls test.h5 -i /foo/bar --ipython')
    parser.add_argument('file', nargs='+',
                        help='filename of HDF5 file')
    parser.add_argument('-d', '--depth', type=int, default=4,
                        help='max depth, defaults to 4')
    parser.add_argument('-nc', '--no-color', action='store_true',
                        help='turn off bash colors')
    parser.add_argument('-i', '--inspect', metavar='GRP',
                        help='print a specific variable (e.g. /data)')
    parser.add_argument('--ipython', action='store_true',
                        help=('load file into an IPython session. '
                              'Works with -i'))
    parser.add_argument('--raw', action='store_true',
                        help=('print the raw HDF5 structure for complex '
                              'data types, such as sparse matrices and pandas '
                              'data frames'))
    parser.add_argument('-f', '--filter', type=str,
                        help=('print only entries that match this regular '
                              'expression'))
    parser.add_argument('-l', '--leaves-only', action='store_true',
                        help=('print only leaves'))
    parser.add_argument('-a', '--all', action='store_true',
                        help=('do not abridge'))
    parser.add_argument('-s', '--summarize', action='store_true',
                        help=('print summary statistics of numpy arrays'))
    parser.add_argument('-c', '--compression', action='store_true',
                        help=('print compression method for each array'))
    parser.add_argument('-v', '--version', action='version',
                        version='deepdish {} (io protocol {})'.format(
                            __version__, IO_VERSION))
    parser.add_argument('--column-width', type=int, default=None)

    args = parser.parse_args()

    colorize = sys.stdout.isatty() and not args.no_color

    settings = {}
    if args.filter:
        settings['filter'] = args.filter

    if args.leaves_only:
        settings['leaves-only'] = True

    if args.summarize:
        settings['summarize'] = True

    if args.compression:
        settings['compression'] = True

    if args.all:
        settings['all'] = True

    def single_file(files):
        if len(files) >= 2:
            s = 'Error: Select a single file when using --inspect'
            print(paint(s, 'red', colorize=colorize))
            sys.exit(1)
        return files[0]

    def run_ipython(fn, group=None, data=None):
        file_desc = paint(fn, 'yellow', colorize=colorize)
        if group is None:
            path_desc = file_desc
        else:
            path_desc = '{}:{}'.format(
                file_desc,
                paint(group, 'white', colorize=colorize))

        welcome = "Loaded {} into '{}':".format(
            path_desc,
            paint('data', 'blue', colorize=colorize))

        # Import deepdish for the session
        import deepdish as dd
        import IPython
        IPython.embed(header=welcome)

    i = 0
    if args.inspect is not None:
        fn = single_file(args.file)

        try:
            data = io.load(fn, args.inspect)
        except ValueError:
            s = 'Error: Could not find group: {}'.format(args.inspect)
            print(paint(s, 'red', colorize=colorize))
            sys.exit(1)
        if args.ipython:
            run_ipython(fn, group=args.inspect, data=data)
        else:
            print(data)
    elif args.ipython:
        fn = single_file(args.file)
        data = io.load(fn)
        run_ipython(fn, data=data)
    else:
        for f in args.file:
            # State that will be incremented
            settings['filtered_count'] = 0

            if args.column_width is None:
                settings['left-column-width'] = max(MIN_AUTOMATIC_COLUMN_WIDTH, min(MAX_AUTOMATIC_COLUMN_WIDTH, _discover_column_width(f)))
            else:
                settings['left-column-width'] = args.column_width

            s = get_tree(f, raw=args.raw, settings=settings)
            if s is not None:
                if i > 0:
                    print()

                if len(args.file) >= 2:
                    print(paint(f, 'yellow', colorize=colorize))
                s.print(colorize=colorize, max_level=args.depth,
                        settings=settings)
                i += 1

            if settings.get('filter'):
                print('Filtered on: {} ({} rows omitted)'.format(
                    paint(args.filter, 'purple', colorize=colorize),
                    paint(str(settings['filtered_count']), 'white',
                          colorize=colorize)))


if __name__ == '__main__':
    main()
