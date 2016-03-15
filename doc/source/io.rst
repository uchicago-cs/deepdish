.. _io:

Saving and loading data
=======================

Saving and loading Python data types is one of the most common operations when
putting together experiments. One method is to use pickling, but this is not
compatible between Python 2 and 3, and the files cannot be easily inspected or
shared with other programming languages.

Deepdish has a function that converts your Python data type into a native HDF5
hierarchy. It stores dictionaries, SimpleNamespaces (for versions of Python that
support them), values, strings and numpy arrays very naturally. It can also
store lists and tuples, but it's not as natural, so prefer numpy arrays whenever
possible. Here's an example saving and HDF5 using :func:`deepdish.io.save`:

>>> import deepdish as dd
>>> d = {'foo': np.arange(10), 'bar': np.ones((5, 4, 3))}
>>> dd.io.save('test.h5', d)

It will try its best to save it in a way native to HDF5::

    $ h5ls test.h5
    bar                      Dataset {5, 4, 3}
    foo                      Dataset {10}

We also offer our own version of ``h5ls`` that works really well with deepdish
saved HDF5 files::

    $ ddls test.h5
    /bar                       array (5, 4, 3) [float64]
    /foo                       array (10,) [int64]

We can now reconstruct the dictionary from the file using
:func:`deepdish.io.load`:

>>> d = dd.io.load('test.h5')

Dictionaries
------------

Dictionaries are saved as HDF5 groups:

>>> d = {'foo': {'bar': np.arange(10), 'baz': np.zeros(3)}, 'qux': np.ones(12)}
>>> dd.io.save('test.h5', d)

Resulting in::

    $ h5ls -r test.h5
    /                        Group
    /foo                     Group
    /foo/bar                 Dataset {10}
    /foo/baz                 Dataset {3}
    /qux                     Dataset {12}

Again, we can use the deepdish tool for better inspection::

    $ ddls test.h5
    /foo                       dict
    /foo/bar                   array (10,) [int64]
    /foo/baz                   array (3,) [float64]
    /qux                       array (12,) [float64]

SimpleNamespaces
----------------
SimpleNamespaces work almost identically to dictionaries and are available in
Python 3.3 and later. Note that for versions of Python that do not support
SimpleNamespaces, deepdish will load them in as dictionaries.

Like dictionaries, SimpleNamespaces are saved as HDF5 groups:

>>> from types import SimpleNamespace as NS
>>> d = NS(foo=NS(bar=np.arange(10), baz=np.zeros(3)), qux=np.ones(12))
>>> dd.io.save('test.h5', d)

For h5ls, the results are identical to the Dictionary example::

    $ h5ls -r test.h5
    /                        Group
    /foo                     Group
    /foo/bar                 Dataset {10}
    /foo/baz                 Dataset {3}
    /qux                     Dataset {12}

Again, we can use the deepdish tool for better inspection. For a version of
Python that supports SimpleNamespaces::

    $ ddls test.h5
    /                          SimpleNamespace
    /foo                       SimpleNamespace
    /foo/bar                   array (10,) [int64]
    /foo/baz                   array (3,) [float64]
    /qux                       array (12,) [float64]

For a version of Python that doesn't support SimpleNamespaces, dictionaries are
used::

    $ ddls test.h5
    /foo                       dict
    /foo/bar                   array (10,) [int64]
    /foo/baz                   array (3,) [float64]
    /qux                       array (12,) [float64]

Numpy arrays
------------
Numpy arrays of any numeric data type are natively stored in HDF5:

>>> d = {'a': np.arange(5),
...      'b': np.array([1.2, 2.3, 3.4]),
...      'c': np.ones(3, dtype=np.int8)}

We can inspect the actual values::

    $ h5ls -d test.h5
    a                        Dataset {5}
        Data:
            (0) 0, 1, 2, 3, 4
    b                        Dataset {3}
        Data:
            (0) 1.2, 2.3, 3.4
    c                        Dataset {3}
        Data:
            (0) 1, 1, 1

Basic data types
----------------
Basic Python data types are stored as attributes or empty groups:

>>> d = {'a': 10, 'b': 'test', 'c': None}
>>> dd.io.save('test.h5', d)

We might not see them through ``h5ls``::

    $ h5ls test.h5
    c                        Group

This is where ``ddls`` excels::

    $ ddls test.h5
    /a                         10 [int64]
    /b                         'test' (4) [unicode]
    /c                         None [python]

Since `c` is specific to Python, it is stored as an empty group with meta
information. The values `a` and `b` however are stored natively as HDF5
attributes::

    $ ptdump -a test.h5
    / (RootGroup) ''
      /._v_attrs (AttributeSet), 7 attributes:
       [CLASS := 'GROUP',
        DEEPDISH_IO_VERSION := 10,
        PYTABLES_FORMAT_VERSION := '2.1',
        TITLE := '',
        VERSION := '1.0',
        a := 10,
        b := 'test']
    /c (Group) 'nonetype:'
      /c._v_attrs (AttributeSet), 3 attributes:
       [CLASS := 'GROUP',
        TITLE := 'nonetype:',
        VERSION := '1.0']

Note that these are still somewhat awkwardly stored, so always prefer using
numpy arrays to store numeric values.

Lists and tuples
----------------
Lists and tuples are shoehorned into the HDF5 key-value structure by letting
each element be its own group (or attribute, depending on the type of the
element):

>>> x = [{'foo': 10}, {'bar': 20}, 30]
>>> dd.io.save('test.h5', x)

The first two elements are stored as ``'i0'`` and ``'i1'``. The third is stored
as an attribute and thus not directly visible by ``h5ls``. However, ``ddls`` will
show it::

    $ ddls test.h5
    /data*                     list
    /data/i0                   dict
    /data/i0/foo               10 [int64]
    /data/i1                   dict
    /data/i1/bar               20 [int64]
    /data/i2                   30 [int64]

Note that this is awkward and if the list is long you easily hit HDF5's
limitation on the number of groups. Therefore, if your list is numeric, always
make it a numpy array first! The asterisk on the "/data" group indicates that
the top level variable that was saved was not a dict or a SimpleNamespace;
during load, deepdish will unpack "/data" so that the saved variable is
returned. See `Fake top-level group`_.

Pandas data structures
----------------------
The pandas_ data structures ``DataFrame``, ``Series`` and ``Panel`` are
natively supported. This is thanks to pandas_ already providing support for this
with the same PyTables backend as deepdish::

    import pandas as pd
    df = pd.DataFrame({'int': np.arange(3), 'name': ['zero', 'one', 'two']})

    dd.io.save('test.h5', df)

We can inspect this as usual::

    $ ddls test.h5
    /data*                     DataFrame (2, 3)

If you are curious of how pandas stores this, we can tell ``ddls`` to forget it
knows how to read data frames by invoking the ``--raw`` command::

    $ ddls test.h5 --raw
    /data*                     dict
    /data/axis0                array (2,) [|S4]
    /data/axis0_variety        'regular' (7) [unicode]
    /data/axis1                array (3,) [int64]
    /data/axis1_variety        'regular' (7) [unicode]
    /data/block0_items         array (1,) [|S3]
    /data/block0_items_vari... 'regular' (7) [unicode]
    /data/block0_values        array (3, 1) [int64]
    /data/block1_items         array (1,) [|S4]
    /data/block1_items_vari... 'regular' (7) [unicode]
    /data/block1_values        pickled [object]
    /data/encoding             'UTF-8' (5) [unicode]
    /data/nblocks              2 [int64]
    /data/ndim                 2 [int64]
    /data/pandas_type          'frame' (5) [unicode]
    /data/pandas_version       '0.15.2' (6) [unicode]

Sparse matrices
---------------
Scipy offers several types of sparse matrices, of which deepdish can save the
types BSR, COO, CSC, CSR and DIA. The types DOK and LIL are currently not
supported (note that these two types are mainly for incrementally building
sparse matrices anyway).

Using deepdish can offer a dramatic space and speed improvement over for instance
``mmwrite`` and ``mmread`` in `scipy.io
<http://docs.scipy.org/doc/scipy/reference/io.html>`__. This is not surprising
since it saves the file in an ASCII format. Instead, deepdish saves the
internal Numpy arrays directly, which means there is no conversion overhead
whatsoever. Here is a comparison on a large (100 billion elements) and sparse
(0.01% sparsity) CSR matrix:

============================  ===========  ==========  ==============  =============
Method                        Compression  Space (MB)  Write time (s)  Read time (s)
============================  ===========  ==========  ==============  =============
scipy's mmwrite                         N         145           79             40
numpy's save                            N         134            1.36           0.75
pickle                                  N         115            0.63           0.17
deepdish (no compression)               N         115            0.52           0.17
numpy's savez_compressed                Y          32            8.88           1.33
pickle (gzip)                           Y          29            5.19           0.86
deepdish (blosc)                        Y          24            0.36           0.37
deepdish (zlib)                         Y          21            9.01           0.83
============================  ===========  ==========  ==============  =============

This particular matrix had only nonzero elements that were set to 1, which
meant even more compression could be applied. The default compression in
deepdish is blosc, since zlib takes much longer to encode and offers only a
modest space improvement.

Just like with pandas data types, you can inspect the storage format using
``ddls --raw``.

Quick inspection
----------------
Using ``ddls`` gives you a quick overview of your data. You can also use it to
print specific entries from the command line::

    $ ddls test.h5 -i /foo
    [0 1 2 3 4 5 6 7 8 9]

Adding ``--ipython`` will start an IPython session that comes pre-loaded
with the selected variable loaded into the variable ``data``. This can be used
even without ``-i``, in which case the whole file is loaded into ``data``.

Fake top-level group
--------------------
Even if the entry object is not a dictionary or SimpleNamespace, HDF5
forces us to create a top-level group to put it in. This group will be
called ``data`` and marked using hidden attributes as fake so that a
dictionary or SimpleNamespace is not added when loaded::

    dd.io.save('test.h5', [np.arange(5), 100])

Note that ``ddls`` will let you know this group is fake by adding a star after
``/data``::

    $ ddls test.h5
    /data*                     list
    /data/i0                   array (5,) [int64]
    /data/i1                   100 [int64]

Partial loading
---------------

A specific level can be loaded as follows::

    dd.io.save('test.h5', dict(foo=dict(bar=np.ones((10, 5)))))

    bar = dd.io.load('test.h5', '/foo/bar')

You can even load slices of arrays::

    bar_slice = dd.io.load('test.h5', '/foo/bar', sel=dd.aslice[:5, -2:])

The file will never be read in full, not even the array, so this technique can
be used to step through very large arrays.

Pickled objects
---------------
Some objects cannot be saved natively as HDF5, such as object classes. Our
suggestion is to convert classes to to dictionary-like structures first, but
sometimes it can be nice to be able to dump anything into a file. This is why
deepdish also offers pickling as a last resort::

    import deepdish as dd

    class Foo(object):
        pass

    foo = Foo()
    dd.io.save('test.h5', dict(foo=foo))

Inspecting this file will yield::

    $ ddls test.h5
    /foo                       pickled [object]

Note that the class `Foo` has to be defined in the file that calls
``dd.io.load``.

Avoid relying on pickling, since it hurts the interoperability provided by
deepdish's HDF5 saving. Each pickled object will raise a
``DeprecationWarning``, so call Python with ``-Wall`` to make sure you aren't
implicitly pickling something. You can of course also use ``ddls`` to inspect
the file to make sure nothing is pickled.

If deepdish fatally fails to save an object, you should first report this as an
issue on GitHub. As a quick fix, you can force pickling by wrapping the object
in :func:`deepdish.io.ForcePickle`::

    dd.io.save('test.h5', {'foo': dd.io.ForcePickle('pickled string')})

Class instances
---------------
Storing classes can be done by converting them to and from dictionary
structures. This is a bit more work than straight up pickling, but the benefit
is that they are inspectable from outside Python, and compatible between Python
2 and 3 (which pickled classes are not!). This process can be facilitated by
subclassing :class:`deepdish.util.SaveableRegistry`:

.. literalinclude:: codefiles/saveable_example.py

Now, we can save an instane of `Foo` directly to an HDF5 file by:

>>> f = Foo(10)
>>> f.save('foo.h5')

And restore it by:

>>> f = Foo.load('foo.h5')

In the example, we also showed how we can subclass `Foo` as `Bar`. Now, we can
do:

>>> b = Bar(10, 20)
>>> b.save('bar.h5')

Along with `x` and `y`, it will save ``'bar'`` as meta-information (since we
called ``Foo.register('bar')``). What this means is that we can reconstruct the
instance from:

>>> b = Foo.load('bar.h5')

Note that we did not need to call `Bar.load` (although it would work too),
which makes it very easy to load subclassed instances of various kinds. The
registered named can be accessed through:

>>> b.name
'bar'

To give the base class a name, we can add
``dd.util.SaveableRegistry.register('foo')`` before the class definition.


Soft Links
----------
In Python, many names can be bound to the same object. Deepdish accounts for
this for many objects (dictionaries, lists, numpy arrays, pandas dataframes,
SimpleNamespaces, etc) by using HDF5 soft links. This means the object itself
is only written once and that these relationships are preserved upon loading.
Recursion inside objects is also handled via soft links.

Here is an example where soft links are used for both purposes:

>>> import deepdish as dd
>>> ones = np.ones((5, 4, 3))
>>> d = {'foo': np.arange(10), 'bar': ones, 'baz': ones}
>>> d['self'] = d   # to demonstrate recursion
>>> dd.io.save('test.h5', d)

Soft links are native to HDF5::

    $ h5ls test.h5
    bar                      Dataset {5, 4, 3}
    baz                      Soft Link {/bar}
    foo                      Dataset {10}
    self                     Soft Link {/}

Notice that the `ones` 3D array was only written once and that ``self`` is a
link to the top level group. With ``ddls``::

    $ ddls test.h5
    /bar                       array (5, 4, 3) [float64]
    /baz                       link -> /bar [SoftLink]
    /foo                       array (10,) [int64]
    /self                      link -> / [SoftLink]

Verify that ``d['bar']`` and ``d['baz']`` refer to the same object:

>>> d = dd.io.load('test.h5')
>>> d['bar'] is d['baz']
True

Also verify that ``d['self']`` is ``d``:

>>> d['self'] is d
True

.. _pandas: http://pandas.pydata.org/
