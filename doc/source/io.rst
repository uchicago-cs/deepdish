.. _io:

Saving and loading data
=======================

Saving and loading Python data types is one of the most common operations when
putting together experiments. One method is to use pickling, but this is not
compatible between Python 2 and 3, and the files cannot be easily inspected or
shared with other programming languages.

Deepdish has a function that converts your Python data type into a native HDF5
hierarchy. It stores dictionaries, values, strings and numpy arrays very
naturally. It can also store lists and tuples, but it's not as natural, so
prefer numpy arrays whenever possible. Here's an example:

>>> import deepdish as dd
>>> d = {'foo': np.arange(10), 'bar': np.ones((5, 4, 3))}
>>> dd.io.save('test.h5', d)

It will try its best to save it in a way native to HDF5::

    $ h5ls test.h5
    bar                      Dataset {5, 4, 3}
    foo                      Dataset {10}

We can also reconstruct the dictionary from file:

>>> d = dd.io.load('test.h5')

See the `deepdish.io.save` for what data structures it supports.

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

We might not seem them through `h5ls`::

    $ h5ls test.h5
    c                        Group

We see `c`, since it is stored as an empty group with meta information. The
values `a` and `b` are stored as natively as HDF5 attributes::

    $ ptdump -a test.h5
    / (RootGroup) ''
      /._v_attrs (AttributeSet), 6 attributes:
       [CLASS := 'GROUP',
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

Note that these are still somewhat awkwardly stored, so prefer using numpy
arrays to store numeric values.

Lists and tuples
----------------
Lists and tuples are shoehorned into the HDF5 key-value structure by letting
each element be its own group (or attribute, depending on the type of the
element):

>>> x = [{'foo': 10}, {'bar': 20}, 30]
>>> dd.io.save('test.h5', x)

The first two elements are stored as ``'i0'`` and ``'i1'``. The third is stored
as an attribute and thus not visible here::

    $ h5ls -r test.h5
    /                        Group
    /_top                    Group
    /_top/i0                 Group
    /_top/i1                 Group

Note that this is awkward and if the list is long you easily hit HDF5's
limitation on the number of groups. Therefore, if your list is numeric, always
make it a numpy array first!

Sparse matrices
---------------
Scipy offers many sparse matrix structures. Currently only CSR and CSC are
supported. This can offer a dramatic space and speed improvement over `mmwrite`
and `mmread` in `scipy.io`, which saves the file in an ASCII format. I tested
this on a very large and sparse CSR matrix:

============================  ======  ==============  =========
File type                     Space   Write time (s)  Read time
============================  ======  ==============  =========
`scipy.io.mmwrite` (`.mtx`)   512 MB           127.0      116.0
`deepdish.io.save` (`.h5`)     69 MB             0.6        0.8
============================  ======  ==============  =========

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

