deepdish changelog
==================

0.3.4
-----
Released: TBD

* Support for Python big integers (>64 bits)
* Support for zero-length arrays

0.3.3
-----
Released: 2016-04-10

* Default compression changed from blosc to zlib, to promote interoperability
* Support for changing the default compression using a ``~/.deepdish.conf`` file
* Specify compression method for individual items by wrapping them in ``dd.io.Compression``
* Added ``ddls --compression``
* Load multiple groups using ``foo, bar = dd.io.load('test.h5', ['/foo', '/bar'])``

0.3.2
-----
Released: 2016-03-18

* Support for numpy scalar arrays (saved as numpy scalars)
* Added ``dd.image.crop_or_pad``
* Added ``ddls --summarize``
* Added ``ddls --filter``
* Added ``ddls --leaves-only``
* Added support for softlinks in ``dd.io.save``
* Added support for ``SimpleNamespace`` objects in ``dd.io.save``
