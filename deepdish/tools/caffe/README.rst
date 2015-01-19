Caffe tools
===========
Tools for working with Caffe. You can run the tool ``maker`` by::

    python -m deepdish.tools.caffe.maker

maker
-----
The ``maker`` creates Caffe prototxt configs and accompanying

Data files
~~~~~~~~~~
Point the environment variable ``MAKER_DATA_DIR`` to your dataset folder. Open
up ``maker.py`` and add new datasets to the global ``DATASETS`` variable.

tester
------
Run a classifier on a particular caffemodel file and hdf5 dataset.
