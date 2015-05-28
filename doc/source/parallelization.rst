.. _parallelization:

Parallelization
===============
This module provides a method of doing parallelization using MPI. It uses
`mpi4py`, but provides convenience functions that make parallelization much
easier.

If you have the following file (`double.py`):

.. literalinclude:: codefiles/double.py

You can parallelize the computation by replacing it with the following
(`double_mpi.py`):

.. literalinclude:: codefiles/double_mpi.py
    :emphasize-lines: 6,9

And run it with::

    $ mpirun -n 8 python double_mpi.py

The way it works is that `deepdish.parallel.main` will be a gate-keeper and
only let through rank 0. The rest of the nodes will stand idly by. Then, when
`deepdish.parallel.imap` is called, the computation is sent to the worker
nodes, computed, and finally sent back to rank 0. Note that the rank 0 node is
not given a task since it needs to be ready to execute the body of the for loop
(the `imap` function will yield values as soon as ready). If the body of the
for loop is light on computation, you might want to tell `mpirun` that you want
one more job than your cores.

The file `double_mpi.py` can also be run without `mpirun` as well.

Note that if your high performance cluster has good MPI support, this will
allow you to parallelize not only across cores, but across machines.

