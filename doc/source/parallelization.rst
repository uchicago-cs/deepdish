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

The way it works is that :meth:`deepdish.parallel.main` will be a gate-keeper that
only lets through rank 0. The rest of the nodes will stand idly by. Then, when
:meth:`deepdish.parallel.imap` is called, the computation is sent to the worker
nodes, computed, and finally sent back to rank 0. Note that the rank 0 node is
not given a task since it needs to be ready to execute the body of the for loop
(the `imap` function will yield values as soon as ready). If the body of the
for loop is light on computation, you might want to tell `mpirun` that you want
one more job than your cores.

The file `double_mpi.py` can also be run without `mpirun` as well.

Note that if your high performance cluster has good MPI support, this will
allow you to parallelize not only across cores, but across machines.

Multiple arguments
------------------
If you have multiple arguments, you can use :meth:`deepdish.parallel.starmap`
to automatically unpack them. Note that `starmap` also returns a generator that
will yield results as soon as they are done, so to gather all into a list we
have to run it through `list` before giving it to `concatenate`:

.. literalinclude:: codefiles/double_starmap.py

We are also showing that you can print the rank using :meth:`deepdish.parallel.rank`.

Unordered
---------
Sometimes we don't care what order the jobs get processed in, in which case we
can use :meth:`deepdish.parallel.imap_unordered` or
:meth:`deepdish.parallel.starmap_unordered`. This is great if we are doing a
commutative reduction on the results or if we are processing or testing
independent samples. The results will be yielded as soon as any batch is done,
which means more responsive output and a smaller memory footprint than its
ordered counterpart.  However, this means that you won't necessarily know which
batch completed, so you might have to put the batch number in as one of the
arguments and return it. In this example, we do something similar which is to
run the indices through the `compute` function:

.. literalinclude:: codefiles/double_unordered.py

For more information, see the :mod:`deepdish.parallel` API documentation.
