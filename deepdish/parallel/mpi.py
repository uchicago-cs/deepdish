from __future__ import division, print_function, absolute_import

import sys
import itertools as itr
import numpy as np

__all__ = ['rank', 'imap_unordered', 'imap',
           'starmap_unordered', 'starmap', 'main']

# Global set of workers - initialized a map function is first called
_g_available_workers = None 
_g_initialized = False

# For docstrings, see deepdish.parallel.fallback

def rank():
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    return rank


def kill_workers():
    from mpi4py import MPI
    all_workers = range(1, MPI.COMM_WORLD.Get_size())
    for worker in all_workers:
        MPI.COMM_WORLD.send(None, dest=worker, tag=666)


def _init():
    global _g_available_workers, _g_initialized
    from mpi4py import MPI
    import atexit
    _g_available_workers = set(range(1, MPI.COMM_WORLD.Get_size()))
    _g_initialized = True
    atexit.register(kill_workers)


def imap_unordered(f, workloads, star=False):
    global _g_available_workers, _g_initialized

    from mpi4py import MPI
    N = MPI.COMM_WORLD.Get_size() - 1
    if N == 0 or not _g_initialized:
        mapf = [map, itr.starmap][star]
        for res in mapf(f, workloads):
            yield res
        return

    for job_index, workload in enumerate(itr.chain(workloads, itr.repeat(None))):
        if workload is None and len(_g_available_workers) == N:
            break

        while not _g_available_workers or workload is None:
            # Wait to receive results
            status = MPI.Status()
            ret = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if status.tag == 2:
                yield ret['output_data']
                _g_available_workers.add(status.source)
                if len(_g_available_workers) == N:
                    break

        if _g_available_workers and workload is not None:
            dest_rank = _g_available_workers.pop()

            # Send off job
            task = dict(func=f, input_data=workload, job_index=job_index, unpack=star)
            MPI.COMM_WORLD.send(task, dest=dest_rank, tag=10)


def imap(f, workloads, star=False):
    global _g_available_workers, _g_initialized
    from mpi4py import MPI
    N = MPI.COMM_WORLD.Get_size() - 1
    if N == 0 or not _g_initialized:
        mapf = [map, itr.starmap][star]
        for res in mapf(f, workloads):
            yield res
        return

    results = []
    indices = []

    for job_index, workload in enumerate(itr.chain(workloads, itr.repeat(None))):
        if workload is None and len(_g_available_workers) == N:
            break

        while not _g_available_workers or workload is None:
            # Wait to receive results
            status = MPI.Status()
            ret = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if status.tag == 2:
                results.append(ret['output_data'])
                indices.append(ret['job_index'])
                _g_available_workers.add(status.source)
                if len(_g_available_workers) == N:
                    break

        if _g_available_workers and workload is not None:
            dest_rank = _g_available_workers.pop()

            # Send off job
            task = dict(func=f, input_data=workload, job_index=job_index, unpack=star)
            MPI.COMM_WORLD.send(task, dest=dest_rank, tag=10)

    II = np.argsort(indices)  
    for i in II:
        yield results[i]


def starmap(f, workloads):
    return imap(f, workloads, star=True)


def starmap_unordered(f, workloads):
    return imap_unordered(f, workloads, star=True)


def worker():
    from mpi4py import MPI
    while True:
        status = MPI.Status()
        ret = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG, status=status) 

        if status.tag == 10:
            # Workload received
            func = ret['func']
            if ret.get('unpack'):
                res = func(*ret['input_data'])
            else:
                res = func(ret['input_data'])

            # Done, let's send it back
            MPI.COMM_WORLD.send(dict(job_index=ret['job_index'], output_data=res), dest=0, tag=2)

        elif status.tag == 666:
            # Kill code
            sys.exit(0)


def main(name=None):
    if name is not None and name != '__main__':
        return False

    from mpi4py import MPI
    rank =  MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        _init()
        return True
    else:
        worker()
        sys.exit(0)
