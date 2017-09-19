"""Multiprocessing customized supporting module"""

import multiprocessing
from time import sleep
import numpy as np
from .core import ulen


def multiproc(func, *par, processes=6):
    """Initiate multiple processes in async mode, return the list of
    workers"""
    pool = multiprocessing.Pool(processes=processes)
    npar = len(par)   # number of parameters
    if npar==1:
        return [pool.apply_async(func,[x]) for x in par[0]]
    # the case of more than one parameter
    plen = [ulen(x) for x in par]   # length of each parameter
    plen0 = np.max(plen)
    pars = []
    for p,n in zip(par,plen):
        if n==1:
            pars.append([p]*plen0)
        elif n==plen0:
            pars.append(p)
        else:
            raise ValueError('lengths of parameters not consistent')
    return [pool.apply_async(func,p) for p in zip(*pars)]


def results(worker, wait=30, hold=True, verbose=True):
    """Hold the terminal for the running multiprocessing pool, and return
    the results"""
    nworkers = len(worker)
    ready = [w.ready() for w in worker]
    ndone = len(np.where(ready)[0])
    if verbose:
        print('{0} workers total, {1} done, {2} running'.format(nworkers,ndone,nworkers-ndone))
    if not hold:
        return [w.get() for w in worker]
    else:
        while not np.all(ready):
            ready = [w.ready() for w in worker]
            ndone = len(np.where(ready)[0])
            if verbose:
                print('{0} workers total, {1} done, {2} running'.format(nworkers,ndone,nworkers-ndone))
            sleep(wait)
        return [w.get() for w in worker]
