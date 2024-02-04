#!/bin/python3
"""Evaluating pi using the mid-point rule, with N_SAMPLES."""
import time
import numpy as np
from mpi4py import MPI

# MPI.Init()

start_time = time.time()

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
nworkers = nproc - 1
rank = comm.Get_rank()

N_SAMPLES = 200000000
integral = np.array(0.0, dtype = np.double)
steps = np.linspace(0, N_SAMPLES, nworkers+1)

def integrand(j):
    """Evaluating the mid-point rule for point j."""
    sample_point = (j+0.5) / N_SAMPLES
    integral_evaluation = 4.0 / (1.0 + sample_point**2)
    return integral_evaluation

if rank != 0:
    for i in range(int(steps[rank-1]), int(steps[rank])):
        integral += integrand(i)

summa=comm.reduce(integral/N_SAMPLES, MPI.SUM, 0)

if rank == 0:
    time_taken = time.time()-start_time
    print("Pi estimate %.15f" % summa)
    print('time taken %.2f' % time_taken)

# MPI.Finalize()