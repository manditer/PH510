#!/bin/python3

from mpi4py import MPI
import numpy as np

# MPI.Init()

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()

N = 300000000
I = np.array(0.0, dtype = np.double)

nworkers = nproc - 1

def integrand(i):
    x = (i+0.5) / N
    I_estim = 4.0 / (1.0 + x*x)
    return(I_estim)

steps = np.linspace(0, N, nworkers+1)

if rank != 0:
    print('rank', rank, 'range', steps[rank-1], steps[rank])
    #range doesnt include last value, stop not inlcuded
    for i in range(int(steps[rank-1]), int(steps[rank])):
        I += integrand(i)


print('Rank ',rank,' computes ', I)
summa=comm.reduce(I/N, MPI.SUM, 0)

if rank == 0:
    print("Integral %.15f" % summa)

# MPI.Finalize()