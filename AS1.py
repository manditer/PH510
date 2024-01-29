#!/bin/python3

from mpi4py import MPI

# MPI.Init()

comm = MPI.COMM_WORLD

nproc = comm.Get_size()
rank = comm.Get_rank()

nworkers = nproc - 1

N = 100000000
delta = 1.0 / N


I = 0.0


def integrand(x):
    result = 4.0 / (1.0 + x*x)
    return result


for i in range(0, N):
    x = (i+0.5) * delta
    I =+ integrand(x)

print('Rank ', rank, ' computes ', I)

summa = comm.reduce(I, MPI.SUM, 0)

if rank == 0:
    print("Integral %.15f" % I)

# MPI.Finalize()