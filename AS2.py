#!/bin/python3
import random
import numpy as np
from mpi4py import MPI

# MPI.Init()

class MonteCarloIntegrationParallel:

    def __init__(self, N, lim_l, lim_u, function, seed=10000):
        self.N = N
        self.lim_l = lim_l
        self.lim_u = lim_u
        self.function = function
        self.seed = seed

        self.comm = MPI.COMM_WORLD
        self.nproc = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.ss = np.random.SeedSequence(seed)
        self.child_ss = self.ss.spawn(self.nproc)

    def monte_carlo(self):

        if self.rank == 0:
            data_points = int(self.N/(self.nproc)) + (self.N % self.nproc)
        else:
            data_points = int(self.N/(self.nproc))

        integral = 0
        integral_squared = 0
        grand_child_ss = self.child_ss[self.rank].spawn(data_points)

        for i in range(data_points):
            rng2 = np.random.default_rng(grand_child_ss[i])
            xrand = rng2.uniform(self.lim_l, self.lim_u)
            point = self.function(xrand)
            integral += point
            integral_squared += point**2

        integral_estim = (self.lim_u-self.lim_l)/self.N * integral
        variance = 1/self.N * (integral_squared/self.N - (integral/self.N))**2
        mean = integral/self.N

        return integral_estim, variance, mean

    def running_in_parallel(self):
        I, var, mean = self.monte_carlo()
        integral = self.comm.reduce(I, MPI.SUM, 0)
        variance = self.comm.reduce(var, MPI.SUM, 0)
        mean = self.comm.reduce(mean, MPI.SUM, 0)

        return integral, variance, mean, self.rank


def func(x):
    return np.sin(x)


sin_monte = MonteCarloIntegrationParallel(100, 0, np.pi, func, seed = 1500)
I, var, mean, rank = sin_monte.running_in_parallel()

if rank == 0:
    print('Integral: %.5f' % I, '\nvariance: : %.5f' % var, '\nmean: %.5f: ' % mean)

# MPI.Finalize()