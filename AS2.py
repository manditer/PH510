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

        self.total_integral = np.array(0.0, dtype=np.double)
        self.total_variance = np.array(0.0, dtype=np.double)
        self.total_mean = np.array(0.0, dtype=np.double)

    def monte_carlo(self):

        integral = 0
        integral_squared = 0

        for i in range(int(self.N/(self.nproc))):
            #random n generation needs modifies
            xrand = random.uniform(self.lim_l, self.lim_u)
            point = self.function(xrand)
            integral += point
            integral_squared += point**2

        integral_estim = (self.lim_u-self.lim_l)/self.N * integral
        variance = 1/self.N * (integral_squared/self.N - (integral/self.N))**2
        mean = integral/self.N

        return integral_estim, variance, mean

    def running_in_parallel(self):
        I, var, mean = self.monte_carlo()
        self.total_integral += I
        self.total_variance += var
        self.total_mean += mean

        integral = self.comm.reduce(self.total_integral, MPI.SUM, 0)
        variance = self.comm.reduce(self.total_variance, MPI.SUM, 0)
        mean = self.comm.reduce(self.total_mean, MPI.SUM, 0)

        return integral, variance, mean, self.rank


def func(x):
    return np.sin(x)


sin_monte = MonteCarloIntegrationParallel(1000, 0, np.pi, func)
I, var, mean, rank = sin_monte.running_in_parallel()

if rank == 0:
    print('Integral: %.5f' % I, '\nvariance: : %.5f' % var, '\nmean: %.5f: ' % mean)

# MPI.Finalize()

