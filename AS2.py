#!/bin/python3
import random
import numpy as np
from mpi4py import MPI
import time


# MPI.Init()

class MonteCarloIntegrationParallel:

    def __init__(self, N, mean, lim_l, lim_u, function, dim, seed=10000):
        self.N = N
        self.mean = mean
        self.lim_l = lim_l
        self.lim_u = lim_u
        self.function = function
        self.seed = seed

        self.comm = MPI.COMM_WORLD
        self.nproc = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.ss = np.random.SeedSequence(seed)
        self.child_ss = self.ss.spawn(self.nproc)
        self.dim = dim

    def monte_carlo(self):

        if self.rank == 0:
            data_points = int(self.N / self.nproc) + (self.N % self.nproc)
        else:
            data_points = int(self.N / self.nproc)

        integral = 0
        integral_squared = 0
        pos_average = np.zeros(self.dim)
        # pos variance doesnt work yet. array of positions? could get too large. Make a function for it
        pos_variance = np.zeros(self.dim)

        grand_child_ss = self.child_ss[self.rank].spawn(data_points)

        for i in range(data_points):
            grand_grand_child_ss = grand_child_ss[i].spawn(self.dim)
            for j in range(self.dim):
                rng2 = np.random.default_rng(grand_grand_child_ss[j])
                rand_points = rng2.uniform(self.lim_l, self.lim_u)

            pos_average += rand_points / self.N
            pos_variance += (rand_points - self.mean) ** 2 / self.N

            point = self.function(rand_points)
            integral += point
            integral_squared += point ** 2

        integral_limits = (self.lim_u[0] - self.lim_l[0])
        for i in range(self.dim - 1):
            integral_limits *= (self.lim_u[i + 1] - self.lim_l[i + 1])

        integral_estim = integral_limits * integral / self.N
        # variance = 1/self.N * (integral_squared/self.N - integral/self.N)
        return integral_estim, pos_average, pos_variance  # variance, mean

    def running_in_parallel(self):
        I, pos_mean, pos_var = self.monte_carlo()
        integral = self.comm.reduce(I, MPI.SUM, 0)
        mean = self.comm.reduce(pos_mean, MPI.SUM, 0)
        variance = self.comm.reduce(pos_var, MPI.SUM, 0)

        return integral, mean, variance, self.rank


def func(x):
    return np.sin(x)


# variance should be sigma**2
# mean should be the same as the mean defined here, 0? position mean? or integral mean?
dim = 6
mean = np.ones(dim) * 0
sigma = 1


def func2(points):
    dim = np.size(points)
    covar_matrix = np.eye(dim) * sigma ** 2
    gauss = 1 / (np.sqrt((2 * np.pi) ** dim * np.linalg.det(covar_matrix))) * np.exp(
        (-1 / 2) * (points - mean).T @ (np.linalg.inv(covar_matrix) @ (points - mean)))
    return gauss


sin_monte = MonteCarloIntegrationParallel(1000000, mean, mean - 4 * sigma, mean + 4 * sigma, func2, dim, seed=1900)
start_time = time.time()
I, mean, var, rank = sin_monte.running_in_parallel()

if rank == 0:
    time_taken = time.time() - start_time
    print('Integral: %.5f' % I, '\nmean pos: ', np.round(mean, 5), '\nvariance pos:', np.round(var, 5))
    print('time taken %.2f' % time_taken)

# MPI.Finalize()
