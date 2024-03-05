#!/bin/python3
import random
import numpy as np
from mpi4py import MPI
import time


# MPI.Init()

class MonteCarloIntegrationParallel:

    def __init__(self, N, lim_l, lim_u, function, dim, seed=10000):
        self.N = N
        self.lim_l = lim_l
        self.lim_u = lim_u
        self.function = function
        self.dim = dim
        self.seed = seed

        self.comm = MPI.COMM_WORLD
        self.nproc = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.ss = np.random.SeedSequence(seed)
        self.child_ss = self.ss.spawn(self.nproc)

    def monte_carlo(self):
        n_data_points = self.n_data_points()
        random_positions, integrals_at_positions = self.random_positions(n_data_points)
        integral, integral_std = self.integral_and_std(integrals_at_positions)

        pos_average = np.mean(random_positions, axis=0) / self.nproc
        pos_variance = np.var(random_positions, axis=0) / self.nproc

        return integral, integral_std / self.nproc, pos_average, pos_variance

    def integral_and_std(self, integrals_at_positions):
        integrals_summed = np.sum(integrals_at_positions)
        integral_limits = (self.lim_u[0] - self.lim_l[0])
        for i in range(self.dim - 1):
            integral_limits *= (self.lim_u[i + 1] - self.lim_l[i + 1])
        integral_estim = integral_limits * integrals_summed / self.N
        integral_std = np.std(integrals_at_positions)

        return integral_estim, integral_std

    def random_positions(self, n_data_points):
        positions = np.zeros((n_data_points, self.dim))
        integrals_at_data_points = np.zeros(n_data_points)
        grand_child_ss = self.child_ss[self.rank].spawn(n_data_points)
        for i in range(n_data_points):
            rng3 = np.random.default_rng(grand_child_ss[i])
            rand_point = rng3.uniform(self.lim_l, self.lim_u)
            for j in range(self.dim):
                positions[i][j] = rand_point[j]
            integrals_at_data_points[i] = self.function(rand_point)

        return positions, integrals_at_data_points

    def n_data_points(self):
        if self.rank == 0:
            data_points = int(self.N / (self.nproc)) + (self.N % self.nproc)
        else:
            data_points = int(self.N / (self.nproc))
        return data_points

    def running_in_parallel(self):
        # comm gather
        I, I_std, pos_mean, pos_var = self.monte_carlo()
        integral = self.comm.reduce(I, MPI.SUM, 0)
        integral_std = self.comm.reduce(I_std, MPI.SUM, 0)
        pos_mean = self.comm.reduce(pos_mean, MPI.SUM, 0)
        pos_var = self.comm.reduce(pos_var, MPI.SUM, 0)

        return integral, integral_std, pos_mean, pos_var, self.rank


def func(x):
    return np.sin(x)


dim = 6
mean = np.ones(dim) * 0
sigma = 1


def func2(points):
    dim = np.size(points)
    covar_matrix = np.eye(dim) * sigma ** 2
    gauss = 1 / (np.sqrt((2 * np.pi) ** dim * np.linalg.det(covar_matrix))) * np.exp(
        (-1 / 2) * (points - mean).T @ (np.linalg.inv(covar_matrix) @ (points - mean)))
    return gauss


sin_monte = MonteCarloIntegrationParallel(2500000, mean - 4 * sigma, mean + 4 * sigma, func2, dim, seed=1900)
start_time = time.time()
I, I_std, pos_mean, pos_var, rank = sin_monte.running_in_parallel()

if rank == 0:
    time_taken = time.time() - start_time
    print('Integral: %.3f' % I, ' +-  %.3f' % I_std, '\nmean pos: ', np.round(pos_mean, 3), '\nvariance pos:',
          np.round(pos_var, 3))
    print('time taken %.2f' % time_taken)

# MPI.Finalize()
