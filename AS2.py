#!/bin/python3
import random
import numpy as np
from mpi4py import MPI
import time


# MPI.Init()

class MonteCarloIntegrationParallel:

    def __init__(self, N, lim_l, lim_u, function, dim, seed=10000, importance_sampling_function=None):
        self.N = N
        self.lim_l = lim_l
        self.lim_u = lim_u
        self.function = function
        self.seed = seed
        self.dim = dim
        self.importance_sampling_function = importance_sampling_function

        self.comm = MPI.COMM_WORLD
        self.nproc = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.ss = np.random.SeedSequence(seed)
        self.child_ss = self.ss.spawn(self.nproc)

    def __monte_carlo(self):
        n_data_points = self.__n_data_points()
        integral, integral_std, pos_average, pos_variance = self.__integral_and_var(n_data_points)

        return integral, integral_std, pos_average, pos_variance, n_data_points

    def __integral_and_var(self, n_data_points):
        integrals_at_data_points = np.zeros(n_data_points)
        positions = np.zeros((n_data_points, self.dim))
        grand_child_ss = self.child_ss[self.rank].spawn(n_data_points)

        # no importance sampling
        if self.importance_sampling_function == None:

            for i in range(n_data_points):
                rng = np.random.default_rng(grand_child_ss[i])
                rand_point = rng.uniform(self.lim_l, self.lim_u)
                for j in range(self.dim):
                    positions[i][j] = rand_point[j]
                integrals_at_data_points[i] = self.function(rand_point)

            volume = (self.lim_u[0] - self.lim_l[0])
            for i in range(self.dim - 1):
                volume *= (self.lim_u[i + 1] - self.lim_l[i + 1])
            # print('volume', volume)
            integrals = volume * integrals_at_data_points / (n_data_points)


        # importance sampling
        else:
            # normalisation

            integrals_at_data_points = 0
            importance_sampling_function_sum = 0
            inte_sum = 0
            for i in range(n_data_points):
                rng = np.random.default_rng(grand_child_ss[i])
                rand_point = rng.uniform(-1, 1, self.dim)

                for j in range(self.dim):
                    positions[i][j] = np.sign(rand_point) * (-np.log(1 - np.abs(rand_point)))[j]

                importance_sampling_function_sum += self.importance_sampling_function(positions[i])
                integrals_at_data_points += np.divide(self.function(positions[i]),
                                                      self.importance_sampling_function(positions[i]))

            integrals_at_data_points = integrals_at_data_points / importance_sampling_function_sum
            integrals = integrals_at_data_points

        integral_estim = np.sum(integrals)
        integral_var = np.var(integrals)
        pos_average = np.mean(positions, axis=0)
        pos_variance = np.var(positions, axis=0)

        return integral_estim, integral_var, pos_average, pos_variance

    def __n_data_points(self):
        if self.rank == 0:
            data_points = int(self.N / (self.nproc)) + (self.N % self.nproc)
        else:
            data_points = int(self.N / (self.nproc))
        return data_points

    def __parallel_mean_and_variance(self, means, variances, d_points):
        tot_mean = 0
        for i in range(self.nproc):
            tot_mean += means[i] * d_points[i] / self.N
        in_group_var = 0
        between_group_var = 0
        for i in range(self.nproc):
            in_group_var = d_points[i] * variances[i] ** 2 / self.N
            between_group_var += d_points[i] * (means[i] - tot_mean) ** 2 / self.N
        tot_var = in_group_var + between_group_var
        return tot_mean, tot_var

    def running_in_parallel(self):
        start_time = time.time()
        I, I_var, pos_mean, pos_var, d_points = self.__monte_carlo()
        # print('rank', self.rank, I)

        Integrals = np.asarray(self.comm.gather(I, root=0), dtype=float)

        Integral_variances = np.asarray(self.comm.gather(I_var, root=0), dtype=float)
        positions = np.asarray(self.comm.gather(pos_mean, root=0), dtype=float)
        position_variances = np.asarray(self.comm.gather(pos_var, root=0), dtype=float)
        d_points = np.asarray(self.comm.gather(d_points, root=0), dtype=float)

        if self.rank == 0:
            integral_mean, integral_variance = self.__parallel_mean_and_variance(Integrals, Integral_variances,
                                                                                 d_points)
            position_mean, position_variance = self.__parallel_mean_and_variance(positions, position_variances,
                                                                                 d_points)
            time_taken = time.time() - start_time
            if self.importance_sampling_function == None:
                print('\n\nNO IMPORTANCE SAMPLING')
            else:
                print('\n\nIMPORTANCE SAMPLING')
            print('Integral in', self.dim, ' dimensions')
            print('bottom limit(s): ', self.lim_l, '\ntop limit(s): ', self.lim_u)
            print('Integral: %.8f' % integral_mean, ' +-  %.8f' % integral_variance, '(var)')
            print('position mean:', np.round(position_mean, 4))
            print('position variance:', np.round(position_variance, 4), '\nposition std:',
                  np.round(np.sqrt(position_variance), 4))
            print('time taken %.2f' % time_taken)
            print('\n\n')


dim = 1
mean = np.ones(dim) * 0
sigma = 1


def importance_sampling_function_gauss(points):
    I = np.exp(-np.abs(points - mean))
    return I


def gaussian(points):
    dim = np.size(points)
    covar_matrix = np.eye(dim) * sigma ** 2
    gauss = 1 / (np.sqrt((2 * np.pi) ** dim * np.linalg.det(covar_matrix))) * np.exp((-1 / 2) * (points - mean).T @
                                                                                     (np.linalg.inv(covar_matrix) @
                                                                                      (points - mean)))
    return gauss


bottom_lims = mean - 5 * sigma
top_lims = mean + 5 * sigma
seed = 6969
samples = 160000

sin_monte = MonteCarloIntegrationParallel(samples, bottom_lims, top_lims, gaussian, dim, seed)
sin_monte.running_in_parallel()

sin_monte = MonteCarloIntegrationParallel(samples, bottom_lims, top_lims, gaussian, dim, seed,
                                          importance_sampling_function_gauss)
sin_monte.running_in_parallel()

# MPI.Finalize()
