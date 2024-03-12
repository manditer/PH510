#!/bin/python3
#numpy library for doing math calculations, MPI for parallelisation
#time module for timing the speed of the monte carlo class
import numpy as np
from mpi4py import MPI
import time

# MPI.Init()

# Monte Carlo class for integrating in parallel, can use either evenly distributed sampling
# or importance sampling given a function to do this with
class MonteCarloIntegrationParallel:

    # initialising monte carlo integrator, estimated using n_samples random data points,
    # given upper and lower limits of function lim_l and lim_u, the function
    # that is being evaluated, the dimensions of the problem, an optional seed
    # for random data point generation
    # and if importance sampling is optionally used, the sampling function and its inverse
    # can also be given
    # Parallelisation is also initialised, and a seed sequence is initialised for each worker
    def __init__(self, n_samples, lim_l, lim_u, function, dim, seed=10000,
                 importance_sampling_function=None, importance_sampling_function_inverse=None):
        self.n_samples = n_samples
        self.lim_l = lim_l
        self.lim_u = lim_u
        self.mean = mean
        self.function = function
        self.seed = seed
        self.dim = dim
        self.importance_sampling_function = importance_sampling_function
        self.importance_sampling_function_inverse = importance_sampling_function_inverse

        self.comm = MPI.COMM_WORLD
        self.nproc = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.ss = np.random.SeedSequence(seed)
        self.child_ss = self.ss.spawn(self.nproc)


    # the main monte_carlo function that calls for other functions.
    # finds the number of data points to be analysed by that worker,
    # spawns appropriate seeds, and finds the integrals
    #and positions depending if importance sampling is or isnt used
    # the means and variances of both are found from the integral and position arrays
    def __main_monte_carlo(self):
        n_data_points = self.__n_data_points()
        grand_child_ss = self.child_ss[self.rank].spawn(n_data_points)

        if self.importance_sampling_function == None:
            integrals, positions = self.__random_sampled_monte_carlo(n_data_points,
                                                                    grand_child_ss)
        else:
            integrals, positions = self.__importance_sampled_monte_carlo(n_data_points,
                                                                        grand_child_ss)

        integral_estim = np.sum(integrals)
        integral_var = np.var(integrals)
        pos_average = np.mean(positions, axis=0)
        pos_variance = np.var(positions, axis=0)

        return integral_estim, integral_var, pos_average, pos_variance, n_data_points


    # monte carlo integration without importance sampling for given number of
    # data points and a seed sequence
    # random sampling points come from a uniform distribution
    # makes an array out of the position and integrals to be sent back
    def __random_sampled_monte_carlo(self, n_data_points, grand_child_ss):
        integrals_at_data_points = np.zeros(n_data_points)
        positions = np.zeros((n_data_points, self.dim))
        for i in range(n_data_points):
            rng = np.random.default_rng(grand_child_ss[i])
            rand_point = rng.uniform(self.lim_l, self.lim_u)
            for j in range(self.dim):
                positions[i][j] = rand_point[j]
            integrals_at_data_points[i] = self.function(rand_point)

        volume = (self.lim_u[0]-self.lim_l[0])
        for i in range(self.dim-1):
            volume *= (self.lim_u[i+1]-self.lim_l[i+1])
        integrals = volume * integrals_at_data_points/n_data_points

        return integrals, positions

    # monte carlo integration with importance sampling for given number of
    # data points and a seed sequence
    # also requires a sampling function and its inverse for the random data point distribution
    # makes an array out of the position and integrals to be sent back
    def __importance_sampled_monte_carlo(self,n_data_points, grand_child_ss):
        integrals_at_data_points = np.zeros((n_data_points, self.dim))
        positions = np.zeros((n_data_points, self.dim))
        importance_sampling_function_sum = 0
        for i in range(n_data_points):
            rng = np.random.default_rng(grand_child_ss[i])
            rand_point = rng.uniform(-1, 1, self.dim)
            function_integral_at_point = np.zeros(self.dim)
            importance_function_integral_at_point = np.zeros(self.dim)

            positions[i] = self.importance_sampling_function_inverse(rand_point)
            function_integral_at_point = self.function(positions[i])
            importance_f_integ = np.array(self.importance_sampling_function(positions[i]))

            for j in range(self.dim):
                division = function_integral_at_point/importance_f_integ
                integrals_at_data_points[i][j] = division

            importance_sampling_function_sum += self.importance_sampling_function(positions[i])

        integrals_at_data_points = integrals_at_data_points/importance_sampling_function_sum
        integrals = integrals_at_data_points/self.dim

        return integrals, positions


    # function that divides the number of sample points given to the monte carlo integrator
    # between the workers
    # each worker gets the same amount of data points, but rank 0 also gets the remainder
    # in this case the remainder is max. 15, which adds a neglible amount of work to rank 0,
    # thus not being divided between workers further
    def __n_data_points(self):
        if self.rank == 0:
            data_points = int(self.n_samples/(self.nproc)) + (self.n_samples % self.nproc)
        else:
            data_points = int(self.n_samples/(self.nproc))
        return data_points


    # function that calculates the parallel mean and variance of an array
    # with several means and an array with several variances
    # as different members of the arrays may have been generated from more data points than others,
    # the weight of value is also included
    def __parallel_mean_and_variance(self, means, variances, d_points):
        tot_mean = 0
        for i in range(self.nproc):
            tot_mean += means[i] * d_points[i] / self.n_samples
        in_group_var = 0
        between_group_var = 0
        for i in range(self.nproc):
            in_group_var = d_points[i]*variances[i]**2 / self.n_samples
            between_group_var += d_points[i]*(means[i] - tot_mean)**2 / self.n_samples
        tot_var = in_group_var + between_group_var
        return tot_mean, tot_var


    # function that manages running the monte carlo in parallel
    # gathers the integrals, positions, variances and data points per worker into arrays
    # worker 0 sends these to be found a global average and variance of and prints out the results
    def running_in_parallel(self):
        start_time = time.time()
        I, I_var, pos_mean, pos_var, d_points = self.__main_monte_carlo()

        Integrals = np.asarray(self.comm.gather(I, root=0), dtype=float)
        Integral_variances = np.asarray(self.comm.gather(I_var, root=0), dtype=float)
        positions = np.asarray(self.comm.gather(pos_mean, root=0), dtype=float)
        position_variances = np.asarray(self.comm.gather(pos_var, root=0), dtype=float)
        d_points = np.asarray(self.comm.gather(d_points, root=0), dtype=float)

        if self.rank == 0:
            integral_mean, integral_variance = self.__parallel_mean_and_variance(Integrals,
                                                Integral_variances, d_points)
            position_mean, position_variance = self.__parallel_mean_and_variance(positions,
                                                position_variances, d_points)
            time_taken = time.time()-start_time
            if self.importance_sampling_function==None:
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
mean = np.ones(dim)*0
sigma = 1


#importance sampling function for a given data point (in 1 or more dimensions)
def importance_sampling_function_gauss(points):
    I = np.exp(-np.abs(points[0]-mean[0]))
    for i in range(dim-1):
        I *= np.exp(-np.abs(points[i+1]-mean[i+1]))
    return I


# inverse of importance sampling function for a given data point (in 1 or more dimensions)
# used for inverse transorm sampling
def importance_sampling_function_inverse(point):
    return np.sign(point)*(-np.log(1-np.abs(point)))+mean


#gaussian function for for a given data point (in 1 or more dimensions)
def gaussian(points):
    dim = np.size(points)
    covar_matrix = np.eye(dim) * sigma**2
    gauss = 1/(np.sqrt((2*np.pi)**dim * np.linalg.det(covar_matrix)))*np.exp((-1/2)*(points-mean).T
    @ (np.linalg.inv(covar_matrix) @(points-mean)))
    return gauss


bottom_lims = mean-5*sigma
top_lims = mean+5*sigma
seed = 69697
samples = 100

sin_monte = MonteCarloIntegrationParallel(samples, bottom_lims, top_lims, gaussian, dim, seed)
sin_monte.running_in_parallel()

sin_monte = MonteCarloIntegrationParallel(samples, bottom_lims, top_lims, gaussian, dim, seed,
             importance_sampling_function_gauss, importance_sampling_function_inverse)
sin_monte.running_in_parallel()


# MPI.Finalize()