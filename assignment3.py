#!/bin/python3
""" numpy library for doing math calculations, MPI for parallelization
time module for timing the speed of the monte carlo class"""
import time
import numpy as np
from mpi4py import MPI


# MPI.Init()


class MonteCarlo:
    """ Monte Carlo class for integrating in parallel, can use either evenly distributed sampling
        or importance sampling given a function to do this with"""

    def __init__(self, n_samples, limits, dim, start_pos, seed=10000):
        """ initialising monte carlo integrator, estimated using n_samples random data points,
            given upper and lower limits of function, the function that is being
            evaluated, the dimensions of the problem, an optional seed for random data point
            generation and if importance sampling is optionally used, the sampling function and
            its inverse can also be given. Parallelization is also initialised, and a seed
            sequence is initialised for each worker"""
        self.n_samples = n_samples
        self.limits = limits
        self.dim = dim
        self.seed = seed
        self.start_pos = start_pos

        self.comm = MPI.COMM_WORLD
        self.nproc = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.seed_sequence = np.random.SeedSequence(seed)
        self.child_ss = self.seed_sequence.spawn(self.nproc)

    def __main_monte_carlo(self):
        """the main monte_carlo function that calls for other functions. Finds the number of
            data points to be analysed by that worker, spawns appropriate seeds, and finds
            the integrals and positions depending on if importance sampling is or isn't used.
            The means and variances of both are found from the integral and position arrays"""
        n_walks_per_worker = self.__n_walks_per_worker()

        #random walk
        end_positions = self.__random_walk(self.start_pos, n_walks_per_worker)
        pos_average = np.array((np.mean(end_positions[0], axis=0),  np.mean(end_positions[1], axis=0)))
        pos_variance = np.array((np.var(end_positions[0], axis=0), np.var(end_positions[1], axis=0)))

        return pos_average, pos_variance, n_walks_per_worker

    def __random_walk(self, start, n_walks):
        end_pos = np.zeros((n_walks, self.dim))
        start_x = start[0]
        start_y = start[1]

        for i in range(n_walks):
            xs = [start_x]
            ys = [start_y]
            current_pos = [start_x, start_y]
            #print('iter', i)
            #print(xs)
            #print(ys)
            while True:
                rand_point = np.random.uniform(0, 1)
                # right
                if rand_point <= 0.25:
                    current_pos[0] = current_pos[0] + 1
                # left
                if 0.25 < rand_point <= 0.5:
                    current_pos[0] = current_pos[0] - 1
                # up
                if 0.5 < rand_point <= 0.75:
                    current_pos[1] = current_pos[1] + 1
                # down
                if 0.75 < rand_point <= 1:
                    current_pos[1] = current_pos[1] - 1
                xs.append(current_pos[0])
                ys.append(current_pos[1])

                if current_pos[0] == self.limits[1] or current_pos[1] == self.limits[1] or \
                        current_pos[0] == self.limits[0] or current_pos[1] == self.limits[0]:
                    break
            end_pos[i][0] = current_pos[0]
            end_pos[i][1] = current_pos[1]

            #print(end_pos[i][0], end_pos[i][1])
        #print(end_pos)
        return end_pos


    def __n_walks_per_worker(self):
        """function that divides the number of sample points given to the monte carlo integrator
            between the workers. Each worker gets the same amount of data points, but rank 0 also
            gets the remainder. In this case the remainder is max. 15, which adds a negligible
            amount of work to rank 0, thus not being divided between workers further"""
        if self.rank == 0:
            data_points = int(self.n_samples / self.nproc) + (self.n_samples % self.nproc)
        else:
            data_points = int(self.n_samples / self.nproc)
        return data_points

    def __parallel_mean_and_variance(self, means, variances, d_points):
        """function that calculates the parallel mean and variance of an array with several means
            and an array with several variances. As different members of the arrays may have been
            generated from more data points than others, the weight of value is also included"""
        tot_mean = 0
        for i in range(self.nproc):
            tot_mean += means[i] * d_points[i] / self.n_samples
        in_group_var = 0
        between_group_var = 0
        for i in range(self.nproc):
            in_group_var = d_points[i] * variances[i] ** 2 / self.n_samples
            between_group_var += d_points[i] * (means[i] - tot_mean) ** 2 / self.n_samples
        tot_var = in_group_var + between_group_var
        return tot_mean, tot_var

    def print_init_info(self):
        """Prints out initialisation information, can be called before doing integral"""
        print('Integral in', self.dim, ' dimensions using MonteCarlo')
        print('Number of data points: ', self.n_samples)
        print('Number of workers: ', self.nproc)
        print('Seed: ', self.seed)

    def running_in_parallel(self):
        """function that manages running the monte carlo in parallel gathers the integrals,
            positions, variances and data points per worker into arrays. Worker 0 sends
            these to be found a global average and variance of and prints out the results"""
        start_time = time.time()
        pos_mean, pos_var, d_points = self.__main_monte_carlo()

        positions = np.asarray(self.comm.gather(pos_mean, root=0), dtype=float)
        position_variances = np.asarray(self.comm.gather(pos_var, root=0), dtype=float)
        d_points = np.asarray(self.comm.gather(d_points, root=0), dtype=float)

        if self.rank == 0:
            position_mean, position_variance = self.__parallel_mean_and_variance(positions,
                                                position_variances, d_points)
            time_taken = time.time() - start_time
            self.print_init_info()
            print('position mean:', np.round(position_mean, 4))
            print('position variance:', np.round(position_variance, 4), '\nposition std:',
                  np.round(np.sqrt(position_variance), 4))
            print('time taken %.2f' % time_taken)
            print('\n\n')


DIMENSIONS = 2
SEED = 1010
LIMITS = np.array((0, 10))
N_DATAPOINTS =  1000000
start_pos = [5, 5]

monte = MonteCarlo(N_DATAPOINTS, LIMITS, DIMENSIONS, start_pos, seed=SEED)
monte.running_in_parallel()

# MPI.Finalize()