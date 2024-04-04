#!/bin/python3
""" numpy library for doing math calculations, MPI for parallelization
time module for timing the speed of the monte carlo class"""
import time
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt


# MPI.Init()


class MonteCarlo:
    """ Monte Carlo class for integrating in parallel, can use either evenly distributed sampling
        or importance sampling given a function to do this with"""

    def __init__(self, n_samples, limits, dim, start_pos, potentials, seed=10000):
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
        self.potentials = potentials

    def __main_monte_carlo(self):
        """the main monte_carlo function that calls for other functions. Finds the number of
            data points to be analysed by that worker, spawns appropriate seeds, and finds
            the integrals and positions depending on if importance sampling is or isn't used.
            The means and variances of both are found from the integral and position arrays"""
        n_walks_per_worker = self.__n_walks_per_worker()

        # random walk
        end_positions = self.__random_walk(self.start_pos, n_walks_per_worker)
        potentials = self.evaluate_potential(end_positions)
        pos_average = np.mean(end_positions, axis=0)
        pos_variance = np.var(end_positions, axis=0)

        return end_positions, potentials, pos_average, pos_variance, n_walks_per_worker

    def __random_walk(self, start, n_walks):
        end_pos = []  # np.zeros((n_walks, self.dim))
        start_x = start[0]
        start_y = start[1]
        seed = self.child_ss[self.rank]
        rng = np.random.default_rng(seed)

        for i in range(n_walks):
            xs = [start_x]
            ys = [start_y]
            current_pos = [start_x, start_y]
            # print('iter', i)
            # print(xs)
            # print(ys)
            while True:
                if current_pos[0] >= self.limits[1] or current_pos[1] >= self.limits[1] or \
                        current_pos[0] <= self.limits[0] or current_pos[1] <= self.limits[0]:
                    break
                rand_point = rng.uniform(0, 1)
                # print(self.rank, rand_point)
                # right
                if rand_point <= 0.25:
                    current_pos[0] = current_pos[0] + 0.5
                # left
                if 0.25 < rand_point <= 0.5:
                    current_pos[0] = current_pos[0] - 0.5
                # up
                if 0.5 < rand_point <= 0.75:
                    current_pos[1] = current_pos[1] + 0.5
                # down
                if 0.75 < rand_point <= 1:
                    current_pos[1] = current_pos[1] - 0.5
                xs.append(current_pos[0])
                ys.append(current_pos[1])
                # end_pos.append(current_pos)
                # end_pos.append(current_pos)

            # print(current_pos)
            # print(end_pos)
            end_pos.append(current_pos)
            # print(end_pos)

        all_positions = np.vstack((xs, ys)).T.tolist()
        # return all_positions
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
        print('start location', self.start_pos)
        print('bottom and top limits', self.limits)
        print('potentials (up, right, down, left)', self.potentials)
        print('Number of data points: ', self.n_samples)
        print('Number of workers: ', self.nproc)
        print('Seed: ', self.seed)

    def plot_green(self, pos_and_prob):
        s = 300
        marker = 's'

        plt.figure()
        plt.scatter(self.start_pos[0], self.start_pos[1], c='w', s=s, edgecolors='black')
        plt.scatter(pos_and_prob[:, 0], pos_and_prob[:, 1], c=pos_and_prob[:, 2], s=s, marker=marker, cmap='Reds')
        plt.grid()
        cbar = plt.colorbar()
        cbar.set_label('Probability')
        plt.xlim(self.limits[0] - 0.5, self.limits[1] + 0.5)
        plt.ylim(self.limits[0] - 0.5, self.limits[1] + 0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(np.linspace(self.limits[0], self.limits[1], np.int(self.limits[0] + self.limits[1] + 1)))
        plt.yticks(np.linspace(self.limits[0], self.limits[1], np.int(self.limits[0] + self.limits[1] + 1)))
        plt.title(f'Greens function, start position {self.start_pos}')
        plt.show()

        plt.figure()
        left = []
        right = []
        bottom = []
        top = []
        for i in range(np.size(pos_and_prob[:, 0])):
            if pos_and_prob[i][0] == self.limits[0]:
                left.append([pos_and_prob[i][1], pos_and_prob[i][2]])
            if pos_and_prob[i][0] == self.limits[1]:
                right.append([pos_and_prob[i][1], pos_and_prob[i][2]])
            if pos_and_prob[i][1] == self.limits[0]:
                bottom.append([pos_and_prob[i][0], pos_and_prob[i][2]])
            if pos_and_prob[i][1] == self.limits[1]:
                top.append([pos_and_prob[i][0], pos_and_prob[i][2]])
        if np.size(left) != 0:
            plt.plot(np.asarray(left)[:, 0], np.asarray(left)[:, 1], label='left')
            plt.scatter(np.asarray(left)[:, 0], np.asarray(left)[:, 1])
        if np.size(right) != 0:
            plt.plot(np.asarray(right)[:, 0], np.asarray(right)[:, 1], label='right')
            plt.scatter(np.asarray(right)[:, 0], np.asarray(right)[:, 1])
        if np.size(bottom) != 0:
            plt.plot(np.asarray(bottom)[:, 0], np.asarray(bottom)[:, 1], label='bottom')
            plt.scatter(np.asarray(bottom)[:, 0], np.asarray(bottom)[:, 1])
        if np.size(top) != 0:
            plt.plot(np.asarray(top)[:, 0], np.asarray(top)[:, 1], label='top')
            plt.scatter(np.asarray(top)[:, 0], np.asarray(top)[:, 1])
        plt.xticks(np.linspace(self.limits[0], self.limits[1], np.int(self.limits[0] + self.limits[1] + 1)))
        plt.xlabel('axis of respective boundary in legend')
        plt.ylabel('probability')
        plt.title('Greens function')
        plt.grid()
        plt.legend()
        plt.show()

    def evaluate_potential(self, positions):
        potential = 0
        unique, counts = np.unique(positions, return_counts=True, axis=0)
        pos_and_prob = np.column_stack((unique, counts / np.sum(counts)))

        for i in range(np.size(pos_and_prob, 0)):
            x = pos_and_prob[i][0]
            y = pos_and_prob[i][1]
            prob = pos_and_prob[i][2]

            if y == self.limits[1]:
                potential += prob * self.potentials[0]
                ##up
            if x == self.limits[1]:
                ##right
                potential += prob * self.potentials[1]
            if y == self.limits[0]:
                ##down
                potential += prob * self.potentials[2]
            if x == self.limits[0]:
                ##left
                potential += prob * self.potentials[3]

        return potential

    def running_in_parallel(self):
        """function that manages running the monte carlo in parallel gathers the integrals,
            positions, variances and data points per worker into arrays. Worker 0 sends
            these to be found a global average and variance of and prints out the results"""
        start_time = time.time()
        positions, potentials, pos_mean, pos_var, d_points = self.__main_monte_carlo()
        positions = np.asarray(self.comm.reduce(positions, MPI.SUM, root=0))
        potentials = np.asarray(self.comm.gather(potentials, root=0), dtype=float)
        position_averages = np.asarray(self.comm.gather(pos_mean, root=0), dtype=float)
        position_variances = np.asarray(self.comm.gather(pos_var, root=0), dtype=float)
        d_points = np.asarray(self.comm.gather(d_points, root=0), dtype=float)

        if self.rank == 0:
            unique, counts = np.unique(positions, return_counts=True, axis=0)
            pos_and_prob = np.column_stack((unique, counts / np.sum(counts)))
            potential = self.evaluate_potential(pos_and_prob)
            self.plot_green(pos_and_prob)

            position_mean, position_variance = self.__parallel_mean_and_variance(position_averages,
                                                                                 position_variances, d_points)
            time_taken = time.time() - start_time
            self.print_init_info()
            print('potential', np.mean(potentials), '+-', np.var(potentials))
            print('position mean:', np.round(position_mean, 4))
            print('position variance:', np.round(position_variance, 4), '\nposition std:',
                  np.round(np.sqrt(position_variance), 4))
            print('time taken %.2f' % time_taken)
            print('\n\n')


DIMENSIONS = 2
SEED = 35
# in cm
LIMITS = np.array((0, 5))
N_DATAPOINTS = 5000
start_pos = np.array((2, 2))
# up, right, down, left
potentials = [2, -4, 0, 2]

monte = MonteCarlo(N_DATAPOINTS, LIMITS, DIMENSIONS, start_pos, potentials, seed=SEED)
monte.running_in_parallel()

# MPI.Finalize()