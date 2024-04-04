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

    def __init__(self, n_samples, limits, dim, seed=10000):
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

        self.comm = MPI.COMM_WORLD
        self.nproc = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.seed_sequence = np.random.SeedSequence(seed)
        self.child_ss = self.seed_sequence.spawn(self.nproc)

    def n_data_points(self):
        """function that divides the number of sample points given to the monte carlo integrator
            between the workers. Each worker gets the same amount of data points, but rank 0 also
            gets the remainder. In this case the remainder is max. 15, which adds a negligible
            amount of work to rank 0, thus not being divided between workers further"""
        if self.rank == 0:
            data_points = int(self.n_samples / self.nproc) + (self.n_samples % self.nproc)
        else:
            data_points = int(self.n_samples / self.nproc)
        return data_points

    def parallel_mean_and_variance(self, means, variances, d_points):
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
        print('Monte Carlo in', self.dim, ' dimensions')
        print('limits', self.limits)
        print('Number of data points: ', self.n_samples)
        print('Number of workers: ', self.nproc)
        print('Seed: ', self.seed)


class Integrator(MonteCarlo):
    def __init__(self, n_samples, limits, dim, seed=10000, function=None, importance_sampling=None):
        super().__init__(n_samples, limits, dim, seed)
        self.function = function
        self.importance_sampling_func_and_inverse = importance_sampling

    def __integrator_main(self):
        """the main monte_carlo function that calls for other functions. Finds the number of
            data points to be analysed by that worker, spawns appropriate seeds, and finds
            the integrals and positions depending on if importance sampling is or isn't used.
            The means and variances of both are found from the integral and position arrays"""
        n_data_points = super().n_data_points()
        grand_child_ss = self.child_ss[self.rank].spawn(n_data_points)

        if self.importance_sampling_func_and_inverse is None:
            integrals, positions = self.__integrator_random_sampled(n_data_points,
                                                                    grand_child_ss)
        else:
            integrals, positions = self.__integrator_importance_sampled(n_data_points,
                                                                        grand_child_ss)

        integral_estim = np.sum(integrals)
        integral_var = np.var(integrals)
        pos_average = np.mean(positions, axis=0)
        pos_variance = np.var(positions, axis=0)

        return integral_estim, integral_var, pos_average, pos_variance, n_data_points

    def __integrator_random_sampled(self, n_data_points, grand_child_ss):
        """Monte carlo integration without importance sampling for given number of data
            points and a seed sequence. Random sampling points come from a uniform distribution
            makes an array out of the position and integrals to be sent back"""
        integrals_at_data_points = np.zeros(n_data_points)
        positions = np.zeros((n_data_points, self.dim))
        for i in range(n_data_points):
            rng = np.random.default_rng(grand_child_ss[i])
            rand_point = rng.uniform(self.limits[0], self.limits[1])
            for j in range(self.dim):
                positions[i][j] = rand_point[j]
            integrals_at_data_points[i] = self.function(rand_point)

        volume = (self.limits[1][0] - self.limits[0][0])
        for i in range(self.dim - 1):
            volume *= (self.limits[1][i + 1] - self.limits[0][i + 1])
        integrals = volume * integrals_at_data_points / n_data_points

        return integrals, positions

    def __integrator_importance_sampled(self, n_data_points, grand_child_ss):
        """monte carlo integration with importance sampling for given number of data
            points and a seed sequence. Also requires a sampling function and its
            inverse for the random data point distribution. Makes an array out of
            the position and integrals to be sent back"""
        integrals_at_data_points = np.zeros((n_data_points, self.dim))
        positions = np.zeros((n_data_points, self.dim))
        importance_sampling_function_sum = 0
        importance_sampling_function = self.importance_sampling_func_and_inverse[0]
        importance_sampling_function_inverse = self.importance_sampling_func_and_inverse[1]
        for i in range(n_data_points):
            rng = np.random.default_rng(grand_child_ss[i])
            positions[i] = importance_sampling_function_inverse(rng.uniform(-1, 1, self.dim))
            function_integral_at_point = self.function(positions[i])
            importance_f_integ = np.array(importance_sampling_function(positions[i]))

            for j in range(self.dim):
                division = function_integral_at_point / importance_f_integ
                integrals_at_data_points[i][j] = division

            importance_sampling_function_sum += importance_sampling_function(positions[i])

        integrals_at_data_points = integrals_at_data_points / importance_sampling_function_sum
        integrals = integrals_at_data_points / self.dim

        return integrals, positions

    def integrator_running_in_parallel(self):
        """function that manages running the monte carlo in parallel gathers the integrals,
            positions, variances and data points per worker into arrays. Worker 0 sends
            these to be found a global average and variance of and prints out the results"""
        start_time = time.time()
        integral, integral_variance, pos_mean, pos_var, d_points = self.__integrator_main()

        integrals = np.asarray(self.comm.gather(integral, root=0), dtype=float)
        integral_variances = np.asarray(self.comm.gather(integral_variance, root=0), dtype=float)
        positions = np.asarray(self.comm.gather(pos_mean, root=0), dtype=float)
        position_variances = np.asarray(self.comm.gather(pos_var, root=0), dtype=float)
        d_points = np.asarray(self.comm.gather(d_points, root=0), dtype=float)

        if self.rank == 0:
            integral_mean, integral_variance = super().parallel_mean_and_variance(integrals,
                                                                                 integral_variances, d_points)
            position_mean, position_variance = super().parallel_mean_and_variance(positions,
                                                                                 position_variances, d_points)
            time_taken = time.time() - start_time
            super().print_init_info()
            print('\nIntegral: %.8f' % integral_mean, ' +-  %.8f' %
                  np.sqrt(integral_variance), '(std)')
            print('position mean:', np.round(position_mean, 4))
            print('position variance:', np.round(position_variance, 4), '\nposition std:',
                  np.round(np.sqrt(position_variance), 4))
            print('time taken %.2f' % time_taken)
            print('\n\n')


class RandomWalk(MonteCarlo):
    def __init__(self, n_samples, limits, dim, seed=10000, start_pos=None, potentials=None):
        super().__init__(n_samples, limits, dim, seed)
        self.start_pos = start_pos
        self.potentials = potentials

    def __main_random_walk(self):
        """the main monte_carlo function that calls for other functions. Finds the number of
            data points to be analysed by that worker, spawns appropriate seeds, and finds
            the integrals and positions depending on if importance sampling is or isn't used.
            The means and variances of both are found from the integral and position arrays"""
        n_walks_per_worker = super().n_data_points()
        end_positions = self.__random_walk(self.start_pos, n_walks_per_worker)
        potentials = self.__random_walk_evaluate_potential(end_positions)
        pos_average = np.mean(end_positions, axis=0)
        pos_variance = np.var(end_positions, axis=0)

        return end_positions, potentials, pos_average, pos_variance, n_walks_per_worker

    def __random_walk(self, start, n_walks):
        end_pos = []
        start_x = start[0]
        start_y = start[1]
        seed = self.child_ss[self.rank]
        rng = np.random.default_rng(seed)

        for i in range(n_walks):
            xs = [start_x]
            ys = [start_y]
            current_pos = [start_x, start_y]
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

            end_pos.append(current_pos)

        all_positions = np.vstack((xs, ys)).T.tolist()
        return end_pos

    def __random_walk_plot_green(self, pos_and_prob):
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

    def __random_walk_evaluate_potential(self, positions):
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

    def random_walk_running_in_parallel(self):
        """function that manages running the monte carlo in parallel gathers the integrals,
            positions, variances and data points per worker into arrays. Worker 0 sends
            these to be found a global average and variance of and prints out the results"""
        start_time = time.time()
        positions, potentials, pos_mean, pos_var, d_points = self.__main_random_walk()
        positions = np.asarray(self.comm.reduce(positions, MPI.SUM, root=0))
        potentials = np.asarray(self.comm.gather(potentials, root=0), dtype=float)
        position_averages = np.asarray(self.comm.gather(pos_mean, root=0), dtype=float)
        position_variances = np.asarray(self.comm.gather(pos_var, root=0), dtype=float)
        d_points = np.asarray(self.comm.gather(d_points, root=0), dtype=float)

        if self.rank == 0:
            unique, counts = np.unique(positions, return_counts=True, axis=0)
            pos_and_prob = np.column_stack((unique, counts / np.sum(counts)))
            potential = self.__random_walk_evaluate_potential(pos_and_prob)

            position_mean, position_variance = super().parallel_mean_and_variance(position_averages,
                                                                                 position_variances, d_points)
            time_taken = time.time() - start_time
            super().print_init_info()
            self.__random_walk_plot_green(pos_and_prob)
            print('potential', np.mean(potentials), '+-', np.var(potentials))
            print('position mean:', np.round(position_mean, 4))
            print('position variance:', np.round(position_variance, 4), '\nposition std:',
                  np.round(np.sqrt(position_variance), 4))
            print('time taken %.2f' % time_taken)
            print('\n\n')


def run_gaussian_integral_mc():
    DIMENSIONS = 1
    MEAN = np.ones(DIMENSIONS) * 0
    SIGMA = 1

    def importance_sampling_function_gauss(points):
        """importance sampling function for a given data point (in 1 or more dimensions)"""
        integrand = np.exp(-np.abs(points[0] - MEAN[0]))
        for i in range(DIMENSIONS - 1):
            integrand *= np.exp(-np.abs(points[i + 1] - MEAN[i + 1]))
        return integrand

    def importance_sampling_function_inverse_gauss(point):
        """inverse of importance sampling function for a given data point (in 1 or more
            dimensions) used for inverse transform sampling"""
        return np.sign(point) * (-np.log(1 - np.abs(point))) + MEAN

    def gaussian(points):
        """gaussian function for a given data point (in 1 or more dimensions)"""
        dim = np.size(points)
        covar_matrix = np.eye(dim) * SIGMA ** 2
        gauss = 1 / (np.sqrt((2 * np.pi) ** dim * np.linalg.det(covar_matrix))) * \
                np.exp((-1 / 2) * (points - MEAN).T @ (np.linalg.inv(covar_matrix) @ (points - MEAN)))
        return gauss

    bottom_lims = MEAN - 5 * SIGMA
    top_lims = MEAN + 5 * SIGMA
    limits = [bottom_lims, top_lims]
    SEED = 1010
    N_DATAPOINTS = 10000

    monte = Integrator(N_DATAPOINTS, limits, DIMENSIONS, seed=SEED, function=gaussian)
    monte.integrator_running_in_parallel()

    monte = Integrator(N_DATAPOINTS, limits, DIMENSIONS, seed=SEED, function=gaussian,
                           importance_sampling=[importance_sampling_function_gauss,
                                                importance_sampling_function_inverse_gauss])
    monte.integrator_running_in_parallel()


def run_random_walk_mc():
    DIMENSIONS = 2
    SEED = 35
    # in cm
    LIMITS = np.array((0, 10))
    N_DATAPOINTS = 5000
    start_pos = np.array((5, 5))
    # up, right, down, left
    potentials = [2, -4, 0, 2]

    monte = RandomWalk(N_DATAPOINTS, LIMITS, DIMENSIONS, seed=SEED, start_pos=start_pos, potentials=potentials)
    monte.random_walk_running_in_parallel()


run_gaussian_integral_mc()
run_random_walk_mc()

# MPI.Finalize()