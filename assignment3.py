#!/bin/python3
""" numpy library for doing math calculations, MPI for parallelization,
plt for plotting, and time module for timing the speeds"""
import time
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt


# MPI.Init()

class MonteCarlo:
    """ Monte Carlo class for initializing parallel running and a few global functions"""

    def __init__(self, n_samples, limits, dim, seed=10000):
        """ initialising the monte carlo, estimated using n_samples random data points,
            given upper and lower limits, the dimensions of the problem, an optional seed
            for random data point generation.  Parallelization is also initialised,
            and a seed sequence is initialised for each worker"""
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
        """function that divides the number of sample points given to the monte carlo
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
            generated from more data points than others, the weights are also included"""
        if self.nproc > 1:
            tot_mean = 0
            tot_var = 0
            for i in range(self.nproc):
                tot_mean += means[i] * d_points[i] / self.n_samples

            for i in range(self.nproc):
                tot_var += ((means[i] - tot_mean) ** 2 + variances[i] ** 2) \
                            * d_points[i] / (self.n_samples*(self.nproc-1))
        else:
            tot_mean = means
            tot_var = variances

        return tot_mean, tot_var

    def print_init_info(self):
        """Prints out initialisation information"""
        print('Monte Carlo in', self.dim, ' dimensions')
        print('limits', self.limits)
        print('Number of data points: ', self.n_samples)
        print('Number of workers: ', self.nproc)
        print('Seed: ', self.seed)
        print('')


class Integrator(MonteCarlo):
    """Monte Carlo integrator, child class of the parallel Monte Carlo class. """

    def __init__(self, n_samples, limits, dim, seed=10000, function=None, importance_sampling=None):
        """ initialising the monte carlo parent class, as well as the function that is being
            estimated, and the possible importance sampling function and its inverse"""
        super().__init__(n_samples, limits, dim, seed)
        self.function = function
        self.importance_sampling_func_and_inverse = importance_sampling

    def __integrator_main(self):
        """the main integrator function that calls for other functions. Finds the number of
            data points to be analysed by that worker, spawns appropriate seeds, and finds
            the integrals and positions depending on if importance sampling is or isn't used.
            The integral and position arrays are returned."""
        n_data_points = super().n_data_points()
        grand_child_ss = self.child_ss[self.rank].spawn(n_data_points)

        if self.importance_sampling_func_and_inverse is None:
            integrals, positions = self.__integrator_random_sampled(n_data_points,
                                                                    grand_child_ss)
        else:
            integrals, positions = self.__integrator_importance_sampled(n_data_points,
                                                                        grand_child_ss)

        return integrals, positions, n_data_points

    def __integrator_random_sampled(self, n_data_points, grand_child_ss):
        """Monte carlo integration without importance sampling for given number of data
            points and a seed sequence. Random sampling points come from a uniform distribution.
            Makes an array out of the positions and integrals to be sent back"""
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
        """function that manages running the MC integrator in parallel: gathers the integrals,
            positions, variances and data points per worker into arrays. Worker 0 sends
            these to be found a global average and variance of and prints out the results"""
        start_time = time.time()
        integral, position, d_points = self.__integrator_main()

        integral_mean = np.mean(integral)
        integral_var = np.var(integral)
        position_mean = np.mean(position)
        position_var = np.var(position)

        integral_means = np.asarray(self.comm.gather(integral_mean, root=0), dtype=float)
        integral_variances = np.asarray(self.comm.gather(integral_var, root=0), dtype=float)
        positions = np.asarray(self.comm.gather(position_mean, root=0), dtype=float)
        position_variances = np.asarray(self.comm.gather(position_var, root=0), dtype=float)
        d_points = np.asarray(self.comm.gather(d_points, root=0), dtype=float)

        if self.rank == 0:
            integral_mean_parallel, integral_variance_parallel = \
                super().parallel_mean_and_variance(integral_means, integral_variances, d_points)
            position_mean_parallel, position_variance_parallel = \
                super().parallel_mean_and_variance(positions, position_variances, d_points)

            time_taken = time.time() - start_time
            super().print_init_info()
            print('\nIntegral: %.8f' % integral_mean_parallel, ' +-  %.8f' %
                  np.sqrt(integral_variance_parallel), '(std)')
            print('position mean:', np.round(position_mean_parallel, 4))
            print('position variance:', np.round(position_variance_parallel, 4), '\nposition std:',
                  np.round(np.sqrt(position_variance_parallel), 4))
            print('time taken %.2f' % time_taken)
            print('\n\n')


def run_gaussian_integral_mc():
    """Defining the gaussian function we are interested in integrating,
    the possible importance sampling function and its inverse, and sending them to the
     integrator class with the given limits etc. to be integrated."""
    dimensions = 1
    mean = np.ones(dimensions) * 0
    sigma = 1

    def importance_sampling_function_gauss(points):
        """importance sampling function for a given data point (in 1 or more dimensions)"""
        integrand = np.exp(-np.abs(points[0] - mean[0]))
        for i in range(dimensions - 1):
            integrand *= np.exp(-np.abs(points[i + 1] - mean[i + 1]))
        return integrand

    def importance_sampling_function_inverse_gauss(point):
        """inverse of importance sampling function for a given data point (in 1 or more
            dimensions) used for inverse transform sampling"""
        return np.sign(point) * (-np.log(1 - np.abs(point))) + mean

    def gaussian(points):
        """gaussian function for a given data point (in 1 or more dimensions)"""
        dim = np.size(points)
        covar_matrix = np.eye(dim) * sigma ** 2
        gauss = 1 / (np.sqrt((2 * np.pi) ** dim * np.linalg.det(covar_matrix))) * \
                np.exp((-1 / 2) * (points - mean).T @ (np.linalg.inv(covar_matrix)
                                                       @ (points - mean)))
        return gauss

    bottom_lims = mean - 5 * sigma
    top_lims = mean + 5 * sigma
    limits = [bottom_lims, top_lims]
    seed = 1010
    n_datapoints = 10000

    monte = Integrator(n_datapoints, limits, dimensions, seed=seed, function=gaussian)
    monte.integrator_running_in_parallel()

    monte = Integrator(n_datapoints, limits, dimensions, seed=seed, function=gaussian,
                       importance_sampling=[importance_sampling_function_gauss,
                                            importance_sampling_function_inverse_gauss])
    monte.integrator_running_in_parallel()


class RandomWalk(MonteCarlo):
    """The parallel random walk class, child class of the Monte Carlo class. Performing a
        random walk on a 2D grid to estimate the Laplace equation. Plotting Greens function based
        on the random walk results and evaluating the potential of a given location on the grid."""

    def __init__(self, n_samples, limits, dim, seed=10000, start_pos=None, potentials=None,
                 poisson=None):
        """Initialising the parent Monte Carlo class. Defining the starting position on the grid,
        potentials of the up, right, down, left edges of the grid, and defining the poisson
        equation. """
        super().__init__(n_samples, limits, dim, seed)
        self.start_pos = start_pos
        self.potentials = potentials
        self.poisson_function = poisson

    def __main_random_walk(self):
        """the main MC random walk function that calls for other functions. Finds the number of
            data points to be analysed by that worker,and finds the end positions, the error, and
            poisson evaluation. The means and variances are found using a parent function."""
        n_walks_per_worker = super().n_data_points()
        end_positions, random_values, poisson_eval = self.__random_walk(self.start_pos,
                                                                        n_walks_per_worker)
        potentials = self.__random_walk_evaluate_potential(end_positions)

        return end_positions, random_values, potentials, n_walks_per_worker, poisson_eval

    def __random_walk(self, start, n_walks):
        """The random walk function for a starting position of the grid, for n- walks.
        The end positions of where the n- random walks reach the limits of the grid are
        put onto the end_pos list to be returned from the function. The randomly created
        values determining the direction the walk moves in is added onto the random_values
        list, for finding an error. The poisson evaluation array is also initialised.
        The n-walks are looped through. If the current position reaches any of the limits,
        the position is added onto the end_positions list. A random value based on the
        seed is created between 0 and 1,which determines which direction the walk moves in.
        The poisson function is evaluated at each step of the random walk as well. """
        end_pos = []
        seed = self.child_ss[self.rank]
        rng = np.random.default_rng(seed)
        poisson_eval = []
        random_values = []
        for _ in range(n_walks):
            current_pos = [start[0], start[1]]
            while True:
                if self.poisson_function is not None:
                    poisson_eval.append(self.poisson_function(current_pos))
                if current_pos[0] > self.limits[1] or current_pos[1] > self.limits[1] or \
                        current_pos[0] < self.limits[0] or current_pos[1] < self.limits[0]:
                    break
                rand_point = rng.uniform(0, 1)
                random_values.append(rand_point)
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
            end_pos.append(current_pos)

        poisson_eval = np.mean(poisson_eval)
        return end_pos, random_values, poisson_eval

    def __random_walk_plot_green(self, pos_and_prob, error):
        """For the end positions and their probabilities stored in the pos_and_prob array,
        greens function is plotted with an error, using two different methods. """
        size = 300
        marker = 's'

        plt.figure()
        plt.scatter(self.start_pos[0], self.start_pos[1], c='w', s=size, edgecolors='black')
        plt.scatter(pos_and_prob[:, 0], pos_and_prob[:, 1], c=pos_and_prob[:, 2], s=size,
                    marker=marker, cmap='Reds')
        plt.grid()
        cbar = plt.colorbar()
        cbar.set_label('Probability')
        plt.xlim(self.limits[0] - 0.5, self.limits[1] + 0.5)
        plt.ylim(self.limits[0] - 0.5, self.limits[1] + 0.5)
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.xticks(np.linspace(self.limits[0], self.limits[1],
                               np.int(self.limits[0] + self.limits[1] + 1)))
        plt.yticks(np.linspace(self.limits[0], self.limits[1],
                               np.int(self.limits[0] + self.limits[1] + 1)))
        error = np.format_float_scientific(error, precision=2)
        plt.title(f'Greens function, start position {self.start_pos}, std {error}')
        plt.show()

        plt.figure()
        left = []
        right = []
        bottom = []
        top = []
        for i in range(np.size(pos_and_prob[:, 0])):
            if pos_and_prob[i][0] < self.limits[0]:
                left.append([pos_and_prob[i][1], pos_and_prob[i][2]])
            if pos_and_prob[i][0] > self.limits[1]:
                right.append([pos_and_prob[i][1], pos_and_prob[i][2]])
            if pos_and_prob[i][1] < self.limits[0]:
                bottom.append([pos_and_prob[i][0], pos_and_prob[i][2]])
            if pos_and_prob[i][1] > self.limits[1]:
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
        plt.xticks(np.linspace(self.limits[0], self.limits[1], np.int(self.limits[0] +
                                                                      self.limits[1] + 1)))
        plt.xlabel('axis of respective boundary in legend cm')
        plt.ylabel('probability')
        plt.title(f'Greens function, start position {self.start_pos}, std {error}')
        plt.grid()
        plt.legend()
        plt.show()

    def __random_walk_evaluate_potential(self, positions):
        """Evaluating the potential for given end positions of the walk.
        The number of unique end positions, and their probabilities, are put on
        a list. If the walk has reached the top side, the top edge potential is added
        to the total potential with its probability weight. This is repeated until all
        end positions have been looped through."""
        potential = 0
        unique, counts = np.unique(positions, return_counts=True, axis=0)
        pos_and_prob = np.column_stack((unique, counts / np.sum(counts)))

        for i in range(np.size(pos_and_prob, 0)):
            x_coords = pos_and_prob[i][0]
            y_coords = pos_and_prob[i][1]
            prob = pos_and_prob[i][2]

            if y_coords > self.limits[1]:
                potential += prob * self.potentials[0]
                ##up
            if x_coords > self.limits[1]:
                ##right
                potential += prob * self.potentials[1]
            if y_coords < self.limits[0]:
                ##down
                potential += prob * self.potentials[2]
            if x_coords < self.limits[0]:
                ##left
                potential += prob * self.potentials[3]

        return potential

    def random_walk_running_in_parallel(self):
        """Function that manages running the MC random walk in parallel. Gathers the end
        positions of the walks, the errors, potentials, number of data points per worker,
        and poisson function evaluations. Finds these in parallel. Prints out the final
        results of the random walk evaluation."""
        start_time = time.time()

        end_positions, random_vals, potentials, d_points, poisson_evals = self.__main_random_walk()
        pos_average = np.mean(end_positions, axis=0)
        pos_variance = np.var(end_positions, axis=0)

        positions = np.asarray(self.comm.reduce(end_positions, MPI.SUM, root=0))
        position_means = np.asarray(self.comm.gather(pos_average, root=0), dtype=float)
        position_variances = np.asarray(self.comm.gather(pos_variance, root=0), dtype=float)

        potentials = np.asarray(self.comm.gather(potentials, root=0), dtype=float)
        d_points = np.asarray(self.comm.gather(d_points, root=0), dtype=float)
        poisson_eval = np.asarray(self.comm.gather(poisson_evals, root=0))
        random_vals = np.asarray(self.comm.reduce(random_vals, MPI.SUM, root=0))

        if self.rank == 0:
            unique, counts = np.unique(positions, return_counts=True, axis=0)
            pos_and_prob = np.column_stack((unique, counts / np.sum(counts)))
            error = np.std(random_vals) / np.sqrt(np.size(random_vals))
            position_mean, position_variance = super().parallel_mean_and_variance(position_means,
                                                        position_variances, d_points)
            time_taken = time.time() - start_time
            self.__random_walk_plot_green(pos_and_prob, error)

            super().print_init_info()
            print('potential boundaries: ', self.potentials)
            print('starting position:', self.start_pos)
            print('potential', np.format_float_scientific(np.mean(potentials), precision=2), '+-',
                  np.format_float_scientific(np.mean(potentials) * error, precision=2))
            print('position mean:', np.round(position_mean, 4))
            print('position variance:', np.round(position_variance, 4), '\nposition std:',
                  np.round(np.sqrt(position_variance), 4))
            if self.poisson_function is not None:
                print('poisson expectation value (',
                      np.format_float_scientific(np.mean(poisson_eval), precision=2),
                      '+-', np.format_float_scientific(np.mean(poisson_eval) * error, precision=2),
                      ') /e0')

            print('time taken %.2f' % time_taken)
            print('\n\n')


def run_random_walk_mc():
    """The random walk function, defining the initial conditions of the grid random walk.
    Defining the different poisson functions in the assignment for evaluation. """
    dimensions = 2
    seed = 35
    # in cm
    limits = np.array((0, 10))
    n_datapoints = 10000
    start_pos = np.array((5,5))
    # up, right, down, left
    potentials = [2, -4, 0, 2]

    permittivity = 1  # 8.8542*10**-12
    # 1, 2 or 3 in accordance to assignment question
    poisson_task = 3

    def poisson_func(location):
        """Defining the poisson functions from the assignment sheet for evaluation."""
        if poisson_task == 1:
            charge_density = 1 / 0.1 ** 2

        if poisson_task == 2:
            ycoord = location[1]
            charge_density = 0.1 * ycoord / (0.1 ** 2)

        if poisson_task == 3:
            radius = np.sqrt(((5-location[0])*0.1)**2 + ((5-location[1])*0.1)**2)
            charge_density = -2 * np.exp(-2 * np.abs(radius))

        evaluation = -charge_density / permittivity
        return evaluation

    monte = RandomWalk(n_datapoints, limits, dimensions, seed=seed, start_pos=start_pos,
                       potentials=potentials, poisson=poisson_func)
    monte.random_walk_running_in_parallel()


#run_gaussian_integral_mc()
run_random_walk_mc()

# MPI.Finalize()