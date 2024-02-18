import random
import numpy as np


class MonteCarloIntegrationParallel:

    def __init__(self, N, lim_l, lim_u, function):
        self.N = N
        self.lim_l = lim_l
        self.lim_u = lim_u
        self.function = function

    def monte_carlo(self):

        integral = 0
        integral_squared = 0
        for i in range(self.N):
            xrand = random.uniform(self.lim_l, self.lim_u)
            point = self.function(xrand)
            integral += point
            integral_squared += point**2

        integral_estim = (self.lim_u-self.lim_l)/self.N * integral

        variance = 1/self.N * (integral_squared/self.N - (integral/self.N)**2)

        return integral_estim, variance


def func(x):
    return np.sin(x)


sin_monte = MonteCarloIntegrationParallel(100, 0, np.pi, func)
print(sin_monte.monte_carlo())
