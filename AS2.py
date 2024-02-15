import random
import numpy as np


def monte_carlo(function, N, lim_l, lim_u):

    integral = 0
    integral_squared = 0
    for i in range(N):
        xrand = random.uniform(lim_l, lim_u)
        point = function(xrand)
        integral += point
        integral_squared += point**2

    integral_estim = (lim_u-lim_l)/N * integral
    variance = 1/N * ((lim_u-lim_l)/N * integral_squared - integral_estim**2)
    #var should not be negative..
    return integral_estim, variance


def func(x):
    return np.sin(x)


print(monte_carlo(func, 10000, 0, np.pi))
