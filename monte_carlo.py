import random
import math
import numpy as np
import matplotlib.pyplot as plt


class MC_Simulation:
    def __init__(self, func):
        self.func = func

    def run(self, iterations, **kwargs):
        sum = 0.0
        for i in range(iterations):
            outcome = self.func(**kwargs)
            sum += outcome
        return sum / iterations


def mc_european(**kwargs):
    s = kwargs.get("s")
    k = kwargs.get("k")
    r = kwargs.get("r")
    sigma = kwargs.get("sigma")
    T = kwargs.get("T")
    u = random.normalvariate(0, 1)
    future_value = s * math.exp(
        (r - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * u
    )
    if kwargs["option_type"].upper() == "CALL":
        option_value = math.exp(-r * T) * max(future_value - k, 0.0)
    elif kwargs["option_type"].upper() == "PUT":
        option_value = math.exp(-r * T) * max(k - future_value, 0.0)
    return option_value


def mc_asian(**kwargs):

    option_type = kwargs.get("option_type")
    timesteps = kwargs.get("timesteps")
    s = kwargs.get("s")
    k = kwargs.get("k")
    r = kwargs.get("r")
    sigma = kwargs.get("sigma")
    T = kwargs.get("T")
    if option_type.upper() == "CALL":
        sign = 1
    elif option_type.upper() == "PUT":
        sign = -1
    else:
        return None

    asset_sum = s
    prev_val = s
    for j in range(timesteps):
        rn = random.normalvariate(0, 1)
        next_val = prev_val * math.exp(
            (r - 0.5 * sigma**2) * T / timesteps
            + sigma * math.sqrt(T / timesteps) * rn
        )
        asset_sum += next_val
        prev_val = next_val
    payoff = max(sign * (asset_sum / (timesteps + 1) - k), 0.0)

    return payoff * math.exp(-r * T)
