import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class BlackScholes:
    def __init__(self, S, K, rf, sigma, T, tipo_opzione, **kwargs):
        self.S = S
        self.K = K
        self.rf = rf
        self.sigma = sigma
        self.T = T
        self.tipo_opzione = tipo_opzione
        if "q" in kwargs.keys():
            self.q = kwargs["q"]
        else:
            self.q = 0
        if "phi" in kwargs.keys():
            self.phi = kwargs["phi"]
            self.interval = kwargs["interval"]
        else:
            self.phi = None

    def get_option_price(self):
        if self.phi:
            self.sigma = self.sigma * (
                (
                    1
                    + (
                        np.sqrt(2 / np.pi)
                        * self.phi
                        / (self.sigma * np.sqrt(self.interval))
                    )
                )
                ** 0.5
            )

        d1 = (
            np.log(self.S / self.K)
            + (self.rf - self.q + (self.sigma**2 / 2) * self.T)
        ) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.tipo_opzione == "Call":
            prezz_1 = self.S * norm.cdf(d1, 0, 1)
            if self.q:
                prezz_1 *= np.exp(-self.q * self.T)
            prezzo_opzione = prezz_1 - self.K * np.exp(
                -self.rf * self.T
            ) * norm.cdf(d2, 0, 1)
            
        elif self.tipo_opzione == "Put":
            prezz_2 = self.S * norm.cdf(-d1, 0, 1)
            if self.q:
                prezz_2 *= np.exp(-self.q * self.T)
            prezzo_opzione = self.K * np.exp(-self.rf * self.T) * norm.cdf(
                -d2, 0, 1
            ) - prezz_2
        return prezzo_opzione
