import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
from math import sin
from typing import Callable

from ape._typing import Array, Matrix


class EvolutionStrategy:
    """Optimization by Evolution Strategy (ES)."""
    
    def _generate(self) -> Array:
        raise NotImplementedError

    def _update(self, samples: Array, fitnesses: Array):
        raise NotImplementedError

    def estimate(self):
        raise NotImplementedError
    
    def optimize(self, f: Callable, iterations: int = 1000) -> list[Array]:
        history_samples : list[Array] = []
        for _ in tqdm(range(iterations)):
            samples = self._generate()
            history_samples.append(samples)
            fitnesses = np.zeros(np.size(samples, 0))
            for j in range(np.size(samples, 0)):
                fitnesses[j] = f(samples[j, :])
            self._update(samples, fitnesses)
        return history_samples


class NaiveEvolutionStrategy(EvolutionStrategy):
    """Evolution Strategy by naive covariance and mean update."""

    def __init__(self, μ: int, λ: int, mean: Array = None, cov: Matrix = None, n: int = None):
        if (mean is None) and (cov is None) and (n is None):
            raise AttributeError('You must specify n or mean and cov')
        if mean is None:
            mean : Array = np.zeros((n,))
        if cov is None:
            cov : Matrix = np.identity(n) 

        self.μ = μ
        self.λ = λ
        self.mean = mean
        self.cov = cov

    def _generate(self) -> Array:
        samples = np.random.multivariate_normal(self.mean, self.cov, self.λ)
        return samples

    def _update(self, samples: Array, fitnesses: Array):
        indices = np.argsort(fitnesses)
        best_indices = indices[:self.μ]
        best_samples = samples[best_indices, :]
        
        centered = best_samples - self.mean
        self.cov = 1/self.μ * (centered.T @ centered)
        self.mean : Array = 1/self.μ * np.sum(best_samples, axis=0)
    
    def estimate(self) -> tuple[Array, Matrix]:
        return self.mean, self.cov
