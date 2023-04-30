import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import trange
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
    
    def optimize(self, f: Callable, iterations: int = 1000):
        history_samples = []
        for _ in trange(iterations):
            samples = self._generate()
            history_samples.append(samples)
            fitnesses = np.zeros(np.size(samples, 0))
            for j in range(np.size(samples, 0)):
                fitnesses[j] = f(samples[j, :])
            self._update(samples, fitnesses)
        return history_samples


class NaiveEvolutionStrategy(EvolutionStrategy):
    """Evolution Strategy by naive covariance and mean update."""

    def __init__(self, mean: Array, cov: Matrix, μ: int, λ: int):
        self.mean = mean
        self.cov = cov
        self.μ = μ
        self.λ = λ

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
