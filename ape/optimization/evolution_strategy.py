from typing import Callable
import numpy as np
from tqdm import tqdm

from ape._typing import Array, Matrix


class EvolutionStrategy:
    """Optimization by Evolution Strategy (ES)."""

    def _generate(self) -> Array:
        raise NotImplementedError

    def _update(self, samples: Array, fitnesses: Array):
        raise NotImplementedError

    def estimate(self):
        """Estimate the local minimum after optimization."""
        raise NotImplementedError

    def optimize(self, func: Callable, iterations: int = 1000) -> list[Array]:
        """Search for local minimum.

        Computes the local minimum by applying the Evolution Strategy based on
        population sampling. Population is sampled from multivariate normal
        distribution whose parameters are updated in each iteration.

        Args:
            func: Function to be optimized.
            iterations: Desired number of iterations (i.e. updates in the algorithm).
        
        Returns:
            A list with the historical population that was used in the algorithm.
        """
        history_samples : list[Array] = []
        for _ in tqdm(range(iterations)):
            samples = self._generate()
            history_samples.append(samples)
            fitnesses = np.zeros(np.size(samples, 0))
            for j in range(np.size(samples, 0)):
                fitnesses[j] = func(samples[j, :])
            self._update(samples, fitnesses)
        return history_samples


class NaiveEvolutionStrategy(EvolutionStrategy):
    """Evolution Strategy by naive covariance and mean update."""

    def __init__(self, μ: int, λ: int, n: int, mean: Array | None = None, cov: Matrix|None = None):
        self.μ = μ
        self.λ = λ
        self.mean : Array = mean if mean is not None else np.zeros((n,))
        self.cov : Matrix = cov if cov is not None else np.identity(n)

    def _generate(self) -> Array:
        samples = np.random.multivariate_normal(self.mean, self.cov, self.λ)
        return samples

    def _update(self, samples: Array, fitnesses: Array):
        indices = np.argsort(fitnesses)
        best_indices = indices[:self.μ]
        best_samples = samples[best_indices, :]

        centered : Matrix = best_samples - self.mean
        self.cov = 1/self.μ * (centered.T @ centered)
        self.mean : Array = 1/self.μ * np.sum(best_samples, axis=0)

    def estimate(self) -> tuple[Array, Matrix]:
        return self.mean, self.cov
