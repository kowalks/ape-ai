import numpy as np

from ape.optimization import NaiveEvolutionStrategy

def test_sample_code_evolution_strategy(tol=1e-6):
    # Cost function
    quadratic = lambda x: x[0]**2 + x[1]**2

    # Parameters
    μ = 12
    λ = μ//2
    iterations = 100

    # Algorithm
    es = NaiveEvolutionStrategy(μ, λ, n=2)
    es.optimize(func=quadratic, iterations=iterations)
    mean, _ = es.estimate()

    assert np.linalg.norm(mean) < tol
