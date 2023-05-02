# ape-ai: a Python framework for AI

## What is it?

*ape-ai* is a Python package that provides implementations of common algorithms in artificial intelligence. This package is a personal project, aimed at providing a simple and easy-to-use library for AI enthusiasts, students, and researchers to explore and experiment with various AI techniques. Additionally, this project serves as a learning platform for the developer to study object-oriented programming patterns and testing libraries.

## Installation

To install *ape-ai*, it's not possible to use pip yet.

## Usage

*ape-ai* currently supports the following algorithms:

- Evolution Strategy (ES)

Here's an example of how to use the ES algorithm to optimize a simple function:

```python
import numpy as np
from ape.optimization import NaiveEvolutionStrategy

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

print('estimated minimum:', mean, flush=True)
```

## Contributing

Contributions are welcome! If you want to contribute to *ape-ai*, please fork the repository and submit a pull request. Make sure to follow the contributing guidelines.
