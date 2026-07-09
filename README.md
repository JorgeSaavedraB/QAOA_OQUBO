<h1 align="center"> OQUBO: Optimization-to-QUBO for Quantum Algorithms </h1>


**OQUBO** is a Python library for transforming constrained binary optimization problems into Quadratic Unconstrained Binary Optimization (QUBO) and Ising Hamiltonian formulations. The repository also provides tools for constructing optimization models, evaluating quantum optimization algorithms, and reproducing the numerical experiments presented in our work.

The project is built around DOcplex models and provides an end-to-end workflow:
- Formulate an optimization problem in DOcplex.
- Convert the constrained problem into a QUBO formulation.
- Transform the **QUBO** into an Ising Hamiltonian.
- Solve or analyze the resulting Hamiltonian using classical or quantum optimization techniques.

## Features:
- Automatic conversion from DOcplex models to QUBO formulations.
- Conversion from QUBO to Ising Hamiltonians.
- Support for both equality constraints and inequality constraints through:
- Standard quadratic penalties.
- Unbalanced Penalization (UP).
- Optional coefficient normalization.

Utilities for:
- Exact brute-force energy evaluation.
- Expectation value computation from measurement counts.
- Warm-start parameter generation.
- Solution feasibility verification.
- Implementations of benchmark optimization problems:
- Portfolio Optimization (PO).
- Maximum Covering Location Problem (MCLP).

## Repository Structure:
```text
.
├── src
│   ├── problems
│   │   ├── Portfolio Optimization model
│   │   └── Maximum Covering Location Problem model
│   │
│   └── utils
│       ├── DOcplex → QUBO conversion
│       ├── QUBO → Ising conversion
│       ├── CPLEX helper functions
│       ├── Brute-force solvers
│       └── Auxiliary utilities
│
├── results
│   ├── Portfolio Optimization experiments
│   └── MCLP experiments
│
└── README.md
```

## Experimental Benchmarks:

The repository contains scripts used to evaluate several QAOA-based optimization strategies on the Portfolio Optimization and Maximum Covering Location Problem benchmarks.

The implemented approaches include:

- Standard QAOA
- QAOA with Linear Ramp (LR) parameter initialization
- Warm-Start (WS) QAOA
- Warm-Start + Linear Ramp
- Unbalanced Penalization (UP)
- Combinations of UP with WS and LR

The experimental scripts report quantities such as:

- Approximation ratio
- Ground-state probability
- Feasible solution rate
-  Energy expectation values
- Number of optimizer evaluations
-  Circuit statistics
-  Execution times

The project relies on the following main libraries:

-  Python 3.x
-  NumPy
-  DOcplex
-  IBM CPLEX
-  Qiskit
-  Qiskit Aer
-  SciPy
-  pandas
-  yfinance
-  matplotlib

## Quick Start

The following example illustrates the basic workflow for generating a QUBO and its corresponding Ising Hamiltonian from a DOcplex optimization model.

```python
from src.problems.PortfolioOptimization import portfolio_optimization
from src.utils.OQUBO import DocplexModeltoQUBO, QUBOtoIsingModel

# Build a DOcplex optimization model
model = portfolio_optimization(
    mu=mu,
    sigma=sigma,
    risk=0.5,
    budget=5
)

# Generate the QUBO formulation
qubo, offset = DocplexModeltoQUBO(model)

# Convert the QUBO into an Ising Hamiltonian
ising_model, ising_offset = QUBOtoIsingModel(model)

print(qubo)
print(ising_model)
```

## References:

If you use this repository in your research, please consider citing the following works.

### Unbalanced Penalization

Montañez et al., Unbalanced penalization: a new approach to encode inequality constraints of combinatorial problems for quantum optimization algorithms, Quantum Science and Technology (2024). DOI: [10.1088/2058-9565/ad35e4](https://iopscience.iop.org/article/10.1088/2058-9565/ad35e4)

### Warm-Start QAOA

Egger, et al., Warm-starting quantum optimization, Quantum 5, 479 (2021). DOI: [10.22331/q-2021-06-17-479](https://quantum-journal.org/papers/q-2021-06-17-479/)

## License:

This repository is intended for academic and research use.
