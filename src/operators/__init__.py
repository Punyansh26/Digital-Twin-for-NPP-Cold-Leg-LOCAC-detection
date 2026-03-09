# Plug-in neural operator architectures
__version__ = "2.0.0"

from .transolver_operator import TransolverOperator
from .clifford_operator   import CliffordNeuralOperator

__all__ = ["TransolverOperator", "CliffordNeuralOperator"]
