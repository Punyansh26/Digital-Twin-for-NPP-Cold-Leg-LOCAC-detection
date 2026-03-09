"""
DeepONet Base — re-exports the original DeepONet architecture.

Acts as a stable import target for the multi-fidelity module and other
components that need to reference the original (RANS-fidelity) operator.
"""

# Expose original classes under new canonical names so callers can write:
#   from src.deeponet.deeponet_base import DeepONetBase
from src.deeponet.model import DeepONet as DeepONetBase  # noqa: F401
from src.deeponet.model import BranchNet, TrunkNet, DeepONetLoss  # noqa: F401

__all__ = ["DeepONetBase", "BranchNet", "TrunkNet", "DeepONetLoss"]
