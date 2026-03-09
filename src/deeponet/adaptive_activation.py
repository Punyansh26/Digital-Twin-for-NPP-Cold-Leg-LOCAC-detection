"""
Adaptive Activation Functions for DeepONet.

Implements trainable-slope GELU activations that allow the network to adapt
its activation function independently during training.

    f(x) = GELU(a * x)

where ``a`` is a learnable scalar (or per-neuron vector) initialized to 1.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGELU(nn.Module):
    """
    Adaptive GELU activation with learnable slope.

    f(x) = GELU(a * x)

    The slope ``a`` is initialized to ``init_a`` and optimized together with
    the rest of the network.  A per-neuron parameter vector is used when
    ``n_units`` is specified, otherwise a single shared scalar is used.

    Args:
        n_units:  Number of neurons (one ``a`` per neuron).
                  Pass ``None`` to use a single shared scalar.
        init_a:   Initial value of the slope (default 1.0).
    """

    def __init__(self, n_units: int = None, init_a: float = 1.0) -> None:
        super().__init__()
        if n_units is not None:
            self.a = nn.Parameter(torch.ones(n_units) * init_a)
        else:
            self.a = nn.Parameter(torch.tensor(float(init_a)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.a * x)


class AdaptiveActivationLayer(nn.Module):
    """
    Linear layer followed by an adaptive GELU activation.

    Drop-in replacement for ``nn.Linear`` + activation in the trunk network.
    The adaptive slope is per-output-neuron.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        bias:         Whether to add a bias term (default True).
        init_a:       Initial activation slope (default 1.0).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_a: float = 1.0,
    ) -> None:
        super().__init__()
        self.linear     = nn.Linear(in_features, out_features, bias=bias)
        self.activation = AdaptiveGELU(n_units=out_features, init_a=init_a)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="linear")
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

    def extra_repr(self) -> str:
        return (
            f"in={self.linear.in_features}, "
            f"out={self.linear.out_features}"
        )
