"""Inference subset of JEPA-Anything's orthogonal factor projection.

Adapted from JEPA-Anything, licensed under Apache-2.0.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn

def validate_factorization(state_dim: int, num_factors: int, factor_dim: int) -> None:
    """Validate the complete-state contract ``state_dim = num_factors * factor_dim``.

    Args:
        state_dim: Width of the complete latent state.
        num_factors: Number of predictive coordinate groups, ``K``.
        factor_dim: Width of each group, ``r``.

    Raises:
        ValueError: If a dimension is non-positive or ``K * r != d``.
    """

    dimensions = {
        "state_dim": state_dim,
        "num_factors": num_factors,
        "factor_dim": factor_dim,
    }
    invalid_types = [
        name
        for name, value in dimensions.items()
        if isinstance(value, bool) or not isinstance(value, int)
    ]
    if invalid_types:
        raise TypeError(f"dimensions must be integers; invalid: {', '.join(invalid_types)}")
    invalid = [name for name, value in dimensions.items() if value <= 0]
    if invalid:
        raise ValueError(f"dimensions must be positive; invalid: {', '.join(invalid)}")
    if num_factors * factor_dim != state_dim:
        raise ValueError(
            "complete OPF state requires num_factors * factor_dim == state_dim "
            f"({num_factors} * {factor_dim} != {state_dim})"
        )

def _validate_basis_shape(basis: Tensor) -> tuple[int, int, int]:
    if basis.ndim != 3:
        raise ValueError(f"basis must have shape (K, r, d), got {tuple(basis.shape)}")
    num_factors, factor_dim, state_dim = basis.shape
    validate_factorization(state_dim, num_factors, factor_dim)
    if not basis.is_floating_point():
        raise TypeError("basis must be floating point")
    return num_factors, factor_dim, state_dim

def _validate_same_device_dtype(value: Tensor, basis: Tensor, name: str) -> None:
    if value.device != basis.device:
        raise ValueError(f"{name} and basis must be on the same device")
    if value.dtype != basis.dtype:
        raise ValueError(f"{name} and basis must have the same dtype")

def decompose_state(state: Tensor, basis: Tensor) -> Tensor:
    """Analyze complete states into predictive factor coordinates.

    Args:
        state: Tensor with trailing shape ``(d,)``.
        basis: Analysis rows with shape ``(K, r, d)``.

    Returns:
        Tensor with trailing shape ``(K, r)``.
    """

    _, _, state_dim = _validate_basis_shape(basis)
    if state.ndim < 1 or state.shape[-1] != state_dim:
        raise ValueError(f"state must end in dimension {state_dim}, got {tuple(state.shape)}")
    _validate_same_device_dtype(state, basis, "state")
    return torch.einsum("...d,krd->...kr", state, basis)

class OrthogonalFactorProjection(nn.Module):
    """Load a frozen OPF analysis basis and expose predictive coordinates."""

    def __init__(self, state_dim: int, num_factors: int, factor_dim: int):
        super().__init__()
        validate_factorization(state_dim, num_factors, factor_dim)
        self.state_dim = state_dim
        self.num_factors = num_factors
        self.factor_dim = factor_dim
        basis = torch.eye(state_dim).reshape(num_factors, factor_dim, state_dim)
        self.raw_basis = nn.Parameter(basis, requires_grad=False)

    def analysis_basis(self) -> Tensor:
        return self.raw_basis

    def decompose(self, state: Tensor) -> Tensor:
        return decompose_state(state, self.analysis_basis())

    def forward(self, state: Tensor) -> Tensor:
        return self.decompose(state)
