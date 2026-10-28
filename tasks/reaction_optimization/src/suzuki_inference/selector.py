"""Sequential expected improvement using a frozen, calibrated JEPA prior."""

from pathlib import Path

import numpy as np
from scipy.special import ndtr

from .prior import DEFAULT_PRIOR, SuzukiPrior, finite_number, integer, load_prior


def expected_improvement(mean: np.ndarray, std: np.ndarray, best_yield: float, xi: float) -> np.ndarray:
    """Gaussian EI in yield percentage points; inputs use calibrated positive std."""
    gain = mean - best_yield - xi
    z = gain / std
    density = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return np.maximum(gain * ndtr(z) + std * density, 0.0)


def recommend(observations: list[dict], prior: SuzukiPrior | str | Path = DEFAULT_PRIOR) -> dict:
    """Return one unobserved condition, breaking EI ties by ascending candidate ID.

    Observations contain integer candidate_index and measured yield in 0–100.
    Feedback updates the incumbent and excludes already measured conditions.
    The prediction table stays frozen throughout the experiment.
    """
    if not isinstance(prior, SuzukiPrior):
        prior = load_prior(prior)
    if not isinstance(observations, list) or not observations:
        raise ValueError("Observations must be a nonempty list")
    ids, yields = [], []
    for row_number, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError(f"Observation {row_number} must be an object")
        candidate_id = integer(observation.get("candidate_index"), "candidate_index")
        observed_yield = finite_number(observation.get("yield"), "yield")
        if not 0 <= candidate_id < prior.candidate_count:
            raise ValueError(f"candidate_index must be in [0, {prior.candidate_count - 1}]")
        if not 0 <= observed_yield <= 100:
            raise ValueError("yield must be a percentage in [0, 100]")
        ids.append(candidate_id)
        yields.append(observed_yield)
    if len(set(ids)) != len(ids):
        raise ValueError("Observed candidate IDs must be unique")
    if len(ids) == prior.candidate_count:
        raise ValueError("All candidate conditions have been observed")
    scores = expected_improvement(prior.mean, prior.std, max(yields), prior.xi)
    scores[ids] = -np.inf
    # np.argmax chooses the first maximum in the ascending-ID prediction table.
    selected = int(np.argmax(scores))
    return {
        "candidate_index": selected,
        "original_table_index": prior.original_table_indices[selected],
        "conditions": dict(prior.conditions[selected]),
        "predicted_yield_percent": float(prior.mean[selected]),
        "calibrated_std_percent_points": float(prior.std[selected]),
        "expected_improvement": float(scores[selected]),
        "observations_used": len(ids),
        "best_observed_yield_percent": max(yields),
        "model_id": prior.model_id,
    }
