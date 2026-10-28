"""Load a portable Suzuki prediction table without experimental yield labels."""

from dataclasses import dataclass
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np

DEFAULT_PRIOR = Path(__file__).resolve().parents[2] / "assets" / "suzuki_prior.json"
CONDITION_FIELDS = ("electrophile", "nucleophile", "catalyst", "ligand", "base", "solvent")


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class SuzukiPrior:
    """Prediction arrays and their ordered condition catalog."""

    mean: np.ndarray
    std: np.ndarray
    original_table_indices: tuple[int, ...]
    conditions: tuple[dict[str, str], ...]
    model_id: str
    xi: float

    @property
    def candidate_count(self) -> int:
        return len(self.mean)


def load_prior(path: str | Path = DEFAULT_PRIOR) -> SuzukiPrior:
    """Validate and load schema version 1; IDs are zero-based and contiguous."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Prior must be an object with schema_version 1")
    candidates = document.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Prior candidates must be a nonempty list")
    acquisition = document.get("acquisition")
    if not isinstance(acquisition, dict) or acquisition.get("method") != "expected_improvement":
        raise ValueError("Prior acquisition must use expected_improvement")
    xi = finite_number(acquisition.get("xi_percent_points"), "xi_percent_points")
    if xi < 0:
        raise ValueError("xi_percent_points must be nonnegative")
    model_id = document.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Prior model_id must be a nonempty string")

    means, deviations, source_ids, conditions = [], [], [], []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate {index} must be an object")
        if integer(candidate.get("candidate_index"), "candidate_index") != index:
            raise ValueError("Candidate IDs must be ordered from zero without gaps")
        source_index = integer(candidate.get("original_table_index"), "original_table_index")
        if source_index < 0:
            raise ValueError("original_table_index must be nonnegative")
        mean = finite_number(candidate.get("predicted_yield_percent"), "predicted_yield_percent")
        deviation = finite_number(candidate.get("calibrated_std_percent_points"), "calibrated_std_percent_points")
        if not 0 <= mean <= 100 or deviation <= 0:
            raise ValueError("Prior yields must be in [0, 100] and deviations must be positive")
        condition = candidate.get("conditions")
        if not isinstance(condition, dict) or any(
            not isinstance(condition.get(field), str) or not condition[field].strip()
            for field in CONDITION_FIELDS
        ):
            raise ValueError(f"Candidate {index} requires six named condition fields")
        means.append(mean)
        deviations.append(deviation)
        source_ids.append(source_index)
        conditions.append({field: condition[field] for field in CONDITION_FIELDS})
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("original_table_index values must be unique")
    mean_array, std_array = np.asarray(means, dtype=np.float64), np.asarray(deviations, dtype=np.float64)
    mean_array.flags.writeable = std_array.flags.writeable = False
    return SuzukiPrior(mean_array, std_array, tuple(source_ids), tuple(conditions), model_id, xi)
