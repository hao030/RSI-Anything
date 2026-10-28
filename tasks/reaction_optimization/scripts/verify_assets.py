"""Verify the released prediction asset and evaluation-record consistency."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suzuki_inference import load_prior


def main() -> None:
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        data = (ROOT / name).read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Prediction asset identity changed: {name}")
    asset = json.loads((ROOT / "assets/suzuki_prior.json").read_text())
    prior = load_prior()
    results = json.loads((ROOT / "results/evaluation.json").read_text())
    if prior.candidate_count != manifest["candidate_count"] or prior.candidate_count != results["candidate_count"]:
        raise ValueError("Candidate count differs between release records")
    if asset["source_calibrated_prior_npz_sha256"] != results["provenance"]["calibrated_prior_npz_sha256"]:
        raise ValueError("Source calibrated-prior identity differs")
    if asset["source_calibrated_prior_npz_sha256"] != manifest["source_calibrated_prior_npz_sha256"]:
        raise ValueError("Manifest source identity differs")
    if prior.model_id != results["model_id"]:
        raise ValueError("Model identity differs")
    if any("yield" in candidate or "observed_yield" in candidate for candidate in asset["candidates"]):
        raise ValueError("Prediction asset contains measured-yield fields")
    context = {tuple(row[field] for field in ("electrophile", "nucleophile", "catalyst")) for row in prior.conditions}
    if context != {("1c, 6-OTf-Q", "2b, Boronic Ester", "Pd(OAc)2")}:
        raise ValueError("Released reaction context differs")
    first_hits = results["first_optimum_observation"]["per_run"]
    if len(first_hits) != results["runs"] or max(first_hits) != results["first_optimum_observation"]["maximum"]:
        raise ValueError("Reported first-hit observation counts differ")
    if abs(sum(first_hits) / len(first_hits) - results["first_optimum_observation"]["mean"]) > 1e-12:
        raise ValueError("Reported mean first-hit observation differs")
    if sum(hit <= 10 for hit in first_hits) != results["optimum_found_by_observation_10"]["runs"]:
        raise ValueError("Reported optimum-discovery count differs")
    print(json.dumps({"status": "pass", "candidate_count": prior.candidate_count, "prediction_checksum_verified": True}))


if __name__ == "__main__":
    main()
