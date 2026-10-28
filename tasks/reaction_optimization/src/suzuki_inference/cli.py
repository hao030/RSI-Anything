"""Read measured yields and write the next recommended Suzuki condition."""

import argparse
import json
from pathlib import Path

from .prior import DEFAULT_PRIOR
from .selector import recommend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True, help="JSON list of candidate_index and measured yield (0–100)")
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR, help="Portable calibrated prediction table")
    parser.add_argument("--output", type=Path, help="New JSON file; defaults to stdout")
    args = parser.parse_args(argv)
    try:
        observations = json.loads(args.observations.read_text(encoding="utf-8"))
        result = recommend(observations, args.prior)
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                output.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0
