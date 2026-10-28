"""Command-line inference for laboratory plans."""
import argparse
import json
from pathlib import Path

FIELDS = {"id", "context", "goal", "constraints", "available_inputs", "action_pool"}


def validate_task(task):
    missing = FIELDS - task.keys()
    if missing:
        raise ValueError("Missing task fields: " + ", ".join(sorted(missing)))
    task = {name: task[name] for name in FIELDS}
    for name in FIELDS - {"constraints"}:
        if not isinstance(task[name], str):
            raise ValueError(f"{name} must be a string")
    if not isinstance(task["constraints"], list) or not all(isinstance(x, str) for x in task["constraints"]):
        raise ValueError("constraints must be a list of strings")
    return task


def requests(path):
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            task = validate_task(row.get("task", row))
            if task["id"] in seen:
                raise ValueError("Duplicate task identifier: " + task["id"])
            seen.add(task["id"])
            yield task, row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode, description in [("world", "Generate four plans and select with JEPA"),
                              ("select", "Select from four supplied plans with JEPA")]:
        command = modes.add_parser(mode, help=description)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if mode == "world":
            command.add_argument("--base-model", required=True, help="Model ID or local backbone folder")
            command.add_argument("--adapter", type=Path, required=True, help="Local policy adapter folder")
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
        command.add_argument("--world-dir", type=Path, required=True, help="JEPA checkpoint folder")
        command.add_argument("--include-factors", action="store_true")
        if mode == "world":
            command.add_argument("--seed", type=int, choices=[1401, 1402, 1403], default=1401)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    generator = None
    if args.mode == "world":
        from .generator import CandidateGenerator
        generator = CandidateGenerator(args.base_model, args.adapter, device=args.device, dtype=args.dtype)
    from .world import JEPAWorld
    world = JEPAWorld(args.world_dir)
    if args.mode == "world" and world.config["seed"] != args.seed:
        raise ValueError("Use a world checkpoint matching --seed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("x", encoding="utf-8") as output:
        for task, row in requests(args.input):
            candidates = generator.candidates(task, args.seed) if generator else row["candidates"]
            forecast = world.forecast(task, candidates, args.include_factors)
            result = {"id": task["id"], "completion": candidates[forecast["index"]]["completion"],
                      "seed": world.config["seed"], "selected_candidate_id": forecast["candidate_id"],
                      "candidates": candidates, "forecast": forecast}
            output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            count += 1
    print(json.dumps({"mode": args.mode, "tasks": count, "output": str(args.output)}))
