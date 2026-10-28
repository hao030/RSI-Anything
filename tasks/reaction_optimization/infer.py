"""Command-line entry point for Suzuki condition selection."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from suzuki_inference.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
