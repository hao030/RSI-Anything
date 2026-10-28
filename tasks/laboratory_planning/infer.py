"""Run JEPA-guided laboratory planning or select existing candidate plans."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lab_inference.cli import main


if __name__ == "__main__":
    main()
