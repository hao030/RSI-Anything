from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    expected: tuple[Path, ...] = ()
    retry_outputs: tuple[Path, ...] = ()

    def as_dict(self):
        return dict(name=self.name, command=list(self.command), cwd=str(self.cwd),
                    env=self.env, expected=[str(p) for p in self.expected],
                    retry_outputs=[str(p) for p in self.retry_outputs])


@dataclass(frozen=True)
class Plan:
    stages: tuple[Stage, ...]
    # Files rendered into the run directory before launching any worker.
    files: dict[str, dict] = field(default_factory=dict)
