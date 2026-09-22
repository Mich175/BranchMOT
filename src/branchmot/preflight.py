"""Preflight checks for a reproducible MOTIP + DanceTrack experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreflightItem:
    name: str
    path: Path
    exists: bool


@dataclass(frozen=True)
class PreflightReport:
    items: tuple[PreflightItem, ...]

    @property
    def ready(self) -> bool:
        return all(item.exists for item in self.items)

    def format(self) -> str:
        lines = ["BranchMOT experiment preflight"]
        for item in self.items:
            marker = "OK" if item.exists else "MISSING"
            lines.append(f"[{marker:7}] {item.name}: {item.path}")
        lines.append("READY" if self.ready else "NOT READY")
        return "\n".join(lines)


def check_experiment(
    *,
    motip_root: str | Path,
    data_root: str | Path,
    checkpoint: str | Path,
    config: str | Path | None = None,
    split: str = "val",
) -> PreflightReport:
    """Check the minimum inputs needed for a MOTIP DanceTrack export."""

    motip = Path(motip_root)
    data = Path(data_root) / "DanceTrack"
    checkpoint_path = Path(checkpoint)
    config_path = (
        Path(config)
        if config is not None
        else motip / "configs" / "r50_deformable_detr_motip_dancetrack.yaml"
    )
    required = (
        ("MOTIP runtime", motip / "models" / "runtime_tracker.py"),
        ("MOTIP entrypoint", motip / "submit_and_evaluate.py"),
        ("MOTIP config", config_path),
        ("MOTIP checkpoint", checkpoint_path),
        (f"DanceTrack {split} split", data / split),
        (f"DanceTrack {split} seqmap", data / f"{split}_seqmap.txt"),
    )
    return PreflightReport(
        tuple(PreflightItem(name, path, path.exists()) for name, path in required)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motip-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    report = check_experiment(
        motip_root=args.motip_root,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        config=args.config,
        split=args.split,
    )
    print(report.format())
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())

