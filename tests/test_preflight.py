from pathlib import Path

from branchmot.preflight import check_experiment


def test_preflight_reports_complete_layout(tmp_path: Path) -> None:
    motip = tmp_path / "MOTIP"
    data = tmp_path / "datasets"
    checkpoint = tmp_path / "weights" / "motip.pth"
    paths = [
        motip / "models" / "runtime_tracker.py",
        motip / "submit_and_evaluate.py",
        motip / "configs" / "r50_deformable_detr_motip_dancetrack.yaml",
        checkpoint,
        data / "DanceTrack" / "val_seqmap.txt",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (data / "DanceTrack" / "val").mkdir()

    report = check_experiment(
        motip_root=motip, data_root=data, checkpoint=checkpoint
    )
    assert report.ready
    assert report.format().endswith("READY")


def test_preflight_lists_missing_inputs(tmp_path: Path) -> None:
    report = check_experiment(
        motip_root=tmp_path / "MOTIP",
        data_root=tmp_path / "data",
        checkpoint=tmp_path / "missing.pth",
    )
    assert not report.ready
    assert report.format().count("MISSING") == 6
