from pathlib import Path

from scripts.run_e2e_smoke import run_project


def test_e2e_project_path_is_absolute(tmp_path):
    project = Path(run_project(tmp_path / "nested" / "run", "train"))

    assert project.is_absolute()
    assert project.name == "train"
    assert project.parent.name == "run"
