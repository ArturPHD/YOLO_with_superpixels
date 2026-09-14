import json

from yolo_superpixels.integrations import run_config


def test_source_tree_metadata_hashes_only_project_sources(tmp_path):
    root = tmp_path / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "runs").mkdir(parents=True)
    (root / ".venv").mkdir(parents=True)
    (root / "third_party" / "SPiT").mkdir(parents=True)
    (root / "src" / "pkg" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (root / "runs" / "result.txt").write_text("ignored\n", encoding="utf-8")
    (root / ".venv" / "site.py").write_text("ignored\n", encoding="utf-8")
    (root / "third_party" / "SPiT" / "external.py").write_text("ignored\n", encoding="utf-8")

    first = run_config.source_tree_metadata(root)
    (root / "runs" / "result.txt").write_text("changed but ignored\n", encoding="utf-8")
    ignored_change = run_config.source_tree_metadata(root)
    (root / "src" / "pkg" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    source_change = run_config.source_tree_metadata(root)

    assert first["files"] == 2
    assert first["sha256"] == ignored_change["sha256"]
    assert first["sha256"] != source_change["sha256"]


def test_superpixel_run_config_includes_source_tree_fingerprint(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "model.py").write_text("x = 1\n", encoding="utf-8")
    weights = root / "spit_weights.pt"
    weights.write_bytes(b"fake-spit-weights")
    monkeypatch.setattr(run_config, "_repo_root", lambda: root)

    run_config.write_superpixel_run_config(
        tmp_path / "run",
        "neck_injection",
        {"token_provider": "spit", "spit_weights": str(weights)},
    )

    payload = json.loads((tmp_path / "run" / "superpixel_config.json").read_text(encoding="utf-8"))
    assert payload["source_tree"]["files"] == 1
    assert len(payload["source_tree"]["sha256"]) == 64
    assert payload["external_files"]["spit_weights"]["sha256"] == run_config.file_metadata(weights)["sha256"]
    assert payload["config"] == {"token_provider": "spit", "spit_weights": str(weights)}


def test_superpixel_run_config_hashes_base_model_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "model.py").write_text("x = 1\n", encoding="utf-8")
    checkpoint = root / "yolo12n.pt"
    checkpoint.write_bytes(b"pretrained-yolo-weights")
    monkeypatch.setattr(run_config, "_repo_root", lambda: root)

    run_config.write_superpixel_run_config(
        tmp_path / "run",
        "structural_supervision",
        {"base_model_path": str(checkpoint)},
    )

    payload = json.loads((tmp_path / "run" / "superpixel_config.json").read_text(encoding="utf-8"))
    assert payload["external_files"]["base_model_path"] == run_config.file_metadata(checkpoint)


def test_spit_metadata_hashes_external_checkout(tmp_path):
    root = tmp_path / "repo"
    spit = root / "third_party" / "SPiT"
    (spit / ".git").mkdir(parents=True)
    (spit / ".git" / "HEAD").write_text("abc123\n", encoding="utf-8")
    (spit / "spit").mkdir()
    (spit / "spit" / "tokenizer.py").write_text("VALUE = 1\n", encoding="utf-8")

    first = run_config._spit_metadata(root)
    (spit / ".git" / "HEAD").write_text("ignored-git-change\n", encoding="utf-8")
    git_change = run_config._spit_metadata(root)
    (spit / "spit" / "tokenizer.py").write_text("VALUE = 2\n", encoding="utf-8")
    source_change = run_config._spit_metadata(root)

    assert len(first["source_tree"]["sha256"]) == 64
    assert first["source_tree"]["sha256"] == git_change["source_tree"]["sha256"]
    assert first["source_tree"]["sha256"] != source_change["source_tree"]["sha256"]
