from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load_benchmark_module():
    path = ROOT / "scripts" / "run_coco_manual_benchmark.py"
    spec = spec_from_file_location("run_coco_manual_benchmark", path)
    assert spec is not None
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_args(tmp_path):
    return SimpleNamespace(
        data="datasets/coco/data.yaml",
        models=["yolo12n.pt", "yolo26n.pt"],
        epochs=70,
        imgsz=640,
        batch=16,
        workers=4,
        device="0",
        run_root=tmp_path,
        stage=["all"],
        dry_run=True,
        rerun=False,
        exist_ok=True,
        input_algorithm="torch_slic",
        input_channel_mode="adjacency",
        input_control_algorithm=["color_affinity", "torch_grid", "torch_voronoi"],
        no_rgb_baseline=False,
        init_extra="tiny",
        edge_representation="zero_centered",
        swap_algorithm=["torch_slic", "color_affinity"],
        swap_checkpoints=["best"],
        include_spit_dense_swap=False,
        neck_fusion=["add", "concat", "cross_attention"],
        neck_token_provider="region",
        neck_region_algorithm="torch_slic",
        neck_image_size=160,
        allow_random_spit=False,
        spit_weights=None,
        aux_algorithm="torch_slic",
        aux_loss_weight=0.05,
        n_segments=300,
        torch_slic_compactness=1.0,
        torch_slic_iters=7,
        soft_max_distance=32,
        soft_boundary_width=6.0,
        soft_affinity_threshold=0.5,
        lr0=None,
        lrf=None,
        optimizer=None,
        freeze=None,
        cos_lr=False,
        patience=None,
        weight_decay=None,
        warmup_epochs=None,
        warmup_momentum=None,
        warmup_bias_lr=None,
    )


def test_manual_coco_benchmark_builds_expected_transfer_learning_jobs(tmp_path):
    benchmark = load_benchmark_module()
    args = make_args(tmp_path)

    jobs = benchmark.build_training_jobs(args)

    assert len(jobs) == 18
    assert [job.stage for job in jobs].count("rgb_baseline") == 2
    assert [job.stage for job in jobs].count("input") == 8
    assert [job.stage for job in jobs].count("neck") == 6
    assert [job.stage for job in jobs].count("auxiliary") == 2
    assert {job.model for job in jobs} == {"yolo12n.pt", "yolo26n.pt"}
    assert {job.variant for job in jobs if job.stage == "input"} == {
        "torch_slic_adjacency",
        "color_affinity_adjacency",
        "torch_grid_adjacency",
        "torch_voronoi_adjacency",
    }
    assert {job.variant for job in jobs if job.stage == "neck"} == {
        "add_region_torch_slic",
        "concat_region_torch_slic",
        "cross_attention_region_torch_slic",
    }

    for job in jobs:
        model_index = job.command.index("--model") + 1
        assert job.command[model_index].endswith(".pt")
        assert job.command[job.command.index("--epochs") + 1] == "70"
    input_jobs = [job for job in jobs if job.stage == "input"]
    assert all(job.command[job.command.index("--edge-representation") + 1] == "zero_centered" for job in input_jobs)
    assert all(job.command[job.command.index("--channel-mode") + 1] == "adjacency" for job in input_jobs)


def test_manual_coco_benchmark_swap_job_uses_trained_input_checkpoints(tmp_path):
    benchmark = load_benchmark_module()
    args = make_args(tmp_path)

    job = benchmark.build_swap_job(args)

    checkpoint_values = [
        value for index, value in enumerate(job.command) if index > 0 and job.command[index - 1] == "--checkpoint"
    ]
    assert len(checkpoint_values) == 2
    assert all("input_" in value for value in checkpoint_values)
    assert any("yolo12n_adjacency_best" in value for value in checkpoint_values)
    assert any("yolo26n_adjacency_best" in value for value in checkpoint_values)
    assert job.command[job.command.index("--channel-mode") + 1] == "adjacency"
    assert job.command[job.command.index("--edge-representation") + 1] == "zero_centered"


def test_manual_coco_benchmark_can_disable_exist_ok(tmp_path):
    benchmark = load_benchmark_module()
    args = make_args(tmp_path)
    args.exist_ok = False

    jobs = benchmark.build_training_jobs(args)

    assert jobs
    assert all("--exist-ok" not in job.command for job in jobs)


def test_manual_coco_benchmark_gpu_device_parser(tmp_path):
    benchmark = load_benchmark_module()

    assert benchmark.parse_gpu_device_ids("cpu") is None
    assert benchmark.parse_gpu_device_ids(None) is None
    assert benchmark.parse_gpu_device_ids("0") == "0"
    assert benchmark.parse_gpu_device_ids("cuda:0") == "0"
    assert benchmark.parse_gpu_device_ids("0,1") == "0,1"


def test_manual_coco_benchmark_completion_requires_marker_and_requested_epoch(tmp_path):
    benchmark = load_benchmark_module()
    run_dir = tmp_path / "run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"fake")
    (run_dir / "results.csv").write_text("epoch,metrics/mAP50-95(B)\n1,0.1\n", encoding="utf-8")

    job = benchmark.BenchmarkJob(
        stage="input",
        model="yolo12n.pt",
        variant="torch_slic",
        command=["python", "train.py"],
        run_dir=run_dir,
        log_path=tmp_path / "run.log",
    )

    assert benchmark.COMPLETION_MARKER in benchmark.train_completion_problem(job, requested_epochs=70)

    marker = {
        "status": "ok",
        "requested_epochs": 70,
        "last_epoch": 1,
        "command_hash": benchmark.command_hash(job.command),
        "source_tree_sha256": benchmark.source_tree_metadata()["sha256"],
        "external_fingerprint": benchmark.external_fingerprint(),
    }
    benchmark.completion_marker_path(run_dir).write_text(
        benchmark.json.dumps(marker),
        encoding="utf-8",
    )

    assert "last epoch 1" in benchmark.train_completion_problem(job, requested_epochs=70)
