# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""scripts/run_pipeline.sh: the checkout roots it exports."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.sh"
ORCHESTRATOR = SCRIPT.parent.parent / "run_orchestrator.py"


def launch(pipeline_dir, python="true", **env):
    """Run a copy of the launcher placed in `pipeline_dir`/scripts on
    `pipeline_dir`/cfg.yaml in full mode, with `python` as the interpreter."""
    scripts = Path(pipeline_dir) / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(SCRIPT, scripts / SCRIPT.name)
    base = {k: v for k, v in os.environ.items() if not k.startswith("TILEWRIGHT_")}
    return subprocess.run(
        [
            "bash",
            str(scripts / SCRIPT.name),
            str(Path(pipeline_dir) / "cfg.yaml"),
            "--mode",
            "full",
        ],
        env={**base, "PYTHON": str(python), **env},
        capture_output=True,
        text=True,
        timeout=120,
    )


def env_printer(tmp_path):
    """An interpreter stand-in that prints the environment it gets."""
    path = tmp_path / "print_env"
    path.write_text("#!/bin/sh\nenv\n")
    path.chmod(0o755)
    return path


def test_outside_a_checkout_the_roots_stay_unset(tmp_path):
    proc = launch(
        tmp_path / "standalone" / "training_pipeline", python=env_printer(tmp_path)
    )
    assert proc.returncode == 0, proc.stderr
    assert "bench root  : unset\n" in proc.stdout
    assert "deploy root : unset\n" in proc.stdout
    assert "TILEWRIGHT_BENCH_ROOT=" not in proc.stdout
    assert "TILEWRIGHT_DEPLOY_ROOT=" not in proc.stdout


def test_the_orchestrator_names_the_root_a_config_refers_to(tmp_path):
    pipeline = tmp_path / "standalone" / "training_pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / ORCHESTRATOR.name).symlink_to(ORCHESTRATOR)
    (pipeline / "cfg.yaml").write_text(
        "config_id: c\narch: gfx950\n"
        "runtime:\n  rocm_libraries_root: ${TILEWRIGHT_BENCH_ROOT}\n"
    )
    proc = launch(pipeline, python=sys.executable)
    assert proc.returncode == 2
    assert "environment variable(s): TILEWRIGHT_BENCH_ROOT\n" in proc.stderr
    assert "TILEWRIGHT_DEPLOY_ROOT" not in proc.stderr


def test_the_roots_come_from_the_environment(tmp_path):
    bench, deploy = tmp_path / "bench", tmp_path / "deploy"
    proc = launch(
        tmp_path / "standalone" / "training_pipeline",
        TILEWRIGHT_BENCH_ROOT=str(bench),
        TILEWRIGHT_DEPLOY_ROOT=str(deploy),
    )
    assert proc.returncode == 0, proc.stderr
    assert f"bench root  : {bench} (" in proc.stdout
    assert f"deploy root : {deploy} (" in proc.stdout


def test_inside_a_checkout_the_roots_default_to_it(tmp_path):
    root = tmp_path / "rocm-libraries"
    deploy = tmp_path / "deploy"
    proc = launch(
        root / "shared" / "tilewright" / "training_pipeline",
        python=env_printer(tmp_path),
        TILEWRIGHT_DEPLOY_ROOT=str(deploy),
    )
    assert proc.returncode == 0, proc.stderr
    assert f"bench root  : {root} (" in proc.stdout
    assert f"deploy root : {deploy} (" in proc.stdout
    assert f"TILEWRIGHT_BENCH_ROOT={root}\n" in proc.stdout
    assert f"TILEWRIGHT_DEPLOY_ROOT={deploy}\n" in proc.stdout
