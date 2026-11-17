# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests of Invoke build tasks that require the source checkout."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


class _RecordingContext:
    """An invoke context that records commands instead of running them."""

    def __init__(self):
        self.commands: list[str] = []

    def run(self, command, **kwargs):
        self.commands.append(command)


def _stub_compiler_checks(tasks, monkeypatch, root):
    checked = []

    def compiler_check(command, **kwargs):
        assert command[1:] == ["--version"]
        assert Path(command[0]).parent == root / "bin"
        assert kwargs["env"]["ROCM_PATH"] == str(root)
        checked.append(Path(command[0]).name)

    monkeypatch.setattr(tasks.subprocess, "run", compiler_check)
    monkeypatch.setenv("ROCM_PATH", str(root))
    return checked


def test_build_client_configures_for_the_target_and_not_the_features(monkeypatch, tmp_path):
    """The detection path itself, not just the helper: passing the probe's answer
    straight to -DGPU_TARGETS is what failed CI, and nothing about the helper
    existing prevents that on its own."""
    tasksPath = Path(__file__).resolve().parents[4] / "tasks.py"
    spec = importlib.util.spec_from_file_location("tensilelite_tasks", tasksPath)
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)

    from tensilelite import GpuArch

    root = tmp_path / "selected-rocm"
    checked = _stub_compiler_checks(tasks, monkeypatch, root)

    def detect(rocm_path):
        assert rocm_path == root
        return "gfx950:sramecc+:xnack-"

    monkeypatch.setattr(GpuArch, "detect_gpu_arch", detect)
    context = _RecordingContext()

    tasks.build_client.body(context, build=False, build_dir=str(tmp_path / "build"))

    configure = next(c for c in context.commands if c.startswith("cmake"))
    assert "-DGPU_TARGETS=gfx950" in configure
    assert "sramecc" not in configure
    assert checked == ["amdclang", "amdclang++"]


def test_build_client_strips_features_from_an_explicit_gpu_target(monkeypatch, tmp_path):
    """tox -e py3 forwards ``--gpu-targets $ARCH`` from get-gpu-arch; a probe
    that still names sramecc must not reach CMake as GPU_TARGETS."""
    tasksPath = Path(__file__).resolve().parents[4] / "tasks.py"
    spec = importlib.util.spec_from_file_location("tensilelite_tasks", tasksPath)
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)

    checked = _stub_compiler_checks(tasks, monkeypatch, tmp_path / "selected-rocm")
    context = _RecordingContext()
    tasks.build_client.body(
        context,
        build=False,
        build_dir=str(tmp_path / "build"),
        gpu_targets="gfx942:sramecc+:xnack-",
    )

    configure = next(c for c in context.commands if c.startswith("cmake"))
    assert "-DGPU_TARGETS=gfx942" in configure
    assert "sramecc" not in configure
    assert checked == ["amdclang", "amdclang++"]
