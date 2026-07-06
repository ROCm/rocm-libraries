# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the dnn-benchmarking setup script."""

import os
import subprocess
from pathlib import Path


SETUP_SCRIPT = Path(__file__).resolve().parents[3] / "setup.sh"


def test_setup_script_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SETUP_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_setup_script_rejects_force_build_with_cuda_mode() -> None:
    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--torch-mode", "cuda", "--force-build"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--force-build is not supported with --torch-mode cuda" in result.stderr


def test_setup_script_rejects_unknown_torch_mode() -> None:
    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--torch-mode", "bogus"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "rocm, cuda, cpu, existing, none" in result.stderr


def test_setup_script_rejects_force_build_with_existing_cuda_venv(
    tmp_path: Path,
) -> None:
    # Reaches the post-activation guard (setup.sh:784), distinct from the early
    # --torch-mode cuda arg check. Build a fake "existing" venv whose
    # interpreter reports a CUDA torch build; --torch-mode existing forces venv
    # reuse, so the dir survives instead of being recreated.
    workspace = tmp_path / "ws"
    venv_bin = workspace / ".venv" / "bin"
    venv_bin.mkdir(parents=True)

    # Stub interpreter: drain the heredoc on stdin and report CUDA torch.
    # Covers both `python3` (version check) and `python` (get_torch_mode).
    stub = "#!/usr/bin/env bash\ncat >/dev/null 2>&1 || true\necho cuda\n"
    for name in ("python", "python3"):
        interp = venv_bin / name
        interp.write_text(stub)
        interp.chmod(0o755)

    activate = venv_bin / "activate"
    activate.write_text(f'export PATH="{venv_bin}:$PATH"\n')

    env = dict(os.environ)
    env["DNN_BENCH_WORKSPACE"] = str(workspace)
    env["PATH"] = f"{venv_bin}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--torch-mode", "existing", "--force-build", "-y"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        input="",
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "existing CUDA torch venv" in result.stderr
