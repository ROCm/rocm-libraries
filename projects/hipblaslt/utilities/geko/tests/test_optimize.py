# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for optimize.py CLI interface.

Usage:
    python3 -m pytest tests/test_optimize.py \\
        --hipblaslt-path /path/to/hipblaslt \\
        --workload /path/to/workload.yaml \\
        [--hw gfx942]

    ``--hw`` sets ``configure.py --architecture`` (default: gfx950).

    Hint: add -rs to see skip reasons, or -v for verbose output.

    ``TestPipelineFast`` exercises ``geko.pipeline.run_optimize`` with Tensile/GPU
    work mocked; it runs in CI without hipBLASLt. ``TestIntegration`` runs configure
    then real optimize (long-running; needs a built hipBLASLt and GPU).
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from geko import pipeline as pipeline_mod
from geko.pipeline import run_optimize

ROOT = Path(__file__).resolve().parents[1]


# Configure step matches test_configure.py; optimize can take much longer (Tensile / GA).
_CONFIGURE_TIMEOUT_S = 600
_OPTIMIZE_TIMEOUT_S = 7200


def _patch_optimize_config(path, n_gen_target=10, pop_size_target=128):
    """Rewrite one generated optimization config with faster GA settings.

    Config format is stable: Backend -> Config -> {n_gen, pop_size}.
    """
    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return

    backend_cfg = data.setdefault("Backend", {}).setdefault("Config", {})
    backend_cfg["n_gen"] = n_gen_target
    backend_cfg["pop_size"] = pop_size_target

    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Fast pipeline test (mocked optim / merge / analyze)
# ---------------------------------------------------------------------------

class TestPipelineFast:
    """Call ``run_optimize`` with heavy steps stubbed; no real tuning or benchmarks."""

    def test_main_completes_with_mocks(self, tmp_path):
        """Full control-flow: validation, state flags, mocked run → merge → analyze."""
        hip = tmp_path / "hipblaslt_stub"
        hip.mkdir()
        workdir = tmp_path / "workdir"
        (workdir / "optimizations").mkdir(parents=True)

        from geko.schemas import RunState

        dummy_input = tmp_path / "workload_stub.yaml"
        dummy_input.write_text("# stub\n")
        state = RunState.create(dummy_input)
        state.configured = True
        state.dump(workdir / "run_state.json")

        merged = MagicMock()

        def _dump_libs(target):
            p = Path(target)
            p.mkdir(parents=True, exist_ok=True)
            (p / "merged_stub.yaml").write_text(
                "library_name: stub\nlibrary_type: tensile\nsolutions: []\nsizes: []\n"
            )

        merged.dump.side_effect = _dump_libs
        mock_run = MagicMock()
        mock_analyze = MagicMock(
            return_value=(
                pd.DataFrame([{"dummy": 1}]),
                pd.DataFrame([{"dummy": 1, "winner": "reference", "lib": "merged_stub.yaml"}]),
            )
        )
        mock_merge = MagicMock(return_value=merged)
        final_lib = MagicMock()
        full_lib = MagicMock()
        mock_from_dataframe = MagicMock(return_value=final_lib)
        mock_from_full_dataframe = MagicMock(return_value=full_lib)

        with patch.object(pipeline_mod.optim, "run", mock_run), patch.object(
            pipeline_mod.optim.utils,
            "check_progress",
            side_effect=[(1, 1, 0), (1, 1, 0)],
        ), patch.object(pipeline_mod.library, "merge_solutions", mock_merge), patch.object(
            pipeline_mod.library,
            "from_dataframe",
            mock_from_dataframe,
        ), patch.object(
            pipeline_mod.library,
            "from_full_dataframe",
            mock_from_full_dataframe,
        ), patch.object(
            pipeline_mod.optim,
            "analyze",
            mock_analyze,
        ):
            run_optimize(str(hip), workdir=str(workdir), devices=[0], verbose=0)

        mock_run.assert_called_once()
        ca = mock_run.call_args
        assert ca.args[0] == Path(hip)
        assert ca.args[1] == workdir / "optimizations"
        assert ca.kwargs["devices"] == [0]

        mock_merge.assert_called_once()
        mock_analyze.assert_called_once()
        mock_from_dataframe.assert_called_once()
        mock_from_full_dataframe.assert_called_once()
        final_lib.dump.assert_called_once_with(workdir / "final_libs")
        full_lib.dump.assert_called_once_with(workdir / "full_libs")

        with (workdir / "run_state.json").open() as f:
            final_state = json.load(f)
        assert final_state["configured"] is True
        assert final_state["optimized"] is True


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntegration:
    """Run configure.py then optimize.py against a real workload and hipBLASLt repo.

    Usage:
        python3 -m pytest tests/test_optimize.py \\
            --hipblaslt-path ~/rocm-libraries/projects/hipblaslt \\
            --workload tests/test_data/workload.yaml \\
            [--hw gfx942]
    """

    def test_full_cli_run(self, hipblaslt_path, workload_path, hw_arch, tmp_path):
        """Configure then optimize subprocesses complete (uses conftest CLI options)."""
        if hipblaslt_path is None or workload_path is None:
            msg = "Skipped: requires --hipblaslt-path and --workload CLI options"
            warnings.warn(msg, stacklevel=1)
            pytest.skip(msg)

        hip = Path(hipblaslt_path)
        workload = Path(workload_path)
        if not hip.is_dir():
            msg = f"Skipped: hipBLASLt path not found: {hip}"
            warnings.warn(msg, stacklevel=1)
            pytest.skip(msg)
        if not workload.is_file():
            msg = f"Skipped: workload file not found: {workload}"
            warnings.warn(msg, stacklevel=1)
            pytest.skip(msg)

        workdir = tmp_path / "workdir"
        client_build = (tmp_path / "build_tmp").resolve()
        run_cwd = str(tmp_path.resolve())
        # Absolute paths so subprocess cwd=tmp_path does not break relative CLI args.
        configure_py = str((ROOT / "scripts" / "configure.py").resolve())
        optimize_py = str((ROOT / "scripts" / "optimize.py").resolve())
        hip_abs = str(hip.resolve())
        workload_abs = str(workload.resolve())
        workdir_abs = str(workdir.resolve())

        cfg = subprocess.run(
            [
                sys.executable,
                configure_py,
                workload_abs,
                "--hipblaslt",
                hip_abs,
                "--workdir",
                workdir_abs,
                "--architecture",
                hw_arch,
            ],
            cwd=run_cwd,
            capture_output=True,
            text=True,
            timeout=_CONFIGURE_TIMEOUT_S,
        )
        assert cfg.returncode == 0, (
            f"configure.py exited with code {cfg.returncode}\n"
            f"--- stdout ---\n{cfg.stdout[-2000:]}\n"
            f"--- stderr ---\n{cfg.stderr[-2000:]}"
        )
        assert workdir.is_dir()
        assert (workdir / "optimizations").is_dir()
        
        config_files = pipeline_mod.optim.utils.list_optimization_configs(workdir / "optimizations")
        assert len(config_files) > 0, "Expected config files but workdir is empty"
        for f in config_files:
            # Patch config so it runs faster, independent of original defaults.
            _patch_optimize_config(Path(f), n_gen_target=10, pop_size_target=128)

        opt = subprocess.run(
            [
                sys.executable,
                optimize_py,
                "--hipblaslt",
                hip_abs,
                "--workdir",
                workdir_abs,
                "--client_build_dir",
                str(client_build),
                "--devices",
                "0",
                "--n_slots",
                "1",
            ],
            cwd=run_cwd,
            capture_output=True,
            text=True,
            timeout=_OPTIMIZE_TIMEOUT_S,
        )
        assert opt.returncode == 0, (
            f"optimize.py exited with code {opt.returncode}\n"
            f"--- stdout ---\n{opt.stdout[-2000:]}\n"
            f"--- stderr ---\n{opt.stderr[-2000:]}"
        )

        state_path = workdir / "run_state.json"
        assert state_path.is_file()
        with state_path.open() as f:
            state = json.load(f)
        assert state.get("configured") is True
        assert state.get("optimized") is True

        generated = list(workdir.rglob("*"))
        assert len(generated) > 0, "Expected output files but workdir is empty"
