# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""High-level workflow tests for config_generator.

Sections (pytest markers — see ``tests/conftest.py``):

* ``cg_integration`` — CLI subprocess and in-process ``run_config_generator`` with artifact checks.
* ``cg_cli_guard`` — Hermetic validation of ``config_generator.main()`` arguments.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from geko.config_generator.config_generator import run
from geko.config_generator.load_input_config import load_prepared_config_from_yaml

from tests.config_generator.artifact_assertions import assert_output_artifacts, gemm_type_string


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_cg_main_module(repo_root: Path):
    """Load scripts/config_generator.py as a module for main() tests."""
    spec = importlib.util.spec_from_file_location(
        "config_generator_cli",
        repo_root / "scripts" / "config_generator.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_variant(base_path: Path, overrides: dict, dest_dir: Path) -> Path:
    """Load *base_path* YAML, apply *overrides*, write a new file under *dest_dir*."""
    with open(base_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    label = "_".join(f"{k}={v}" for k, v in sorted(overrides.items())) or "base"
    out = dest_dir / f"{base_path.stem}_{label}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None, sort_keys=False)
    return out


def _effective_gemm_type(cfg_path: Path) -> str:
    """Return gemm type string after load_prepared_config_from_yaml."""
    raw = load_prepared_config_from_yaml(cfg_path)
    return gemm_type_string(raw)


# Small YAML variant matrix (not a full Cartesian sweep).
_VARIANT_OVERRIDES = [
    {},
    {"StreamK": False},
]


# ---------------------------------------------------------------------------
# Integration: CLI + API + artifacts
# ---------------------------------------------------------------------------


@pytest.mark.cg_integration
@pytest.mark.slow
class TestConfigGeneratorIntegration:
    """Full generator runs; requires ``--config`` and ``--hipblaslt-path``."""

    def test_cli_produces_valid_artifacts(
        self,
        config_path: str | None,
        hipblaslt_path: str | None,
        repo_root: Path,
        tmp_path: Path,
    ) -> None:
        """Run config_generator.py subprocess per variant; assert YAML/shell layout."""
        if not config_path or not hipblaslt_path:
            pytest.skip("Requires --config and --hipblaslt-path")
        base = Path(config_path)
        if not base.is_file():
            pytest.skip(f"Config not found: {base}")
        if not Path(hipblaslt_path).is_dir():
            pytest.skip(f"hipBLASLt path not found: {hipblaslt_path}")

        script = repo_root / "scripts" / "config_generator.py"
        hip = str(Path(hipblaslt_path).resolve())  # resolve before cwd changes
        for overrides in _VARIANT_OVERRIDES:
            variant = _write_variant(base, overrides, tmp_path)
            out_dir = tmp_path / f"cli_out_{variant.stem}"
            gemm = _effective_gemm_type(variant)

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--hipblaslt",
                    hip,
                    "--config",
                    str(variant),
                    "-o",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=900,
                cwd=str(repo_root),
            )
            assert result.returncode == 0, (
                f"CLI failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
            assert_output_artifacts(str(out_dir), gemm)

    def test_run_config_generator_produces_valid_artifacts(
        self,
        config_path: str | None,
        hipblaslt_path: str | None,
        tmp_path: Path,
    ) -> None:
        """Call run_config_generator in-process for each variant; same artifact assertions as CLI."""
        if not config_path or not hipblaslt_path:
            pytest.skip("Requires --config and --hipblaslt-path")
        base = Path(config_path)
        if not base.is_file():
            pytest.skip(f"Config not found: {base}")
        if not Path(hipblaslt_path).is_dir():
            pytest.skip(f"hipBLASLt path not found: {hipblaslt_path}")

        for overrides in _VARIANT_OVERRIDES:
            variant = _write_variant(base, overrides, tmp_path)
            out_dir = tmp_path / f"api_out_{variant.stem}"
            gemm = _effective_gemm_type(variant)
            out_dir.mkdir(parents=True, exist_ok=True)

            cfg = load_prepared_config_from_yaml(variant)
            run(cfg, hipblaslt_path, out_dir)
            assert_output_artifacts(str(out_dir), gemm)

    def test_run_config_generator_without_shell_scripts(
        self,
        config_path: str | None,
        hipblaslt_path: str | None,
        tmp_path: Path,
    ) -> None:
        """write_shell_scripts=False yields YAML + log only (no .sh or run-all)."""
        if not config_path or not hipblaslt_path:
            pytest.skip("Requires --config and --hipblaslt-path")
        base = Path(config_path)
        if not base.is_file():
            pytest.skip(f"Config not found: {base}")
        if not Path(hipblaslt_path).is_dir():
            pytest.skip(f"hipBLASLt path not found: {hipblaslt_path}")

        variant = _write_variant(base, {}, tmp_path)
        out_dir = tmp_path / f"api_no_sh_{variant.stem}"
        gemm = _effective_gemm_type(variant)
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = load_prepared_config_from_yaml(variant)
        run(
            cfg, hipblaslt_path, out_dir, write_shell_scripts=False
        )
        assert_output_artifacts(str(out_dir), gemm, expect_shell_scripts=False)


# ---------------------------------------------------------------------------
# CLI guardrails (hermetic)
# ---------------------------------------------------------------------------


@pytest.mark.cg_cli_guard
class TestConfigGeneratorMainGuardrails:
    """``config_generator.main()`` rejects bad paths before touching hipBLASLt."""

    def test_main_missing_config_file(self, repo_root: Path, tmp_path: Path) -> None:
        """Raises FileNotFoundError when the YAML path is missing."""
        cg = _load_cg_main_module(repo_root)
        hip = tmp_path / "hip"
        hip.mkdir()
        missing = tmp_path / "nope.yaml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            cg.main(
                str(hip),
                output_path=str(tmp_path / "out"),
                config=str(missing),
            )

    def test_main_missing_hipblaslt_dir(self, repo_root: Path, tmp_path: Path) -> None:
        """Raises FileNotFoundError when hipBLASLt path is not a directory."""
        cg = _load_cg_main_module(repo_root)
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("k: v\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="hipBLASLt path not found"):
            cg.main(
                str(tmp_path / "not_a_dir"),
                output_path=str(tmp_path / "out"),
                config=str(cfg),
            )

    def test_main_requires_config_or_log_flags(self, repo_root: Path, tmp_path: Path) -> None:
        """Raises ValueError when neither tuning YAML nor arch+log path pair is given."""
        cg = _load_cg_main_module(repo_root)
        hip = tmp_path / "hip"
        hip.mkdir()
        with pytest.raises(ValueError, match="Provide --config"):
            cg.main(str(hip), config=None, arch=None, gemm_log_path=None)
