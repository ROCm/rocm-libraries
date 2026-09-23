# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for log/workload-driven config (SIZE_OPTION 2, GEMM_LOG_PATH)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from geko.config_generator.load_input_config import (
    apply_input_config_defaults,
    gemm_configs_from_gemm_dataframe,
    gemm_configs_from_gemm_log_path,
    load_prepared_config_from_yaml,
    validate_input_config,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_load_prepared_requires_arch_for_gemm_log_mode() -> None:
    wf = _repo_root() / "tests" / "test_data" / "workload.yaml"
    with pytest.raises(ValueError, match="No ARCH defined"):
        load_prepared_config_from_yaml(None, gemm_log_path=str(wf))


def test_validate_log_mode_minimal() -> None:
    """Log-driven mode skips TRANSA/dtypes; ARCH is validated before defaults."""
    wf = _repo_root() / "tests" / "test_data" / "workload.yaml"
    assert wf.is_file()
    cfg = {
        "ARCH": "gfx950",
        "GEMM_LOG_PATH": str(wf),
        "SIZE_OPTION": 2,
    }
    validate_input_config(cfg)


def test_validate_size_option_two_requires_path() -> None:
    with pytest.raises(ValueError, match="SIZE_OPTION 2 requires GEMM_LOG_PATH"):
        validate_input_config({"ARCH": "gfx950", "SIZE_OPTION": 2})


def test_gemm_configs_missing_log_file() -> None:
    """Parser raises when the workload file path does not exist."""
    missing = _repo_root() / "nonexistent_workload.yaml"
    assert not missing.is_file()
    with pytest.raises(FileNotFoundError):
        gemm_configs_from_gemm_log_path(missing)


def test_gemm_configs_from_workload_yaml() -> None:
    wf = _repo_root() / "tests" / "test_data" / "workload.yaml"
    gcs = gemm_configs_from_gemm_log_path(wf)
    assert len(gcs) >= 1
    assert all(len(gc.sizes) >= 1 for gc in gcs)


def test_apply_defaults_log_mode_roundtrip() -> None:
    """Defaults fill tuning keys; GemmProblems are not set by apply_input_config_defaults alone."""
    wf = _repo_root() / "tests" / "test_data" / "workload.yaml"
    cfg = {"ARCH": "gfx950", "GEMM_LOG_PATH": str(wf), "SIZE_OPTION": 2}
    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    assert "search_space" in cfg
    assert cfg.get("GemmProblems") is None


def test_validate_rejects_unknown_arch() -> None:
    with pytest.raises(ValueError, match="Unknown ARCH"):
        validate_input_config(
            {
                "ARCH": "gfx000",
                "TRANSA": "N",
                "TRANSB": "N",
                "DataType": "B",
                "DestDataType": "B",
                "ComputeDataType": "S",
            }
        )


def test_apply_defaults_macro_tile_opt_tensile_size_option_zero_allowed() -> None:
    cfg = {
        "ARCH": "gfx950",
        "TRANSA": "N",
        "TRANSB": "N",
        "DataType": "B",
        "DestDataType": "B",
        "ComputeDataType": "S",
        "MACROTILE_OPT": True,
        "backend": "tensile",
    }
    validate_input_config(cfg)
    apply_input_config_defaults(cfg)


def test_apply_defaults_macro_tile_opt_tensile_nonzero_size_option_rejected() -> None:
    cfg = {
        "ARCH": "gfx950",
        "TRANSA": "N",
        "TRANSB": "N",
        "DataType": "B",
        "DestDataType": "B",
        "ComputeDataType": "S",
        "MACROTILE_OPT": True,
        "backend": "tensile",
        "SIZE_OPTION": 1,
    }
    validate_input_config(cfg)
    with pytest.raises(NotImplementedError, match="only supported for SIZE_OPTION=0"):
        apply_input_config_defaults(cfg)


def test_apply_defaults_sets_non_ga_kernel_cap_and_mt_du_none() -> None:
    cfg = {
        "ARCH": "gfx950",
        "TRANSA": "N",
        "TRANSB": "N",
        "DataType": "B",
        "DestDataType": "B",
        "ComputeDataType": "S",
        "backend": "tensile",
        "MACROTILE_OPT": False,
    }
    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    assert cfg["MT_DU"] is None
    assert "MAX_NUM_KERNELS_PER_CONFIG" in cfg


def test_apply_defaults_env_overrides_and_invalid_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {
        "ARCH": "gfx950",
        "TRANSA": "N",
        "TRANSB": "N",
        "DataType": "B",
        "DestDataType": "B",
        "ComputeDataType": "S",
        "backend": "tensile",
    }
    validate_input_config(cfg)

    monkeypatch.setenv("StreamK", "false")
    monkeypatch.setenv("MI_FILTER", "7")
    monkeypatch.setenv("DUCTILE_VALIDATION_PROFILE", "bad-int")
    apply_input_config_defaults(cfg)

    assert cfg["StreamK"] is False
    assert cfg["MI_FILTER"] == 7


def test_gemm_configs_from_dataframe_empty_and_lowercase_cols() -> None:
    assert gemm_configs_from_gemm_dataframe(pd.DataFrame()) == []

    df = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "compute_type": "f32_r",
                "m": 16,
                "n": 16,
                "batch_count": 1,
                "k": 16,
            }
        ]
    )
    out = gemm_configs_from_gemm_dataframe(df)
    assert len(out) == 1
    assert out[0].sizes[0] == [16, 16, 1, 16]


def test_load_prepared_with_arch_and_gemm_log_path_and_empty_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wf = tmp_path / "w.yaml"
    wf.write_text("[]\n")

    monkeypatch.setattr("geko.config_generator.load_input_config.gemm_configs_from_gemm_log_path", lambda _p: [])
    with pytest.raises(ValueError, match="No GEMM entries found"):
        load_prepared_config_from_yaml(None, arch="gfx950", gemm_log_path=wf)


def test_load_prepared_non_log_mode_populates_gemm_problem(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    yaml.safe_dump(
        {
            "ARCH": "gfx950",
            "TRANSA": "N",
            "TRANSB": "N",
            "DataType": "B",
            "DestDataType": "B",
            "ComputeDataType": "S",
            "SIZE_OPTION": 0,
            "Sizes": [[32, 32, 1, 32]],
        },
        cfg_path.open("w"),
        sort_keys=False,
    )

    cfg = load_prepared_config_from_yaml(cfg_path)
    assert "GemmProblems" in cfg
    assert len(cfg["GemmProblems"]) == 1


@pytest.mark.parametrize("dt", ["C", "Z"])
def test_heuristic_rejected_for_complex_dtype(dt: str) -> None:
    cfg = {
        "ARCH": "gfx942",
        "TRANSA": "N",
        "TRANSB": "N",
        "DataType": dt,
        "DestDataType": dt,
        "ComputeDataType": "S" if dt == "C" else "Z",
        "search_space": "heuristic",
    }
    validate_input_config(cfg)
    with pytest.raises(NotImplementedError, match="Heuristic search space"):
        apply_input_config_defaults(cfg)


def test_heuristic_rejected_for_complex_gemm_problems() -> None:
    from geko.schemas import GemmConfig, GemmType

    gt = GemmType.from_tensile("N", "N", "C", "C", "C")
    cfg = {
        "ARCH": "gfx942",
        "search_space": "heuristic",
        "GemmProblems": [GemmConfig(gt, [[64, 64, 1, 64]])],
    }
    with pytest.raises(NotImplementedError, match="Heuristic search space"):
        apply_input_config_defaults(cfg)
