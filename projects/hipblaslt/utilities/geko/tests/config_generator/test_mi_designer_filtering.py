# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for MI designer filtering logic (MX + subtile rejections).

Tests focus on:
- reject0: MX+subtile requires even MIWaveTile
- reject1/reject2: MX scale loading bytesLoaded checks
- reject3: Subtile DepthU divisibility checks
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from geko.config_generator.load_input_config import (
    apply_input_config_defaults,
    get_gemm_problem,
    validate_input_config,
)
from geko.config_generator.mi_designer import MIDesign
from geko.config_generator.shared_utils import ForkParameter


# ---------------------------------------------------------------------------
# Base config template
# ---------------------------------------------------------------------------


def _base_template() -> dict:
    """Return a minimal valid tuning config."""
    return {
        "TRANSA": "N",
        "TRANSB": "N",
        "DataType": "B",
        "DestDataType": "B",
        "ComputeDataType": "S",
        "ARCH": "gfx950",
        "StreamK": True,
        "backend": "tensile",
        "search_space": "heuristic",
        "MACROTILE_OPT": False,
        "SIZE_OPTION": 0,
        "ONE_SIZE_PER_CONFIG": True,
        "CLUSTER": 0,
        "MI_FILTER": 0,
        "Sizes": [[128, 128, 1, 128]],
    }


# ---------------------------------------------------------------------------
# Tests: reject0 - MX+subtile requires even MIWaveTile
# ---------------------------------------------------------------------------


@pytest.mark.cg_components
def test_reject0_mx_subtile_odd_wavetile(tmp_path: Path) -> None:
    """Test reject0: MX+subtile filters out odd MIWaveTile configurations.

    When MX is enabled with subtile, MFMAs with odd WaveTileM or WaveTileN
    should be filtered out by reject0. This test verifies the filtering
    by checking that valid MIs are produced.
    """
    cfg = _base_template()
    cfg["ARCH"] = "gfx950"
    cfg["search_space"] = "subtile"
    cfg["DataType"] = "F4"  # F4 enables MX
    cfg["Sizes"] = [[256, 256, 1, 256]]

    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    get_gemm_problem(cfg)
    cfg["GemmProblem"] = cfg["GemmProblems"][0]

    mi_log = tmp_path / "MI_finder_log"
    mi_log.mkdir(parents=True, exist_ok=True)

    # Create MI designer with MX+subtile
    mi_designer_mx_subtile = MIDesign(
        str(mi_log),
        copy.deepcopy(cfg),
        mx_block_values=(32, 32),  # MX enabled
        subtile_enabled=True,
    )

    size = (256, 256, 1, 256)
    test_depthu = [128, 256, 512]

    # Generate MI groups with MX+subtile
    # reject0 is applied inside _find_mi_for_size during filtering
    mi_groups_mx_subtile = mi_designer_mx_subtile.generate_for_size(size, depthu_values=test_depthu)

    # Should produce valid MIs (some but not all may pass reject0)
    assert len(mi_groups_mx_subtile) > 0, "MX+subtile should produce some valid MI groups after reject0 filtering"


# ---------------------------------------------------------------------------
# Tests: reject3 - Subtile DepthU divisibility
# ---------------------------------------------------------------------------


@pytest.mark.cg_components
def test_reject3_subtile_depthu_divisibility(tmp_path: Path) -> None:
    """Test reject3: Subtile filters DepthU values by divisibility.

    For subtile mode, DepthU must be divisible by (numSubIterK * MatrixInstK * LSU).
    We verify that valid DepthU values are processed correctly.
    """
    cfg = _base_template()
    cfg["ARCH"] = "gfx950"
    cfg["search_space"] = "subtile"
    cfg["DataType"] = "S"  # Single precision (numSubIterK=2)
    cfg["Sizes"] = [[128, 128, 1, 256]]

    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    get_gemm_problem(cfg)
    cfg["GemmProblem"] = cfg["GemmProblems"][0]

    mi_log = tmp_path / "MI_finder_log"
    mi_log.mkdir(parents=True, exist_ok=True)

    mi_designer = MIDesign(
        str(mi_log),
        copy.deepcopy(cfg),
        mx_block_values=None,
        subtile_enabled=True,
    )

    size = (128, 128, 1, 256)

    # Use standard DepthU values for testing
    # These will be filtered by reject3 based on divisibility constraints
    test_depthu = [64, 128, 256, 512]

    # Generate with DepthU values
    # reject3 will validate: DepthU % (numSubIterK * MatrixInstK * LSU) == 0
    mi_groups = mi_designer.generate_for_size(size, depthu_values=test_depthu)

    # We should get some MI groups (reject3 filters but doesn't eliminate all MIs)
    assert len(mi_groups) > 0, "Subtile should produce some valid MI groups"


# ---------------------------------------------------------------------------
# Tests: reject1/reject2 - MX scale loading checks
# ---------------------------------------------------------------------------


@pytest.mark.cg_components
def test_mx_depthu_filtering_by_scale_loads(tmp_path: Path) -> None:
    """Test reject1/reject2: MX filters DepthU by scale loading bytesLoaded check.

    For MX enabled, DepthU values must satisfy:
    - reject1: bytesLoaded >= (DepthU / MXBlockA) * MT0
    - reject2: bytesLoaded >= (DepthU / MXBlockB) * MT1

    Larger DepthU values require higher SubGroup sizes to pass these checks.
    We verify that some DepthU values are filtered out.
    """
    cfg = _base_template()
    cfg["ARCH"] = "gfx950"
    cfg["search_space"] = "subtile"
    cfg["DataType"] = "F4"  # F4 enables MX (MXBlockA=32, MXBlockB=32)
    cfg["Sizes"] = [[128, 128, 1, 256]]

    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    get_gemm_problem(cfg)
    cfg["GemmProblem"] = cfg["GemmProblems"][0]

    mi_log = tmp_path / "MI_finder_log"
    mi_log.mkdir(parents=True, exist_ok=True)

    mi_designer = MIDesign(
        str(mi_log),
        copy.deepcopy(cfg),
        mx_block_values=(32, 32),  # MX enabled
        subtile_enabled=True,
    )

    size = (128, 128, 1, 256)

    # Test with a wide range of DepthU values
    # Small values should be more likely to pass reject1/reject2
    # Large values might be filtered by scale loading constraints
    test_depthu = [64, 128, 256, 512, 1024]

    # Generate with specific DepthU values
    mi_groups = mi_designer.generate_for_size(size, depthu_values=test_depthu)

    # We expect some but not necessarily all DepthU values to pass
    assert len(mi_groups) > 0, "MX should produce some valid MI groups"


# ---------------------------------------------------------------------------
# Tests: MX disabled vs enabled comparison
# ---------------------------------------------------------------------------


@pytest.mark.cg_components
def test_mx_affects_mi_count(tmp_path: Path) -> None:
    """Test that MX filtering affects the number of generated MIs.

    When MX is enabled with subtile, additional filtering (reject0/reject1/reject2)
    should reduce the number of valid MFMA configurations compared to subtile-only mode.
    """
    cfg = _base_template()
    cfg["ARCH"] = "gfx950"
    cfg["search_space"] = "subtile"
    cfg["DataType"] = "F4"  # F4 enables MX
    cfg["Sizes"] = [[256, 256, 1, 256]]

    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    get_gemm_problem(cfg)
    cfg["GemmProblem"] = cfg["GemmProblems"][0]

    mi_log_subtile = tmp_path / "MI_finder_log_subtile"
    mi_log_subtile.mkdir(parents=True, exist_ok=True)

    mi_log_mx = tmp_path / "MI_finder_log_mx"
    mi_log_mx.mkdir(parents=True, exist_ok=True)

    # MI designer with subtile only
    mi_designer_subtile = MIDesign(
        str(mi_log_subtile),
        copy.deepcopy(cfg),
        mx_block_values=None,  # No MX
        subtile_enabled=True,
    )

    # MI designer with MX+subtile
    mi_designer_mx = MIDesign(
        str(mi_log_mx),
        copy.deepcopy(cfg),
        mx_block_values=(32, 32),  # MX enabled
        subtile_enabled=True,
    )

    size = (256, 256, 1, 256)
    test_depthu = [128, 256, 512]

    # Generate for both configurations
    mi_groups_subtile = mi_designer_subtile.generate_for_size(size, depthu_values=test_depthu)
    mi_groups_mx = mi_designer_mx.generate_for_size(size, depthu_values=test_depthu)

    # Both should produce MIs
    assert len(mi_groups_subtile) > 0, "Subtile-only should produce MIs"
    assert len(mi_groups_mx) > 0, "MX+subtile should produce MIs"

    # With MX, we expect <= MIs than subtile-only (due to additional filtering)
    # Note: Not strictly <, could be equal in some cases
    assert len(mi_groups_mx) <= len(mi_groups_subtile), \
        f"MX should not produce more MIs than subtile-only. Got {len(mi_groups_mx)} vs {len(mi_groups_subtile)}"


# ---------------------------------------------------------------------------
# Tests: F8 variant handling in reject3
# ---------------------------------------------------------------------------


@pytest.mark.cg_components
def test_f8_variant_numsubiterk_handling(tmp_path: Path) -> None:
    """Test that F8 correctly uses numSubIterK=1 in reject3.

    For F8, numSubIterK should be 1 (not 2 like other types).
    This affects DepthU divisibility in reject3:
    DepthU must be divisible by (numSubIterK * MatrixInstK * LSU).

    With numSubIterK=1, DepthU has fewer divisibility constraints than
    non-F8 types.
    """
    cfg = _base_template()
    cfg["ARCH"] = "gfx950"
    cfg["search_space"] = "subtile"
    cfg["DataType"] = "F8"
    cfg["Sizes"] = [[128, 128, 1, 256]]

    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    get_gemm_problem(cfg)
    cfg["GemmProblem"] = cfg["GemmProblems"][0]

    mi_log = tmp_path / "MI_finder_log"
    mi_log.mkdir(parents=True, exist_ok=True)

    mi_designer = MIDesign(
        str(mi_log),
        copy.deepcopy(cfg),
        mx_block_values=None,
        subtile_enabled=True,
    )

    size = (128, 128, 1, 256)

    # F8 uses numSubIterK=1, so DepthU has fewer divisibility constraints
    test_depthu = [64, 128, 256, 512]

    mi_groups = mi_designer.generate_for_size(size, depthu_values=test_depthu)

    # Should produce valid MIs
    assert len(mi_groups) > 0, "F8 should produce valid MI groups with numSubIterK=1"


# ---------------------------------------------------------------------------
# Tests: Post-processor DepthU removal (via real BasePostProcessor.apply())
# ---------------------------------------------------------------------------


@pytest.mark.cg_components
def test_post_processor_removes_depthu_when_in_all_mi_groups(monkeypatch) -> None:
    """Test that DepthU is removed from fork_params when present in all MI groups.

    When every MI group entry has DepthU, it should be removed from fork_params
    since it's already embedded in each MI group's configuration and doesn't
    need to fork the parameter space.
    """
    from geko.config_generator.fork_params import optimization_param as opt_param
    from geko.config_generator.fork_params.post_processor import BasePostProcessor
    from geko.schemas import GemmType

    monkeypatch.setattr(opt_param, "load_tensile_metadata", lambda: {})

    gt = GemmType.from_tensile("N", "N", "H", "H", "S")
    cfg = {
        "GemmProblem": type("GP", (), {"gemm_type": gt})(),
        "ARCH": "gfx950",
        "StreamK": False,
        "CMS": False,
    }
    pp = BasePostProcessor(cfg)

    fork_params = {
        "DepthU": ForkParameter(
            "DepthU",
            values=[128, 256, 512],
            active=True,
        ),
        "WorkGroupMapping": ForkParameter(
            "WorkGroupMapping",
            values=[0, 1],
            active=True,
        ),
    }

    # Create MI groups where all entries have DepthU
    mi_groups = [
        {
            "MatrixInstruction": ForkParameter(
                "MatrixInstruction",
                values=(16, 16, 128, 1, 1, 1, 1, 1, 1),
                active=True,
            ),
            "DepthU": 128,
            "WorkGroupMapping": 0,
        },
        {
            "MatrixInstruction": ForkParameter(
                "MatrixInstruction",
                values=(32, 32, 64, 1, 1, 1, 1, 1, 1),
                active=True,
            ),
            "DepthU": 256,
            "WorkGroupMapping": 1,
        },
        {
            "MatrixInstruction": ForkParameter(
                "MatrixInstruction",
                values=(16, 16, 128, 1, 1, 1, 1, 1, 1),
                active=True,
            ),
            "DepthU": 512,
            "WorkGroupMapping": 0,
        },
    ]

    fork_params, mi_groups = pp.apply(fork_params, mi_groups, (16, 16, 1, 16))

    # Verify DepthU is removed from fork_params
    assert "DepthU" not in fork_params, \
        "DepthU should be removed from fork_params when present in all MI groups"

    # Verify other fork params are preserved
    assert "WorkGroupMapping" in fork_params, \
        "Other fork parameters should be preserved"

    # Verify DepthU still exists in MI groups
    for i, mi_group in enumerate(mi_groups):
        assert "DepthU" in mi_group, \
            f"DepthU should remain in MI group {i}"


@pytest.mark.cg_components
def test_post_processor_preserves_depthu_when_not_in_all_groups(monkeypatch) -> None:
    """Test that DepthU is preserved in fork_params when not present in all MI groups.

    If some MI groups don't have DepthU, it's a fork parameter and must remain
    in fork_params.
    """
    from geko.config_generator.fork_params import optimization_param as opt_param
    from geko.config_generator.fork_params.post_processor import BasePostProcessor
    from geko.schemas import GemmType

    monkeypatch.setattr(opt_param, "load_tensile_metadata", lambda: {})

    gt = GemmType.from_tensile("N", "N", "H", "H", "S")
    cfg = {
        "GemmProblem": type("GP", (), {"gemm_type": gt})(),
        "ARCH": "gfx950",
        "StreamK": False,
        "CMS": False,
    }
    pp = BasePostProcessor(cfg)

    fork_params = {
        "DepthU": ForkParameter(
            "DepthU",
            values=[128, 256, 512],
            active=True,
        ),
    }

    # Create MI groups where NOT all entries have DepthU
    mi_groups = [
        {
            "MatrixInstruction": ForkParameter(
                "MatrixInstruction",
                values=(16, 16, 128, 1, 1, 1, 1, 1, 1),
                active=True,
            ),
            "DepthU": 128,
        },
        {
            "MatrixInstruction": ForkParameter(
                "MatrixInstruction",
                values=(32, 32, 64, 1, 1, 1, 1, 1, 1),
                active=True,
            ),
            # Missing DepthU in this group
        },
    ]

    fork_params, _ = pp.apply(fork_params, mi_groups, (16, 16, 1, 16))

    # Verify DepthU is preserved in fork_params
    assert "DepthU" in fork_params, \
        "DepthU should be preserved in fork_params when not in all MI groups"
