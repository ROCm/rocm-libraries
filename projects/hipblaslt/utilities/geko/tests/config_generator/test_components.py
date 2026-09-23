# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Component-level tests: MI designer, optimization params, fork assembly.

Section marker (see ``tests/conftest.py``):

* ``cg_components`` — exercises the same building blocks as ``run_config_generator`` before clustering/output.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from geko.config_generator.fork_param_generator import generate_fork_params
from geko.config_generator.fork_params import get_optimization_params, get_post_processor
from geko.config_generator.load_input_config import (
    apply_input_config_defaults,
    get_gemm_problem,
    validate_input_config,
)
from geko.config_generator.mi_designer import MIDesign


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_template() -> dict:
    """Return a minimal valid tuning config dict before validate_input_config /
    apply_input_config_defaults and field overrides.
    """
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
# MI + optimization + fork_param pipeline
# ---------------------------------------------------------------------------


@pytest.mark.cg_components
@pytest.mark.parametrize(
    "arch,search_space,transa,transb,size",
    [
        ("gfx950", "heuristic", "N", "N", (128, 128, 1, 128)),
        ("gfx950", "generic", "N", "N", (256, 256, 1, 256)),
        ("gfx942", "heuristic", "N", "T", (128, 256, 1, 128)),
        ("gfx942", "generic", "T", "N", (64, 64, 1, 64)),
        # M and N both < 8 (tiny tiles)
        ("gfx950", "heuristic", "N", "N", (4, 4, 1, 128)),
        ("gfx950", "generic", "N", "N", (1, 7, 1, 128)),
        ("gfx942", "heuristic", "N", "T", (2, 6, 1, 128)),
        # M and N both >= 8192 (large problems)
        ("gfx950", "heuristic", "N", "N", (8192, 8192, 1, 4096)),
        ("gfx950", "generic", "N", "N", (12288, 8192, 1, 2048)),
        ("gfx942", "heuristic", "N", "T", (8192, 16384, 1, 1024)),
    ],
)
def test_mi_opt_fork_pipeline_non_empty(
    arch: str,
    search_space: str,
    transa: str,
    transb: str,
    size: tuple[int, int, int, int],
    hipblaslt_path: str | None,
    tensilelite_sys_path: None,
    tmp_path: Path,
) -> None:
    """Non-empty MI groups, fork params, and nkernels > 0 for varied arch/layout/size."""
    if not hipblaslt_path or not Path(hipblaslt_path).is_dir():
        pytest.skip("Requires --hipblaslt-path")

    cfg = _base_template()
    cfg["ARCH"] = arch
    cfg["search_space"] = search_space
    cfg["TRANSA"] = transa
    cfg["TRANSB"] = transb
    cfg["Sizes"] = [list(size)]
    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    get_gemm_problem(cfg)
    cfg["GemmProblem"] = cfg["GemmProblems"][0]

    mi_log = tmp_path / "MI_finder_log"
    mi_log.mkdir(parents=True, exist_ok=True)

    mi_designer = MIDesign(str(mi_log), copy.deepcopy(cfg))
    opt_params = get_optimization_params(cfg)
    post_processor = get_post_processor(cfg)

    M, N, B, K = size
    mi_groups = mi_designer.generate_for_size(size)
    assert len(mi_groups) > 0, "MIDesign.generate_for_size returned no MI groups"

    fork_dict, opt_groups = opt_params.generate_for_size(size)
    assert len(fork_dict) > 0, "Optimization params produced empty fork dict"
    assert any(opt_groups), "Optimization params produced no group dimensions"

    fork_params, num_mis, nkernels = generate_fork_params(
        mi_designer,
        opt_params,
        cfg,
        size,
        post_processor=post_processor,
    )
    assert "Groups" in fork_params
    assert num_mis > 0
    assert nkernels > 0
