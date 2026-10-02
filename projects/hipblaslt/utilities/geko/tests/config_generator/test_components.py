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
from geko.config_generator.config_sections_generator import ConfigSectionGenerator
from geko.config_generator.config_generator import mi_design_mx_options
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

    # Extract MX block values from ConfigSectionGenerator (same flow as config_generator)
    csg = ConfigSectionGenerator(cfg)
    mx_block_values = None
    if csg._problem_type.get("MXBlockA") and csg._problem_type.get("MXBlockB"):
        mx_block_values = (csg._problem_type["MXBlockA"], csg._problem_type["MXBlockB"])

    # Extract subtile_enabled from config
    subtile_enabled = cfg.get("search_space") == "subtile"

    if mx_block_values is not None and not subtile_enabled:
        subtile_enabled = True

    # Create MI designer with MX and subtile flags only
    mi_designer = MIDesign(
        str(mi_log),
        copy.deepcopy(cfg),
        mx_block_values=mx_block_values,
        subtile_enabled=subtile_enabled,
    )
    opt_params = get_optimization_params(cfg)
    post_processor = get_post_processor(cfg)

    # Extract DepthU values and wavefront size from opt_params (same flow as fork_param_generator)
    fork_dict, opt_groups = opt_params.generate_for_size(size)

    depthu_values = None
    if "DepthU" in fork_dict:
        depthu_values = fork_dict["DepthU"].values

    wavefront_size = 64  # Default
    if "WavefrontSize" in fork_dict:
        wavefront_size = fork_dict["WavefrontSize"].values[0]

    # Call generate_for_size with per-size DepthU and wavefront_size
    M, N, B, K = size
    mi_groups = mi_designer.generate_for_size(size, depthu_values=depthu_values, wavefront_size=wavefront_size)
    assert len(mi_groups) > 0, "MIDesign.generate_for_size returned no MI groups"

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


# ---------------------------------------------------------------------------
# MT_DU origami sentinels and the gfx1250 generic prefetch axes
# ---------------------------------------------------------------------------


def _prepared(**overrides) -> dict:
    cfg = _base_template()
    cfg.update(overrides)
    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    get_gemm_problem(cfg)
    cfg["GemmProblem"] = cfg["GemmProblems"][0]
    return cfg


@pytest.mark.cg_components
@pytest.mark.parametrize(
    "arch,search_space,backend,streamk,library_type,sentinels",
    [
        ("gfx950", "generic", "ductile", True, "OOB", True),
        ("gfx950", "generic", "ductile", False, "OOB", False),
        ("gfx1250", "heuristic", "tensile", True, "OOB", True),
        ("gfx1250", "heuristic", "tensile", False, "OOB", False),
        ("gfx1250", "generic", "ductile", True, "OOB", True),
        ("gfx1250", "generic", "ductile", True, "Equality", False),
        ("gfx1250", "generic", "ductile", False, "OOB", False),
    ],
)
def test_mt_du_origami_sentinels_need_streamk(
    arch: str,
    search_space: str,
    backend: str,
    streamk: bool,
    library_type: str,
    sentinels: bool,
    hipblaslt_path: str | None,
    tensilelite_sys_path: None,
    tmp_path: Path,
) -> None:
    """MT_DU pins WGMXCC -1 / SKXCC 0 only where the runtime can pick WGM itself."""
    size = (2048, 2048, 1, 512)
    cfg = _prepared(ARCH=arch, search_space=search_space, backend=backend, StreamK=streamk,
                    LIBRARY_TYPE=library_type, TRANSA="T", MACROTILE_OPT=True,
                    MT_DU=[128, 128, 64], Sizes=[list(size)])
    mi_log = tmp_path / "MI_finder_log"
    mi_log.mkdir()
    fork_params, _, _ = generate_fork_params(
        MIDesign(str(mi_log), copy.deepcopy(cfg)),
        get_optimization_params(cfg),
        cfg,
        size,
        post_processor=get_post_processor(cfg),
    )
    wgmxcc = fork_params.get("WorkGroupMappingXCC")
    skxcc = fork_params.get("StreamKXCCMapping")
    if sentinels:
        assert wgmxcc.values == [-1]
        assert skxcc.values == [0]
    else:
        assert wgmxcc is None or -1 not in wgmxcc.values


@pytest.mark.cg_components
@pytest.mark.parametrize(
    "arch,streamk,library_type,prefetch_gl2",
    [
        ("gfx1250", False, "OOB", [0]),
        ("gfx1250", True, "OOB", [0, 1, 2]),
        ("gfx1250", True, "Equality", [0, 1, 2]),
        ("gfx1250v0", True, "OOB", [0]),
    ],
)
def test_gfx1250_generic_prefetch_axes(
    arch: str,
    streamk: bool,
    library_type: str,
    prefetch_gl2: list,
    hipblaslt_path: str | None,
    tensilelite_sys_path: None,
) -> None:
    """PrefetchGL2 is off where GSU stays -1; per-tensor PGR is auto with EPS off."""
    size = (4096, 4096, 1, 4096)
    cfg = _prepared(ARCH=arch, search_space="generic", backend="ductile", StreamK=streamk,
                    LIBRARY_TYPE=library_type, TRANSA="T", Sizes=[list(size)])
    fork_dict, _ = get_optimization_params(cfg).generate_for_size(size)
    values = {name: fp.values for name, fp in fork_dict.items()}
    assert values["PrefetchGL2"] == prefetch_gl2
    assert values["PrefetchGlobalReadA"] == [-1]
    assert values["PrefetchGlobalReadB"] == [-1]
    assert values["ExpandPointerSwap"] == [False]


@pytest.mark.cg_components
@pytest.mark.parametrize("arch,subtile", [("gfx950", True), ("gfx1250", False)])
def test_mt_du_depthu_pin_holds_for_mx(
    arch: str,
    subtile: bool,
    hipblaslt_path: str | None,
    tensilelite_sys_path: None,
    tmp_path: Path,
) -> None:
    """MT_DU's DepthU holds for MXFP4 whether the MI groups carry DepthU (gfx950 subtile) or not."""
    size = (4096, 4096, 1, 4096)
    cfg = _prepared(ARCH=arch, search_space="generic", backend="ductile", TRANSA="T", DataType="F4",
                    MACROTILE_OPT=True, MT_DU=[256, 256, 512], Sizes=[list(size)])
    mx_block_values, subtile_enabled = mi_design_mx_options(cfg)
    assert subtile_enabled is subtile
    assert mx_block_values == ((32, 32) if subtile else None)
    mi_log = tmp_path / "MI_finder_log"
    mi_log.mkdir()
    fork_params, num_mis, _ = generate_fork_params(
        MIDesign(str(mi_log), copy.deepcopy(cfg), mx_block_values=mx_block_values, subtile_enabled=subtile_enabled),
        get_optimization_params(cfg),
        cfg,
        size,
        post_processor=get_post_processor(cfg),
    )
    assert num_mis > 0
    mi_groups = fork_params["Groups"].values[0]
    if subtile:
        assert "DepthU" not in fork_params
        assert {tuple(g["DepthU"].values) for g in mi_groups} == {(512,)}
    else:
        assert fork_params["DepthU"].values == [512]
        assert all("DepthU" not in g for g in mi_groups)
