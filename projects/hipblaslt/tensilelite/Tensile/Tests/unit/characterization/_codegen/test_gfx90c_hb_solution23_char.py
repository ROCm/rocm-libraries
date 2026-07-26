# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Full-generator contract for the canonical gfx90c HB solution 23."""

import re
from pathlib import Path

import pytest

from codegen_harness import emit_kernels_from_logic


pytestmark = pytest.mark.unit

SOLUTION_NAME = (
    "Cijk_Ailk_Bljk_HB_MT064x008x32_EPS1_GRVW02_LPB00_PGR1_SNLL1_"
    "TT02_02_USFGRO01_VW02_WG32_04_01"
)
SOLUTION_7_NAME = (
    "Cijk_Ailk_Bljk_HB_MT016x016x16_EPS1_GRVW02_LPB00_PGR1_SNLL1_"
    "TT02_02_USFGRO01_VW02_WG08_08_01"
)
LOGIC = (
    Path(__file__).resolve().parents[6]
    / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx90c/Equality/"
    "vega10_Cijk_Ailk_Bljk_HB.yaml"
)


def test_solution_name_selection_fails_closed():
    with pytest.raises(ValueError, match="missing=.*not-a-solution"):
        emit_kernels_from_logic(LOGIC, solution_names={"not-a-solution"})


def test_solution23_emits_packed_mac_in_compute_body():
    results = emit_kernels_from_logic(
        LOGIC, solution_names={SOLUTION_NAME}, solution_indices={23}
    )
    assert len(results) == 1
    basename, source, error = results[0]
    assert error == 0
    assert "MT64x8x32" in basename
    assert ".set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0" in source
    assert ".set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+1" in source
    assert ".set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0" in source
    assert ".set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+1" in source

    accumulator_init = source.index("initC")
    result_write = source.index("/* Stores for OptNLL */", accumulator_init)
    compute_body = source[accumulator_init:result_write]
    assert "v_pk_fma_f16" in compute_body


def test_solution7_beta_epilogue_uses_relative_packed_accumulator_offsets():
    results = emit_kernels_from_logic(
        LOGIC, solution_names={SOLUTION_7_NAME}, solution_indices={7}
    )
    assert len(results) == 1
    basename, source, error = results[0]
    assert error == 0
    assert "MT16x16x16" in basename
    assert "C*beta" in source
    assert "vgprValuC+-" not in source
    edge_batch = source.rindex(
        "rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), "
        "(0, 0, 1, 0), (0, 0, 1, 1)]"
    )
    beta_math = source.index("C*beta", edge_batch)
    assert source[edge_batch:beta_math].count("*= alpha packed accumulator") == 2
    edge_path = source[source.rindex("edge=1", 0, edge_batch):edge_batch]
    assert "buffer_load_short_d16 v9" in edge_path
    assert "buffer_load_short_d16_hi v9" in edge_path
    assert "buffer_load_short_d16 v12" in edge_path
    assert "buffer_load_short_d16_hi v12" in edge_path


def test_solution7_edge_beta_waitcnt_order_regression():
    results = emit_kernels_from_logic(
        LOGIC, solution_names={SOLUTION_7_NAME}, solution_indices={7}
    )
    assert len(results) == 1
    basename, source, error = results[0]
    assert error == 0

    assert "buffer_load_short_d16_hi v9" in source
    assert "buffer_load_short_d16_hi v12" in source

    # Locate the last high-half load pair instruction for edge C loading
    d16_hi_idx = source.rindex("buffer_load_short_d16_hi v12")
    d16_hi_line_end = source.find("\n", d16_hi_idx)
    sub_source_after_load = source[d16_hi_line_end:]

    # Assert s_waitcnt vmcnt(0) occurs before the rC scaling section
    wait_match = re.search(r"s_waitcnt\s+vmcnt\(0\)", sub_source_after_load)
    assert wait_match is not None, "Missing s_waitcnt vmcnt(0) after buffer_load_short_d16_hi v12"
    wait_idx = d16_hi_line_end + wait_match.start()

    # Find first compute instruction using v9 or v12
    first_use_match = re.search(r"v_[a-z0-9_]+\s+.*?\b(v9|v12)\b", sub_source_after_load)
    assert first_use_match is not None, "Missing compute instruction using v9 or v12"
    first_use_idx = d16_hi_line_end + first_use_match.start()

    assert wait_idx < first_use_idx, (
        f"s_waitcnt vmcnt(0) at {wait_idx} must occur before first use "
        f"of loaded registers at {first_use_idx}"
    )
