# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unroll-loop exits of Stream-K kernels with AssertSummationElementMultiple % (2*DepthU) == 0.

ASEM % (2*DepthU) == 0 lets the generator assume an even unroll-loop count that
is never 1: it drops the single-iteration escape (PGR>=2, toPGR1) and closes
the loop with the once-per-2-iterations exit. Stream-K
(TileProcessingStrategy=StreamK) gives each workgroup an arbitrary range of DU iterations,
so neither assumption holds there and every exit must be kept.
"""

from pathlib import Path

import pytest

_LOGIC = Path(__file__).resolve().parent / "streamk_asem_data" / "gfx950_sk3_userargs_mt32x64.yaml"

from loop_exit_test_helpers import (emit, dec_blocks as _dec_blocks,
                                    assemble as _assemble, has_single_iteration_escape,
                                    pristine_default_solution)

pytestmark = pytest.mark.unit


def _emit_variant(tmp_path, name, sol):
    return None, emit(tmp_path, name, _LOGIC, sol), 0


@pytest.mark.parametrize("pgr", [1, 2])
def test_streamk_asem_multiple_of_2du_keeps_every_exit(tmp_path, pgr):
    """Both loop copies decrement by 1 and keep their exit test, and PGR2 keeps
    the single-iteration escape. Relying on ASEM here raised NameError
    ('decValue') at codegen for PGR1, and for PGR2 dropped the escape and
    decremented one copy by a StaggerU-selected 2/1."""
    shape = {"ExpandPointerSwap": pgr == 1, "UnrollLoopSwapGlobalReadOrder": int(pgr == 2)}
    _, src, err = _emit_variant(tmp_path, f"sk{pgr}",
                                {"PrefetchGlobalRead": pgr,
                                 "AssertSummationElementMultiple": 256, **shape})
    assert err == 0
    assert "s[sgprStreamKLocalEnd]" in src
    decs = _dec_blocks(src)
    assert len(decs) == 2 and all(d == "1" for d, _ in decs)
    assert all(n.startswith("s_cmp_") for _, n in decs)
    assert has_single_iteration_escape(src) == (pgr == 2)
    _assemble(tmp_path, f"sk{pgr}", src)


@pytest.mark.parametrize("dp", [{"StreamK": 0, "GlobalSplitU": 1}], ids=["fixed-gsu1"])
def test_data_parallel_asem_multiple_of_2du_drops_single_iteration_escape(tmp_path, dp):
    """Fixed plain GEMM GSU=1 proves an even count; restoring the one-trip
    escape for that proven case breaks this preservation test."""
    _, src, err = _emit_variant(tmp_path, "dp",
                                {"PrefetchGlobalRead": 2,
                                 "AssertSummationElementMultiple": 256, **dp})
    assert err == 0
    assert "s[sgprStreamKLocalEnd]" not in src
    assert not has_single_iteration_escape(src)
    _assemble(tmp_path, "dp", src)


@pytest.mark.parametrize("overrides,want_escape", [
    ({"StreamK": 3}, True),
    ({"StreamK": 4}, True),
    ({"StreamK": 5}, True),
    ({"StreamK": 0, "GlobalSplitU": 1}, False),
    ({"StreamK": 0, "GlobalSplitU": 2}, True),
    ({"StreamK": 0, "GlobalSplitU": -1}, True),
    ({"StreamK": 0, "GlobalSplitU": 1,
      "InternalSupportParams": {"SupportUserGSU": True, "SupportCustomWGM": True,
                                "SupportCustomStaggerU": False, "UseUniversalArgs": True,
                                "UseSFC": False}}, True),
    ({"StreamKForceDPOnly": 1}, True),
], ids=["sk3", "sk4", "sk5", "fixed-gsu1", "gsu2", "auto-gsu", "runtime-gsu1", "persistent-dp"])
def test_open_loop_uses_per_workgroup_even_trip_proof(tmp_path, overrides, want_escape):
    """Entrance must not infer a per-WG even count from globally even K.

    Production pass: shared fixed-split proof at openLoop. Fail again: revert
    only the entrance site to the ASEM-only gate (closeLoop can stay fixed).
    """
    source = emit(tmp_path, "entry", _LOGIC,
                  {"PrefetchGlobalRead": 2, "AssertSummationElementMultiple": 256, **overrides})
    assert has_single_iteration_escape(source) is want_escape


@pytest.mark.parametrize("pgr", [1, 2], ids=["pgr1", "pgr2"])
@pytest.mark.parametrize("copies", [1, 2], ids=["one-copy", "two-copies"])
def test_open_loop_without_even_asem_keeps_required_escape(tmp_path, pgr, copies):
    """Preserve the no-proof entrance; replacing its gate with false loses it."""
    shape = {"ExpandPointerSwap": copies == 2 and pgr == 1,
             "UnrollLoopSwapGlobalReadOrder": int(copies == 2 and pgr == 2)}
    source = emit(tmp_path, "no_even_fact", _LOGIC,
                  {"StreamK": 0, "GlobalSplitU": 1, "PrefetchGlobalRead": pgr,
                   "AssertSummationElementMultiple": 128, **shape})
    assert has_single_iteration_escape(source) is (pgr == 2)


def test_open_loop_custom_schedule_keeps_single_iteration_escape(tmp_path, monkeypatch):
    """Entrance-site feature test independent of the custom main-loop schedule's separate close-loop defect.

    Only the unrelated closer sees ASEM=1, preserving its real emission and
    CMS validation. The real entrance still sees the ASEM=2U fact. Reverting
    only openLoop's fixed-split proof removes the required edge again.
    """
    import copy
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    close = KernelWriterAssembly.closeLoop

    def ordinary_close(writer, kernel, *args, **kwargs):
        close_kernel = copy.copy(kernel)
        close_kernel["AssertSummationElementMultiple"] = 1
        return close(writer, close_kernel, *args, **kwargs)

    monkeypatch.setattr(KernelWriterAssembly, "closeLoop", ordinary_close)
    logic = Path(__file__).parent / "test_data" / "half_exit_gfx950_cms_mt256x256.yaml"
    source = emit(tmp_path, "custom_entry", logic,
                  {"AssertSummationElementMultiple": 128, "UseCustomMainLoopSchedule": 1})
    assert has_single_iteration_escape(source)
