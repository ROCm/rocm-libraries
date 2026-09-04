# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""ForceDPOnly=0 PGR=1: SIA=4 drains tensorcnt before tdm*Group0 mutate."""

import os
import re

import pytest

from config_harness import (
    assert_assembles,
    assert_cluster_barrier_balanced,
    assert_pgr1_persist_dp_close_wait,
    assert_real_gfx1250_kernels,
    emit_kernels_from_config,
    solutions_from_config,
)

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "streamk_cluster_multicast_pgr1_cd22.yaml",
)

_TDM_DESC_MUT = re.compile(
    r"s_(?:add_u32|addc_u32|xor_b32)\s+s\[sgprtdm(?:A|MXSA)Group0"
)


def _state(sol):
    return sol._state if hasattr(sol, "_state") else sol


def _by_sia(results):
    out = {}
    for base, src, err in results:
        if "SIA4" in src:
            out[4] = (base, src, err)
        elif "SIA0" in src:
            out[0] = (base, src, err)
    return out


def _prologue(src):
    """ISA from the prologue TDM label through openLoopL."""
    lines = src.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("label_NoBranch")), None)
    end = next((i for i, ln in enumerate(lines) if ln.startswith("label_openLoopL:")), None)
    assert start is not None and end is not None and start < end, (
        "missing prologue label_NoBranch / label_openLoopL"
    )
    return lines[start:end]


def _first_tdm_and_desc_mut(window):
    tdm_i = next((i for i, ln in enumerate(window) if "tensor_load_to_lds" in ln), None)
    mut_i = next(
        (i for i, ln in enumerate(window) if tdm_i is not None and i > tdm_i and _TDM_DESC_MUT.search(ln)),
        None,
    )
    return tdm_i, mut_i


def test_streamk_pgr1_f8_cd22_sia4_vs_sia0_descriptor_war():
    """PGR=1 SIA=4 waits tensorcnt before mutating in-flight TDM descriptors."""
    sols = solutions_from_config(_CONFIG, arch=_ARCH, limit_solutions=8)
    assert len(sols) == 2, f"expected SIA=0 and SIA=4, got {len(sols)}"
    sias = sorted(_state(s)["ScheduleIterAlg"] for s in sols)
    assert sias == [0, 4], sias
    for st in (_state(s) for s in sols):
        assert st["PrefetchGlobalRead"] == 1, st["PrefetchGlobalRead"]
        assert list(st["ClusterDim"]) == [2, 2], list(st["ClusterDim"])
        assert st["StreamK"] == 3, st["StreamK"]
        assert st["StreamKForceDPOnly"] == 0, st["StreamKForceDPOnly"]

    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert_real_gfx1250_kernels(results)
    by_sia = _by_sia(results)
    assert set(by_sia) == {0, 4}, f"expected SIA0 and SIA4 kernels, got {results!r}"

    for sia, (base, src, _err) in by_sia.items():
        assert_assembles(src, base)
        assert_cluster_barrier_balanced(src, base)
        assert_pgr1_persist_dp_close_wait(src, base)
        # PGR=1 must not emit skipPGR2 labels or branch targets. Instruction
        # comments may mention skipPGR2 to document that the path is omitted
        # (e.g. "no skipPGR2 leftover for next kernel").
        isa = "\n".join(ln.split("//", 1)[0] for ln in src.splitlines())
        assert "skipPGR2" not in isa, f"Kernel {base!r} PGR=1 must not emit skipPGR2"
        assert "PGR1" in src and f"SIA{sia}" in src and "CD2_2" in src, base

        lines = src.splitlines()
        for i, ln in enumerate(lines):
            if "s_barrier_signal -3" not in ln and "s_barrier_wait -3" not in ln:
                continue
            neigh = lines[max(0, i - 1) : i + 2]
            assert not any("s_wait_tensorcnt" in n for n in neigh), (
                f"Kernel {base!r}: s_wait_tensorcnt must not sit on cluster -3. "
                f"around={neigh!r}"
            )

        pro = _prologue(src)
        tdm_i, mut_i = _first_tdm_and_desc_mut(pro)
        assert tdm_i is not None, f"Kernel {base!r}: missing prologue tensor_load_to_lds"
        assert mut_i is not None, f"Kernel {base!r}: missing prologue TDM descriptor mutation"
        xor_i = next((i for i, ln in enumerate(pro) if "s_xor_b32 s[sgprtdmAGroup0+1]" in ln), None)
        assert xor_i is not None, f"Kernel {base!r}: missing prologue tdmAGroup0+1 XOR"

        if sia == 4:
            waits = [ln for ln in pro[tdm_i:mut_i] if "s_wait_tensorcnt" in ln]
            assert waits, (
                f"Kernel {base!r}: SIA=4 must s_wait_tensorcnt after prologue TDM "
                f"and before mutating tdm*Group0 (WAR on in-flight descriptor). "
                f"window={pro[tdm_i:mut_i + 1]!r}"
            )
            xor_waits = [ln for ln in pro[tdm_i:xor_i] if "s_wait_tensorcnt" in ln]
            assert xor_waits, (
                f"Kernel {base!r}: SIA=4 must wait_tensorcnt before prologue XOR. "
                f"window={pro[tdm_i:xor_i + 1]!r}"
            )
        else:
            assert any("s_wait_tensorcnt" in ln for ln in pro[tdm_i:xor_i]), (
                f"Kernel {base!r}: SIA=0 must wait_tensorcnt before prologue XOR. "
                f"window={pro[tdm_i:xor_i + 1]!r}"
            )
            # Tensile issues both TDMs, then increments, then waits. Do not
            # require a wait between TDM A and the first s_add (SIA=0 passes).
            assert any("tensor_load_to_lds" in ln and "MXSA" in ln for ln in pro[tdm_i:mut_i + 1]) or (
                "s_wait_tensorcnt" in "\n".join(pro[tdm_i:mut_i])
            ), (
                f"Kernel {base!r}: SIA=0 prologue should issue both TDMs before "
                f"the first descriptor mutation, or wait first. "
                f"window={pro[tdm_i:mut_i + 1]!r}"
            )
