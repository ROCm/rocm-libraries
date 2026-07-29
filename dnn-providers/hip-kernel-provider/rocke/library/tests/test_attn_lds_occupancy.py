# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LDS-footprint tripwires for the gfx942 tiled attention kernels.

The transposed-x8 epilogue stores per-lane straight to global and returns before
touching the ``Acc_lds`` staging buffer, so on that path the buffer is dead. It used
to be allocated anyway, which was free while an unreferenced ``addrspace(3)`` global
was dead-stripped, and stopped being free once LDS allocations were placed in a
single pool that reserves bytes per allocation regardless of use. At BLOCK_M=64
HD=128 the dead buffer is 16 KB, which is the difference between two workgroups per
CU and one.

These tests pin the *consequence* rather than the mechanism: how many workgroups the
emitted kernel's static LDS admits per CU. That way they trip on any future change
that reintroduces dead LDS, whoever introduces it and by whatever route, and they do
not have to duplicate the builder's footprint arithmetic.

The footprint is read from the compiled kernel's ``group_segment_fixed_size``, not
summed from the ``addrspace(3)`` globals in the IR. Those are different numbers
whenever the backend is free to overlap allocations that are not simultaneously
live, and it is the segment size that actually decides occupancy. Summing the
declared globals would also make these tests fail on a tree where such overlapping
happens, which is the opposite of what they are for.

The counterpart matters just as much: the epilogues that *do* stage through
``Acc_lds`` must be untouched. That is checked by emission rather than by footprint,
because the two sides of the predicate differ in much more than the accumulator --
different QK/PV geometry, a different V layout, no score tile -- so no footprint
comparison across that boundary can isolate the buffer. Emission is the sharper
check anyway: the allocation is skipped by binding the name to ``None``, so an
epilogue that reads it on a path where it was skipped fails while building.

CPU-only; no GPU required.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from kernels import UnifiedAttentionProblem, build_unified_attention_2d_tiled
from kernels.common import attention_unified as au
from kernels.common.attention_unified import _tiled_2d_impl


@pytest.fixture
def gfx942(monkeypatch):
    old = au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = "gfx942"
    for var in (
        "HIPDNN_GFX942_K_SLICED_RING",
        "HIPDNN_GFX942_K_LDSSEQ",
        "HIPDNN_GFX942_BF16_WIDE",
        "HIPDNN_GFX942_D128_SMALLTILE_DK",
        "HIPDNN_GFX942_FLASH_MLIM",
        "HIPDNN_GFX942_FLASH_WIDE",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        yield
    finally:
        au._RESOLVED_ATTENTION_ARCH = old
        au._2D_LAUNCH_META.clear()


def _lds_capacity() -> int:
    from rocke.core.arch import ArchTarget

    try:
        return ArchTarget.from_gfx("gfx942").lds_capacity_bytes
    except KeyError:
        return 65536


def _readelf() -> str | None:
    for cand in ("/opt/rocm/llvm/bin/llvm-readelf", "llvm-readelf", "readelf"):
        found = (
            cand if cand.startswith("/") and shutil.which(cand) else shutil.which(cand)
        )
        if found:
            return found
    return None


def _lds_bytes(spec, tmp_path) -> int:
    """``group_segment_fixed_size`` of the compiled kernel, in bytes.

    Compiles rather than lowers, because the number that decides occupancy is the
    one in the kernel metadata, and it can be smaller than the sum of the declared
    LDS globals.
    """
    from rocke import compile_kernel

    readelf = _readelf()
    if readelf is None:
        pytest.skip("needs llvm-readelf to read group_segment_fixed_size")

    kernel = build_unified_attention_2d_tiled(spec, arch="gfx942")
    artifact = compile_kernel(kernel, arch="gfx942", capture_ir_text=False)
    path = tmp_path / f"{spec.kernel_name()[:80]}.hsaco"
    path.write_bytes(artifact.hsaco)

    out = subprocess.run(
        [readelf, "--notes", str(path)], capture_output=True, text=True, timeout=180
    ).stdout
    match = re.search(r"group_segment_fixed_size[^0-9]*(\d+)", out)
    assert match, (
        "could not read group_segment_fixed_size from the kernel metadata note; "
        "the note format or the reader changed"
    )
    return int(match.group(1))


def _wg_per_cu(spec, tmp_path) -> int:
    return _lds_capacity() // _lds_bytes(spec, tmp_path)


def _prod_spec(dtype, head_size, block_size):
    return au._tiled_spec_from_problem(
        UnifiedAttentionProblem(
            total_q=4096,
            num_seqs=1,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=head_size,
            block_size=block_size,
            max_seqlen_q=4096,
            max_seqlen_k=4096,
            dtype=dtype,
        )
    )


def _narrow_spec(dtype):
    """A non-transposed spec: its epilogue stages output through Acc_lds."""
    spec_cls, _build, _supports = _tiled_2d_impl("gfx942")
    return spec_cls(
        head_size=128,
        block_size=64,
        num_query_heads=32,
        num_kv_heads=8,
        dtype=dtype,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        num_warps=2,
        tile_size=64,
    )


# ---------------------------------------------------------------------------
# The regression tripwires
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_d128_prefill_admits_two_workgroups_per_cu(gfx942, dtype, tmp_path):
    """D128 prefill must not be LDS-limited to a single workgroup.

    One resident workgroup per CU is the state a dead staging buffer put this path
    in. Whether that is worth more than the bytes themselves is not something this
    test claims; it pins the occupancy tier, which is the property that regressed.
    """
    spec = _prod_spec(dtype, 128, 64)
    assert spec.use_mfma_32x32x8 and spec.use_transposed_qk_32x32
    got = _lds_bytes(spec, tmp_path)
    wg = _lds_capacity() // got
    assert wg >= 2, (
        f"{dtype} D128 prefill admits only {wg} workgroup(s)/CU "
        f"(group_segment_fixed_size {got} B of {_lds_capacity()} B); a dead "
        f"allocation has probably crept back in"
    )


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_d64_prefill_admits_three_workgroups_per_cu(gfx942, dtype, tmp_path):
    spec = _prod_spec(dtype, 64, 16)
    assert spec.use_mfma_32x32x8 and spec.use_transposed_qk_32x32
    got = _lds_bytes(spec, tmp_path)
    wg = _lds_capacity() // got
    assert wg >= 3, (
        f"{dtype} D64 prefill admits only {wg} workgroup(s)/CU "
        f"(group_segment_fixed_size {got} B of {_lds_capacity()} B)"
    )


# ---------------------------------------------------------------------------
# The counterpart: paths that DO use the accumulator keep it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_accumulator_using_epilogue_still_builds(gfx942, dtype):
    """Emission is the over-reach guard for the epilogues that read the buffer.

    The allocation is skipped by binding the name to ``None`` rather than by making
    it smaller, so an epilogue that reads it on a path where it was skipped fails
    while emitting rather than landing on another buffer's bytes. Building the
    accumulator-using epilogues is what exercises that: if the predicate were ever
    widened to cover them, these builds would raise.

    Note this covers the two epilogues reachable from this builder. It does not
    enumerate every future epilogue, so the predicate still has to be revisited if
    one is added.
    """
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    spec = _narrow_spec(dtype)
    assert not (spec.use_mfma_32x32x8 and spec.use_transposed_qk_32x32)
    kernel = build_unified_attention_2d_tiled(spec, arch="gfx942")
    assert _lower_kernel_to_llvm_python(kernel, arch="gfx942", llvm_flavor="llvm22")
