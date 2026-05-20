################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Tail-loop emit content assertions.

These tests pin the *content* of the subtile tail-loop emit body
(kPosBase init, lane mask, single-iter forcing, PGR>0 entry gating),
complementing the structural / placement / label assertions in
`test_SubtileBasedLogicalScheduler.py`.

They drive `KernelWriter._emitTailLoopScaffoldSubtile` directly via a
minimal real `KernelWriterAssembly` writer — the same emit-site that
`kernelBodySubtile` invokes.
"""
import re

import pytest

from unittest.mock import MagicMock

from Tensile.Tests.unit._subtile_tailloop_fixtures import (
    build_minimal_subtile_kwa,
    setdefault_tail_scaffold_kernel_keys,
    wrap_with_skiptoend,
)


# ── Mock kernel + writer ──

def _mock_dtype(num_bytes=2):
    """Create a mock DataType with numBytes() returning the given size."""
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    mock.numRegisters.return_value = num_bytes / 4
    mock.isFloat4.return_value = num_bytes == 0.5
    mock.is6bitFloat.return_value = False
    mock.is8bitFloat.return_value = num_bytes == 1
    mock.isHalf.return_value = num_bytes == 2
    mock.isBFloat16.return_value = num_bytes == 2
    mock.isSingle.return_value = num_bytes == 4
    return mock


def _create_kernel(MT0=256, MT1=256, *, fp4=False, depthU=None, no_tail_loop=False):
    """Minimal kernel dict driving tail-loop emit logic."""
    mxblock = 32 if fp4 else 0
    bpe = 0.5 if fp4 else 2
    matrixInstK = 128 if fp4 else 32
    if depthU is None:
        depthU = 256 if fp4 else 64

    dtype = _mock_dtype(bpe)
    problemType = {
        "DataTypeA": dtype,
        "DataTypeB": dtype,
        "ComputeDataType": _mock_dtype(4),
    }
    if fp4:
        problemType["MXBlockA"] = mxblock
        problemType["MXBlockB"] = mxblock

    kernel = {
        "DepthU": depthU,
        "_DepthUA": depthU,
        "_DepthUB": depthU,
        "MacroTileA": MT0,
        "MacroTileB": MT1,
        "MacroTile0": MT0,
        "MacroTile1": MT1,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": matrixInstK,
        "MIWaveGroup": [2, 2],
        "WavefrontSize": 64,
        "SourceSwap": False,
        "MIArchVgpr": False,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "NonTemporalMXSA": 0,
        "NonTemporalMXSB": 0,
        "ProblemType": problemType,
        "NoTailLoop": no_tail_loop,
        "AssertSummationElementMultiple": 32,
    }
    if fp4:
        kernel["_DepthUMXSA"] = depthU // mxblock
        kernel["_DepthUMXSB"] = depthU // mxblock
    return kernel


def _augment_kernel_for_tail_scaffold(kernel, pgr):
    """Thin wrapper around the shared setdefaults helper.

    Kept here so callers in this file read naturally
    (`_augment_kernel_for_tail_scaffold(kernel, pgr)`); the
    actual setdefault work lives in the shared fixtures module
    so it stays in lockstep with the parallel population in
    `test_SubtileBasedLogicalScheduler.py`.
    """
    return setdefault_tail_scaffold_kernel_keys(kernel, pgr)


def _build_minimal_kwa(kernel):
    """Thin wrapper around the shared kwa-builder.

    See `_subtile_tailloop_fixtures.build_minimal_subtile_kwa` for
    the actual SGPR / VGPR / tile-info setup the scaffold relies on.
    """
    return build_minimal_subtile_kwa(kernel)


def _emit_tail_loop_asm(*, fp4: bool, no_tail_loop: bool, pgr: int) -> str:
    """Drive `KernelWriter._emitTailLoopScaffoldSubtile` and return the
    flat asm string. The scaffold runs for all PGR values; PGR>0 layers
    the entry-gate branches (c=0 reset / small-counter realign /
    +1 DU SRD advance) on top of the PGR=0 baseline.
    """
    kernel = _create_kernel(256, 256, fp4=fp4,
                            depthU=256 if fp4 else 64,
                            no_tail_loop=no_tail_loop)
    _augment_kernel_for_tail_scaffold(kernel, pgr)

    kwa = _build_minimal_kwa(kernel)
    tPA = {"is_sparse": False, "tpsMetadata": None}
    tPB = {"is_sparse": False, "tpsMetadata": None}
    module = kwa._emitTailLoopScaffoldSubtile(kernel, tPA, tPB)

    return wrap_with_skiptoend(module)


def _extract_tail_section(asm: str) -> str:
    """Extract the tail-loop section of the emitted asm.

    The tail body lives between the tail-start sentinel and the post-tail
    terminator. The scaffold (`_emitTailLoopScaffoldSubtile`) emits a
    "Tail Loop" banner via `addComment2`; the orphan template in
    `LogicalScheduler.py` instead emits a `TAILLOOP` comment. Either is
    accepted as the start so this helper works against both emit paths.
    Returns "" if no tail block is present.

    End candidates must include the trailing ``:`` so we match the actual
    label *definition* and not earlier branch references to it. Otherwise
    e.g. the ``s_cbranch_scc1 label_SkipTailLoopL`` issued by
    ``calculateLoopNumIter`` (the early-exit when numIter==0) would
    truncate the section before the tail body — hiding the kPosBase init
    and lane mask emitted after ``openLoop``.
    """
    candidates_start = [
        asm.find("Tail Loop"),
        asm.find("TAILLOOP"),
        asm.find("TailLoopBeginL"),
    ]
    candidates_start = [c for c in candidates_start if c >= 0]
    if not candidates_start:
        return ""
    tail_start = min(candidates_start)
    # End at whichever post-tail label definition appears next. The
    # trailing ``:`` is mandatory — see docstring.
    candidates = [
        asm.find("SkipTailLoopL:", tail_start),
        asm.find("label_SkipTailLoopL:", tail_start),
        asm.find("TailLoopEndL:", tail_start),
        asm.find("label_TailLoopEndL:", tail_start),
        asm.find("TailLoopEnd:", tail_start),
        asm.find("SkipToEnd:", tail_start),
    ]
    candidates = [c for c in candidates if c > tail_start]
    tail_end = min(candidates) if candidates else len(asm)
    return asm[tail_start:tail_end]


# ── Tests: PGR=0 ─────────────────────────────────────────────────────────────

class TestTailEmitContent_PGR0:
    """Tail-body content assertions for PGR=0 kernels."""

    @pytest.fixture
    def fp4_pgr0_asm(self):
        return _emit_tail_loop_asm(fp4=True, no_tail_loop=False, pgr=0)

    @pytest.fixture
    def bf16_pgr0_asm(self):
        return _emit_tail_loop_asm(fp4=False, no_tail_loop=False, pgr=0)

    def test_omits_srd_rewind_A(self, fp4_pgr0_asm):
        """Tail body must NOT rewind SrdA. The mainloop's per-iter
        `s_add_u32 SrdA, ..., depthUBytes` already leaves SrdA at the
        K-tail's first byte; a rewind would either move SRD before the
        buffer base (K < DU) or double-count the last DU (K >= DU)."""
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail, "No tail block emitted; cannot test SRD rewind absence"
        assert not re.search(r"s_sub_u32.*SrdA", tail), (
            "Tail must NOT emit `s_sub_u32 … SrdA` rewind. "
            "Tail body excerpt:\n" + tail[:1500]
        )

    def test_omits_srd_rewind_B(self, fp4_pgr0_asm):
        """Same no-rewind requirement for SrdB (see test_omits_srd_rewind_A)."""
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert not re.search(r"s_sub_u32.*SrdB", tail), (
            "Tail must NOT emit `s_sub_u32 … SrdB` rewind"
        )

    def test_omits_srd_rewind_MXSA(self, fp4_pgr0_asm):
        """MX kernels must not rewind the scale tensor SRDs either.

        Same reasoning as the data-tensor SRD: the per-iter MX scale
        SRD update in `SubtileScaleEmit.py::emitScaleGRPtrUpdate`
        already leaves SrdMXSA at the K-tail's first scale byte.
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert not re.search(r"s_sub_u32.*SrdMXSA", tail), (
            "FP4 tail must NOT emit `s_sub_u32 … SrdMXSA` rewind"
        )

    def test_omits_srd_rewind_MXSB(self, fp4_pgr0_asm):
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert not re.search(r"s_sub_u32.*SrdMXSB", tail), (
            "FP4 tail must NOT emit `s_sub_u32 … SrdMXSB` rewind"
        )

    def test_emits_loop_counter_zero_before_closeloop(self, fp4_pgr0_asm):
        """Tail body must zero `LoopCounterL` right before `closeLoop`.

        The legacy `closeLoop(finalLoop=True, tailLoop=True)` emits a
        per-iter `s_sub_i32 LoopCounterL, ..., MatrixInstK` and a
        `s_cbranch_scc0 label_TailLoopBeginL` back-edge that re-runs
        the body while LoopCounterL > 0. The subtile scaffold processes
        the entire K_tail in a single body pass via the `mmak` loop, so
        repeated body iterations would re-accumulate accD and narrow
        the per-MFMA lane mask each pass. Zeroing LoopCounterL forces
        the sub to underflow past 0, dropping the back-edge.
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail, "No tail block emitted; cannot test counter zeroing"
        zero_match = re.search(
            r"s_mov_b32\s+s\[sgprLoopCounterL\],\s*0\b[^\n]*single-iter tail",
            tail,
        )
        assert zero_match, (
            "Tail must emit `s_mov_b32 sgprLoopCounterL, 0` with the "
            "single-iter rationale before the closeLoop decrement. "
            "Tail body excerpt:\n" + tail[-1500:]
        )

        sub_match = re.search(
            r"s_sub_i32\s+s\[sgprLoopCounterL\]", tail
        )
        assert sub_match, (
            "closeLoop must still emit `s_sub_i32 sgprLoopCounterL` "
            "(this test pins the relative ordering)."
        )
        assert zero_match.start() < sub_match.start(), (
            "tail-single-iter zeroing must appear BEFORE the closeLoop "
            "decrement (so the sub underflows past 0 on the first iter)."
        )

    def test_emits_kReg_first_init(self, fp4_pgr0_asm):
        """Per-lane K position must be materialized via
        `v_and_b32 v[?], 63, vgprSerial` + shift chain (kPosBase init)."""
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert re.search(r"v_and_b32.*\b63\b", tail) or \
               re.search(r"v_and_b32.*0x3f", tail), (
            "Tail must compute kReg_first via v_and_b32 with mask 63 "
            "(lane id within wave). Tail excerpt:\n" + tail[:1500]
        )

    def test_emits_lane_mask_for_valuA(self, fp4_pgr0_asm):
        """Each MFMA-input vgpr-pair gets a lane mask zeroing lanes
        whose K-position >= LoopCounterL: `v_cmp_ge_i32 ..., LoopCounterL`
        + `v_cndmask_b32 v[ValuA_*], v[ValuA_*], 0, ...`."""
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert re.search(r"v_cmp_ge_i32.*LoopCounterL", tail), (
            "Tail must compare per-lane K-pos to LoopCounterL via v_cmp_ge_i32"
        )
        assert re.search(r"v_cndmask_b32.*ValuA", tail) or \
               re.search(r"v_cndmask_b32.*valuA", tail), (
            "Tail must zero out-of-range valuA vgprs via v_cndmask_b32"
        )

    def test_emits_lane_mask_for_valuB(self, fp4_pgr0_asm):
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert re.search(r"v_cndmask_b32.*ValuB", tail) or \
               re.search(r"v_cndmask_b32.*valuB", tail), (
            "Tail must zero out-of-range valuB vgprs via v_cndmask_b32"
        )

    def test_emits_lane_mask_for_valuMXSA_MXSB(self, fp4_pgr0_asm):
        """MX kernels need the same lane mask for the scale tensor vgprs."""
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        has_mxsa_mask = (re.search(r"v_cndmask_b32.*ValuMXSA", tail) or
                         re.search(r"v_cndmask_b32.*valuMXSA", tail))
        has_mxsb_mask = (re.search(r"v_cndmask_b32.*ValuMXSB", tail) or
                         re.search(r"v_cndmask_b32.*valuMXSB", tail))
        assert has_mxsa_mask, "Tail must lane-mask valuMXSA"
        assert has_mxsb_mask, "Tail must lane-mask valuMXSB"

    def test_one_kpos_cmp_per_mmak(self, fp4_pgr0_asm):
        """`v_cmp_ge_i32 ..., LoopCounterL` must be emitted at most once
        per mmak (subIterK), not once per (mmak, mma1, mma0) MFMA.

        `_emitTailKPosCmpSubtile` hoists the per-mmak setup (kPosCur
        add + cmp + mask SGPR alloc) out of the (mma1, mma0) inner loop,
        leaving only `_emitTailLaneMaskApplySubtile`'s cndmask chain
        inside. The assertion `cmp_count * 4 <= cndmask_count` is a
        loose ceiling that catches a re-inlining regression without
        coupling to the exact tile-grid shape.
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        cmpLines = re.findall(r"v_cmp_ge_i32.*LoopCounterL", tail)
        cndmaskValuLines = re.findall(r"v_cndmask_b32.*[vV]alu", tail)
        assert cmpLines, "Tail must emit at least one v_cmp_ge_i32 vs LoopCounterL"
        assert cndmaskValuLines, "Tail must emit at least one v_cndmask_b32 against an MFMA input"
        assert len(cmpLines) * 4 <= len(cndmaskValuLines), (
            "Expected cmp count to be much smaller than cndmask count "
            "(cmp hoisted to per-mmak setup, cndmask remains per "
            "(mmak, mma1, mma0) MFMA). Got cmps=%u, cndmasks=%u. The "
            "per-mmak cmp was likely re-inlined into the (mma1, mma0) "
            "loop, which would balloon the cmp count back up to "
            "~cndmask_groups." % (len(cmpLines), len(cndmaskValuLines))
        )

    def test_emits_one_ds_read_per_scale_group(self, fp4_pgr0_asm):
        """Tail LR must emit one ds_read_b32 per scale group; group
        count = lrLocalSubtileGrid[0] * lrLocalSubtileGrid[1]
        (NOT `ceil(localSubtileGrid[0]/2) * localSubtileGrid[1]`,
        which is hard-wired to 1 for MXScaleTilePair regardless of MT
        and would leave the higher scale-group VGPRs uninitialised).
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail, "No tail block emitted; cannot count scale ds_reads"

        mxsa_reads = re.findall(r"ds_read_b32[^\n]*scaleMXSA\[group(\d+)\]", tail)
        mxsb_reads = re.findall(r"ds_read_b32[^\n]*scaleMXSB\[group(\d+)\]", tail)

        # MT256x256 DU=256 with 2x2 WG: lrLocalSubtileGrid = [4, 1]
        # → 4 scale groups per scale tensor.
        expected_groups = ["0", "1", "2", "3"]
        assert mxsa_reads == expected_groups, (
            "Tail must emit ds_read_b32 for MXSA groups 0,1,2,3 (one per "
            "lrLocalSubtileGrid entry). Got: %r. Tail excerpt:\n%s"
            % (mxsa_reads, tail[:3000])
        )
        assert mxsb_reads == expected_groups, (
            "Tail must emit ds_read_b32 for MXSB groups 0,1,2,3. Got: %r"
            % (mxsb_reads,)
        )

    def test_fp4_mt128x128_emits_all_scale_groups(self):
        """Same regression pin as `test_emits_one_ds_read_per_scale_group`
        but at MT128x128 DU=256 — the smallest MT that exposes the bug
        (lrLocalSubtileGrid = [2, 1] → 2 groups per scale tensor).

        Driving the scaffold via a custom kernel (rather than reusing the
        MT256x256 default fixture) so the failure mode at the geometry
        boundary `2 < numGroups <= 4` is locked down independently.
        """
        kernel = _create_kernel(MT0=128, MT1=128, fp4=True, depthU=256,
                                no_tail_loop=False)
        _augment_kernel_for_tail_scaffold(kernel, pgr=0)
        kwa = _build_minimal_kwa(kernel)
        tPA = {"is_sparse": False, "tpsMetadata": None}
        tPB = {"is_sparse": False, "tpsMetadata": None}
        module = kwa._emitTailLoopScaffoldSubtile(kernel, tPA, tPB)
        asm = str(module)
        tail = _extract_tail_section(asm)
        assert tail, "No tail block emitted for MT128x128 FP4 fixture"

        mxsa_reads = re.findall(r"ds_read_b32[^\n]*scaleMXSA\[group(\d+)\]", tail)
        mxsb_reads = re.findall(r"ds_read_b32[^\n]*scaleMXSB\[group(\d+)\]", tail)
        assert mxsa_reads == ["0", "1"], (
            "MT128x128 FP4 tail must emit ds_read for MXSA group0 AND "
            "group1 (lrLocalSubtileGrid = [2, 1]). Got: %r" % (mxsa_reads,)
        )
        assert mxsb_reads == ["0", "1"], (
            "MT128x128 FP4 tail must emit ds_read for MXSB group0 AND "
            "group1. Got: %r" % (mxsb_reads,)
        )

    def test_omits_byte_shift_mask_when_asem_32(self, fp4_pgr0_asm):
        """Step 2-4 of the legacy non-subtile tail (ASEM<32 byte-shift mask
        via `s_lshlrev_b64`) is unnecessary when ASEM>=32. It must not appear.

        This passes today (the imported template doesn't emit step 2-4 either)
        but is a regression guard against accidental porting from the legacy
        emitter.
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        if not tail:
            pytest.skip("Tail block not emitted; nothing to assert")
        assert "s_lshlrev_b64" not in tail.lower(), (
            "ASEM=32 path must not emit the legacy byte-shift mask "
            "(s_lshlrev_b64 sequence)"
        )

    def test_fp4_tail_has_no_scale_lds_prezero(self, fp4_pgr0_asm):
        """FP4 + MX-scale tail must NOT emit an LDS pre-zero before the
        scale GR. The host pads + pre-swizzles MXSA/MXSB on gfx950 so
        over-read bytes are already zero; an earlier pre-zero attempt
        actually regressed FP4 tail validation."""
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail, "No tail block emitted"

        assert "tail pre-zero" not in tail, (
            "FP4 tail must not emit any scale-LDS pre-zero (padded global "
            "covers the LR-read footprint). Found:\n" + tail[:1500]
        )

        gr_a_idx = tail.find("scaleMXSA: DTL b128 load")
        gr_b_idx = tail.find("scaleMXSB: DTL b128 load")
        gr_ab_idx = tail.find("buffer_load_dwordx")
        assert 0 <= gr_ab_idx < gr_a_idx, (
            "AB GR must precede scale GR in tail. A/B@%d MXSA@%d"
            % (gr_ab_idx, gr_a_idx)
        )
        assert 0 <= gr_a_idx < gr_b_idx, (
            "Scale GR order in tail must be MXSA then MXSB. MXSA@%d MXSB@%d"
            % (gr_a_idx, gr_b_idx)
        )

    def test_fp4_mt128x128_tail_has_no_scale_lds_prezero(self):
        """Same regression pin as
        `test_fp4_tail_has_no_scale_lds_prezero` at MT128x128 DU=256,
        the smallest MT that exposed the FP4 partial-K mismatch and
        for which an earlier pre-zero attempt was wired in."""
        kernel = _create_kernel(MT0=128, MT1=128, fp4=True, depthU=256,
                                no_tail_loop=False)
        _augment_kernel_for_tail_scaffold(kernel, pgr=0)
        kwa = _build_minimal_kwa(kernel)
        tPA = {"is_sparse": False, "tpsMetadata": None}
        tPB = {"is_sparse": False, "tpsMetadata": None}
        module = kwa._emitTailLoopScaffoldSubtile(kernel, tPA, tPB)
        asm = str(module)
        tail = _extract_tail_section(asm)
        assert tail, "No tail block emitted for MT128x128 FP4 fixture"

        assert "tail pre-zero" not in tail, (
            "MT128x128 FP4 tail must not emit any scale-LDS pre-zero. "
            "Got:\n" + tail[:1500]
        )

        gr_a_idx = tail.find("scaleMXSA: DTL b128 load")
        gr_b_idx = tail.find("scaleMXSB: DTL b128 load")
        assert 0 <= gr_a_idx < gr_b_idx, (
            "Scale GR order in MT128x128 tail must be MXSA then MXSB. "
            "MXSA@%d MXSB@%d" % (gr_a_idx, gr_b_idx)
        )

    def test_bf16_tail_omits_scale_emit(self, bf16_pgr0_asm):
        """BF16 (no MX scale) must not mention MXSA/MXSB in the tail,
        nor any scale-LDS pre-zero (the helper has been removed entirely).
        """
        tail = _extract_tail_section(bf16_pgr0_asm)
        if not tail:
            pytest.skip("Tail block not emitted; nothing to check")
        assert "tail pre-zero" not in tail, (
            "BF16 tail must not emit any scale-LDS pre-zero; got:\n"
            + tail[:1500]
        )
        assert "scaleMXSA" not in tail and "scaleMXSB" not in tail, (
            "BF16 tail must not mention MXSA/MXSB scale tensors"
        )

    def test_NoTailLoop_true_omits_all_tail_emit(self):
        """Aligned-K kernels (NoTailLoop=True) must not emit any tail
        body, but must still emit the `SkipTailLoopL` label so any
        branches targeting it resolve."""
        asm = _emit_tail_loop_asm(fp4=True, no_tail_loop=True, pgr=0)
        assert "Tail Loop" not in asm, "NoTailLoop=True must skip emit body"
        assert "TAILLOOP" not in asm
        assert "TailLoopBeginL" not in asm
        assert "TailLoopEndL" not in asm
        assert "SkipTailLoopL" in asm, \
            "SkipTailLoopL must remain emitted even with NoTailLoop=True"
        assert not re.search(r"s_sub_u32.*SrdA.*K_rem", asm)


# ── Tests: PGR=2 ─────────────────────────────────────────────────────────────

class TestTailEmitContent_PGR2:
    """PGR=2 (scheduler-managed prefetch) tail-emit assertions.

    The PGR>0 path reuses the PGR=0 scaffold and layers three mutually-
    exclusive entry-gate branches on top, keyed off origCounter:
      - c == 0:                reset (zero accD, undo preLoop GR_INC).
      - 0 < origCounter < PGR: small-counter LWA realign.
      - origCounter >= PGR:    +1 DU SRD advance.
    The assertions below pin the structural presence of each gate;
    they avoid coupling to alloc-order-dependent sgpr/imm operands.
    """

    @pytest.fixture
    def fp4_pgr2_asm(self):
        return _emit_tail_loop_asm(fp4=True, no_tail_loop=False, pgr=2)

    @pytest.fixture
    def bf16_pgr2_asm(self):
        return _emit_tail_loop_asm(fp4=False, no_tail_loop=False, pgr=2)

    def test_emits_SkipTailLoopL_label(self, fp4_pgr2_asm):
        """PGR=2 must reuse the same SkipTailLoopL label as PGR=0."""
        assert "SkipTailLoopL:" in fp4_pgr2_asm

    def test_emits_origCounter_snapshot(self, fp4_pgr2_asm):
        r"""PGR>0 must snapshot `OrigLoopCounter` before
        `calculateLoopNumIter` resets it; the entry-gate decisions
        downstream need the original K // DU value.
        """
        assert re.search(
            r"s_mov_b32\s+s\[?\d+\]?,\s*s\[sgprOrigLoopCounter\][^\n]*snapshot K//DU",
            fp4_pgr2_asm
        ), (
            "PGR=2 must snapshot OrigLoopCounter before "
            "calculateLoopNumIter overwrites it. Asm head:\n"
            + fp4_pgr2_asm[:2500]
        )

    def test_emits_c0_reset_compare_and_branch(self, fp4_pgr2_asm):
        """For origCounter==0, branch to `PGRTailC0Reset<L>` rather
        than skip the tail. On gfx950 `buffer_load_*_lds` with oob=1
        suppresses (not zeroes) the LDS write, so NLL MFMA'd MT0 with
        garbage in OOB subIterK slots; the lane-masked tail re-issue
        is what restores correctness.
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail, (
            "Tail section not extractable.\n"
            "Full asm head:\n" + fp4_pgr2_asm[:2500]
        )
        # Match on the literal `, 0` immediate of the c=0 cmp.
        assert re.search(r"s_cmp_eq_u32[^\n]*\b0\b[^\n]*origCounter == 0", tail), (
            "Tail must emit `s_cmp_eq_u32 ..., 0` for the c=0 reset path"
        )
        assert re.search(
            r"s_cbranch_scc1[^\n]*PGRTailC0Reset", tail
        ), (
            "Tail must conditionally branch to PGRTailC0ResetL on the "
            "c=0 compare hit (NOT SkipTailLoopL — the c=0 path must "
            "fall through to the tail body after resetting accD)"
        )
        # The c=0 path must NOT branch to SkipTailLoopL: pin by checking
        # the c=0 cmp's matching branch points at PGRTailC0Reset (above)
        # and is not followed by a SkipTailLoop branch within the same
        # gating block.
        c0_cmp_pos = tail.find("origCounter == 0")
        if c0_cmp_pos >= 0:
            after_c0 = tail[c0_cmp_pos:c0_cmp_pos + 400]
            assert "SkipTailLoop" not in after_c0, (
                "c=0 path must not branch to SkipTailLoopL"
            )

    def test_emits_c0_reset_label_and_accD_zero(self, fp4_pgr2_asm):
        """The c=0 reset block must zero accD via `initVgprTilesToZero`
        (same helper kernelBodySubtile uses at kernel start).
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        c0_label_pos = tail.find("PGRTailC0ResetL")
        assert c0_label_pos >= 0, "tail must define PGRTailC0ResetL"
        c0_block = tail[c0_label_pos:c0_label_pos + 4000]
        assert "Init D vgprTiles to zero" in c0_block, (
            "c=0 reset must invoke `initVgprTilesToZero(D)`.\n"
            "c=0 block excerpt:\n" + c0_block[:1500]
        )

    def _extract_c0_block(self, tail):
        """Extract the entire `label_PGRTailC0ResetL:` block up through
        (but not including) the `label_PGRTailEntryL:` label that
        terminates it.

        The c=0 reset block contains the accD-init MFMA sweep (which on
        FP4 + MX scale fixture configurations can run to 16 `v_mfma_i32`
        instructions for a 256-accVGPR D tile) plus the per-tensor
        s_sub/s_subb chain and the LWA XOR chain — easily several KB
        of asm. Extracting "from the label definition to the next label
        definition" keeps the assertions robust to the variable-length
        init sweep. (We anchor on the `:` suffix to find the actual
        label *definition* rather than the upstream `s_cbranch ...
        label_PGRTailC0ResetL` reference that points at it.)
        """
        label_def = "label_PGRTailC0ResetL:"
        c0_label_pos = tail.find(label_def)
        if c0_label_pos < 0:
            return ""
        # Skip past the label definition itself so we find the
        # TERMINATING label (PGRTailEntryL), not the label that opens
        # this block.
        c0_body_start = c0_label_pos + len(label_def)
        c0_end = tail.find("label_PGRTailEntryL:", c0_body_start)
        if c0_end < 0:
            return tail[c0_label_pos:]
        return tail[c0_label_pos:c0_end]

    def test_emits_c0_srd_subtract_with_borrow(self, fp4_pgr2_asm):
        """For PGR>=2, the c=0 reset block must subtract 1 DU from
        each Srd<tc> (with borrow) to undo preLoop's GR_INC advance.
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        c0_block = self._extract_c0_block(tail)
        assert c0_block, "PGRTailC0ResetL block not found in tail"
        assert re.search(
            r"s_sub_u32[^\n]*sgprSrdA[^\n]*undo preLoop GR_INC", c0_block
        ), (
            "c=0 reset must emit `s_sub_u32 SrdA, ..., depthUBytes` to "
            "undo preLoop's GR_INC advance.\n"
            "c=0 block excerpt:\n" + c0_block[:1500]
        )
        assert re.search(
            r"s_subb_u32[^\n]*sgprSrdA\+1", c0_block
        ), "c=0 reset must propagate borrow to SrdA+1"
        assert re.search(
            r"s_sub_u32[^\n]*sgprSrdB[^\n]*undo preLoop GR_INC", c0_block
        )
        assert re.search(
            r"s_subb_u32[^\n]*sgprSrdB\+1", c0_block
        )

    def test_emits_c0_lwa_xor_realign(self, fp4_pgr2_asm):
        """For PGR>=2, the c=0 reset block must XOR LWA back to buf 0
        to undo preLoop's GR_INC LWA swap.
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        c0_block = self._extract_c0_block(tail)
        assert c0_block, "PGRTailC0ResetL block not found in tail"
        assert re.search(
            r"s_xor_b32[^\n]*LocalWriteBaseAddrA[^\n]*SwapA[^\n]*undo preLoop GR_INC",
            c0_block
        ), (
            "c=0 reset must XOR LocalWriteBaseAddrA with SwapA.\n"
            "c=0 block tail:\n" + c0_block[-1500:]
        )
        assert re.search(
            r"s_xor_b32[^\n]*LocalWriteBaseAddrB[^\n]*SwapB[^\n]*undo preLoop GR_INC",
            c0_block
        )

    def test_emits_srd_advance_A_with_carry(self, fp4_pgr2_asm):
        """Large-counter path (origCounter >= PGR): advance SrdA by
        one DU (with carry) so the tail GR re-issues at K_aligned.
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert re.search(r"s_add_u32[^\n]*sgprSrdA[^\n]*advance SrdA by 1 DU", tail), (
            "Tail must emit `s_add_u32 SrdA, ..., depthUBytes`"
        )
        assert re.search(r"s_addc_u32[^\n]*sgprSrdA\+1", tail), (
            "SRD advance must propagate carry to SrdA+1"
        )

    def test_emits_srd_advance_B_with_carry(self, fp4_pgr2_asm):
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert re.search(r"s_add_u32[^\n]*sgprSrdB[^\n]*advance SrdB by 1 DU", tail)
        assert re.search(r"s_addc_u32[^\n]*sgprSrdB\+1", tail)

    def test_emits_srd_advance_MXSA_MXSB(self, fp4_pgr2_asm):
        """MX scale tensors must also advance one DU."""
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert re.search(
            r"s_add_u32[^\n]*sgprSrdMXSA[^\n]*advance SrdMXSA by 1 DU", tail
        ), "MX FP4 tail must advance SrdMXSA"
        assert re.search(
            r"s_add_u32[^\n]*sgprSrdMXSB[^\n]*advance SrdMXSB by 1 DU", tail
        ), "MX FP4 tail must advance SrdMXSB"

    def test_emits_small_counter_lwa_realign(self, fp4_pgr2_asm):
        """Small-counter (PGR=2 origCounter==1) path must XOR LWA back
        to match the LR buffer; preLoop's lone gr_inc left it out of
        sync with LR (NGLL never ran for counter<=1).
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert re.search(
            r"s_xor_b32[^\n]*LocalWriteBaseAddrA[^\n]*SwapA",
            tail
        ), "Tail must XOR LocalWriteBaseAddrA with SwapA"
        assert re.search(
            r"s_xor_b32[^\n]*LocalWriteBaseAddrB[^\n]*SwapB",
            tail
        ), "Tail must XOR LocalWriteBaseAddrB with SwapB"

    def test_omits_old_srd_rewind_in_main_tail_body(self, fp4_pgr2_asm):
        """No unconditional SRD rewind in the main tail body. Any
        `s_sub_u32 Srd<tc>` must carry the `undo preLoop GR_INC`
        attribution (i.e. live inside the c=0 reset block).
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        for m in re.finditer(r"s_sub_u32[^\n]*sgprSrdA[^\n]*", tail):
            assert "undo preLoop GR_INC" in m.group(0), (
                "Found `s_sub_u32 SrdA` outside the c=0 reset block. "
                "Offending line:\n" + m.group(0)
            )

    def test_pgr2_reuses_lane_mask(self, fp4_pgr2_asm):
        """PGR=2 reuses the PGR=0 lane-mask emit unchanged."""
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert re.search(r"v_cndmask_b32.*ValuA", tail) or \
               re.search(r"v_cndmask_b32.*valuA", tail), (
            "PGR=2 tail must lane-mask valuA"
        )
        assert re.search(r"v_cmp_ge_i32.*LoopCounterL", tail), (
            "PGR=2 tail must compare per-lane K-pos to LoopCounterL"
        )

    def test_pgr2_reuses_kReg_first_init(self, fp4_pgr2_asm):
        """PGR=2 reuses the per-lane K-position init from PGR=0."""
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert re.search(r"v_and_b32.*\b63\b", tail) or \
               re.search(r"v_and_b32.*0x3f", tail)

    def test_pgr2_loop_counter_zero_before_closeloop(self, fp4_pgr2_asm):
        """PGR=2 reuses the single-iter forcing
        (`s_mov_b32 sgprLoopCounterL, 0` before closeLoop).
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert re.search(
            r"s_mov_b32\s+s\[sgprLoopCounterL\],\s*0\b[^\n]*single-iter tail",
            tail
        ), "PGR=2 tail must force single-iter via LoopCounterL=0"


# ── Tests: PGR=1 ─────────────────────────────────────────────────────────────

class TestTailEmitContent_PGR1:
    """PGR=1 (single-buffer prefetch) tail-emit assertions.

    PGR=1 has no NGLL block and no preLoop `gr_inc`, so the LDS double-
    buffers never get out of sync (LWA and LR both stay at buf 0 until a
    mainloop iter flips both). The small-counter realign branch is
    therefore inert for PGR=1 (`origCounter < 1` only means `==0`, which
    already returned via skip-tail). The SRD-advance branch DOES still
    fire (mainloop runs `counter-1` iters, leaving SRD at `(N-1)*DU`).
    """

    @pytest.fixture
    def bf16_pgr1_asm(self):
        return _emit_tail_loop_asm(fp4=False, no_tail_loop=False, pgr=1)

    def test_emits_SkipTailLoopL_label(self, bf16_pgr1_asm):
        assert "SkipTailLoopL:" in bf16_pgr1_asm

    def test_emits_origCounter_snapshot(self, bf16_pgr1_asm):
        assert re.search(
            r"s_mov_b32\s+s\[?\d+\]?,\s*s\[sgprOrigLoopCounter\][^\n]*snapshot K//DU",
            bf16_pgr1_asm
        )

    def test_emits_c0_reset_compare_and_branch(self, bf16_pgr1_asm):
        """PGR=1 also routes c=0 through the reset path. SRD-sub /
        LWA-XOR are no-ops (no preLoop GR_INC) but accD must still be
        zeroed to undo NLL's garbage accumulation.
        """
        tail = _extract_tail_section(bf16_pgr1_asm)
        assert tail
        assert re.search(r"s_cmp_eq_u32[^\n]*\b0\b[^\n]*origCounter == 0", tail)
        assert re.search(
            r"s_cbranch_scc1[^\n]*PGRTailC0Reset", tail
        )

    def test_emits_srd_advance_AB(self, bf16_pgr1_asm):
        tail = _extract_tail_section(bf16_pgr1_asm)
        assert tail
        assert re.search(r"s_add_u32[^\n]*sgprSrdA[^\n]*advance SrdA by 1 DU", tail)
        assert re.search(r"s_add_u32[^\n]*sgprSrdB[^\n]*advance SrdB by 1 DU", tail)

    def test_omits_c0_srd_subtract_pgr1(self, bf16_pgr1_asm):
        """PGR=1 has no preLoop GR_INC, so the c=0 reset block must NOT
        emit `s_sub_u32 Srd<tc>`.
        """
        tail = _extract_tail_section(bf16_pgr1_asm)
        assert tail
        c0_label_pos = tail.find("PGRTailC0Reset")
        assert c0_label_pos >= 0
        c0_block = tail[c0_label_pos:c0_label_pos + 4000]
        assert not re.search(
            r"s_sub_u32[^\n]*sgprSrdA[^\n]*undo preLoop GR_INC", c0_block
        ), "PGR=1 c=0 reset must NOT emit `s_sub_u32 SrdA`"

    def test_emits_c0_accD_zero_pgr1(self, bf16_pgr1_asm):
        """PGR=1 c=0 reset still zeroes accD via `initVgprTilesToZero`."""
        tail = _extract_tail_section(bf16_pgr1_asm)
        assert tail
        c0_label_pos = tail.find("PGRTailC0Reset")
        assert c0_label_pos >= 0
        c0_block = tail[c0_label_pos:c0_label_pos + 4000]
        assert "Init D vgprTiles to zero" in c0_block, (
            "PGR=1 c=0 reset must invoke `initVgprTilesToZero(D)` to "
            "undo NLL's garbage accumulation."
        )

    def test_emits_small_counter_compare(self, bf16_pgr1_asm):
        """The compare against PGR is still emitted; for PGR=1 it
        resolves to `< 1` and is inert at runtime (origCounter==0
        already branched away via the c=0 reset path above). Emitted
        for structural symmetry with PGR=2.
        """
        tail = _extract_tail_section(bf16_pgr1_asm)
        assert tail
        # Source is the unnamed snapshot sgpr (rendered as `s29`);
        # accept any `s\[?\d+\]?, 1`.
        assert re.search(
            r"s_cmp_lt_u32\s+s\[?\d+\]?,\s*1\b[^\n]*origCounter < PGR", tail
        )
