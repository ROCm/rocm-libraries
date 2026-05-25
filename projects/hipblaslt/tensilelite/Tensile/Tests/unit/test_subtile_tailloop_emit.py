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


def _collect_per_mmak_early_exit_thresholds(tail: str):
    """Walk the tail asm in order and return the `MIK*(subIterK+1)`
    threshold value for each per-mmak early-exit cmp, in emission
    order. Accepts both forms emitted by
    `_emitSubtileScalarCmpLitOrStaged`:

      * Inline (threshold in [-16..64]):
          `s_cmp_le_u32 ..., 0xN ... MIK*(subIterK+1)?`
      * Staged (threshold > 64 or < -16):
          `s_mov_b32 sN, 0xT ... stage literal T (non-inline) for cmp src1`
          `s_cmp_le_u32 ..., sN ... MIK*(subIterK+1)?`

    Pairs each cmp with its immediately-preceding staging mov by
    matching sgpr name (so a future ordering change doesn't silently
    cross-pair).
    """
    thresholds = []
    pending_stage = {}  # sgpr name -> staged literal value
    for line in tail.split("\n"):
        m_stage = re.search(
            r"s_mov_b32\s+s(\d+)\s*,\s*(0x[0-9a-fA-F]+)[^\n]*"
            r"stage literal\s+(-?\d+)\s+\(non-inline\) for cmp src1", line)
        if m_stage:
            pending_stage["s" + m_stage.group(1)] = int(m_stage.group(2), 0)
            continue
        if "MIK*(subIterK+1)" not in line:
            continue
        m_inline = re.search(
            r"s_cmp_le_u32[^\n]*sgprLoopCounterL[^\n]*,\s*(0x[0-9a-fA-F]+)",
            line)
        if m_inline:
            thresholds.append(int(m_inline.group(1), 0))
            continue
        m_staged = re.search(
            r"s_cmp_le_u32[^\n]*sgprLoopCounterL[^\n]*,\s*(s\d+)\b", line)
        assert m_staged, "per-mmak cmp didn't match inline or staged form: %r" % line
        sg = m_staged.group(1)
        assert sg in pending_stage, (
            "per-mmak cmp references %s but no preceding "
            "`stage literal ... (non-inline) for cmp src1` s_mov_b32 "
            "was found for it.\nLine: %r" % (sg, line))
        thresholds.append(pending_stage.pop(sg))
    return thresholds


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

    def test_omits_closeloop_emit(self, fp4_pgr0_asm):
        """Tail scaffold must NOT emit `closeLoop(... finalLoop=True)`.

        Per nakajee review (PR #7636): the subtile tail body processes
        the entire K_tail in a single pass via the `mmak` loop above
        (every lane mask was emitted against the current
        `LoopCounterL = K mod DU` snapshot). The `closeLoop` emit
        would produce a per-iter `s_sub_i32 LoopCounterL, ..., MIK` +
        `s_cbranch_scc0 label_TailLoopBeginL` back-edge that re-runs
        the body while LoopCounterL > 0. Since the body is one-shot
        for subtile, those instructions are dead code; the only
        useful effect (the `TailLoopEndL:` label) has no in-bound
        branches. We also no longer need the `s_mov_b32 LoopCounterL,
        0` that previously forced fall-through.
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail, "No tail block emitted; cannot test closeLoop omission"
        assert not re.search(
            r"s_sub_i32\s+s\[sgprLoopCounterL\][^\n]*dec counterL", tail
        ), (
            "Tail must NOT emit the closeLoop per-iter decrement "
            "`s_sub_i32 sgprLoopCounterL ... dec counterL` "
            "(closeLoop is unnecessary for the single-pass subtile "
            "tail body). Tail excerpt:\n" + tail[-1500:]
        )
        assert not re.search(
            r"s_mov_b32\s+s\[sgprLoopCounterL\],\s*0\b[^\n]*single-iter tail",
            tail,
        ), (
            "Tail must NOT emit the `s_mov_b32 sgprLoopCounterL, 0` "
            "force-fall-through (it only existed to neutralize the "
            "now-removed closeLoop decrement). Tail excerpt:\n"
            + tail[-1500:]
        )
        assert "TailLoopEndL" not in tail, (
            "Tail must NOT define `TailLoopEndL` (closeLoop emitted "
            "it, but nothing branches there). Tail excerpt:\n"
            + tail[-1500:]
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
        add + cmp + mask SGPR alloc) out of the (mma1, mma0) inner
        loop; the cndmask chain is inlined directly into the tail
        scaffold's mmak loop with a `seenVgpr` set so each unique
        ValuA/B/MXSA/MXSB vgpr is masked at most once per mmak. The
        assertion `cmp_count * 4 <= cndmask_count` is a loose ceiling
        that catches a re-inlining regression without coupling to the
        exact tile-grid shape.
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

    def test_emits_per_mmak_early_exit(self, bf16_pgr0_asm):
        """After each non-final mmak the scaffold must emit
        `s_cmp_le_u32 LoopCounterL, MIK*(mmak+1)` + `s_cbranch_scc1
        label_SkipTailLoopL`. Once LoopCounterL is consumed by the
        mmaks already issued, remaining mmaks operate on lanes that
        the cndmask + sub-lane refine have already zeroed -- branching
        past them skips wasted MFMA cycles.
        """
        tail = _extract_tail_section(bf16_pgr0_asm)
        assert tail, "No tail block emitted"
        cmpMatches = re.findall(
            r"s_cmp_le_u32[^\n]*sgprLoopCounterL[^\n]*MIK\*\(subIterK\+1\)",
            tail)
        brMatches = re.findall(
            r"s_cbranch_scc1[^\n]*label_SkipTailLoopL[^\n]*early-exit tail",
            tail)
        assert cmpMatches, (
            "Tail must emit at least one per-mmak `s_cmp_le_u32 LoopCounterL, "
            "MIK*(subIterK+1)` early-exit guard.\nTail excerpt:\n" + tail[-1500:]
        )
        assert len(cmpMatches) == len(brMatches), (
            "Per-mmak early exit must pair s_cmp_le_u32 with s_cbranch_scc1; "
            "got %d cmps vs %d branches" % (len(cmpMatches), len(brMatches))
        )

    def test_per_mmak_early_exit_threshold_progression(self, fp4_pgr0_asm):
        """Each emitted per-mmak early exit must compare against
        `MIK * (subIterK + 1)`, so the threshold strictly increases
        with subIterK. Pins the formula rather than a fixed value
        (so future tile-grid changes don't silently corrupt it).

        gfx950 VOPC/SOPC inline-constant range is -16..64, so a
        threshold > 64 (FP4 MIK=128 hits every entry) is staged via
        `s_mov_b32 sN, <lit> ... stage literal <lit> (non-inline) for
        cmp src1` immediately before the cmp; thresholds in range
        appear directly as a hex literal in the cmp's src1.
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        thresholds = _collect_per_mmak_early_exit_thresholds(tail)
        assert thresholds, (
            "Tail must emit at least one per-mmak `s_cmp_le_u32 LoopCounterL, "
            "MIK*(subIterK+1)`"
        )
        miK = 128  # FP4 fixture: MatrixInstK
        for idx, thr in enumerate(thresholds):
            assert thr == miK * (idx + 1), (
                "Per-mmak early-exit threshold[%u]=%#x must equal "
                "MIK*(subIterK+1)=%#x" % (idx, thr, miK * (idx + 1))
            )

    def test_per_mmak_early_exit_omits_after_final_mmak(self, bf16_pgr0_asm):
        """The final mmak's natural exit is the closeLoop sub +
        single-iter zero, so emitting an early-exit after it would be
        wasted asm. Pin that the early-exit branch count == mmak count
        - 1 (one branch between every consecutive mmak pair, none
        after the last).

        The bf16 fixture has localMMATileGrid[1] == 2 -> exactly one
        early-exit branch in the tail. If the production tile grid
        changes, this pin still holds via the parameterised count.
        """
        tail = _extract_tail_section(bf16_pgr0_asm)
        assert tail
        brCount = len(re.findall(
            r"s_cbranch_scc1[^\n]*label_SkipTailLoopL[^\n]*early-exit tail",
            tail))
        # bf16 fixture: MIK=32, DepthU=64 -> 2 mmak iters, so 1 exit.
        assert brCount == 1, (
            "bf16 fixture should emit exactly 1 per-mmak early exit "
            "(localMMATileGrid[1]=2 -> mmak in {0,1}, exit only after "
            "mmak=0). Got %d.\nTail excerpt:\n%s"
            % (brCount, tail[-2000:])
        )

    def test_per_mmak_early_exit_absent_when_NoTailLoop(self):
        """NoTailLoop=True elides the entire tail body, so no per-mmak
        early-exit should be emitted anywhere.
        """
        asm = _emit_tail_loop_asm(fp4=False, no_tail_loop=True, pgr=0)
        assert not re.search(
            r"s_cmp_le_u32[^\n]*sgprLoopCounterL[^\n]*MIK\*\(subIterK\+1\)",
            asm
        ), "NoTailLoop must not emit per-mmak early exits"

    def test_per_mmak_early_exit_inline_when_consumedK_fits(self, bf16_pgr0_asm):
        """BF16 fixture has MIK=32, DepthU=64 -> the only per-mmak
        threshold is `32 * 1 = 32`, which fits the gfx950 inline range
        [-16..64]. The scaffold must emit the cmp with the literal
        directly in src1 -- NO staging `s_mov_b32 sN, 0x20 ... stage
        literal ... for cmp src1` should appear before the cmp.
        """
        tail = _extract_tail_section(bf16_pgr0_asm)
        assert tail
        thresholds = _collect_per_mmak_early_exit_thresholds(tail)
        assert thresholds == [32], (
            "BF16 fixture expected one cmp at consumedK=32. Got %r" % (thresholds,)
        )
        assert not re.search(
            r"s_mov_b32[^\n]*stage literal\s+32\s+\(non-inline\)", tail), (
            "consumedK=32 fits the gfx950 inline range; the scaffold "
            "must NOT stage it through a scratch sgpr."
        )
        assert re.search(
            r"s_cmp_le_u32[^\n]*sgprLoopCounterL[^\n]*,\s*0x20\b"
            r"[^\n]*MIK\*\(subIterK\+1\)", tail), (
            "consumedK=32 must appear as a direct inline literal 0x20 "
            "in the s_cmp_le_u32 src1."
        )

    def test_per_mmak_early_exit_staged_when_consumedK_exceeds_inline(self, fp4_pgr0_asm):
        """FP4 fixture has MIK=128 -> every per-mmak threshold is
        128, 256, ... (all > 64). The scaffold must stage each via
        `s_mov_b32 sN, 0xT ... stage literal T (non-inline) for cmp
        src1` before the cmp, and the cmp's src1 must reference an
        sgpr (not a raw literal).
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        cmpLines = [ln for ln in tail.split("\n")
                    if "MIK*(subIterK+1)" in ln and "s_cmp_le_u32" in ln]
        assert cmpLines, "Expected at least one per-mmak early-exit cmp in FP4 tail"
        for ln in cmpLines:
            assert re.search(
                r"s_cmp_le_u32[^\n]*sgprLoopCounterL[^\n]*,\s*s\d+\b", ln), (
                "FP4 per-mmak early-exit cmp must use a staged sgpr "
                "(consumedK >= 128 exceeds the gfx950 inline range).\n"
                "Got: %r" % ln
            )
        stageCount = len(re.findall(
            r"s_mov_b32[^\n]*stage literal\s+\d+\s+\(non-inline\) for cmp src1",
            tail))
        assert stageCount == len(cmpLines), (
            "Each staged cmp must have exactly one preceding `stage "
            "literal ... (non-inline) for cmp src1` s_mov_b32. "
            "Got %d stages for %d cmps." % (stageCount, len(cmpLines))
        )

    def test_per_mmak_early_exit_boundary_consumedK_64(self):
        """Direct-vs-staged boundary: consumedK == 64 sits at the high
        end of the gfx950 inline range and must remain inline. Drives
        a custom bf16 fixture with DU=128 -> mmak ∈ {0,1,2,3} so the
        first non-final boundary (mmak=0 -> consumedK=32) and the
        second (mmak=1 -> consumedK=64) both fit inline, while
        mmak>=2 (consumedK >= 96) must stage. Pins that 64 is treated
        as the inclusive upper bound.
        """
        kernel = _create_kernel(MT0=256, MT1=256, fp4=False, depthU=128,
                                no_tail_loop=False)
        _augment_kernel_for_tail_scaffold(kernel, pgr=0)
        kwa = _build_minimal_kwa(kernel)
        tPA = {"is_sparse": False, "tpsMetadata": None}
        tPB = {"is_sparse": False, "tpsMetadata": None}
        module = kwa._emitTailLoopScaffoldSubtile(kernel, tPA, tPB)
        tail = _extract_tail_section(wrap_with_skiptoend(module))
        assert tail
        thresholds = _collect_per_mmak_early_exit_thresholds(tail)
        # DU=128, MIK=32 -> 4 mmaks -> 3 non-final boundaries:
        #   consumedK = 32, 64, 96.
        assert thresholds == [32, 64, 96], (
            "Expected DU=128 bf16 fixture to emit 3 per-mmak early "
            "exits at consumedK in {32,64,96}; got %r" % (thresholds,)
        )
        # 32 and 64 must NOT be staged; 96 MUST be staged.
        assert not re.search(
            r"s_mov_b32[^\n]*stage literal\s+32\s+\(non-inline\)", tail), (
            "consumedK=32 must not be staged (inline range)."
        )
        assert not re.search(
            r"s_mov_b32[^\n]*stage literal\s+64\s+\(non-inline\)", tail), (
            "consumedK=64 must not be staged: 64 is the inclusive "
            "upper bound of the gfx950 inline-constant range."
        )
        assert re.search(
            r"s_mov_b32[^\n]*stage literal\s+96\s+\(non-inline\) for cmp src1",
            tail), (
            "consumedK=96 exceeds the gfx950 inline range and must be "
            "staged through a scratch sgpr."
        )


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

    def test_omits_origCounter_snapshot_after_hoist(self, fp4_pgr2_asm):
        r"""PGR>0 tail-entry gating is now hoisted ABOVE
        `calculateLoopNumIter` (per sebvince #PR-7683 design), so the
        `s_mov_b32 sX, sgprOrigLoopCounter ... snapshot K//DU` SGPR
        snapshot is no longer needed -- the gate cmps read
        `sgprOrigLoopCounter` directly while it still holds the
        original K//DU value (calculateLoopNumIter zeroes it).
        """
        assert not re.search(
            r"s_mov_b32\s+s\[?\d+\]?,\s*s\[sgprOrigLoopCounter\][^\n]*snapshot K//DU",
            fp4_pgr2_asm
        ), (
            "PGR=2 must NOT emit the legacy OrigLoopCounter snapshot "
            "(the gating block reads sgprOrigLoopCounter directly now "
            "that it sits before calculateLoopNumIter).\nAsm head:\n"
            + fp4_pgr2_asm[:2500]
        )

    def test_gating_reads_OrigLoopCounter_directly(self, fp4_pgr2_asm):
        r"""After the hoist, the c=0 cmp must source from
        `sgprOrigLoopCounter` directly (not a scratch snapshot sgpr).
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert re.search(
            r"s_cmp_eq_u32\s+s\[sgprOrigLoopCounter\][^\n]*\b0\b"
            r"[^\n]*origCounter == 0", tail
        ), (
            "PGR=2 c=0 gate must compare sgprOrigLoopCounter directly "
            "against 0 (no intermediate snapshot sgpr).\nTail head:\n"
            + tail[:1500]
        )

    def test_gating_block_precedes_TailLoopBeginL(self, fp4_pgr2_asm):
        r"""Per sebvince #PR-7683 design, the PGR>0 SRD-advance / LWA-
        XOR gating block must sit BEFORE `TailLoopBeginL` (the openLoop
        label) AND BEFORE the calculateLoopNumIter K%DU==0 early-exit
        cmp/branch. Pin by checking the c=0 origCounter cmp appears
        before the SkipTailLoopL early-exit cmp/branch.
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        c0_pos = tail.find("origCounter == 0")
        early_exit_pos = tail.find("skip to end of tail loop")
        assert c0_pos >= 0, "c=0 origCounter gate not found in tail"
        assert early_exit_pos >= 0, (
            "calculateLoopNumIter's K%DU==0 early-exit not found in tail"
        )
        assert c0_pos < early_exit_pos, (
            "PGR>0 gating block must precede the K%%DU==0 early-exit "
            "cmp/branch (c0_pos=%d, early_exit_pos=%d).\n"
            "Tail head:\n%s" % (c0_pos, early_exit_pos, tail[:2500])
        )
        # And the gating block must precede TailLoopBeginL too.
        tail_begin_pos = tail.find("TailLoopBeginL")
        if tail_begin_pos >= 0:
            assert c0_pos < tail_begin_pos, (
                "PGR>0 gating must precede TailLoopBeginL "
                "(c0_pos=%d, tail_begin=%d)." % (c0_pos, tail_begin_pos)
            )

    def test_emits_c0_reset_compare_and_branch(self, fp4_pgr2_asm):
        """For origCounter==0, the tail scaffold's c=0 compare branches
        directly into the tail body (PGRTailEntry<L>), skipping the
        small-counter realign and large-counter SRD-advance paths.

        Background: with the upstream SkipSubtileMainLoop<L> gate in
        kernelBodySubtile (added per reviewer comment), origCounter==0
        skips the entire preLoop / mainloop / NGLL / NLL block. SRDs
        stay at K=0, LWA/LRA stay at buf 0, accD stays at zero — so the
        tail body just needs to run as-is, no undo needed. The legacy
        PGRTailC0Reset<L> block (which used to live here) is gone.
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail, (
            "Tail section not extractable.\n"
            "Full asm head:\n" + fp4_pgr2_asm[:2500]
        )
        # Match on the literal `, 0` immediate of the c=0 cmp.
        assert re.search(r"s_cmp_eq_u32[^\n]*\b0\b[^\n]*origCounter == 0", tail), (
            "Tail must emit `s_cmp_eq_u32 ..., 0` for the c=0 gate"
        )
        assert re.search(
            r"s_cbranch_scc1[^\n]*PGRTailEntry", tail
        ), (
            "Tail must conditionally branch to PGRTailEntry<L> on the "
            "c=0 compare hit (the legacy PGRTailC0Reset<L> block has "
            "been replaced by an upstream skip gate)"
        )
        # The c=0 path must NOT branch to SkipTailLoopL or
        # PGRTailC0Reset (legacy): pin by checking the c=0 cmp's
        # matching branch points at PGRTailEntry.
        assert "PGRTailC0Reset" not in tail, (
            "Tail must no longer reference the legacy PGRTailC0Reset "
            "block (handled upstream by SkipSubtileMainLoop<L>)"
        )
        c0_cmp_pos = tail.find("origCounter == 0")
        if c0_cmp_pos >= 0:
            after_c0 = tail[c0_cmp_pos:c0_cmp_pos + 400]
            assert "SkipTailLoop" not in after_c0, (
                "c=0 path must not branch to SkipTailLoopL"
            )

    def test_omits_c0_reset_label(self, fp4_pgr2_asm):
        """The tail must NOT define a PGRTailC0Reset<L> label any more.

        The legacy c=0 reset block (which used to zero accD via
        initVgprTilesToZero, subtract one DU from each SRD, and XOR LWA
        back to buf 0) has been removed: those undos are now obviated
        by the upstream SkipSubtileMainLoop<L> gate, which skips the
        preLoop/mainloop/NGLL/NLL block entirely when origCounter==0.
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert "PGRTailC0ResetL:" not in tail, (
            "Tail must not define a PGRTailC0ResetL label any more.\n"
            "Tail head:\n" + tail[:1500]
        )

    def test_omits_c0_srd_subtract(self, fp4_pgr2_asm):
        """The tail must NOT emit the legacy `s_sub_u32 Srd<tc>,
        depthUBytes` "undo preLoop GR_INC" instructions anywhere
        (they used to live in the c=0 reset block; now obviated).
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert not re.search(
            r"s_sub_u32[^\n]*sgprSrd[AB][^\n]*undo preLoop GR_INC", tail
        ), (
            "Tail must not emit the legacy `s_sub_u32 SrdA/B ... undo "
            "preLoop GR_INC` (obviated by upstream skip gate)."
        )

    def test_omits_c0_lwa_xor_undo(self, fp4_pgr2_asm):
        """The tail must NOT emit the legacy `s_xor_b32 LWA ... undo
        preLoop GR_INC` instructions (legacy c=0 reset path).
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert not re.search(
            r"s_xor_b32[^\n]*LocalWriteBaseAddr[AB][^\n]*undo preLoop GR_INC",
            tail
        ), (
            "Tail must not emit the legacy `s_xor_b32 LWA ... undo "
            "preLoop GR_INC` (obviated by upstream skip gate)."
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

    def test_pgr2_omits_closeloop_emit(self, fp4_pgr2_asm):
        """PGR=2 also omits `closeLoop(... finalLoop=True)` (per
        nakajee review on PR #7636). See PGR=0 counterpart
        `test_omits_closeloop_emit` for full rationale: the
        `s_sub_i32 LoopCounterL` decrement / back-edge / TailLoopEndL
        emit are all unreachable / dead for the single-pass subtile
        tail body, and the `s_mov_b32 LoopCounterL, 0` workaround is
        gone with them.
        """
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert not re.search(
            r"s_sub_i32\s+s\[sgprLoopCounterL\][^\n]*dec counterL", tail
        ), "PGR=2 tail must NOT emit closeLoop per-iter decrement"
        assert not re.search(
            r"s_mov_b32\s+s\[sgprLoopCounterL\],\s*0\b[^\n]*single-iter tail",
            tail
        ), (
            "PGR=2 tail must NOT emit the `s_mov_b32 sgprLoopCounterL, 0` "
            "force-fall-through (no longer needed without closeLoop)."
        )
        assert "TailLoopEndL" not in tail, (
            "PGR=2 tail must NOT define `TailLoopEndL` "
            "(closeLoop is no longer emitted)."
        )


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

    def test_omits_origCounter_snapshot_after_hoist(self, bf16_pgr1_asm):
        """PGR=1 must NOT emit the legacy OrigLoopCounter snapshot;
        the gating block reads sgprOrigLoopCounter directly now that
        it sits before calculateLoopNumIter (per sebvince #PR-7683
        design).
        """
        assert not re.search(
            r"s_mov_b32\s+s\[?\d+\]?,\s*s\[sgprOrigLoopCounter\][^\n]*snapshot K//DU",
            bf16_pgr1_asm
        )

    def test_emits_c0_reset_compare_and_branch(self, bf16_pgr1_asm):
        """PGR=1 also routes c=0 through the tail-entry gate. With the
        upstream SkipSubtileMainLoop<L> gate, the preLoop/mainloop/NLL
        block is skipped for origCounter==0; SRDs stay at K=0, LWA/LRA
        at buf 0, accD at zero, so the c=0 path branches directly to
        the tail body (PGRTailEntry<L>) — no in-tail reset needed.
        """
        tail = _extract_tail_section(bf16_pgr1_asm)
        assert tail
        assert re.search(r"s_cmp_eq_u32[^\n]*\b0\b[^\n]*origCounter == 0", tail)
        assert re.search(
            r"s_cbranch_scc1[^\n]*PGRTailEntry", tail
        )
        assert "PGRTailC0Reset" not in tail, (
            "PGR=1 tail must not reference the legacy PGRTailC0Reset "
            "block (handled upstream by SkipSubtileMainLoop<L>)"
        )

    def test_emits_srd_advance_AB(self, bf16_pgr1_asm):
        tail = _extract_tail_section(bf16_pgr1_asm)
        assert tail
        assert re.search(r"s_add_u32[^\n]*sgprSrdA[^\n]*advance SrdA by 1 DU", tail)
        assert re.search(r"s_add_u32[^\n]*sgprSrdB[^\n]*advance SrdB by 1 DU", tail)

    def test_omits_c0_reset_label_pgr1(self, bf16_pgr1_asm):
        """PGR=1 tail must not define a PGRTailC0Reset<L> label."""
        tail = _extract_tail_section(bf16_pgr1_asm)
        assert tail
        assert "PGRTailC0ResetL:" not in tail, (
            "PGR=1 tail must not define a PGRTailC0ResetL label "
            "(obviated by upstream SkipSubtileMainLoop<L> gate)."
        )

    def test_omits_c0_undo_instructions_pgr1(self, bf16_pgr1_asm):
        """PGR=1 tail must not emit the legacy `undo preLoop GR_INC`
        SRD-sub or LWA-XOR instructions anywhere.
        """
        tail = _extract_tail_section(bf16_pgr1_asm)
        assert tail
        assert not re.search(
            r"s_sub_u32[^\n]*sgprSrd[AB][^\n]*undo preLoop GR_INC", tail
        )
        assert not re.search(
            r"s_xor_b32[^\n]*LocalWriteBaseAddr[AB][^\n]*undo preLoop GR_INC",
            tail
        )

    def test_emits_small_counter_compare(self, bf16_pgr1_asm):
        """The compare against PGR is still emitted; for PGR=1 it
        resolves to `< 1` and is inert at runtime (origCounter==0
        already branched away via the c=0 reset path above). Emitted
        for structural symmetry with PGR=2.

        After the sebvince #PR-7683 hoist, the source is
        `sgprOrigLoopCounter` directly (the snapshot scratch sgpr is
        gone), so the cmp src0 must spell out `sgprOrigLoopCounter`.
        """
        tail = _extract_tail_section(bf16_pgr1_asm)
        assert tail
        assert re.search(
            r"s_cmp_lt_u32\s+s\[sgprOrigLoopCounter\],\s*1\b"
            r"[^\n]*origCounter < PGR", tail
        )


# ── Tests: Srd<tc>+2 tightening at tail entry ────────────────────────────────

class TestTailSrdTightenSubtile:
    """`_emitTailSrdTightenSubtile` emit shape (nakajee #PR-7661 OOR review
    + follow-up tightening to bpr=4 alignment).

    The tightening fires once at tail entry (after the PGR>0 entry
    gating, before openLoop), shrinks `SrdA+2` and `SrdB+2` by
    `DepthU*bpe - roundUp(K_remain*bpe, bpr=4)`, and is gated to
    non-MX, non-swizzled, bpe in {1,2}, symmetric A/B kernels. Earlier
    m-rows are over-protected (handled by lane mask + sub-lane refine);
    the tightening's job is exclusively the last m-row's last GR thread
    which would otherwise read past A/B's allocated K bytes.

    Under align-UP to bpr=4, `delta = DepthU*bpe - alignedBytes` is
    provably >= 0, so there is NO runtime cbranch/skip-label -- when
    delta=0 the SSubs are harmless no-ops (see nakajee #PR-7661
    review cleanup (2)).
    """

    @pytest.fixture
    def bf16_pgr0_asm(self):
        return _emit_tail_loop_asm(fp4=False, no_tail_loop=False, pgr=0)

    @pytest.fixture
    def bf16_pgr2_asm(self):
        return _emit_tail_loop_asm(fp4=False, no_tail_loop=False, pgr=2)

    @pytest.fixture
    def fp4_pgr0_asm(self):
        return _emit_tail_loop_asm(fp4=True, no_tail_loop=False, pgr=0)

    def test_emits_srd_tighten_banner(self, bf16_pgr0_asm):
        """bf16 subtile must emit the SRD tighten comment banner so
        the rest of the structural pins below have a stable anchor.
        """
        tail = _extract_tail_section(bf16_pgr0_asm)
        assert tail
        assert "OOR review" in tail or "nakajee #PR-7661" in tail, (
            "Tail must emit the `_emitTailSrdTightenSubtile` banner "
            "referencing the nakajee #PR-7661 OOR review.\n"
            "Tail head:\n" + tail[:2000]
        )

    def test_emits_alignedBytes_chain(self, bf16_pgr0_asm):
        """The aligned-K-bytes chain is the runtime fingerprint of the
        helper: `s_lshl_b32 <s>, sgprLoopCounterL, 0x1` (bf16 bpe=2)
        then `s_add_u32 <s>, <s>, 3` and `s_and_b32 <s>, <s>, 0xfffffffc`
        for bpr=4 alignment (nakajee #PR-7661 follow-up cleanup #1:
        align to bpr=4, not loadBytes=16, since BufferLoad's natural
        granularity is bpr).
        """
        tail = _extract_tail_section(bf16_pgr0_asm)
        assert tail
        assert re.search(
            r"s_lshl_b32\s+s\[?\d+\]?,\s*s\[sgprLoopCounterL\]\s*,\s*0x1\b"
            r"[^\n]*K_remain \* bpe \(bpe=2\)",
            tail
        ), "Tail must compute `K_remain * bpe` via s_lshl_b32 ..., 0x1"
        assert re.search(
            r"s_add_u32\s+s\[?\d+\]?,\s*s\[?\d+\]?,\s*3\b"
            r"[^\n]*\+ \(bpr-1\) for roundUp",
            tail
        ), "Tail must add (bpr-1)=3 before align-mask"
        assert re.search(
            r"s_and_b32\s+s\[?\d+\]?,\s*s\[?\d+\]?,\s*0xfffffffc\b"
            r"[^\n]*alignedBytes = roundUp\(K_remain\*bpe, 4\)",
            tail
        ), (
            "Tail must mask to align up to bpr=4 boundary "
            "(alignMaskInv = 0xfffffffc), per nakajee follow-up #1"
        )

    def test_no_runtime_skip_branch(self, bf16_pgr0_asm):
        """Under align-UP to bpr=4 the delta is provably non-negative
        (`alignedBytes <= roundUp((DepthU-1)*bpe, 4) <= DepthU*bpe`
        for bpe in {1, 2}). The runtime `s_cmp_lt_u32 alignedBytes,
        DepthU*bpe` + `s_cbranch_scc0 TailSrdTightenSkip<L>` short-
        circuit from the original commit is dead and must be removed
        (nakajee #PR-7661 follow-up cleanup #2). When delta=0 the
        two `s_sub_u32` lines become harmless no-ops.
        """
        tail = _extract_tail_section(bf16_pgr0_asm)
        assert tail
        # Negative pin: scan only the tightening region (lives
        # between the banner and the first `s_sub_u32 SrdA+2` line)
        # so we don't mis-flag an unrelated `s_cmp_lt_u32` elsewhere
        # in the tail.
        banner_pos = max(
            tail.find("OOR review"),
            tail.find("nakajee #PR-7661"),
        )
        srdA_match = re.search(
            r"s_sub_u32\s+s\[sgprSrdA\+2\][^\n]*Srd A\+2 -= delta",
            tail)
        assert banner_pos >= 0 and srdA_match, (
            "Tail must have both the SRD tighten banner and the SrdA+2 "
            "sub. tail head:\n" + tail[:2000]
        )
        region = tail[banner_pos:srdA_match.end()]
        assert not re.search(
            r"s_cmp_lt_u32[^\n]*alignedBytes < DepthU", region
        ), (
            "Tail must NOT emit `s_cmp_lt_u32 alignedBytes < DepthU*bpe` "
            "in the tighten region -- delta is provably >= 0 under "
            "align-UP to bpr=4 (nakajee #PR-7661 cleanup #2). Region:\n"
            + region
        )
        assert "TailSrdTightenSkip" not in tail, (
            "Tail must NOT reference `TailSrdTightenSkip<L>` -- the "
            "runtime skip-label was removed in the bpr=4 tightening "
            "follow-up (delta is provably >= 0)."
        )

    def test_emits_srd_tighten_ssub_chain(self, bf16_pgr0_asm):
        """The actual tightening: `s_sub_u32 SrdA+2, SrdA+2, <delta>`
        and the matching `s_sub_u32 SrdB+2, SrdB+2, <delta>`. Pin the
        `delta = DepthU*bpe - alignedBytes` precompute too so the
        emit order isn't accidentally rearranged.
        """
        tail = _extract_tail_section(bf16_pgr0_asm)
        assert tail
        # bf16 fixture has DepthU=64 -> depthUBytes=128 (0x80).
        assert re.search(
            r"s_sub_u32\s+s\[?\d+\]?,\s*(?:128|0x80)\s*,\s*s\[?\d+\]?"
            r"[^\n]*delta = DepthU\*bpe - alignedBytes",
            tail
        ), (
            "Tail must precompute `delta = depthUBytes - alignedBytes`"
            " before applying it to Srd<tc>+2"
        )
        assert re.search(
            r"s_sub_u32\s+s\[sgprSrdA\+2\]\s*,\s*s\[sgprSrdA\+2\]\s*,\s*s\[?\d+\]?"
            r"[^\n]*Srd A\+2 -= delta",
            tail
        ), "Tail must subtract delta from Srd A+2"
        assert re.search(
            r"s_sub_u32\s+s\[sgprSrdB\+2\]\s*,\s*s\[sgprSrdB\+2\]\s*,\s*s\[?\d+\]?"
            r"[^\n]*Srd B\+2 -= delta",
            tail
        ), "Tail must subtract delta from Srd B+2"

    def test_ssub_chain_ordering(self, bf16_pgr0_asm):
        """Strict ordering inside the tighten region: the delta
        precompute lands BEFORE the SrdA+2 sub, which lands BEFORE
        the SrdB+2 sub. Pins emit order so a refactor doesn't
        scramble the dependent SSub chain (delta is the src of both
        Srd subs).
        """
        tail = _extract_tail_section(bf16_pgr0_asm)
        assert tail
        delta_match = re.search(
            r"s_sub_u32\s+s\[?\d+\]?,\s*(?:128|0x80)\s*,\s*s\[?\d+\]?"
            r"[^\n]*delta = DepthU\*bpe - alignedBytes",
            tail)
        srdA_match = re.search(
            r"s_sub_u32\s+s\[sgprSrdA\+2\][^\n]*Srd A\+2 -= delta",
            tail)
        srdB_match = re.search(
            r"s_sub_u32\s+s\[sgprSrdB\+2\][^\n]*Srd B\+2 -= delta",
            tail)
        assert delta_match and srdA_match and srdB_match, (
            "Missing one of: delta precompute / SrdA+2 sub / SrdB+2 sub. "
            "tail head:\n" + tail[:2000]
        )
        assert delta_match.start() < srdA_match.start() < srdB_match.start(), (
            "Tail must order delta < SrdA+2 sub < SrdB+2 sub. "
            "Got delta=%d, srdA=%d, srdB=%d" %
            (delta_match.start(), srdA_match.start(), srdB_match.start())
        )

    def test_srd_tighten_omitted_for_fp4(self, fp4_pgr0_asm):
        """MX kernels (MXBlockA/B > 0) must not emit the bf16/fp16
        SRD tighten helper: the MX path has its own
        `_emitTailSrdTightenSubtileMX` for MXSA/MXSB (statically
        gated to DepthU > 256). MX data SRD tightening for swizzled
        A/B remains deferred (DTV path, separate tail emitter).
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert "OOR review" not in tail, (
            "MX FP4 tail must NOT emit the bf16/fp16 SRD tighten."
        )
        assert "TailSrdTightenSkip" not in tail, (
            "MX FP4 tail must NOT define TailSrdTightenSkip<L>"
        )

    def test_srd_tighten_omitted_for_NoTailLoop(self):
        """`NoTailLoop=True` (aligned K) emits no tail body at all,
        so the SRD tighten helper must short-circuit early.
        """
        asm = _emit_tail_loop_asm(fp4=False, no_tail_loop=True, pgr=0)
        assert "OOR review" not in asm
        assert "TailSrdTightenSkip" not in asm

    def test_srd_tighten_fires_for_pgr2(self, bf16_pgr2_asm):
        """PGR=2 emits the SRD tighten in the same slot (after the
        PGR>0 entry gating, before openLoop). Pin its presence so
        a refactor of the PGR>0 gate ordering doesn't drop it.
        """
        tail = _extract_tail_section(bf16_pgr2_asm)
        assert tail
        assert re.search(
            r"s_sub_u32\s+s\[sgprSrdA\+2\][^\n]*Srd A\+2 -= delta",
            tail
        ), (
            "PGR=2 bf16 tail must still emit SrdA+2 tighten. "
            "Tail head:\n" + tail[:2000]
        )

    def test_srd_tighten_ordering_after_PGRTailEntry(self, bf16_pgr2_asm):
        """For PGR>0, the tighten must fire AFTER `PGRTailEntry<L>:`
        (so the c=0 / small-counter realign / large-counter advance
        paths have all converged) and BEFORE the tail GR call (so
        `globalReadDoSubtile` sees the tightened SRD).
        """
        tail = _extract_tail_section(bf16_pgr2_asm)
        assert tail
        entry_pos = tail.find("PGRTailEntryL:")
        if entry_pos < 0:
            entry_pos = tail.find("label_PGRTailEntryL:")
        srdA_match = re.search(
            r"s_sub_u32\s+s\[sgprSrdA\+2\][^\n]*Srd A\+2 -= delta", tail)
        assert entry_pos >= 0 and srdA_match, (
            "PGR=2 tail must have both PGRTailEntryL: label and "
            "SrdA+2 tighten. tail head:\n" + tail[:2000]
        )
        assert entry_pos < srdA_match.start(), (
            "PGR=2 tail must emit SRD tighten AFTER PGRTailEntryL: "
            "so c=0 / small-counter / large-counter all converge "
            "before the tightening fires. entry=%d, srdA=%d"
            % (entry_pos, srdA_match.start())
        )


# ── Tests: MX scale Srd<MXS{A,B}>+2 tightening (DepthU > 256) ────────────────

class TestTailSrdTightenSubtileMX:
    """`_emitTailSrdTightenSubtileMX` emit shape.

    Per nakajee #PR-7661 spec the MX scale SRD+2 needs an extra
    tightening step at tail entry when `DepthU > 256` (= MX K-padding
    unit). For `DepthU <= 256` the host's
    `rearrangePaddedMXScaleLayout` already pads K-blocks out to the
    next 256-K boundary, so the natural NumRecords already covers any
    K_remain on the last m-row -- the MX helper must be a static
    no-op (no instructions, no comments).

    For `DepthU > 256` (only `DepthU=512` in our current MX yaml
    gauntlet), the helper emits a roundUp(K_remain, 256) chain plus
    `s_sub_u32 SrdMXS{A,B}+2, -, <delta_K>`. `bytesPerKElement_MX`
    is 1 for all current MXSA_B4/MXSB_B4/MXSA_B8/MXSB_B8 layouts so
    the K-element delta is applied directly without scaling.
    """

    def _emit_fp4_asm(self, *, depthU, pgr=0):
        kernel = _create_kernel(256, 256, fp4=True, depthU=depthU,
                                no_tail_loop=False)
        _augment_kernel_for_tail_scaffold(kernel, pgr)
        kwa = _build_minimal_kwa(kernel)
        tPA = {"is_sparse": False, "tpsMetadata": None}
        tPB = {"is_sparse": False, "tpsMetadata": None}
        module = kwa._emitTailLoopScaffoldSubtile(kernel, tPA, tPB)
        return wrap_with_skiptoend(module)

    def test_mx_tighten_static_noop_when_depthU_eq_padK(self):
        """`DepthU == 256` is the boundary case: every K_remain in
        [1, 255] rounds up to 256 = DepthU, so delta_K = 0 always.
        The helper must short-circuit (emit nothing) -- no banner,
        no SrdMXSA/B+2 sub, no temp sgpr alloc.
        """
        asm = self._emit_fp4_asm(depthU=256)
        tail = _extract_tail_section(asm)
        assert tail, "fp4 fixture must produce a tail body"
        assert "MX follow-up" not in tail, (
            "DepthU=256 must NOT emit the MX SRD tighten banner "
            "(static no-op: K_remain < 256 == padK already covered "
            "by host padding)."
        )
        assert not re.search(r"s_sub_u32[^\n]*SrdMXSA\+2", tail), (
            "DepthU=256 must NOT emit `s_sub_u32 SrdMXSA+2` (static "
            "no-op)."
        )
        assert not re.search(r"s_sub_u32[^\n]*SrdMXSB\+2", tail), (
            "DepthU=256 must NOT emit `s_sub_u32 SrdMXSB+2` (static "
            "no-op)."
        )

    def test_mx_tighten_emits_when_depthU_gt_padK(self):
        """`DepthU=512` exceeds the MX K-padding unit (256), so the
        helper must emit the roundUp chain and the SrdMXSA+2 /
        SrdMXSB+2 sub. Pin the full chain: banner, roundUp
        precompute, delta = DepthU - remainK_MX, and per-operand
        sub.
        """
        asm = self._emit_fp4_asm(depthU=512)
        tail = _extract_tail_section(asm)
        assert tail
        assert "MX follow-up" in tail, (
            "DepthU=512 fp4 tail must emit the MX SRD tighten banner. "
            "Tail head:\n" + tail[:2500]
        )
        assert re.search(
            r"s_add_u32\s+s\[?\d+\]?,\s*s\[sgprLoopCounterL\]\s*,\s*255\b"
            r"[^\n]*K_remain \+ \(MX_pad_K - 1\)",
            tail
        ), (
            "DepthU=512 tail must add (256-1)=255 to LoopCounterL for "
            "the MX roundUp chain"
        )
        assert re.search(
            r"s_and_b32\s+s\[?\d+\]?,\s*s\[?\d+\]?,\s*0xffffff00\b"
            r"[^\n]*remainK_MX = roundUp\(K_remain, 256\)",
            tail
        ), (
            "DepthU=512 tail must mask with 0xffffff00 to align K to "
            "256-element MX padding boundary"
        )
        assert re.search(
            r"s_sub_u32\s+s\[?\d+\]?,\s*(?:512|0x200)\s*,\s*s\[?\d+\]?"
            r"[^\n]*delta_K = DepthU - remainK_MX",
            tail
        ), (
            "DepthU=512 tail must precompute `delta_K = 512 - remainK_MX`"
        )
        assert re.search(
            r"s_sub_u32\s+s\[sgprSrdMXSA\+2\]\s*,\s*s\[sgprSrdMXSA\+2\]"
            r"\s*,\s*s\[?\d+\]?[^\n]*SrdMXSA\+2 -= delta",
            tail
        ), "DepthU=512 tail must subtract delta from SrdMXSA+2"
        assert re.search(
            r"s_sub_u32\s+s\[sgprSrdMXSB\+2\]\s*,\s*s\[sgprSrdMXSB\+2\]"
            r"\s*,\s*s\[?\d+\]?[^\n]*SrdMXSB\+2 -= delta",
            tail
        ), "DepthU=512 tail must subtract delta from SrdMXSB+2"

    def test_mx_tighten_skipped_for_non_mx(self):
        """Non-MX (bf16) kernels must NOT emit the MX SRD tighten
        helper regardless of DepthU.
        """
        asm = _emit_tail_loop_asm(fp4=False, no_tail_loop=False, pgr=0)
        tail = _extract_tail_section(asm)
        assert tail
        assert "MX follow-up" not in tail, (
            "Non-MX (bf16) tail must NOT emit the MX SRD tighten "
            "banner"
        )
        assert not re.search(r"s_sub_u32[^\n]*SrdMXSA\+2", tail), (
            "Non-MX tail must NOT emit `s_sub_u32 SrdMXSA+2`"
        )
        assert not re.search(r"s_sub_u32[^\n]*SrdMXSB\+2", tail), (
            "Non-MX tail must NOT emit `s_sub_u32 SrdMXSB+2`"
        )

    def test_mx_tighten_fires_for_pgr2_at_depthU_512(self):
        """The MX tightening must also fire under PGR=2 (the same
        slot as the bf16 tightener: after `PGRTailEntry<L>:`, before
        the tail GR). Pins both PGR=0 and PGR=2 emit the helper.
        """
        asm = self._emit_fp4_asm(depthU=512, pgr=2)
        tail = _extract_tail_section(asm)
        assert tail
        assert re.search(
            r"s_sub_u32\s+s\[sgprSrdMXSA\+2\][^\n]*SrdMXSA\+2 -= delta",
            tail
        ), (
            "PGR=2 DepthU=512 fp4 tail must still emit SrdMXSA+2 "
            "tighten. Tail head:\n" + tail[:2500]
        )
        entry_pos = tail.find("PGRTailEntryL:")
        if entry_pos < 0:
            entry_pos = tail.find("label_PGRTailEntryL:")
        srdMXSA_match = re.search(
            r"s_sub_u32\s+s\[sgprSrdMXSA\+2\][^\n]*SrdMXSA\+2 -= delta",
            tail)
        assert entry_pos >= 0 and srdMXSA_match, (
            "PGR=2 DU=512 tail must have both PGRTailEntryL: label and "
            "SrdMXSA+2 tighten. tail head:\n" + tail[:2500]
        )
        assert entry_pos < srdMXSA_match.start(), (
            "PGR=2 DU=512 tail must emit MX SRD tighten AFTER "
            "PGRTailEntryL: so the c=0 / small-counter / large-counter "
            "paths converge before the tightening fires. entry=%d, "
            "srdMXSA=%d" % (entry_pos, srdMXSA_match.start())
        )


# ── Tests: MX data Srd<{A,B}>+2 tightening (DepthU > 256) ─────────────────────

class TestTailSrdTightenSubtileMXData:
    """`_emitTailSrdTightenSubtileMXData` emit shape.

    Companion to `_emitTailSrdTightenSubtileMX`: where the scale
    tightener clips `SrdMXS{A,B}+2`, the data tightener clips
    `Srd{A,B}+2` on the MX **data** tensor side. nakajee #PR-7661
    review pointed out that the data side needs the same K=256-padded
    clip as the scale side -- otherwise the natural DepthU-shaped data
    over-read on the last m-row can fault past the data buffer's
    allocated bytes (the per-lane-mask + 0-scale absorb keeps the MFMA
    result correct under garbage data, but garbage reads can still
    page fault past the data buffer).

    Static gate (must emit nothing): `DepthU <= 256`. The host MX
    K-padding already covers any K_remain < 256 in that regime, so
    the helper short-circuits with zero instructions / zero comments.

    Active gate: `DepthU > 256` AND at least one MX side AND
    non-swizzled MX operand AND bpe in {0.5, 1}. The current MX
    yamls all use mxfp4/mxfp4 (bpe=0.5 on both sides), so the helper
    emits a roundUp(K_remain, 256) chain, a single delta_K =
    DepthU - remainK_MX precompute, ONE `s_lshr_b32 delta, delta, 1`
    (delta_K * 0.5 = delta_K >> 1), and one `s_sub_u32 Srd{A,B}+2`
    per MX operand. Swizzled MX operands bail (separate DTV emit
    path; deferred per nakajee review reply).
    """

    def _emit_fp4_asm(self, *, depthU, pgr=0):
        kernel = _create_kernel(256, 256, fp4=True, depthU=depthU,
                                no_tail_loop=False)
        _augment_kernel_for_tail_scaffold(kernel, pgr)
        kwa = _build_minimal_kwa(kernel)
        tPA = {"is_sparse": False, "tpsMetadata": None}
        tPB = {"is_sparse": False, "tpsMetadata": None}
        module = kwa._emitTailLoopScaffoldSubtile(kernel, tPA, tPB)
        return wrap_with_skiptoend(module)

    def test_mxdata_tighten_static_noop_when_depthU_eq_padK(self):
        """`DepthU == 256` is the static no-op boundary: every
        K_remain in [1, 255] rounds up to 256 == DepthU, so
        delta_K = 0 always. The helper must short-circuit -- no
        banner, no `Srd{A,B}+2` sub, no temp sgpr alloc.
        """
        asm = self._emit_fp4_asm(depthU=256)
        tail = _extract_tail_section(asm)
        assert tail, "fp4 fixture must produce a tail body"
        assert "MX data follow-up" not in tail, (
            "DepthU=256 must NOT emit the MX data SRD tighten banner"
        )

    def test_mxdata_tighten_emits_when_depthU_gt_padK(self):
        """`DepthU=512` (only DepthU>256 fp4 config in our gauntlet)
        must emit the roundUp chain, the per-MX-operand bpe shift,
        and per-MX-operand SrdA+2 / SrdB+2 sub.

        Pinned shape (mxfp4 / mxfp4, uniform bpe=0.5 on both sides):
          banner ("MX data follow-up")
          s_add_u32  remKMx, LoopCounterL, 255
          s_and_b32  remKMx, remKMx, 0xffffff00
          s_sub_u32  delta, 0x200, remKMx
          s_lshr_b32 delta, delta, 1            # bpe=1/2
          s_sub_u32  SrdA+2, SrdA+2, delta
          s_sub_u32  SrdB+2, SrdB+2, delta
        """
        asm = self._emit_fp4_asm(depthU=512)
        tail = _extract_tail_section(asm)
        assert tail
        assert "MX data follow-up" in tail, (
            "DepthU=512 fp4 tail must emit the MX data SRD tighten "
            "banner. Tail head:\n" + tail[:2500]
        )
        assert re.search(
            r"s_add_u32\s+s\[?\d+\]?,\s*s\[sgprLoopCounterL\]\s*,\s*255\b"
            r"[^\n]*K_remain \+ \(MX_pad_K - 1\)",
            tail
        ), "DepthU=512 fp4 tail must add 255 to LoopCounterL for roundUp"
        assert re.search(
            r"s_and_b32\s+s\[?\d+\]?,\s*s\[?\d+\]?,\s*0xffffff00\b"
            r"[^\n]*remainK_MX = roundUp\(K_remain, 256\)",
            tail
        ), "DepthU=512 fp4 tail must mask with 0xffffff00 for MX K=256 align"
        assert re.search(
            r"s_sub_u32\s+s\[?\d+\]?,\s*(?:512|0x200)\s*,\s*s\[?\d+\]?"
            r"[^\n]*delta_K = DepthU - remainK_MX",
            tail
        ), "DepthU=512 fp4 tail must precompute delta_K = 512 - remainK_MX"
        # bpe=0.5 → single shr by 1 (uniformShr path)
        assert re.search(
            r"s_lshr_b32\s+s\[?\d+\]?,\s*s\[?\d+\]?,\s*0x1\b"
            r"[^\n]*delta_bytes = delta_K \* bpe_data \(bpe=1/2\)",
            tail
        ), (
            "DepthU=512 fp4 (bpe=0.5) tail must shift delta right by 1 "
            "to scale K-element delta to bytes"
        )
        assert re.search(
            r"s_sub_u32\s+s\[sgprSrdA\+2\]\s*,\s*s\[sgprSrdA\+2\]"
            r"\s*,\s*s\[?\d+\]?[^\n]*clip MX data past remainK_MX",
            tail
        ), "DepthU=512 fp4 tail must subtract delta from SrdA+2"
        assert re.search(
            r"s_sub_u32\s+s\[sgprSrdB\+2\]\s*,\s*s\[sgprSrdB\+2\]"
            r"\s*,\s*s\[?\d+\]?[^\n]*clip MX data past remainK_MX",
            tail
        ), "DepthU=512 fp4 tail must subtract delta from SrdB+2"

    def test_mxdata_tighten_skipped_for_non_mx(self):
        """Non-MX (bf16) kernels must NOT emit the MX data SRD
        tighten helper regardless of DepthU.
        """
        asm = _emit_tail_loop_asm(fp4=False, no_tail_loop=False, pgr=0)
        tail = _extract_tail_section(asm)
        assert tail
        assert "MX data follow-up" not in tail, (
            "Non-MX (bf16) tail must NOT emit the MX data SRD tighten "
            "banner"
        )

    def test_mxdata_tighten_after_scale_tighten(self):
        """Emit order at DepthU=512 fp4: the MX **scale** tighten
        (`MX follow-up`) must precede the MX **data** tighten
        (`MX data follow-up`); the scaffold call site adds them in
        scale-then-data order so a refactor that swaps them gets
        caught here.
        """
        asm = self._emit_fp4_asm(depthU=512)
        tail = _extract_tail_section(asm)
        assert tail
        scale_pos = tail.find("MX follow-up")
        data_pos  = tail.find("MX data follow-up")
        assert scale_pos >= 0 and data_pos >= 0, (
            "DepthU=512 fp4 tail must emit BOTH scale and data "
            "tightener banners; got scale_pos=%d data_pos=%d. "
            "Tail head:\n%s" % (scale_pos, data_pos, tail[:2500])
        )
        assert scale_pos < data_pos, (
            "MX scale tighten must precede MX data tighten in the "
            "scaffold (got scale=%d, data=%d)" % (scale_pos, data_pos)
        )

    def test_mxdata_tighten_fires_for_pgr2_at_depthU_512(self):
        """The MX data tightening must also fire under PGR=2, in the
        same slot as the scale tightener (after `PGRTailEntry<L>:`,
        before the tail GR). Mirrors the scale-side PGR2 pin.
        """
        asm = self._emit_fp4_asm(depthU=512, pgr=2)
        tail = _extract_tail_section(asm)
        assert tail
        entry_pos = tail.find("PGRTailEntryL:")
        if entry_pos < 0:
            entry_pos = tail.find("label_PGRTailEntryL:")
        data_match = re.search(
            r"s_sub_u32\s+s\[sgprSrdA\+2\]\s*,\s*s\[sgprSrdA\+2\]"
            r"\s*,\s*s\[?\d+\]?[^\n]*clip MX data past remainK_MX",
            tail
        )
        assert entry_pos >= 0 and data_match, (
            "PGR=2 DU=512 fp4 tail must have BOTH PGRTailEntryL: label "
            "and SrdA+2 MX-data tighten. Tail head:\n" + tail[:2500]
        )
        assert entry_pos < data_match.start(), (
            "PGR=2 DU=512 fp4 must emit MX data SRD tighten AFTER "
            "PGRTailEntryL: (entry=%d, srdA-data=%d)"
            % (entry_pos, data_match.start())
        )
