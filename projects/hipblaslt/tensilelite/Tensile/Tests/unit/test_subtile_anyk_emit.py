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
"""Tail-loop emit content assertions for the subtile any-K tail.

Drives `KernelWriter._emitTailLoopScaffoldSubtile` at ASEM ∈ {32, 8,
4, 2, 1} via `setdefault_tail_scaffold_kernel_keys(..., asem=...)` and
pins the K%32 / K%8 / K%4 / K%2 / odd-K emit shape.

When the sub-lane byte refine fires (ASEM<numMIInUnroll for non-MX
integer-bpe operands) it owns the per-lane K-tail mask end-to-end:
the coarse `kPos vs LoopCounterL` cmp + per-VGPR cndmask is removed
and a single per-(operand, ir) mask chain feeds a `v_and` against
each boundary VGPR. The mask chain folds in mod = `elementsPerVgpr`-1
down to 0 byte slots, statically skipped when `ASEM*bpe % bpr == 0`
and runtime-gated by `LoopCounterL & (elementsPerVgpr-1)` otherwise.

With `SubtileTailMaskPrecompute` enabled (the default) the whole
per-(operand, mmak, ir) mask chain is emitted ONCE before the
per-mmak MFMA loop into dedicated scratch VGPRs; the per-mmak step
then collapses to pure `v_and_b32 vIdx, vMask[..], vIdx` (no cmps,
no cndmasks). The K%2 / K%1 emit pins below verify both the
hoisted-chain shape and the v_and-only apply step.
"""
import re

import pytest

from Tensile.Tests.unit._subtile_tailloop_fixtures import (
    build_minimal_subtile_kwa,
    setdefault_tail_scaffold_kernel_keys,
    wrap_with_skiptoend,
)
from Tensile.Tests.unit.test_subtile_tailloop_emit import (
    _create_kernel,
    _extract_tail_section,
)


# ── Driver ──────────────────────────────────────────────────────────────────

def _emit_anyk_tail_asm(*, asem: int, pgr: int = 0, MT0: int = 128, MT1: int = 128,
                        fusedForm: bool = True, depthU: int = 64) -> str:
    """Emit a BF16 subtile tail scaffold at the requested ASEM.

    `fusedForm` (default True) drives `SubtileTailMaskFusedForm` --
    the BF16 mask chain that fuses a single shared diff +
    per-i boundary[ir] init with sFull/sZero cmps per (mmak, ir).
    Set False to pin the legacy per-(operand, mmak, ir) bpe-parametric
    chain (still used for fp8 / int8 byte-refine paths and the
    `SubtileTailMaskFusedForm=False` reversibility escape hatch).
    """
    kernel = _create_kernel(MT0=MT0, MT1=MT1, fp4=False,
                            depthU=depthU, no_tail_loop=False)
    setdefault_tail_scaffold_kernel_keys(kernel, pgr, asem=asem)
    kernel["SubtileTailMaskFusedForm"] = fusedForm

    kwa = build_minimal_subtile_kwa(kernel)
    tPA = {"is_sparse": False, "tpsMetadata": None}
    tPB = {"is_sparse": False, "tpsMetadata": None}
    module = kwa._emitTailLoopScaffoldSubtile(kernel, tPA, tPB)
    return wrap_with_skiptoend(module)


# ── Tests: K%8 (ASEM=8) ──────────────────────────────────────────────────────

class TestAnyKEmit_K8:
    """K%8 reuses the K32 emit path unchanged: at numMIInUnroll=8 for
    bf16 the byte refine gate (`ASEM<numMIInUnroll`) is not satisfied,
    so the coarse per-mmak cmp + per-VGPR cndmask is the only mask.
    """

    def test_k8_emit_matches_k32_shape(self):
        asm_k32 = _emit_anyk_tail_asm(asem=32, pgr=0)
        asm_k8 = _emit_anyk_tail_asm(asem=8, pgr=0)
        tail_k32 = _extract_tail_section(asm_k32)
        tail_k8 = _extract_tail_section(asm_k8)
        assert tail_k32, "K32 baseline emitted no tail block"
        assert tail_k8, "K8 emit produced no tail block"

        cmp_k32 = re.findall(r"v_cmp_ge_i32.*LoopCounterL", tail_k32)
        cmp_k8 = re.findall(r"v_cmp_ge_i32.*LoopCounterL", tail_k8)
        cnd_k32 = re.findall(r"v_cndmask_b32.*[vV]alu", tail_k32)
        cnd_k8 = re.findall(r"v_cndmask_b32.*[vV]alu", tail_k8)

        assert len(cmp_k8) == len(cmp_k32), (
            f"K%8 (ASEM=8) lane-mask cmp count {len(cmp_k8)} must "
            f"equal K%32 (ASEM=32) baseline {len(cmp_k32)}. K%8 is "
            f"expected to reuse the K32 emit path unchanged."
        )
        assert len(cnd_k8) == len(cnd_k32), (
            f"K%8 (ASEM=8) lane-mask cndmask count {len(cnd_k8)} must "
            f"equal K%32 (ASEM=32) baseline {len(cnd_k32)}."
        )

    def test_k8_no_byte_refine(self):
        """ASEM=8 = numMIInUnroll → byte-refine gate must reject; no
        `byteRefine` comments should appear.
        """
        tail = _extract_tail_section(_emit_anyk_tail_asm(asem=8, pgr=0))
        assert tail
        assert "byteRefine" not in tail, (
            "K%8 emit must NOT engage the sub-lane byte refine "
            "(ASEM>=numMIInUnroll → coarse mask suffices)."
        )


# ── Tests: K%4 (ASEM=4) ──────────────────────────────────────────────────────

class TestAnyKEmit_K4:
    """ASEM=4 fires the byte refine (4<numMIInUnroll=8) but the partial
    mod>0 chain collapses statically because `ASEM*bpe = 8` is a
    multiple of `bpr = 4`. Only the mod=0 step is emitted; no runtime
    gate, no mod>0 mask byte, and the coarse cmp+cndmask is gone (#5).
    """

    def test_k4_fused_form_emits_init_and_per_mmak_chain(self):
        """ASEM=4 (byte refine path, BF16). With the fused form
        adopted as default, the chain emits:
          * init: single `v_sub_i32 vDiff, sgprLoopCounterL, vKPosBase`
            (signed) + per-i boundaryMask cndmask sequence.
          * per (mmak, ir): two cmps (sFull / sZero) + two cndmasks
            (full / boundary[ir] / 0). NO mod>0 mask byte mov, NO
            mod=0 chain, NO byteRefine seed mov.
        Also: no coarse `if K_idx >= sizeL` cndmask (byte refine
        subsumes it).
        """
        tail = _extract_tail_section(_emit_anyk_tail_asm(asem=4, pgr=0))
        assert tail, "K%4 emit produced no tail block"

        # Fused form fires (default for BF16/BF16).
        assert "subLaneMask fused" in tail, (
            "K%4 (ASEM=4) emit must engage the fused mask init "
            "(default for bf16/bf16 byte refine)."
        )
        # Legacy chain primitives must be absent. The `byteRefine`
        # tag is still used in the v_and apply comments (shared by
        # both forms; pinned by the apply tests above), but the
        # legacy seed `mask seed = full keep` mov and the
        # per-(operand, mmak, ir) byteRefine CHAIN start tag must
        # not appear under the fused default form.
        assert re.search(
            r"v_mov_b32\s+v\d+,\s*0xffffffff[^\n]*mask seed = full keep",
            tail, re.IGNORECASE,
        ) is None, (
            "K%4 emit must NOT contain the legacy `mask seed = full "
            "keep` (fused form skips the per-(mmak, ir) seed mov)."
        )
        assert re.search(
            r"byteRefine\[[AB] ir=\d+ mmak=\d+\]: mask seed",
            tail,
        ) is None, (
            "K%4 emit must NOT contain the legacy `byteRefine "
            "[<op> ir=N mmak=M]: mask seed` chain start (fused "
            "form replaces the per-(operand, mmak, ir) chain with "
            "the shared diff + boundary precompute)."
        )
        # No coarse `kPosCur = kPosBase + mmak * miK` cndmask either.
        assert re.search(
            r"v_cndmask_b32[^\n]*if K_idx >= sizeL", tail
        ) is None, (
            "K%4 emit must NOT contain the legacy coarse "
            "`v_cndmask_b32 ... if K_idx >= sizeL` per-VGPR cndmask "
            "(fused form's sFull/sZero subsume it)."
        )
        # Fused per-(mmak, ir) chain: 2 cndmasks per chain at
        # numMmaks=2 (DU=64 / MIK=32), vgprPerInUnroll=4 -> 16
        # cndmasks shared across A/B (bpeA==bpeB). Plus 8 cndmasks in
        # the init (per-i boundary, 2 cndmasks per i, 4 vgprs).
        # Pin the per-(mmak, ir) chain markers as a substring count.
        fused_chain_markers = re.findall(
            r"subLaneMask\[A mmak=\d+ ir=\d+\] = sFull",
            tail,
        )
        assert len(fused_chain_markers) >= 4, (
            "K%4 emit (fused form) must have >=4 per-(A, mmak, ir) "
            "sFull cndmasks (2 mmaks * 4 ir). Got %d.\nTail excerpt:\n"
            % len(fused_chain_markers) + tail[:1500]
        )

    def test_k4_legacy_form_emits_mod0_only(self):
        """`SubtileTailMaskFusedForm=False` reverts to the legacy
        bpe-parametric chain. ASEM=4 still statically skips the mod>0
        chain (ASEM*bpe=8 is a multiple of bpr=4); only the mod=0
        step + seed remain. Regression pin for the reversibility
        escape hatch.
        """
        tail = _extract_tail_section(
            _emit_anyk_tail_asm(asem=4, pgr=0, fusedForm=False))
        assert tail, "K%4 legacy emit produced no tail block"
        assert "byteRefine" in tail, (
            "K%4 legacy emit must contain legacy `byteRefine` "
            "comments (fused form disabled)."
        )
        # No mod>0 chain.
        assert re.search(
            r"v_mov_b32\s+v\d+,\s*0xffff\b[^\n]*keep mask",
            tail, re.IGNORECASE,
        ) is None, (
            "K%4 legacy emit must NOT contain mod>0 keep-mask 0xFFFF "
            "mov (static skip drops it)."
        )
        # mod=0 chain fires per (operand, ir).
        seed_movs = re.findall(
            r"v_mov_b32\s+v\d+,\s*0xffffffff[^\n]*mask seed = full keep",
            tail, re.IGNORECASE,
        )
        assert len(seed_movs) >= 4, (
            "K%4 legacy emit missing per-(operand, ir) mask seed."
        )


# ── Tests: K%2 (ASEM=2) ──────────────────────────────────────────────────────

class TestAnyKEmit_K2:
    """ASEM=2 fires the byte refine. `ASEM*bpe = 4` is a multiple of
    `bpr = 4`, so the partial mod>0 chain is statically dropped: only
    the mod=0 step is emitted per (operand, ir). The coarse cmp +
    per-VGPR cndmask is removed (#5); the byte refine's mod=0 v_and
    against each boundary VGPR is the sole K-tail mask.
    """

    def test_k2_emits_fused_form_chain(self):
        """ASEM=2 fires the byte refine. Under the fused default form:
        chain is single `vDiff` init + per-i boundaryMask init +
        sFull/sZero cmps per (mmak, ir). The per-mmak apply step
        remains a pure v_and_b32 (precompute pattern preserved so the
        chain runs entirely in the GR-wait shadow).

        Strictly more cmps than the K%32 baseline (which only has the
        per-mmak coarse cmp): init adds vgprPerInUnroll-many boundary
        cmps + per-(mmak, ir) sFull/sZero cmps.
        """
        asm_k32 = _emit_anyk_tail_asm(asem=32, pgr=0)
        asm_k2 = _emit_anyk_tail_asm(asem=2, pgr=0)
        tail_k2 = _extract_tail_section(asm_k2)
        tail_k32 = _extract_tail_section(asm_k32)
        assert tail_k2, "K%2 emit produced no tail block"
        assert tail_k32, "K%32 baseline emit produced no tail block"

        # Cmp count: fused form emits VCmpLtI32 (boundary init) +
        # VCmpGTI32 (sFull) + VCmpLeI32 (sZero). The K%32 baseline
        # uses VCmpGEI32 (coarse). Pin "strictly more cmps total".
        cmp_k2 = re.findall(r"v_cmp_(?:gt|le|lt|ge)_i32", tail_k2)
        cmp_k32 = re.findall(r"v_cmp_(?:gt|le|lt|ge)_i32", tail_k32)
        assert len(cmp_k2) > len(cmp_k32), (
            f"K%2 fused-form emit must add per-(mmak, ir) sFull/"
            f"sZero cmps + boundary init cmps beyond the K%32 "
            f"baseline; got cmp_k2={len(cmp_k2)} vs "
            f"cmp_k32={len(cmp_k32)}."
        )

        # Init shows up exactly once.
        assert "subLaneMask fused: diff = LoopCounterL - kPosBase" in tail_k2, (
            "K%2 emit must contain the fused init `diff = "
            "LoopCounterL - kPosBase`."
        )
        # Per-(mmak, ir) sFull cmp tagged with mmak/ir indices.
        sfull_cmps = re.findall(
            r"subLaneMask\[A mmak=\d+ ir=\d+\]: sFull",
            tail_k2,
        )
        assert len(sfull_cmps) >= 4, (
            "K%2 emit must contain per-(mmak, ir) sFull cmp "
            "comments. Got %d.\nTail excerpt:\n"
            % len(sfull_cmps) + tail_k2[:1500]
        )
        # Per-(mmak, ir) sZero cmp paired with sFull.
        szero_cmps = re.findall(
            r"subLaneMask\[A mmak=\d+ ir=\d+\]: sZero",
            tail_k2,
        )
        assert len(szero_cmps) == len(sfull_cmps), (
            "K%2 emit must pair sFull and sZero cmps per (mmak, ir); "
            "got %d sFull vs %d sZero." %
            (len(sfull_cmps), len(szero_cmps))
        )

        # Per-VGPR mask application: `v_and_b32 vIdx, vMask, vIdx`
        # with the precompute-mode comment (unchanged from legacy).
        and_a = re.search(
            r"v_and_b32\s+v(\d+),\s+v\d+,\s+v\1"
            r"[^\n]*apply precomputed mask to ValuA",
            tail_k2,
        )
        and_b = re.search(
            r"v_and_b32\s+v(\d+),\s+v\d+,\s+v\1"
            r"[^\n]*apply precomputed mask to ValuB",
            tail_k2,
        )
        assert and_a is not None, (
            "K%2 emit missing per-VGPR `v_and_b32 vIdx, vMask, vIdx` "
            "(apply precomputed mask to ValuA) for boundary VGPR. "
            "Tail excerpt:\n" + tail_k2[:2000]
        )
        assert and_b is not None, (
            "K%2 emit missing per-VGPR `v_and_b32 vIdx, vMask, vIdx` "
            "(apply precomputed mask to ValuB) for boundary VGPR. "
            "Tail excerpt:\n" + tail_k2[:2000]
        )

        # Precompute hoist: the fused init + per-(mmak, ir) chain
        # must appear BEFORE the first per-mmak ds_read wait. The
        # per-mmak apply step (`v_and_b32 ... apply precomputed mask`)
        # lives AFTER that marker.
        ds_wait_idx = tail_k2.find("tail LR mmak=0: wait for ds_reads")
        assert ds_wait_idx > 0, (
            "K%2 emit missing per-mmak `tail LR mmak=0: wait for "
            "ds_reads ...` marker."
        )
        diff_init_idx = tail_k2.find(
            "subLaneMask fused: diff = LoopCounterL - kPosBase")
        assert 0 < diff_init_idx < ds_wait_idx, (
            "K%%2 emit must hoist the fused `diff` init BEFORE the "
            "first per-mmak ds_read wait (precompute block).\n"
            "diff@%d, ds_wait@%d" % (diff_init_idx, ds_wait_idx)
        )
        first_v_and_apply = tail_k2.find(
            "apply precomputed mask to ValuA")
        assert first_v_and_apply > ds_wait_idx, (
            "K%2 emit must place the `v_and_b32 ... apply precomputed "
            "mask to ValuA` AFTER the per-mmak ds_read wait."
        )

        # d = LoopCounterL % numMIInUnroll computed once at init
        # (numMIInUnroll = 8 for bf16 MI16x16x32 -> mask = 7).
        assert re.search(
            r"v_and_b32\s+v\d+,\s*7,\s+s\[sgprLoopCounterL\]"
            r"[^\n]*d = LoopCounterL % 8",
            tail_k2,
        ), (
            "K%2 fused emit missing `v_and_b32 vDLaneRem, 7, "
            "sgprLoopCounterL` (d = LoopCounterL % 8). Tail "
            "excerpt:\n" + tail_k2[:2000]
        )

        # Negative: byte-shift refinement must NOT use a 64-bit lshl.
        assert re.search(r"v_lsh(?:l|lrev)_b64", tail_k2) is None, (
            "K%2 emit must NOT contain v_lshl(rev)_b64."
        )

        # Static partial-mod skip: ASEM=2 → ASEM*bpe=4 = bpr, so the
        # mod>0 chain (and its skip label, runtime gate, hi mask
        # seed) must NOT be emitted.
        assert "SubtileTailByteShiftPartialSkip" not in tail_k2, (
            "ASEM=2 emit must NOT contain the runtime partial-mod "
            "skip label (static skip should drop it)."
        )
        assert re.search(
            r"v_mov_b32\s+v\d+,\s*0xffff\b[^\n]*keep mask",
            tail_k2, re.IGNORECASE,
        ) is None, (
            "ASEM=2 emit must NOT contain the mod=1 `0xFFFF` keep-"
            "mask mov (static skip should drop it)."
        )

        # #5 negative pin: the legacy coarse per-VGPR cndmask is
        # subsumed by the byte refine's v_and when the refine fires.
        assert re.search(
            r"v_cndmask_b32[^\n]*if K_idx >= sizeL", tail_k2
        ) is None, (
            "ASEM=2 emit must NOT contain the legacy coarse "
            "`v_cndmask_b32 ... if K_idx >= sizeL` per-VGPR cndmask "
            "(byte refine subsumes it; #5)."
        )


# ── Tests: K%1 (odd K, ASEM=1) ───────────────────────────────────────────────

class TestAnyKEmit_K1:
    """ASEM=1 (odd K). `ASEM*bpe = 2` is not a multiple of `bpr = 4`,
    so the mod>0 partial chain is emitted unconditionally (no runtime
    s_and/s_cbranch gate -- the 3-instr scalar gate to skip a 4-instr
    chain in the one-shot precompute setup is not worth the branch
    overhead). The chain shape is:

      v_mov_b32 vMask, 0xFFFFFFFF                         // mask seed
      v_mov_b32 vSeed, 0xFFFF                             // mod=1 keep
      v_add_u32 vKpos, mod=1 offset, kPosBase
      v_cmp_ge_i32 sMask, vKpos, sgprLoopCounterL
      v_cndmask_b32 vMask, vMask, vSeed, sMask
      v_add_u32 vKpos, mod=0 offset, kPosBase
      v_cmp_ge_i32 sMask, vKpos, sgprLoopCounterL
      v_cndmask_b32 vMask, vMask, 0, sMask
      v_and_b32 vIdx, vMask, vIdx                         // each idx
    """

    def test_k1_no_narrow_load_and_emits_fused_chain(self):
        """ASEM=1 (full odd-K path). Default fused form chain.
        Asserts:
          * no narrow d16 load (still illegal on gfx950);
          * no legacy runtime-gate skip label or partial-mod residue
            (already dropped pre-fused-form and stay dropped);
          * fused init emitted (vDiff, halfKeep, boundary[i]);
          * fused per-(mmak, ir) sFull/sZero chain emitted;
          * boundary cndmask `(d<hi) ? halfKeep : full` present in
            init (the 3-state mask construction).
        """
        asm = _emit_anyk_tail_asm(asem=1, pgr=0)
        tail = _extract_tail_section(asm)
        assert tail, "ASEM=1 emit produced no tail block"

        # Narrow d16 load must not be emitted (illegal on gfx950).
        assert re.search(
            r"buffer_load_d16_b16|buffer_load_short_d16",
            asm,
        ) is None, (
            "ASEM=1 emit must NOT contain the narrow DTL load: "
            "`buffer_load_*_d16 ... lds` is not legal on gfx950."
        )

        # Runtime gate must NOT be emitted: legacy dropped this gate
        # pre-fused-form, and the fused form doesn't have a mod>0
        # chain to gate either way.
        assert "SubtileTailByteShiftPartialSkip" not in tail, (
            "ASEM=1 emit must NOT contain the legacy partial-mod skip "
            "label (already dropped pre-fused; fused form has "
            "no mod>0 chain)."
        )
        assert re.search(
            r"s_and_b32[^\n]*LoopCounterL[^\n]*0x1"
            r"[^\n]*partial-mod residue",
            tail,
        ) is None, (
            "ASEM=1 emit must NOT compute the runtime partial-mod "
            "residue (legacy chain artifact)."
        )

        # Fused init: `v_sub_i32 vDiff, sgprLoopCounterL, vKPosBase`
        # and `v_mov_b32 vHalfKeep, 0x0000FFFF`.
        assert re.search(
            r"v_sub_i32\s+v\d+,\s+s\[sgprLoopCounterL\],\s+v\d+"
            r"[^\n]*diff = LoopCounterL - kPosBase",
            tail,
        ), (
            "ASEM=1 emit missing fused init `v_sub_i32 vDiff, "
            "sgprLoopCounterL, vKPosBase`. Tail excerpt:\n"
            + tail[:2000]
        )
        assert re.search(
            r"v_mov_b32\s+v\d+,\s*0x0000FFFF[^\n]*halfKeep",
            tail, re.IGNORECASE,
        ), (
            "ASEM=1 emit missing fused init `v_mov_b32 vHalfKeep, "
            "0x0000FFFF`. Tail excerpt:\n" + tail[:2000]
        )

        # Boundary init: per-i cndmask `(d<hi) ? halfKeep : full`.
        boundary_cnd = re.findall(
            r"subLaneMask boundary\[\d+\] = \(d<\d+\) \? halfKeep : full",
            tail,
        )
        assert len(boundary_cnd) >= 1, (
            "ASEM=1 emit missing boundary init cndmask "
            "`boundary[i] = (d<hi) ? halfKeep : full`. Tail excerpt:\n"
            + tail[:2500]
        )

        # Per-(mmak, ir) sFull cndmask: `mask = sFull ? full : boundary[ir]`.
        sfull_cnd = re.search(
            r"v_cndmask_b32\s+v(\d+),\s+v\d+,\s*-1,\s+s\[\d+:\d+\]"
            r"[^\n]*= sFull \? full : boundary",
            tail,
        )
        assert sfull_cnd is not None, (
            "ASEM=1 emit missing fused sFull cndmask "
            "`mask = sFull ? full : boundary[ir]`. Tail excerpt:\n"
            + tail[:2500]
        )

        # Per-(mmak, ir) sZero cndmask: `mask = sZero ? 0 : prev`.
        mod0_cnd = re.search(
            r"v_cndmask_b32\s+v(\d+),\s+v\1,\s*0,\s+s\[\d+:\d+\]"
            r"[^\n]*= sZero \? 0 : prev",
            tail,
        )
        assert mod0_cnd is not None, (
            "ASEM=1 emit missing mod=0 cndmask `v_cndmask_b32 "
            "vMask, vMask, 0, s[<lo>:<hi>]`. Tail excerpt:\n"
            + tail[:2000]
        )

        # Per-VGPR mask application for both ValuA and ValuB.
        # Precompute-mode comment ("apply precomputed mask to ...").
        and_a = re.search(
            r"v_and_b32\s+v(\d+),\s+v\d+,\s+v\1"
            r"[^\n]*apply precomputed mask to ValuA",
            tail,
        )
        and_b = re.search(
            r"v_and_b32\s+v(\d+),\s+v\d+,\s+v\1"
            r"[^\n]*apply precomputed mask to ValuB",
            tail,
        )
        assert and_a is not None, (
            "ASEM=1 emit missing `v_and_b32 vIdx, vMask, vIdx` "
            "(apply precomputed mask to ValuA[..]) per-VGPR mask "
            "application. Tail excerpt:\n" + tail[:2000]
        )
        assert and_b is not None, (
            "ASEM=1 emit missing `v_and_b32 vIdx, vMask, vIdx` "
            "(apply precomputed mask to ValuB[..]) per-VGPR mask "
            "application. Tail excerpt:\n" + tail[:2000]
        )

        # ASEM=1 fused chain runs INSIDE the precompute block
        # (before any per-mmak ds_read wait): both the init `diff`
        # subtract AND the per-(mmak, ir) sFull/sZero cmps must
        # precede the first per-mmak ds_read wait.
        ds_wait_idx = tail.find("tail LR mmak=0: wait for ds_reads")
        assert ds_wait_idx > 0, (
            "ASEM=1 emit missing per-mmak `tail LR mmak=0: wait for "
            "ds_reads` marker."
        )
        diff_init_pos = tail.find(
            "subLaneMask fused: diff = LoopCounterL - kPosBase")
        assert 0 < diff_init_pos < ds_wait_idx, (
            "ASEM=1 emit must place the fused `diff = "
            "LoopCounterL - kPosBase` init INSIDE the precompute "
            "block (before the first per-mmak ds_read wait)."
        )
        last_sfull = list(re.finditer(
            r"subLaneMask\[A mmak=\d+ ir=\d+\]: sFull",
            tail,
        ))
        assert last_sfull, "ASEM=1 emit missing fused sFull markers"
        assert last_sfull[-1].start() < ds_wait_idx, (
            "ASEM=1 emit must place the LAST per-(mmak, ir) sFull cmp "
            "INSIDE the precompute block (before the first per-mmak "
            "ds_read wait)."
        )

        # #5 negative pin: the legacy coarse per-VGPR cndmask is
        # subsumed by the byte refine's v_and when the refine fires.
        assert re.search(
            r"v_cndmask_b32[^\n]*if K_idx >= sizeL", tail
        ) is None, (
            "ASEM=1 emit must NOT contain the legacy coarse "
            "`v_cndmask_b32 ... if K_idx >= sizeL` per-VGPR "
            "cndmask (byte refine subsumes it; #5)."
        )

        # Negative: legacy hi16 step (yesterday's intermediate
        # shape) must not be present.
        legacy_and_ffff = re.search(
            r"v_and_b32\s+v\d+,\s*0xffff,\s+v\d+"
            r"[^\n]*hi16 -> 0",
            tail, re.IGNORECASE,
        )
        assert legacy_and_ffff is None, (
            "ASEM=1 emit must NOT contain the legacy per-VGPR "
            "`v_and_b32 v<tmp>, 0xFFFF, v<idx>  // ... hi16 -> 0` "
            "step (replaced by per-(operand, ir) mask chain)."
        )
        legacy_cnd_hi_per_vgpr = re.search(
            r"v_cndmask_b32\s+v(\d+),\s+v\1,\s+v\d+,\s+s\[\d+:\d+\]"
            r"[^\n]*hi16 Valu",
            tail,
        )
        assert legacy_cnd_hi_per_vgpr is None, (
            "ASEM=1 emit must NOT contain the legacy per-VGPR "
            "`v_cndmask_b32 v<idx>, v<idx>, v<tmp>, s[..]` for hi16 "
            "clear (replaced by per-(operand, ir) mask chain)."
        )

        # Negative: the prior helper's `SubtileTailByteHi16Skip`
        # label is gone — its purpose is owned by the unified
        # `SubtileTailByteShiftPartialSkip` runtime gate.
        assert "SubtileTailByteHi16Skip" not in tail, (
            "ASEM=1 emit must NOT contain the legacy "
            "`SubtileTailByteHi16Skip` label (the unified mask "
            "chain uses `SubtileTailByteShiftPartialSkip` instead)."
        )


# ── Precompute + apply split: per-(mmak, ir) mask precompute ─────────────────

class TestAnyKEmit_Precompute:
    """`SubtileTailMaskPrecompute` (default True) hoists every
    per-(operand, mmak, ir) byte-mask chain ABOVE the per-mmak loop
    into dedicated scratch VGPRs. The per-mmak step collapses to a
    pure `v_and_b32 vIdx, vMask[..], vIdx`. These tests pin the
    hoist structure, the v_and-only apply shape, and the config
    switch's two modes (precompute on/off).
    """

    def test_precompute_block_before_per_mmak_loop(self):
        """ASEM=2 (byte refine path, fused default form). Every
        init step + per-(mmak, ir) sFull/sZero chain step must
        appear BEFORE the first `tail LR mmak=0 ... wait for ds_reads`
        marker -- that marker opens the per-mmak loop body, and the
        precompute lives upstream of it. After the marker, the
        per-mmak apply step must NOT emit any further chain
        primitive -- only `v_and_b32`.
        """
        tail = _extract_tail_section(_emit_anyk_tail_asm(asem=2, pgr=0))
        assert tail

        loop_marker = "tail LR mmak=0: wait for ds_reads"
        marker_idx = tail.find(loop_marker)
        assert marker_idx > 0, "Missing per-mmak loop marker"

        precompute_section = tail[:marker_idx]
        apply_section = tail[marker_idx:]

        # Init runs in the precompute section.
        assert ("subLaneMask fused: diff = LoopCounterL - kPosBase"
                in precompute_section), (
            "Precompute section must contain the fused `diff` init"
        )
        # mmak=0 AND mmak=1 sFull cmps in the precompute section
        # (fused form emits all subIterK chains up front).
        assert re.search(
            r"subLaneMask\[A mmak=0 ir=\d+\]: sFull",
            precompute_section,
        ), "Precompute section must contain the mmak=0 sFull cmp"
        assert re.search(
            r"subLaneMask\[A mmak=1 ir=\d+\]: sFull",
            precompute_section,
        ), (
            "Precompute section must contain the mmak=1 sFull cmp "
            "(all subIterK chains hoisted up front)"
        )

        # Apply section must NOT contain any chain primitive.
        assert re.search(
            r"v_cmp_(?:gt|le|lt|ge)_i32", apply_section
        ) is None, (
            "Per-mmak apply section must NOT emit any v_cmp "
            "(hoisted out into the precompute block)."
        )
        assert re.search(
            r"subLaneMask\[A mmak=\d+ ir=\d+\]", apply_section
        ) is None, (
            "Per-mmak apply section must NOT emit any fused "
            "chain markers (hoisted out into the precompute block)."
        )

    def test_precompute_hoisted_above_dtl_wait_and_barrier(self):
        """The K-tail mask precompute reads only `LoopCounterL` and
        `kPosBaseVgpr`; it does NOT consume any DTL/LDS data. To let
        the cmp/cndmask chain co-issue with the buffer-load latency
        the precompute must be emitted ABOVE the
        `tail GR: wait for DTL writes to LDS` swait + the
        `tail GR: LDS sync before LR` barrier (rather than
        serializing behind them). Tests both ASEM=1 and ASEM=2 under
        the fused default form.
        """
        for asem in (1, 2):
            tail = _extract_tail_section(_emit_anyk_tail_asm(asem=asem, pgr=0))
            assert tail, "ASEM=%d emit produced no tail block" % asem

            dtl_wait_marker = "tail GR: wait for DTL writes to LDS"
            barrier_marker = "tail GR: LDS sync before LR"
            dtl_wait_idx = tail.find(dtl_wait_marker)
            barrier_idx = tail.find(barrier_marker)
            assert dtl_wait_idx > 0, (
                "ASEM=%d emit missing `%s` swait marker"
                % (asem, dtl_wait_marker))
            assert barrier_idx > dtl_wait_idx, (
                "ASEM=%d emit missing `%s` barrier after swait"
                % (asem, barrier_marker))

            # Fused form: pin the `diff` init marker (sole entry
            # point of the chain) above the DTL wait.
            diff_pos = tail.find(
                "subLaneMask fused: diff = LoopCounterL - kPosBase")
            assert diff_pos > 0, (
                "ASEM=%d emit missing fused `diff` init comment"
                % asem)
            assert diff_pos < dtl_wait_idx, (
                "ASEM=%d emit must hoist the fused `diff` init "
                "ABOVE the `%s` swait so cmp/cndmask can co-issue "
                "with the buffer-load latency.\n"
                "diff@%d, dtl_wait@%d"
                % (asem, dtl_wait_marker, diff_pos, dtl_wait_idx))

            # Every per-(mmak, ir) cmp also lives above the wait.
            for m in re.finditer(
                r"subLaneMask\[A mmak=\d+ ir=\d+\]: s(?:Full|Zero)",
                tail,
            ):
                assert m.start() < dtl_wait_idx, (
                    "ASEM=%d emit has fused chain marker at "
                    "offset %d AFTER the DTL wait at offset %d -- "
                    "the precompute chain must be fully hoisted "
                    "above the wait."
                    % (asem, m.start(), dtl_wait_idx))

    def test_precompute_shares_mask_vgpr_between_A_and_B(self):
        """bf16/bf16 has bpeA==bpeB so the per-(mmak, ir) mask is
        identical for A and B; the precompute must allocate ONE
        mask VGPR per (mmak, ir) and reuse it across A's and B's
        v_and apply steps (halves the VGPR cost vs per-operand
        masks).
        """
        tail = _extract_tail_section(_emit_anyk_tail_asm(asem=2, pgr=0))
        assert tail

        # Collect the source-VGPR of each `v_and ... apply
        # precomputed mask to Valu{A,B}` line, keyed by (op, mmak,
        # ir). Each operand-side group at the same (mmak, ir) must
        # share its mask VGPR with the OTHER operand at the same
        # (mmak, ir).
        masks_by_key = {}
        for op in ("A", "B"):
            for m in re.finditer(
                r"v_and_b32\s+v\d+,\s+v(\d+),\s+v\d+"
                r"[^\n]*apply precomputed mask to Valu" + op
                + r"\[\d+\][^\n]*",
                tail,
            ):
                # The seed comment baked the mmak/ir; recover them
                # from the line by looking at the same comment.
                # Each line's full text contains
                # 'byteRefine[<op> ir=<ir> mmak=<mmak>]:'.
                line_match = re.search(
                    r"byteRefine\[(A|B) ir=(\d+) mmak=(\d+)\]",
                    m.group(0))
                assert line_match, "Apply line missing byteRefine tag"
                key = (line_match.group(2), line_match.group(3))
                masks_by_key.setdefault((op, key), set()).add(
                    int(m.group(1)))

        # Per (op, (ir, mmak)) the apply lines all use one mask
        # VGPR.
        for (op, key), mask_set in masks_by_key.items():
            assert len(mask_set) == 1, (
                "Apply lines for (%s, ir=%s, mmak=%s) must reference "
                "exactly one mask VGPR; got %r"
                % (op, key[0], key[1], sorted(mask_set))
            )

        # A and B must share their mask VGPR at the same (ir, mmak).
        keys = set(k for (_, k) in masks_by_key.keys())
        for key in keys:
            ma = next(iter(masks_by_key[("A", key)]))
            mb = next(iter(masks_by_key[("B", key)]))
            assert ma == mb, (
                "A and B at (ir=%s, mmak=%s) must share the same "
                "precomputed mask VGPR (bpeA==bpeB → dedupe); got "
                "A=v%d, B=v%d." % (key[0], key[1], ma, mb)
            )

    def test_precompute_disabled_reverts_to_legacy_per_mmak_inline(self):
        """`SubtileTailMaskPrecompute=False` must revert to the
        legacy per-mmak inline chain (apply comment loses the
        `precomputed` qualifier, and the chain instructions appear
        AFTER each per-mmak ds_read wait, not before).
        """
        from Tensile.Tests.unit._subtile_tailloop_fixtures import (
            build_minimal_subtile_kwa,
            setdefault_tail_scaffold_kernel_keys,
            wrap_with_skiptoend,
        )

        kernel = _create_kernel(MT0=128, MT1=128, fp4=False,
                                depthU=64, no_tail_loop=False)
        setdefault_tail_scaffold_kernel_keys(kernel, pgr=0, asem=2)
        kernel["SubtileTailMaskPrecompute"] = False
        kwa = build_minimal_subtile_kwa(kernel)
        module = kwa._emitTailLoopScaffoldSubtile(
            kernel,
            {"is_sparse": False, "tpsMetadata": None},
            {"is_sparse": False, "tpsMetadata": None})
        tail = _extract_tail_section(wrap_with_skiptoend(module))
        assert tail

        # Legacy apply comment (no "precomputed").
        assert re.search(
            r"v_and_b32[^\n]*apply mask to ValuA\b", tail
        ) is not None, (
            "Legacy mode must emit `apply mask to ValuA[..]` "
            "(no `precomputed` qualifier)."
        )
        assert re.search(
            r"v_and_b32[^\n]*apply precomputed mask to Valu", tail
        ) is None, (
            "Legacy mode must NOT emit `apply precomputed mask to "
            "...` (that comment is only used when precompute fires)."
        )

        # Chain emission must come AFTER each per-mmak ds_read wait
        # (inline within the loop body), not BEFORE.
        loop_marker = "tail LR mmak=0: wait for ds_reads"
        marker_idx = tail.find(loop_marker)
        assert marker_idx > 0
        precompute_section = tail[:marker_idx]
        assert "mask seed = full keep" not in precompute_section, (
            "Legacy mode must NOT emit any chain seed BEFORE the "
            "per-mmak loop (no precompute block)."
        )

    def test_no_tail_loop_emits_no_precompute(self):
        """NoTailLoop=True elides the entire tail body; the
        precompute block must not appear (nothing references it).
        """
        from Tensile.Tests.unit._subtile_tailloop_fixtures import (
            build_minimal_subtile_kwa,
            setdefault_tail_scaffold_kernel_keys,
            wrap_with_skiptoend,
        )

        kernel = _create_kernel(MT0=128, MT1=128, fp4=False,
                                depthU=64, no_tail_loop=True)
        setdefault_tail_scaffold_kernel_keys(kernel, pgr=0, asem=2)
        kwa = build_minimal_subtile_kwa(kernel)
        module = kwa._emitTailLoopScaffoldSubtile(
            kernel,
            {"is_sparse": False, "tpsMetadata": None},
            {"is_sparse": False, "tpsMetadata": None})
        asm = wrap_with_skiptoend(module)

        assert "byteRefine[" not in asm, (
            "NoTailLoop=True must NOT emit any byteRefine chain "
            "(no tail body → no precompute)."
        )
        assert "apply precomputed mask" not in asm, (
            "NoTailLoop=True must NOT emit any precomputed-mask apply."
        )

    def test_coarse_path_unaffected_by_precompute_switch(self):
        """ASEM>=numMIInUnroll uses the coarse per-mmak cmp+cndmask
        path (byte refine does not fire). The precompute switch
        only governs the byte-refine path, so the coarse emit shape
        must be identical whether precompute is enabled or not.
        """
        from Tensile.Tests.unit._subtile_tailloop_fixtures import (
            build_minimal_subtile_kwa,
            setdefault_tail_scaffold_kernel_keys,
            wrap_with_skiptoend,
        )

        def _emit_with(precompute):
            kernel = _create_kernel(MT0=128, MT1=128, fp4=False,
                                    depthU=64, no_tail_loop=False)
            setdefault_tail_scaffold_kernel_keys(kernel, pgr=0, asem=32)
            kernel["SubtileTailMaskPrecompute"] = precompute
            kwa = build_minimal_subtile_kwa(kernel)
            module = kwa._emitTailLoopScaffoldSubtile(
                kernel,
                {"is_sparse": False, "tpsMetadata": None},
                {"is_sparse": False, "tpsMetadata": None})
            return _extract_tail_section(wrap_with_skiptoend(module))

        tail_on = _emit_with(True)
        tail_off = _emit_with(False)
        assert tail_on == tail_off, (
            "Coarse-path (ASEM=32 byte-refine inapplicable) tail "
            "emit must be identical with SubtileTailMaskPrecompute "
            "on vs off."
        )


# ── Regression net: K%32 emit must stay identical ────────────────────────────

class TestAnyKEmit_K32Unchanged:
    """K%32 (ASEM=32) regression net: structural fingerprint (cmp /
    cndmask presence + absence of any-K helper opcodes) so the K32
    emit cannot silently drift when the any-K helpers change.
    """

    K32_FINGERPRINT_HAS_LANE_CMP = True
    K32_FINGERPRINT_HAS_CNDMASK_VALUA = True
    K32_FINGERPRINT_HAS_CNDMASK_VALUB = True
    K32_FINGERPRINT_NO_LSHL_B64 = True
    K32_FINGERPRINT_NO_D16_B16 = True
    K32_FINGERPRINT_NO_AND_7 = True

    def test_k32_unchanged_with_new_helpers(self):
        asm = _emit_anyk_tail_asm(asem=32, pgr=0)
        tail = _extract_tail_section(asm)
        assert tail, "K%32 emit produced no tail block"

        cmp_match = re.search(r"v_cmp_ge_i32.*LoopCounterL", tail)
        assert (cmp_match is not None) == self.K32_FINGERPRINT_HAS_LANE_CMP, (
            "K%32 emit lane cmp shape changed"
        )
        cnd_a = re.search(r"v_cndmask_b32.*[vV]aluA", tail)
        assert (cnd_a is not None) == self.K32_FINGERPRINT_HAS_CNDMASK_VALUA, (
            "K%32 emit cndmask valuA shape changed"
        )
        cnd_b = re.search(r"v_cndmask_b32.*[vV]aluB", tail)
        assert (cnd_b is not None) == self.K32_FINGERPRINT_HAS_CNDMASK_VALUB, (
            "K%32 emit cndmask valuB shape changed"
        )

        no_lshl = re.search(r"v_lsh(?:l|lrev)_b64", tail) is None
        assert no_lshl == self.K32_FINGERPRINT_NO_LSHL_B64, (
            "K%32 emit must NOT contain v_lshl(rev)_b64"
        )
        no_d16 = re.search(
            r"buffer_load_d16_b16|buffer_load_short_d16|BufferLoadD16B16",
            tail,
        ) is None
        assert no_d16 == self.K32_FINGERPRINT_NO_D16_B16, (
            "K%32 emit must NOT contain buffer_load_d16_b16"
        )
        no_and_7 = re.search(
            r"s_and_b32[^\n]*(?:\b7\b|\b8\s*-\s*1\b)", tail
        ) is None
        assert no_and_7 == self.K32_FINGERPRINT_NO_AND_7, (
            "K%32 emit must NOT compute K_remain mod 8 via "
            "`s_and_b32 ..., 7`"
        )


# ── Direct predicate test: `_subtileTailByteShiftApplies` ────────────────────

def _build_predicate_kernel(*, asem, mxa, mxb, bpeA, bpeB):
    """Minimal kernel-dict driving `_subtileTailByteShiftApplies`.

    The predicate only reads `AssertSummationElementMultiple`,
    `ProblemType.MXBlockA`, `ProblemType.MXBlockB`, and
    `ProblemType.DataType{A,B}.numBytes()`. No other state is needed.
    """
    class _Bpe:
        def __init__(self, n):
            self._n = n

        def numBytes(self):
            return self._n

    return {
        "AssertSummationElementMultiple": asem,
        "ProblemType": {
            "MXBlockA": mxa,
            "MXBlockB": mxb,
            "DataTypeA": _Bpe(bpeA),
            "DataTypeB": _Bpe(bpeB),
        },
    }


# bpeA / bpeB legend: bf16/fp16 = 2, mxfp8/int8 = 1, mxfp4 = 0.5,
# fp32 = 4. The relaxed gate accepts any operand pair with integer
# bpe in [1, bpr]; sub-byte (mxfp4: 0.5) and >register (>4) are out.
@pytest.mark.parametrize(
    "asem,numMIInUnroll,mxa,mxb,bpeA,bpeB,expected",
    [
        # Inside the asem<numMIInUnroll window, no MX, integer bpe.
        (4,  8, 0,  0, 2,   2,   True),    # bf16 / fp16 (homogeneous)
        (4,  8, 0,  0, 1,   1,   True),    # int8/fp8 (homogeneous; #3
                                            # gate now accepts integer bpe)
        (4,  8, 0,  0, 2,   1,   True),    # mixed bf16/fp8 (#3 enables
                                            # per-operand handling)
        (4,  8, 0,  0, 4,   4,   True),    # fp32 (1 element/VGPR);
                                            # mod=0-only chain is valid
        # MX path: predicate must reject regardless of asem/bpe.
        (4, 32, 32, 0, 1,   2,   False),   # mxfp8 A
        (4, 32, 0, 32, 2,   0.5, False),   # mxfp4 B
        # asem >= numMIInUnroll: coarse mask covers everything.
        (8,  8, 0,  0, 2,   2,   False),
        (32, 8, 0,  0, 2,   2,   False),
        # Sub-byte / non-integer bpe: helper assumes byte-aligned mod
        # boundaries, so mxfp4 (numBytes=0.5) must be rejected.
        (4,  8, 0,  0, 0.5, 0.5, False),
        (4,  8, 0,  0, 2,   0.5, False),   # mixed bf16 / mxfp4
    ],
)
def test_subtile_tail_byte_shift_applies_predicate(
    asem, numMIInUnroll, mxa, mxb, bpeA, bpeB, expected
):
    """Pin the predicate's truth table across asem / MX / dtype.

    The gate now accepts any non-MX operand pair with integer bpe in
    `[1, bpr]` so the helper can fire on fp8/int8/fp16/bf16/fp32; only
    the MX path and ASEM>=numMIInUnroll force a False.
    """
    from Tensile.KernelWriter import KernelWriter

    kernel = _build_predicate_kernel(
        asem=asem, mxa=mxa, mxb=mxb, bpeA=bpeA, bpeB=bpeB
    )
    actual = KernelWriter._subtileTailByteShiftApplies(kernel, numMIInUnroll)
    assert actual is expected, (
        f"Predicate(asem={asem}, numMIInUnroll={numMIInUnroll}, "
        f"mxa={mxa}, mxb={mxb}, bpeA={bpeA}, bpeB={bpeB}) returned "
        f"{actual}, expected {expected}."
    )
