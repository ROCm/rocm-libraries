# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Warp-specialized producer/consumer 3-stage GEMM pipeline (``wsp3``).

A warp-specialized producer/consumer GEMM for gfx950 / MI355X (CDNA4), built to
push square fp16 GEMM past the ~0.52x-rocBLAS ceiling that the standard
``mem``/``compv3``/``compv4`` pipelines (and CK-Tile's own comp_v4 example)
top out at. Full design + phased plan in
``ck_dsl/_wsp3/BUILD_SPEC.md``.

Architecture (target ~0.85-0.9x rocBLAS):
  * 12 warps / 768 threads, ``launch_bounds(768, 2)``.
      - warps 0..3   = PRODUCERS: async global->LDS (``async_buffer_load_lds``).
      - warps 4..11  = CONSUMERS: LDS->reg (``ds_read``) + ``mfma_f32_16x16x32_f16``.
    Role is wave-uniform (``warp_id``-based), so the two complementary
    ``scf.if`` role-loops carry a MATCHED per-iteration ``s_barrier_bare``
    count and rendezvous correctly (no named/split barrier — those ICE
    gfx950; no scf.if-results needed since cross-role state lives only in LDS).
  * 3-stage LDS ring (As[3][2][2], Bs[3][4][2]), ~147 KB, XOR swizzle.
  * Per-iteration rendezvous: ``s_waitcnt(vmcnt=0); sched_barrier(0);
    s_barrier_bare()`` — drains producer async writes WITHOUT serializing the
    next iteration's in-flight loads (why ``sync()`` is wrong here).

Correctness hazards (see BUILD_SPEC §3): H1 producer-write-visible-before-read,
H2 consumer-read-done-before-overwrite (3-deep ring gives 2-iter WAR slack),
H3 per-operand ds_read landed before its MFMA (per-MFMA ``s_waitcnt(lgkmcnt=0)``).

Phased build (each milestone: builds + verifies bad==0 + a TFLOPS number):
  Ph1 depth-1 warp-split skeleton (correctness)         ~0.15-0.25x   <-- NEXT
  Ph2 depth-2 ring (first async overlap)                ~0.35-0.45x
  Ph3 depth-3 ring (full producer/consumer)                    ~0.50-0.58x
  Ph4 PLR + operand reuse                               ~0.60-0.68x
  Ph5 fine scheduling (s_setprio/sched_barrier)         ~0.68-0.75x
  Ph6 swizzle + wide transposed reads                   ~0.75-0.80x
  Ph7 occupancy/register tune                           ~0.80-0.85x
  Ph8 persistent + chiplet remap                        ~0.85-0.90x (target)
"""

from __future__ import annotations

from ...core.ir import KernelDef
from .gemm_universal import UniversalGemmSpec


def build_wsp3_gemm(spec: UniversalGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Build the warp-specialized producer/consumer 3-stage GEMM.

    Phase 0 (DONE): the ``s_barrier_bare`` IR op + this dispatch scaffold are
    in place and golden-gate-proven (no existing kernel digest changes).

    Phase 1 (NEXT): emit the depth-1 warp-split skeleton — producers load
    tile T into LDS, full barrier, consumers MFMA tile T — and verify bit-exact
    (bad==0) on 256^3 and 4096^3 before adding ring depth. See module docstring
    and BUILD_SPEC.md for the exact emission plan.
    """
    if arch != "gfx950":
        raise ValueError(f"wsp3 pipeline is gfx950-only for now (got {arch!r})")
    raise NotImplementedError(
        "wsp3 Phase 1 (warp-split skeleton) not yet emitted — scaffold + "
        "s_barrier_bare IR op are in place (Phase 0 complete). See "
        "ck_dsl/_wsp3/BUILD_SPEC.md."
    )
