# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pin the per-mmak interleave restructure of the subtile tail scaffold.

The historic scaffold allocated every A/B mma vgprTile up front and
emitted LR for the whole grid before the per-mmak MFMA loop, which
held ~184 VGPRs live across mmak on MT320x288x64 BF16 and blew the
wave-64 256-VGPR budget. The restructure (this commit) interleaves
the slice alloc + ds_read + MFMA grid per mmak so peak pressure
tracks one slice (~92 VGPRs for the same MT).

This file pins three invariants of the restructure:

1. The tail body emits `ds_read` per mmak iteration (not once up
   front for every mmak's worth of slices).
2. The emit-time `vgprPool.size()` is identical before and after
   the scaffold runs (no slice leak; the per-mmak alloc/free is
   matched).
3. The aligned-K hot path (`NoTailLoop=True`) is byte-identical to
   before: no `ds_read` for A/B is emitted at all.
"""
import re

from Tensile.Tests.unit._subtile_tailloop_fixtures import (
    build_minimal_subtile_kwa,
    setdefault_tail_scaffold_kernel_keys,
    wrap_with_skiptoend,
)
from Tensile.Tests.unit.test_subtile_tailloop_emit import (
    _create_kernel,
    _extract_tail_section,
)


def _drive_scaffold(*, asem, fp4=False, no_tail_loop=False,
                    MT0=128, MT1=128, depthU=None):
    """Run `_emitTailLoopScaffoldSubtile` against a minimal writer
    and return (writer, full_asm, live_before, live_after).

    `live_before` / `live_after` are the "checked-out and not yet
    checked-in" VGPR counts at scaffold entry / exit, computed as
    `vgprPool.size() - vgprPool.available()` (the pool's
    high-water minus the count of already-released slots). The
    scaffold's per-mmak alloc/free pattern must net to zero in this
    metric; using raw `size()` would also count high-water growth
    from the scaffold's internal scratch allocs, which is expected.
    """
    kernel = _create_kernel(MT0=MT0, MT1=MT1, fp4=fp4,
                            depthU=depthU,
                            no_tail_loop=no_tail_loop)
    setdefault_tail_scaffold_kernel_keys(kernel, pgr=0, asem=asem)
    kwa = build_minimal_subtile_kwa(kernel)
    live_before = kwa.vgprPool.size() - kwa.vgprPool.available()
    tPA = {"is_sparse": False, "tpsMetadata": None}
    tPB = {"is_sparse": False, "tpsMetadata": None}
    module = kwa._emitTailLoopScaffoldSubtile(kernel, tPA, tPB)
    live_after = kwa.vgprPool.size() - kwa.vgprPool.available()
    return kwa, wrap_with_skiptoend(module), live_before, live_after


# ── Per-mmak LR emit shape ──────────────────────────────────────────────────

def test_tail_ds_read_emitted_inside_mmak_loop():
    """The bf16 fixture has `localMMATileGrid[1] == 2` so the
    scaffold's per-mmak loop runs for two iterations. Each iteration
    must emit its own `ds_read*` block for the K-slice; the historic
    pre-loop bulk LR for all mmak slices is gone, so the ordering
    is `cmp/cndmask` → `ds_read` interleaved per mmak.

    Pin: at least one `ds_read*` instruction appears AFTER the first
    `tail MFMA` line (which is the mmak=0 MFMA grid) -- i.e. the LR
    for mmak=1 follows mmak=0's MFMA. With the legacy emit order
    (LR for all mmak before any MFMA) every `ds_read*` would precede
    the first MFMA.
    """
    _, asm, _, _ = _drive_scaffold(asem=32)
    tail = _extract_tail_section(asm)
    assert tail, "No tail block emitted"

    first_mfma = tail.find("tail MFMA")
    assert first_mfma > 0, (
        "Tail body must contain at least one MFMA instruction; "
        "got tail excerpt:\n" + tail[:1500]
    )
    after_first_mfma = tail[first_mfma:]
    ds_after = re.findall(r"ds_read", after_first_mfma)
    assert ds_after, (
        "Expected at least one ds_read AFTER the first tail MFMA "
        "(the mmak=1 slice LR follows mmak=0's MFMA grid in the "
        "per-mmak interleave). Tail body:\n" + tail[-2000:]
    )


def test_tail_per_mmak_lr_waitcnt_pairs_with_each_mmak():
    """Each mmak must wait for its slice's `ds_read*` before the
    masking + MFMA block consumes the destination VGPRs. The
    scaffold emits one `s_waitcnt` per mmak tagged with `tail LR
    mmak=<n>`; pin both `mmak=0` and `mmak=1` (bf16 fixture's two
    mmaks). This catches a regression that collapses the per-mmak
    wait back into a single pre-loop wait (which would leak the
    `dscnt=0` ordering with byte-refine VGPRs of later mmak).
    """
    _, asm, _, _ = _drive_scaffold(asem=32)
    tail = _extract_tail_section(asm)
    assert tail
    waits = re.findall(r"s_waitcnt[^\n]*tail LR mmak=(\d+)", tail)
    assert "0" in waits, (
        "Tail must emit `s_waitcnt ... tail LR mmak=0`. "
        "Tail excerpt:\n" + tail[:2000]
    )
    assert "1" in waits, (
        "Tail must emit `s_waitcnt ... tail LR mmak=1`. "
        "Tail excerpt:\n" + tail[:2000]
    )


def test_tail_emits_no_aggregate_lr_wait_pre_mmak_loop():
    """The legacy emit produced a single `tail LR: wait for
    ds_reads` waitcnt right before the mmak loop (covering every
    mmak's slice in one wait). With the per-mmak interleave that
    aggregate wait is replaced by the per-mmak `tail LR mmak=<n>`
    waits exercised above; the aggregate comment must not appear
    in the tail body.
    """
    _, asm, _, _ = _drive_scaffold(asem=32)
    tail = _extract_tail_section(asm)
    assert tail
    assert "tail LR: wait for ds_reads before lane mask" not in tail, (
        "The aggregate pre-loop `tail LR: wait for ds_reads ...` "
        "comment must not appear -- the per-mmak interleave emits "
        "its own waitcnt inside each mmak iteration. Tail excerpt:"
        "\n" + tail[:2000]
    )


# ── VGPR pool balance ───────────────────────────────────────────────────────

def test_tail_scaffold_does_not_leak_vgprs():
    """The per-mmak alloc/free in the scaffold must net to zero:
    the writer's vgprPool live count (size - available) at scaffold
    exit equals the live count at scaffold entry. Catches
    regressions where the slice free is dropped (or where a code
    path between the alloc and the free branches out without
    releasing the slice's VGPRs back).

    The kPosBase VGPR is allocated AND released inside the scaffold
    (`self.vgprPool.checkIn(kPosBaseVgpr)` at the end of the
    NoTailLoop=False branch), so it doesn't bias the pool.
    """
    _, _, before, after = _drive_scaffold(asem=32)
    assert before == after, (
        f"vgprPool live count before scaffold ({before}) differs "
        f"from after ({after}); the per-mmak slice alloc/free must "
        f"net to zero."
    )


def test_tail_scaffold_pool_balance_byte_refine_asem2():
    """Same vgprPool balance pin under the sub-lane byte refine
    branch (ASEM=2). The byte refine allocates 3 scratch VGPRs
    (`kPosCur`, `maskVgpr`, `seedVgpr`) and releases them inside
    the helper; combined with the per-mmak slice alloc/free, the
    net pool delta must still be zero.
    """
    _, _, before, after = _drive_scaffold(asem=2)
    assert before == after, (
        f"vgprPool live count before scaffold ({before}) differs "
        f"from after ({after}) under ASEM=2 byte-refine path."
    )


def test_tail_scaffold_pool_balance_fp4_asem32():
    """FP4 + MX scale path: the MX scale tiles use the bulk alloc
    (no per-mmak slice), but the A/B per-mmak alloc/free still runs.
    Pin pool balance for the same regression net.
    """
    _, _, before, after = _drive_scaffold(asem=32, fp4=True)
    assert before == after, (
        f"vgprPool live count before scaffold ({before}) differs "
        f"from after ({after}) under FP4 MX path."
    )


# ── Aligned-K hot path stays byte-identical ─────────────────────────────────

def test_no_tail_loop_emits_no_ab_ds_read():
    """When `NoTailLoop=True` the tail body is elided entirely (the
    scaffold only emits the `SkipTailLoopL` join label). In
    particular the A/B ds_read instructions must NOT appear -- the
    aligned-K hot path must stay byte-identical to before the
    per-mmak interleave restructure.
    """
    _, asm, before, after = _drive_scaffold(asem=32, no_tail_loop=True)
    assert "Tail Loop" not in asm
    assert re.search(r"ds_read[^\n]*Subtile[AB]", asm) is None, (
        "NoTailLoop=True must NOT emit any A/B `ds_read*` "
        "instructions; the aligned-K hot path is byte-identical "
        "to before the restructure."
    )
    assert before == after, (
        f"NoTailLoop pool live count delta must be zero: "
        f"{before} != {after}"
    )


def test_no_tail_loop_emits_no_per_mmak_lr_wait():
    """Companion to `test_no_tail_loop_emits_no_ab_ds_read`: the
    `tail LR mmak=<n>` waitcnt comment must not appear when
    NoTailLoop=True (it's only emitted inside the per-mmak loop
    that NoTailLoop=True skips).
    """
    _, asm, _, _ = _drive_scaffold(asem=32, no_tail_loop=True)
    assert re.search(r"tail LR mmak=", asm) is None, (
        "NoTailLoop=True must NOT contain a `tail LR mmak=<n>` "
        "waitcnt; the per-mmak loop should not run at all."
    )
