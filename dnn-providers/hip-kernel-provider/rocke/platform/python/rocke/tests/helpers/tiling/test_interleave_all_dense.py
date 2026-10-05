"""Proof sweep — every dense MMA instruction's interleavability (CPU-only, no GPU).

Approach A: drive the REAL interleaved construction (the library ``InterleavedStyle``) from each dense
traits row and PROVE, for the rows the construction can drive, that

  * the operand K-contiguous <-> free-dim-contiguous transpose is cross-lane-free (``reorder`` tier),
  * the C-shuffle (native accumulator -> store order) is cross-lane-free, and
  * the MMA soundness (positional per-atom K-match) holds,

for a single atom AND a 2x2-atom tile (``free_sub>1`` / multi-patch, e.g. the 32x32 atom's ``P=4``).

Rows that CANNOT interleave are PROVEN to fail the SOT §9 interleavability gate (never skipped
silently) -- only the non-square ``wmma_f32_32x16x128_f4`` does. gfx11 WMMA holds the whole K per lane and
DUPLICATES the operand across the two 16-lane halves; that duplicate is modelled as a replication axis and
proven cross-lane-free (per-lane element-set preservation + a within-one-copy ``reorder``). The exact
population counts are asserted so a catalogue change (a new atom, a changed trait) trips the sweep.

Interleavability depends only on (shape, wave, element size), so bf8/fp8, bf16/f16 and f32/xf32 are
equivalent — but every distinct dense row is still exercised.
"""
from __future__ import annotations

import pytest

from rocke.helpers.tiling.traits import load_mma_traits
from rocke.helpers.tiling.layouts import make_tile_desc
from rocke.helpers.tiling.mma.styles import AtomNumbers, InterleavedStyle
from rocke.helpers.tiling.transforms import as_forward_map, classify_transform, validate_operands

# The interleaved operand/accumulator descriptors are the promoted `InterleavedStyle`; the atom quantities
# come from the library `AtomNumbers.from_traits`. The one piece not yet promoted is the C store-order
# (writeback) descriptor -- inlined below (a plain `make_tile_desc`) so the sweep depends only on the
# library, never on an example kernel. Promoting that writeback descriptor is a tracked follow-up.
_STYLE = InterleavedStyle()


def _c_store_desc(atom, wave_m, wave_n, m_sub, n_sub):
    """The C-shuffle TARGET (row-major store order): register order `(mna, m_local, n_local)` with the
    n_sub-wide N run contiguous. The lane's `P` disjoint M runs are a `block_repeat` on M above the lane
    level, so `make_tile_desc` expresses it directly; at `P == 1` it is the single-rectangle desc."""
    return make_tile_desc(
        shape=[wave_m, wave_n],
        thread_tile=[atom.c_inner * m_sub, n_sub],
        thread_dist=[atom.c_lane_rows, atom.n],
        thread_order=[0, 1],               # N fastest -> lane = mo*atom.n + n_in, matching native C
        block_repeat=[atom.c_patches, 1],  # the lane's P disjoint M sub-tiles
        wave_size=atom.wave_size,
    )


def _operand_gate(t) -> bool:
    """§9 operand precondition, PER OPERAND and REPLICATION-AWARE. A's free axis is M (free_lanes=m),
    B's is N (free_lanes=n); both share ``k_lanes = K/ABK``. For EACH operand ``free_lanes * k_lanes``
    must DIVIDE the wave; the quotient is the broadcast/replication factor (1 for CDNA/gfx12; 2 for
    gfx11 WMMA, which holds the whole K per lane and DUPLICATES the operand across the two 16-lane
    halves -- identical data, no cross-lane). Fails when EITHER operand OVER-subscribes the wave -- the
    non-square f4 row does so on A (32*2 = 64 > 32) while B is fine (16*2 = 32), so the row is out."""
    k_lanes = t.k // t.k_ab_per_lane
    return all(
        free_lanes * k_lanes <= t.wave_size and t.wave_size % (free_lanes * k_lanes) == 0
        for free_lanes in (t.m, t.n)  # A's free = M, B's free = N
    )


def _accum_gate(t) -> bool:
    """§9 accumulator precondition: ``c_lane_rows * atom.n == wave``."""
    return (t.m // t.c_m_per_lane) * t.n == t.wave_size


def _operand_reorder_ok(t, a, free_atoms: int, k_sub: int) -> tuple[bool, str]:
    """Is the operand K-contiguous <-> free-dim-contiguous transpose cross-lane-free? REPLICATION-AWARE:
    a replicated (gfx11) operand duplicates each element across copies, so full-wave ``classify_transform``
    false-positives to ``cross_lane`` (it pairs the duplicate with a different copy). Prove it directly:
    (a) every lane keeps its exact element set (the DEFINITION of a within-lane reorder), AND (b) within ONE
    replication copy the transform classifies as ``reorder``. Non-replicated atoms use the plain full-wave check."""
    rd, mma = _STYLE.operand_descs(t, free_sub=free_atoms, k_sub=k_sub, free_lanes=a.m)  # A operand (free = M)
    broadcast = a.wave_size // (a.m * a.k_lanes)
    if broadcast == 1:
        plan = classify_transform(rd.layout, mma.layout)
        return plan.tier == "reorder", plan.tier
    frd, fmma = as_forward_map(rd.layout), as_forward_map(mma.layout)

    def _per_lane(m):
        d: dict[int, set] = {}
        for (lane, _r), coord in m.items():
            d.setdefault(lane, set()).add(coord)
        return d

    if _per_lane(frd) != _per_lane(fmma):
        return False, "an element changes lane (not a within-lane reorder)"
    copy = a.m  # one replication copy == the first m (A free-lane) lanes
    one = lambda m: {(lane, r): c for (lane, r), c in m.items() if lane < copy}
    tier = classify_transform(one(frd), one(fmma)).tier
    return tier == "reorder", tier


_DENSE = sorted(
    (t for t in load_mma_traits().by_op_id.values() if t.family == "dense"),
    key=lambda t: t.op_id,
)
_IDS = [t.op_id for t in _DENSE]

# PROVEN-below boundary (asserted, not assumed): the ONLY dense row that cannot interleave.
#   wmma_f32_32x16x128_f4 -- non-square (m=32 != n=16) AND free x K over-subscribes the wave
#   (32*2 = 64 > 32, so no integer replication factor exists). Both failures are real.
# (The 8 gfx11-era wmma_*_w32 rows hold the whole K per lane and DUPLICATE the operand across the two
#  16-lane halves; modelling that duplicate as a replication axis makes them interleave cross-lane-free --
#  proven via per-lane element-set preservation + a within-one-copy `reorder`.)
_CANNOT_INTERLEAVE = {
    "wmma_f32_32x16x128_f4",
}


@pytest.mark.parametrize("t", _DENSE, ids=_IDS)
def test_dense_mma_interleavable(t):
    a = AtomNumbers.from_traits(t)
    op_ok, acc_ok, square = _operand_gate(t), _accum_gate(t), (t.m == t.n)

    # --- Boundary: a gate fails -> PROVE it cannot interleave (documented, never silent) ---
    if not (op_ok and acc_ok and square):
        assert t.op_id in _CANNOT_INTERLEAVE, (
            f"{t.op_id} newly fails an interleavability gate (op={op_ok}, acc={acc_ok}, "
            f"square={square}) -- investigate and document it"
        )
        return

    # --- Gates pass + square -> PROVE the construction is cross-lane-free (wave32/gfx11 AND wave64) ---
    # Operand K-contiguous <-> free-dim-contiguous transpose, single atom and multi-atom (DPT/k_sub>1).
    for free_atoms, k_sub in [(1, 1), (2, 2)]:
        ok, tier = _operand_reorder_ok(t, a, free_atoms, k_sub)
        assert ok, f"{t.op_id}: operand K<->free is {tier} at (free_atoms={free_atoms}, k_sub={k_sub})"

    # MMA soundness (positional per-atom K-match) for the base single-atom operands. Per-operand free
    # lanes: A's free axis is M (free_lanes=a.m), B's is N (free_lanes=a.n) -- equal for a square atom.
    _, a_mma = _STYLE.operand_descs(t, free_sub=1, k_sub=1, free_lanes=a.m)
    _, b_mma = _STYLE.operand_descs(t, free_sub=1, k_sub=1, free_lanes=a.n)
    ok, why = validate_operands(a_mma.layout, b_mma.layout)
    assert ok, f"{t.op_id}: MMA soundness failed -- {why}"

    # C-shuffle (native accumulator -> store), single atom AND 2x2-atom tile (free_sub>1 / multi-patch).
    for m_sub, n_sub in [(1, 1), (2, 2)]:
        cn = _STYLE.accumulator_desc(t, m_sub=m_sub, n_sub=n_sub)
        cs = _c_store_desc(a, a.m * m_sub, a.n * n_sub, m_sub, n_sub)
        plan = classify_transform(cn.layout, cs.layout)
        assert plan.tier == "reorder", (
            f"{t.op_id}: C-shuffle is {plan.tier} at (m_sub={m_sub}, n_sub={n_sub}) -- {plan.reason}"
        )


def test_proof_population():
    """Lock the sweep population so a catalogue change (new atom / changed trait) is caught."""
    driven = cannot = 0
    for t in _DENSE:
        op_ok, acc_ok, square = _operand_gate(t), _accum_gate(t), (t.m == t.n)
        if op_ok and acc_ok and square:
            driven += 1
        else:
            cannot += 1
    assert (len(_DENSE), driven, cannot) == (73, 72, 1), (
        f"population shifted: total={len(_DENSE)} driven={driven} cannot={cannot}"
    )
    # Every cannot-interleave row is one we documented (and vice-versa).
    measured_cannot = {
        t.op_id for t in _DENSE
        if not (_operand_gate(t) and _accum_gate(t) and t.m == t.n)
    }
    assert measured_cannot == _CANNOT_INTERLEAVE, measured_cannot ^ _CANNOT_INTERLEAVE
