# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The single validation the driver runs before issuing an MMA (``_validate_mma_issue``: operand
soundness, pairwise K-match, accumulator store-coordinate consistency, and atom contiguity), the
``canonical_layouts`` helper it checks against, and the rectangular-wave regression. Offline -- no GPU,
no IR builder (the validation is pure label-checking; a fragment with no value is enough).

The headline case: a mislabeled accumulator (C) whose register-to-(m,n) map disagrees with what the
machine actually produces was silently accepted and stored to the wrong coordinates. The accumulator
check now catches it."""

from __future__ import annotations

import pathlib
import types

import pytest

from rocke.core.ir import F16, F32
from rocke.helpers.tiling import make_fragment
from rocke.helpers.tiling.analysis.soundness import verify_mma_soundness
from rocke.helpers.tiling.mma.driver import _validate_mma_issue
from rocke.helpers.tiling.mma.mma_operation import TileMma
from rocke.helpers.tiling.mma.plan import TileMmaPlan, Tiling
from rocke.helpers.tiling.mma.styles import CanonicalStyle, InterleavedStyle
from rocke.helpers.tiling.mma.warp_encoding import canonical_layouts
from rocke.helpers.tiling.tiling_recorder import PipelineOp
from rocke.helpers.tiling.transforms import mma_operand_layout_sound, mma_pair_k_aligned


def _mma(shape=(32, 32, 16), atom=(16, 16, 16), **kw):
    return TileMma(
        shape,
        a="f16",
        b="f16",
        c="f32",
        target="gfx90a",
        tiling=Tiling(atom_shape=atom),
        **kw,
    )


# ---- the canonical_layouts helper ---------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,atom",
    [
        ((16, 16, 16), (16, 16, 16)),  # single atom
        ((32, 32, 16), (16, 16, 16)),  # square multi-atom
        ((32, 16, 16), (16, 16, 16)),  # rectangular m_sub != n_sub
        ((16, 16, 32), (16, 16, 16)),  # k_sub > 1
        ((48, 16, 32), (16, 16, 16)),  # rectangular + k_sub > 1
    ],
)
def test_canonical_layouts_byte_identical_to_plan_accessors(shape, atom):
    # p is canonical-style (no style=), so its style-faithful a_layout/b_layout/c_layout ARE the canonical
    # encodings -- this pins the helper to what CanonicalStyle produces. Under a non-canonical style the
    # accessors differ from canonical_layouts by design.
    p = TileMmaPlan(
        shape,
        a="f16",
        b="f16",
        c="f32",
        target="gfx90a",
        tiling=Tiling(atom_shape=atom),
    )
    assert canonical_layouts(p.traits, p.subtiles) == (
        p.a_layout,
        p.b_layout,
        p.c_layout,
    )


def test_canonical_layouts_matches_canonical_style_operands():
    # Drift guard: the canonical yardstick MUST equal what the default CanonicalStyle produces, or the
    # soundness gate would check default-style operands against a diverged reference and never notice.
    p = TileMmaPlan(
        (32, 32, 16),
        a="f16",
        b="f16",
        c="f32",
        target="gfx90a",
        tiling=Tiling(atom_shape=(16, 16, 16)),
    )
    a_canon, b_canon, c_canon = canonical_layouts(p.traits, p.subtiles)
    style = CanonicalStyle()
    m_sub, n_sub, k_sub = p.subtiles
    assert (
        a_canon
        == style.operand_desc(p.traits, role="A", free_sub=m_sub, k_sub=k_sub).layout
    )
    assert (
        b_canon
        == style.operand_desc(p.traits, role="B", free_sub=n_sub, k_sub=k_sub).layout
    )
    assert c_canon == style.accumulator_desc(p.traits, m_sub=m_sub, n_sub=n_sub).layout


# ---- the driver's issue-time validation ---------------------------------------------------------


def test_accepts_machine_native_accumulator():
    mma = _mma()
    a = make_fragment(mma.a_desc, F16)
    b = make_fragment(mma.b_desc, F16)
    c = make_fragment(mma.c_desc, F32)
    _validate_mma_issue(a, b, c, mma.plan)  # the machine-native C -- must not raise


def test_rejects_swapped_accumulator_canonical():
    # Canonical operands with a transposed accumulator (c_desc.swap_dims(0,1)): lane 0 / register 1 claims
    # it owns output (0,1), but the machine derives (1,0). The accumulator check pins the register-to-(m,n)
    # map, so the mislabel is rejected -- this store-to-wrong-coordinate was silently accepted before.
    mma = _mma()
    a = make_fragment(mma.a_desc, F16)
    b = make_fragment(mma.b_desc, F16)
    c_bad = make_fragment(mma.c_desc.swap_dims(0, 1), F32)
    with pytest.raises(ValueError, match="accumulator .C. not consistent"):
        _validate_mma_issue(a, b, c_bad, mma.plan)


def test_rejects_canonical_accumulator_under_interleaved():
    # An interleaved plan derives its own accumulator from the interleaved operands. Feeding a CANONICAL
    # C is a mislabel; the accumulator check catches it. The canonical C is built from a canonical plan --
    # the interleaved object no longer exposes a canonical accessor (its c_desc IS the interleaved C).
    mma = _mma(style=InterleavedStyle())
    a = make_fragment(mma.a_desc, F16)
    b = make_fragment(mma.b_desc, F16)
    c_bad = make_fragment(_mma().c_desc, F32)
    with pytest.raises(ValueError, match="accumulator .C. not consistent"):
        _validate_mma_issue(a, b, c_bad, mma.plan)


def test_accepts_interleaved_demo_on_canonical_plan():
    # The shipped interleaved demo builds a CANONICAL plan and hand-feeds interleaved A/B/C. The validation
    # MUST accept it: the accumulator check derives C from the PASSED (interleaved) operands, so a correct
    # hand-built C is accepted even though it differs from the plan's own c_desc. (The canonical-vs-style
    # accumulator distinction is exercised by test_rejects_canonical_accumulator_under_interleaved.)
    from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import (
        _wave_descs_interleaved,
    )

    mma = _mma()  # canonical plan, 32x32x16 -> m_sub = n_sub = 2
    a_desc, b_desc, c_desc = _wave_descs_interleaved(2, 2, 1)
    a = make_fragment(a_desc, F16)
    b = make_fragment(b_desc, F16)
    c = make_fragment(c_desc, F32)
    _validate_mma_issue(a, b, c, mma.plan)  # must not raise


def test_rejects_reorderable_k_misorder():
    # A reorder-fixable K-misorder: A's K registers are permuted vs B, so the K SET per lane is unchanged
    # (A stays a sound operand) but the positional K order differs -- the pair-check is a WARNING, not an
    # error. The driver issues no reorder, so it must still REJECT; a valid kernel would transform_fragment
    # one operand first. This is the one path where an un-reordered warning would otherwise miscompile.
    mma = _mma(shape=(16, 16, 32))  # k_sub = 2 gives K register buckets to permute
    a_reordered = mma.a_desc.reorder_registers((1, 2, 0))
    m_sub, n_sub, _ = mma.subtiles
    a_canon = mma.a_desc.layout
    # Self-check the construction: A is still SOUND, and the pair is genuinely a reorder-fixable WARNING --
    # so this test cannot pass for the wrong reason if the layout math shifts.
    assert (
        mma_operand_layout_sound(a_reordered.layout, a_canon, role="A").severity == "ok"
    )
    assert (
        mma_pair_k_aligned(
            a_reordered.layout,
            mma.b_desc.layout,
            a_free_atoms=m_sub,
            b_free_atoms=n_sub,
        ).severity
        == "warning"
    )
    a = make_fragment(a_reordered, F16)
    b = make_fragment(mma.b_desc, F16)
    c = make_fragment(mma.c_desc, F32)
    with pytest.raises(ValueError, match="not K-aligned"):
        _validate_mma_issue(a, b, c, mma.plan)


def test_rejects_unsound_operand():
    # An operand register permutation that scrambles the M labeling makes A no longer a sound MMA operand
    # (a machine output-row carries more than one M). The validation rejects it with the operand-soundness
    # error -- and must do so BEFORE the accumulator check, which flows operands through the machine and is
    # undefined on an unsound one.
    mma = _mma(shape=(32, 16, 32))
    a_unsound = mma.a_desc.reorder_registers((1, 0, 2, 3))
    a_canon = mma.a_desc.layout
    assert (
        mma_operand_layout_sound(a_unsound.layout, a_canon, role="A").severity
        == "error"
    )  # self-check
    a = make_fragment(a_unsound, F16)
    b = make_fragment(mma.b_desc, F16)
    c = make_fragment(mma.c_desc, F32)
    with pytest.raises(ValueError, match="operand not sound"):
        _validate_mma_issue(a, b, c, mma.plan)


# ---- regressions the fix had to preserve --------------------------------------------------------


def test_soundness_gate_accepts_rectangular_wave():
    # A rectangular wave tile (m_sub != n_sub) has whole-wave K-lists of unequal length for A vs B.
    # verify_mma_soundness must PASS it: it threads the per-atom free-atom counts so the K-match is per
    # atom, not a whole-fragment compare that would false-error. (This case previously had no coverage.)
    mma = _mma(shape=(32, 16, 16))  # m_sub = 2, n_sub = 1
    a_canon, b_canon, c_canon = mma.a_desc.layout, mma.b_desc.layout, mma.c_desc.layout
    op = PipelineOp(
        kind="mma",
        seq=0,
        a_enc=mma.a_desc.layout,
        b_enc=mma.b_desc.layout,
        c_enc=mma.c_desc.layout,
        a_canon=a_canon,
        b_canon=b_canon,
        c_canon=c_canon,
        a_free_atoms=2,
        b_free_atoms=1,
    )
    pipeline = types.SimpleNamespace(ops=[op])
    assert verify_mma_soundness(pipeline) == 1  # no MmaSoundnessError


def test_driver_is_sole_fragment_slicer():
    # The atom-contiguity guard lives only in the driver because the driver is the ONLY site that
    # hand-slices a fragment by atom (_read_subvector / _write_subvector / _subtile_triples). A new
    # hand-slicer elsewhere would need its own contiguity guard -- this fails loud if one appears.
    import rocke.helpers.tiling as tiling_pkg

    root = pathlib.Path(tiling_pkg.__file__).parent
    slicers = ("_read_subvector", "_write_subvector", "_subtile_triples")
    hits = {
        p.name for p in root.rglob("*.py") if any(s in p.read_text() for s in slicers)
    }
    assert hits == {
        "driver.py"
    }, f"fragment-slicing helpers leaked outside driver.py: {sorted(hits)}"
