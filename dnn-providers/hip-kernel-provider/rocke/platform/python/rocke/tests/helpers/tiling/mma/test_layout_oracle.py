# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""M1 layout oracle -- our C encoding must field-equal rocke's make_c_warp_dstr_encoding.

This is the headline M1 correctness gate. rocke's ``make_c_warp_dstr_encoding`` is the
Source-of-Truth for the MFMA accumulator layout (and is itself validated to reproduce
``MfmaAtom.lane_to_output``). If our calculator's C encoding is field-equal to it, our
encoding reproduces the hardware layout transitively.
"""

from __future__ import annotations

import pytest

rocke_atoms = pytest.importorskip(
    "rocke.helpers.atoms",
    reason="rocke substrate not importable here; the oracle runs where platform/python "
    "is available (e.g. the gfx90a host)",
)

from rocke.helpers.tiling.mma.warp_encoding import c_warp_encoding  # noqa: E402
from rocke.helpers.tiling.traits import load_mma_traits  # noqa: E402


def _assert_field_equal(ours, oracle) -> None:
    assert ours.replication_lengths == oracle.Rs, "Rs (replication) differs"
    assert ours.hierarchical_lengths == oracle.Hs, "Hs (hierarchical) differs"
    assert ours.lane_to_rh_major == oracle.Ps2RHs_major, "Ps2RHs_major differs"
    assert ours.lane_to_rh_minor == oracle.Ps2RHs_minor, "Ps2RHs_minor differs"
    assert ours.register_to_rh_major == oracle.Ys2RHs_major, "Ys2RHs_major differs"
    assert ours.register_to_rh_minor == oracle.Ys2RHs_minor, "Ys2RHs_minor differs"


def test_c_encoding_field_equals_rocke_oracle_16x16x16_f16() -> None:
    catalog = load_mma_traits()
    ours = c_warp_encoding(
        catalog.select(
            target="gfx90a", input_dtype="f16", output_dtype="f32", m=16, n=16, k=16
        )
    )
    oracle = rocke_atoms.make_c_warp_dstr_encoding(rocke_atoms.MfmaAtom.f16_16x16x16())
    _assert_field_equal(ours, oracle)


def test_c_encoding_field_equals_rocke_oracle_32x32x8_f16() -> None:
    factory = getattr(rocke_atoms.MfmaAtom, "f16_32x32x8", None)
    if factory is None:
        pytest.skip("rocke MfmaAtom has no f16_32x32x8 factory in this build")
    catalog = load_mma_traits()
    ours = c_warp_encoding(
        catalog.select(
            target="gfx90a", input_dtype="f16", output_dtype="f32", m=32, n=32, k=8
        )
    )
    oracle = rocke_atoms.make_c_warp_dstr_encoding(factory())
    _assert_field_equal(ours, oracle)
