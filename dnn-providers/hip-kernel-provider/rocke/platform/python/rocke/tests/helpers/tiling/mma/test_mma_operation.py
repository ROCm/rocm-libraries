# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the public Mma object (target-aware resolution, IR-free)."""

from __future__ import annotations

import pytest

from rocke.helpers.tiling.mma import TileMma, Tiling


def test_mma_resolves_gfx90a_16x16x16_f16() -> None:
    mma_op = TileMma((16, 16, 16), a="f16", b="f16", c="f32", target="gfx90a")
    assert mma_op.op_id == "mfma_f32_16x16x16f16"
    assert mma_op.wave_size == 64
    # Layouts are the calculator's encodings (IR-free).
    assert mma_op.c_layout.hierarchical_lengths == ((1, 4, 4), (16,))
    assert mma_op.a_layout.hierarchical_lengths == ((16,), (1, 4, 4))


def test_mma_is_target_agnostic_authorship_swap_target() -> None:
    # Same authored intent, different bound target -> distinct valid resolution.
    on_gfx90a = TileMma((16, 16, 16), a="f16", b="f16", c="f32", target="gfx90a")
    on_gfx942 = TileMma((16, 16, 16), a="f16", b="f16", c="f32", target="gfx942")
    assert on_gfx90a.op_id == on_gfx942.op_id == "mfma_f32_16x16x16f16"
    # (both CDNA MFMA here; the point is authorship carries no gfx branch)
    assert on_gfx90a.wave_size == 64


def test_default_atom_picks_native_32x32_single_atom() -> None:
    # With no atom knob, a 32x32x8 wave tile resolves the NATIVE 32x32 atom -> single atom,
    # NOT a grid of 16x16 atoms. This is the behaviour we override below.
    mma_op = TileMma((32, 32, 8), a="f16", b="f16", c="f32", target="gfx90a")
    assert mma_op.op_id == "mfma_f32_32x32x8f16"
    assert mma_op.atom_shape == (32, 32, 8)
    assert mma_op.subtiles == (1, 1, 1)


def test_atom_override_derives_shape_and_dtypes() -> None:
    # Name the exact intrinsic; TileMma derives M/N/K + dtypes from its traits (no shape/dtypes given).
    mma_op = TileMma(target="gfx90a", atom_override="mfma_f32_16x16x16f16")
    assert mma_op.op_id == "mfma_f32_16x16x16f16"
    assert mma_op.shape == (16, 16, 16)
    assert mma_op.atom_shape == (16, 16, 16)
    assert mma_op.subtiles == (1, 1, 1)
    assert mma_op.wave_size == 64


def test_atom_override_with_wave_shape_is_a_grid() -> None:
    # atom_override still derives dtypes; an explicit shape is the wave tile (a multiple of the atom).
    mma_op = TileMma((32, 32, 16), target="gfx90a", atom_override="mfma_f32_16x16x16f16")
    assert mma_op.atom_shape == (16, 16, 16)
    assert mma_op.subtiles == (2, 2, 1)


def test_atom_override_conflicts_with_tiling_atom_shape() -> None:
    with pytest.raises(ValueError, match="atom_override OR tiling.atom_shape"):
        TileMma(
            target="gfx90a", atom_override="mfma_f32_16x16x16f16",
            tiling=Tiling(atom_shape="mfma_f32_16x16x16f16"),
        )


def test_forcing_16x16_atom_on_32x32_tile() -> None:
    # Force the small atom via the tiling knob: a 32x32x16 wave tile over a 16x16x16 atom is
    # a 2x2x1 subtile grid, NOT the native 32x32 atom.
    mma_op = TileMma(
        (32, 32, 16), a="f16", b="f16", c="f32", target="gfx90a",
        tiling=Tiling(atom_shape=(16, 16, 16)),
    )
    assert mma_op.op_id == "mfma_f32_16x16x16f16"
    assert mma_op.shape == (32, 32, 16)
    assert mma_op.atom_shape == (16, 16, 16)
    assert mma_op.subtiles == (2, 2, 1)
    # The wave layouts carry the subtile counts: M gets an outer m_iter=2 level, N an
    # n_iter=2 level (K stays single: k_iter=1).
    assert mma_op.a_layout.hierarchical_lengths == ((2, 16), (1, 4, 4))
    assert mma_op.c_layout.hierarchical_lengths == ((2, 1, 4, 4), (2, 16))


def test_atom_selected_by_explicit_intrinsic_name() -> None:
    # Escape hatch: name the exact backend intrinsic; it resolves to the same atom as the
    # shape tuple, and the wave tile subtiles over it.
    mma_op = TileMma(
        (32, 32, 16), a="f16", b="f16", c="f32", target="gfx90a",
        tiling=Tiling(atom_shape="mfma_f32_16x16x16f16"),
    )
    assert mma_op.op_id == "mfma_f32_16x16x16f16"
    assert mma_op.atom_shape == (16, 16, 16)
    assert mma_op.subtiles == (2, 2, 1)


def test_atom_name_unknown_fails_fast() -> None:
    with pytest.raises(ValueError) as excinfo:
        TileMma(
            (16, 16, 16), a="f16", b="f16", c="f32", target="gfx90a",
            tiling=Tiling(atom_shape="mfma_f32_16x16x16_does_not_exist"),
        )
    assert "mfma_f32_16x16x16_does_not_exist" in str(excinfo.value)


def test_atom_name_wrong_target_fails_fast() -> None:
    # A real gfx950-only wide atom named while bound to gfx90a must fail fast.
    with pytest.raises(ValueError) as excinfo:
        TileMma(
            (16, 16, 32), a="f16", b="f16", c="f32", target="gfx90a",
            tiling=Tiling(atom_shape="mfma_f32_16x16x32_f16"),
        )
    assert "mfma_f32_16x16x32_f16" in str(excinfo.value)


def test_atom_name_dtype_mismatch_fails_fast() -> None:
    with pytest.raises(ValueError) as excinfo:
        TileMma(
            (16, 16, 16), a="f16", b="f16", c="f32", target="gfx90a",
            tiling=Tiling(atom_shape="mfma_i32_16x16x16i8"),
        )
    msg = str(excinfo.value)
    assert "mfma_i32_16x16x16i8" in msg
    assert "dtype mismatch" in msg or "not available" in msg


def test_mma_exposes_per_operand_tiledescs() -> None:
    # The MMA object hands back ready-to-use TileDescs (wave shape + resolved layout).
    mma_op = TileMma(
        (32, 32, 16), a="f16", b="f16", c="f32", target="gfx90a",
        tiling=Tiling(atom_shape=(16, 16, 16)),
    )
    assert mma_op.a_desc().shape == (32, 16)          # (wave M, wave K)
    assert mma_op.b_desc().shape == (32, 16)          # (wave N, wave K)
    assert mma_op.c_desc.shape == (32, 32)            # (wave M, wave N)
    assert mma_op.a_desc().layout == mma_op.a_layout  # desc bundles the resolved layout
    assert mma_op.c_desc.layout == mma_op.c_layout


def test_mma_mismatched_ab_dtype_fails_fast() -> None:
    with pytest.raises(ValueError) as excinfo:
        TileMma((16, 16, 16), a="f16", b="bf16", c="f32", target="gfx90a")
    assert "matching A/B dtypes" in str(excinfo.value)


def test_mma_unsupported_shape_fails_fast_with_candidates() -> None:
    with pytest.raises(ValueError) as excinfo:
        TileMma((13, 16, 16), a="f16", b="f16", c="f32", target="gfx90a")
    assert "no MMA intrinsic" in str(excinfo.value)
    assert "available on 'gfx90a'" in str(excinfo.value)
