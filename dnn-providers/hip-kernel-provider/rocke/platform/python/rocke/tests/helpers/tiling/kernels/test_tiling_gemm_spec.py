# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline tests for TilingGemmSpec validation -- the spec's fail-fast raises.

These exercise the spec/knobs layer WITHOUT a GPU: __post_init__ dimension checks, atom
resolution, and the wave/atom multiple + M/N-scope guards are all pure ``rocke.helpers.tiling``
(no torch, no compile).
"""

from __future__ import annotations

import pytest

from rocke.helpers.tiling.kernels.tiling_gemm_demo import TilingGemmSpec, is_valid_spec


def test_spec_rejects_nonpositive_tile() -> None:
    with pytest.raises(ValueError) as excinfo:
        TilingGemmSpec(tile=(16, 0, 16))
    assert "tile must be 3 positive ints" in str(excinfo.value)


def test_spec_rejects_bad_atom_rank() -> None:
    with pytest.raises(ValueError) as excinfo:
        TilingGemmSpec(tile=(16, 16, 16), atom=(16, 16))
    assert "atom must be 3 positive ints" in str(excinfo.value)


def test_is_valid_spec_accepts_16x16x16() -> None:
    ok, why = is_valid_spec(TilingGemmSpec(tile=(16, 16, 16)), arch="gfx90a")
    assert ok, why


def test_is_valid_spec_rejects_unresolvable_atom() -> None:
    # 13 is not an MFMA atom dimension -> no intrinsic resolves on gfx90a.
    ok, why = is_valid_spec(TilingGemmSpec(tile=(13, 16, 16)), arch="gfx90a")
    assert not ok
    assert "no MMA intrinsic" in why


def test_is_valid_spec_accepts_cooperative_mn_tile() -> None:
    # M/N spanning several atoms (cooperative M/N grid) is now supported.
    ok, why = is_valid_spec(
        TilingGemmSpec(tile=(32, 32, 16), atom=(16, 16, 16)), arch="gfx90a"
    )
    assert ok, why


def test_is_valid_spec_rejects_wave_k_not_multiple_of_atom_k() -> None:
    ok, why = is_valid_spec(
        TilingGemmSpec(tile=(16, 16, 24), atom=(16, 16, 16)), arch="gfx90a"
    )
    assert not ok
    assert "not an integer multiple of atom K" in why


def test_spec_rejects_bad_order() -> None:
    from rocke.helpers.tiling.mma import Tiling

    with pytest.raises(ValueError) as excinfo:
        Tiling(atom_shape=(16, 16, 16), order="diagonal")
    assert "unknown subtile order" in str(excinfo.value)
