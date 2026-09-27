# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The wide-GLOBAL-load fully-in-bounds gate (Principle 4). An unmasked wide global load is emitted
ONLY when the tile is provably in bounds on the stride-1 run axis; a ragged/odd tensor (or a runtime
clip that can't be proven aligned) falls to the scalar masked path -- never an out-of-bounds wide
read. LDS is exempt (its allocation is the full tile by construction). Offline, no GPU."""

from __future__ import annotations

from rocke.helpers.tiling.descriptors import make_tensor_desc, make_window
from rocke.helpers.tiling.emit import _contiguous_run
from rocke.helpers.tiling.memory import cooperative_load_desc


class _DT:
    def __init__(self, name: str) -> None:
        self.name = name


_F16 = _DT("f16")
# A (free, K) cooperative-load tile: free (axis 0) is stride-1, tile extent 128.
_COOP = cooperative_load_desc(128, 16, 4, vw=8)


def _vw(lengths, *, bounds=None, is_lds=False, origin=(0, 0)):
    tensor = make_tensor_desc(lengths, (1, lengths[0]), _F16)  # axis 0 stride-1
    return _contiguous_run(_COOP, make_window(tensor, origin, bounds), _F16, is_lds)


def test_aligned_tensor_gets_wide_load() -> None:
    # length 256 is a multiple of the tile extent 128 -> no tile overhangs -> wide (vw 8).
    assert _vw((256, 16)) == 8


def test_ragged_run_axis_scalarizes() -> None:
    # length 250 is NOT a multiple of 128: the far tile overhangs, so an unmasked wide read would be
    # OOB -> scalarize (the masked path handles the clip safely).
    assert _vw((250, 16)) == 1


def test_aligned_explicit_bounds_gets_wide_load() -> None:
    # The gate keys on effective clip vs alignment, NOT on bounds-presence: an aligned bound is wide.
    assert _vw((256, 16), bounds=(128, 16)) == 8


def test_ragged_explicit_bounds_scalarizes() -> None:
    assert _vw((256, 16), bounds=(200, 16)) == 1


def test_runtime_clip_cannot_be_proven_aligned_scalarizes() -> None:
    # A non-int (SSA-like) clip on the run axis can't be proven a tile multiple -> scalarize.
    assert _vw((256, 16), bounds=(object(), 16)) == 1


def test_lds_access_is_exempt_from_the_gate() -> None:
    # Even a "ragged" length is wide for LDS: the allocation is the full tile, so no overhang.
    assert _vw((250, 16), is_lds=True) == 8


# ---- #32: origin-alignment on the wide-load gate (compile-time mid-tile origin caught; SSA trusted) ----

def test_aligned_int_origin_stays_wide() -> None:
    assert _vw((256, 16), origin=(128, 0)) == 8            # origin 128 is on the tile grid (128)


def test_compile_time_mid_tile_origin_scalarizes() -> None:
    # origin 8 is NOT a multiple of the tile extent 128: the tile straddles the grid and can overhang
    # the far edge even though the extent (256) is aligned -> scalarize (masked-safe).
    assert _vw((256, 16), origin=(8, 0)) == 1


def test_ssa_origin_is_trusted_grid_aligned() -> None:
    # A runtime (SSA-like, non-int) origin is trusted grid-aligned -> stays wide. PARTIAL guard: an SSA
    # mid-tile origin (attention) is not caught here; that is deferred with attention.
    assert _vw((256, 16), origin=(object(), 0)) == 8


# ---- #30: the N-D operand gate fires only for a tensor that carries a CONTRACTION axis ----

def test_loads_as_operand_predicate() -> None:
    from rocke.helpers.tiling.emit import _loads_as_operand

    a_op = make_tensor_desc((16, 8), (8, 1), _F16, ("free", "contraction"))       # A (M,K)
    b_op = make_tensor_desc((8, 16), (1, 8), _F16, ("contraction", "free"))       # B (K,N)
    c_out = make_tensor_desc((16, 16), (16, 1), _F16, ("free", "free"))           # C output (M,N)
    plain = make_tensor_desc((16, 8), (8, 1), _F16)                               # no roles
    rank3 = make_tensor_desc((4, 16, 8), (128, 8, 1), _F16, ("batch", "free", "contraction"))

    assert _loads_as_operand(a_op) is True
    assert _loads_as_operand(b_op) is True
    assert _loads_as_operand(c_out) is False   # C reload: all free, not an operand -> gate skipped (was wrongly rejected)
    assert _loads_as_operand(plain) is False   # positional, no roles
    assert _loads_as_operand(rank3) is True    # has a contraction -> operand (assert_mma_operand then rejects rank!=2)


# ---- #31: masking LDS is structurally unrepresentable -- the LDS verbs carry NO bounds parameter ----

def test_lds_verbs_have_no_bounds_parameter() -> None:
    import inspect

    from rocke.helpers.tiling.memory import lds_read, lds_store

    assert "bounds" not in inspect.signature(lds_store).parameters, "lds_store must not accept bounds"
    assert "bounds" not in inspect.signature(lds_read).parameters, "lds_read must not accept bounds"
