# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The LayoutStyle LDS-bridge extension point (follow-up #28): the memory-bridge descriptor pair a
LDS-staging style produces is on the base protocol (``lds_bridge``), not exposed ad-hoc off a concrete
subclass. A non-staging style returns None; the staging style returns (lds_read_landing, mma_ready).
Offline, no GPU."""

from __future__ import annotations

import pytest

from rocke.helpers.tiling.mma.styles import CanonicalStyle, InterleavedStyle
from rocke.helpers.tiling.traits import load_mma_traits

_T = load_mma_traits().get("mfma_f32_16x16x16f16")


def test_canonical_style_has_no_lds_bridge() -> None:
    # Canonical loads MMA-ready directly (no LDS landing), so the optional bridge is None.
    assert CanonicalStyle().lds_bridge(_T, role="A", free_sub=1, k_sub=1) is None


def test_interleaved_lds_bridge_is_on_the_protocol() -> None:
    style = InterleavedStyle()
    bridge = style.lds_bridge(_T, role="A", free_sub=2, k_sub=1)
    assert bridge is not None
    read_landing, mma_ready = bridge
    # the second half of the bridge IS the mma-ready operand_desc -- no divergence between them
    assert mma_ready == style.operand_desc(_T, role="A", free_sub=2, k_sub=1)
    # the read landing differs from mma-ready -- the in-register reorder is the price of the wide LDS read
    assert read_landing != mma_ready


def test_lds_bridge_role_selects_operand_free_axis() -> None:
    style = InterleavedStyle()
    for role in ("A", "B"):
        bridge = style.lds_bridge(_T, role=role, free_sub=1, k_sub=1)
        assert bridge[1] == style.operand_desc(_T, role=role, free_sub=1, k_sub=1)


def test_lds_bridge_rejects_bad_role() -> None:
    with pytest.raises(ValueError, match="role must be 'A' or 'B'"):
        InterleavedStyle().lds_bridge(_T, role="C", free_sub=1, k_sub=1)


# ---- #33: per-operand soundness runs at TileMmaPlan CONSTRUCTION (same timing as the C-oracle) ----

def test_shipped_styles_build_a_plan_without_raising() -> None:
    # Both profiles pass the construction-time operand-soundness check by construction -- for a
    # single-atom wave AND a subtiled one. The (32,32,32) wave over a (16,16,16) atom makes
    # m_sub/n_sub/k_sub == 2, so the check's subtiling path is covered by THIS test, not only
    # transitively via the driver-path tests.
    from rocke.helpers.tiling.mma.plan import TileMmaPlan, Tiling

    configs = (
        ((16, 16, 16), None),                              # single atom (m_sub=n_sub=k_sub=1)
        ((32, 32, 32), Tiling(atom_shape=(16, 16, 16))),   # 2x2x2 subtiled wave tile
    )
    for style in (CanonicalStyle(), InterleavedStyle()):
        for shape, tiling in configs:
            TileMmaPlan(
                shape, a="f16", b="f16", c="f32", target="gfx90a", style=style, tiling=tiling
            )


def test_unsound_operand_rejected_at_plan_construction(monkeypatch) -> None:
    # Force operand_soundness to report an error and confirm the PLAN rejects it at build -- proving the
    # gate fires at construction (matching the C-oracle), not only later at TileMmaDriver.__call__.
    import rocke.helpers.tiling.transforms as transforms
    from rocke.helpers.tiling.mma.plan import TileMmaPlan
    from rocke.helpers.tiling.transforms._core import Diagnostic

    monkeypatch.setattr(
        transforms, "operand_soundness", lambda *a, **k: Diagnostic("error", "forced-unsound (test)")
    )
    with pytest.raises(ValueError, match="not sound"):
        TileMmaPlan((16, 16, 16), a="f16", b="f16", c="f32", target="gfx90a")
