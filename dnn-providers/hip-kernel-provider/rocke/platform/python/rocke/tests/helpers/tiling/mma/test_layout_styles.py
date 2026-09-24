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
