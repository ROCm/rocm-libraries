# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for ``Tensile.SolutionStructs.Utilities.isSubtileIterateMode``.

No GPU required -- exercises pure Python logic.
"""

import pytest
from unittest.mock import MagicMock

from Tensile.SolutionStructs.Utilities import isSubtileIterateMode

pytestmark = pytest.mark.unit


def _mock_dtype(num_bytes):
    dt = MagicMock()
    dt.numBytes.return_value = num_bytes
    return dt


def _make_state(use_subtile, enable_tdm_a, enable_tdm_b, depth_u, bpe_a=2, bpe_b=2):
    """Build a minimal kernel-like dict for isSubtileIterateMode."""
    return {
        "UseSubtileImpl": use_subtile,
        "enableTDMA": enable_tdm_a,
        "enableTDMB": enable_tdm_b,
        "DepthU": depth_u,
        "ProblemType": {
            "DataTypeA": _mock_dtype(bpe_a),
            "DataTypeB": _mock_dtype(bpe_b),
        },
    }


# -- basic boundary tests around 1024B limit --

class TestBoundary:
    """isSubtileIterateMode triggers at DepthU * bpeGR > 1024."""

    def test_exactly_at_limit_is_false(self):
        # 512 * 2 = 1024 == limit -> not exceeded
        state = _make_state(True, True, True, 512, bpe_a=2, bpe_b=2)
        assert isSubtileIterateMode(state, "A") is False
        assert isSubtileIterateMode(state, "B") is False

    def test_one_byte_over_limit_is_true(self):
        # 513 * 2 = 1026 > 1024
        state = _make_state(True, True, True, 513, bpe_a=2, bpe_b=2)
        assert isSubtileIterateMode(state, "A") is True
        assert isSubtileIterateMode(state, "B") is True

    def test_large_depth_u_is_true(self):
        state = _make_state(True, True, True, 1024, bpe_a=2, bpe_b=2)
        assert isSubtileIterateMode(state, "A") is True
        assert isSubtileIterateMode(state, "B") is True

    def test_small_depth_u_is_false(self):
        state = _make_state(True, True, True, 64, bpe_a=2, bpe_b=2)
        assert isSubtileIterateMode(state, "A") is False
        assert isSubtileIterateMode(state, "B") is False


# -- feature-gate tests --

class TestFeatureGates:
    """Each of the three boolean conditions must hold."""

    def test_false_when_subtile_disabled(self):
        state = _make_state(False, True, True, 1024)
        assert isSubtileIterateMode(state, "A") is False

    def test_false_when_tdm_disabled_for_tensor(self):
        state = _make_state(True, False, True, 1024)
        assert isSubtileIterateMode(state, "A") is False
        # B is still enabled
        assert isSubtileIterateMode(state, "B") is True

    def test_false_when_subtile_key_missing(self):
        state = {
            "enableTDMA": True,
            "enableTDMB": True,
            "DepthU": 1024,
            "ProblemType": {
                "DataTypeA": _mock_dtype(2),
                "DataTypeB": _mock_dtype(2),
            },
        }
        # UseSubtileImpl missing -> .get returns False
        assert isSubtileIterateMode(state, "A") is False


# -- asymmetric bpe --

class TestAsymmetricBpe:
    """A and B can have different bytes-per-element."""

    def test_a_exceeds_but_b_does_not(self):
        # A: 256 * 8 = 2048 > 1024, B: 256 * 2 = 512 <= 1024
        state = _make_state(True, True, True, 256, bpe_a=8, bpe_b=2)
        assert isSubtileIterateMode(state, "A") is True
        assert isSubtileIterateMode(state, "B") is False

    def test_b_exceeds_but_a_does_not(self):
        # A: 256 * 2 = 512 <= 1024, B: 256 * 8 = 2048 > 1024
        state = _make_state(True, True, True, 256, bpe_a=2, bpe_b=8)
        assert isSubtileIterateMode(state, "A") is False
        assert isSubtileIterateMode(state, "B") is True
