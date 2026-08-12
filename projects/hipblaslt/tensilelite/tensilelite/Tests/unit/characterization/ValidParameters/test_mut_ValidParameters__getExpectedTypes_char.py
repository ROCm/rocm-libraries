# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mutation-targeted characterization tests for
``Tensile.Common.ValidParameters._getExpectedTypes``.

These pin the helper's current behavior: it skips the ``-1`` sentinel, rejects
empty lists and invalid scalar values (raising ``ValueError``), and builds a
dictionary whose values are exact concrete type sets. The ``bool`` case is
deliberate because the
implementation uses ``type()``, not ``isinstance()``, so ``bool`` stays distinct
from ``int``.
"""

import pytest

from Tensile.Common.ValidParameters import _getExpectedTypes

pytestmark = pytest.mark.unit


def test_get_expected_types_builds_type_map_for_non_sentinel_lists():
    """Non-empty allowed-values lists produce entries in the returned type map."""
    valid_params = {
        "SkipMe": -1,
        "Ints": [1, 2],
        "Bools": [False, True],
        "Mixed": [1, "x", 2.0],
    }

    result = _getExpectedTypes(valid_params)

    assert result == {
        "Ints": {int},
        "Bools": {bool},
        "Mixed": {int, str, float},
    }


def test_get_expected_types_skips_minus_one_sentinel():
    """The ``-1`` sentinel is omitted from the returned type map."""
    result = _getExpectedTypes({"AnyValue": -1})

    assert result == {}


def test_get_expected_types_includes_singleton_allowed_values():
    """A one-element allowed-values list still contributes its concrete type."""
    result = _getExpectedTypes({"Singleton": [7]})

    assert result == {"Singleton": {int}}


def test_get_expected_types_empty_list_raises():
    """An empty allowed-values list raises ``ValueError`` with the current message."""
    with pytest.raises(ValueError) as excinfo:
        _getExpectedTypes({"Empty": []})

    assert str(excinfo.value) == "Invalid parameter value: Empty = []"


def test_get_expected_types_invalid_scalar_raises():
    """A scalar value other than the ``-1`` sentinel raises ``ValueError``."""
    with pytest.raises(ValueError) as excinfo:
        _getExpectedTypes({"NotSentinel": 1})

    assert str(excinfo.value) == "Invalid parameter value: NotSentinel = 1"
