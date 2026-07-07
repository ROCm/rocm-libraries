# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mutation-targeted characterization tests for
``Tensile.Common.ValidParameters._getExpectedTypes``.

These pin the helper's current behavior: it skips sentinel/empty entries and
builds a dictionary whose values are exact concrete type sets. The ``bool`` case
is deliberate because the implementation uses ``type()``, not ``isinstance()``,
so ``bool`` stays distinct from ``int``.
"""

import pytest

from Tensile.Common.ValidParameters import _getExpectedTypes

pytestmark = pytest.mark.unit


def test_get_expected_types_builds_type_map_for_non_sentinel_lists():
    """Non-empty allowed-values lists produce entries in the returned type map."""
    valid_params = {
        "SkipMe": -1,
        "Empty": [],
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
