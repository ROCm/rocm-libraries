# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mutation-targeted characterization tests for ``DataType._populateLookupTable``."""

import pytest
from rocisa.enum import DataTypeEnum

from Tensile.Common.DataType import _populateLookupTable

pytestmark = pytest.mark.unit


def test_populate_lookup_table_index_mismatch_prints_exact_diagnostic(capsys):
    """Index-mismatch rejection prints the current diagnostic before raising."""
    bad = [{"enum": DataTypeEnum.Double, "char": "D"}]

    with pytest.raises(RuntimeError) as excinfo:
        _populateLookupTable(bad, {})

    assert str(excinfo.value) == "Enum value does not match index in properties list"
    assert capsys.readouterr().out == "Double : 1 does not match index 0 in properties list\n"
