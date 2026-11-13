# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sparse int8 uses the fp8 LDS pad / block rules; dense int8 does not.

Sparse int8 kernels read LDS through the same LDS-transpose and TDM path as
fp8, so their LdsPad and LdsBlockSizePerPad come from the fp8 solver. The
Solution code is shared with hipBLASLt, so the switch has to stay off for
dense int8, otherwise its kernels would change.
"""

import pytest

from tensilelite.Common.DataType import DataType
from tensilelite.SolutionStructs.Solution import _usesFp8LdsConfig

pytestmark = pytest.mark.unit


def _state(sparse):
    return {"ProblemType": {"Sparse": sparse}}


@pytest.mark.parametrize("sparse", [0, 1, 2])
@pytest.mark.parametrize("name", ["F8", "B8", "F8N", "B8N"])
def test_fp8_always_uses_fp8_config(name, sparse):
    assert _usesFp8LdsConfig(DataType(name), _state(sparse))


@pytest.mark.parametrize("sparse", [1, 2])
def test_sparse_int8_uses_fp8_config(sparse):
    assert _usesFp8LdsConfig(DataType("I8"), _state(sparse))


def test_dense_int8_keeps_its_own_path():
    assert not _usesFp8LdsConfig(DataType("I8"), _state(0))


@pytest.mark.parametrize("sparse", [0, 1, 2])
@pytest.mark.parametrize("name", ["H", "B", "S", "I"])
def test_other_types_are_unaffected(name, sparse):
    assert not _usesFp8LdsConfig(DataType(name), _state(sparse))
