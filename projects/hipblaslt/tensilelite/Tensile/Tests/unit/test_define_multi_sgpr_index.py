################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Unit tests for ``KernelWriterAssembly.defineMultiSgprIndex``.

``defineMultiSgprIndex`` maps epilogue kernel-arg names onto a single
``checkOutMulti`` block. Only the *start* of that block is aligned, so the
name -> index mapping has to be one-to-one: a repeated name silently rebinds
``self.sgprs[name]`` to the later index, which is aligned only by luck. A
64-bit arg (e.g. ``AddressGate``) that lands on an odd SGPR then fails
assembly with "invalid register alignment", far from the duplicate registration
that caused it.
"""

import pytest

from rocisa.enum import RegisterType
from rocisa.register import RegisterPool

from Tensile.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit


class _Writer:
    """Minimal stand-in exposing only what defineMultiSgprIndex touches."""

    def __init__(self, size=64):
        self.sgprPool = RegisterPool(8, RegisterType.Sgpr, False)
        self.sgprPool.add(0, size, "init")
        self.sgprs = {}


def _define(writer, names, sizes, align=4):
    return KernelWriterAssembly.defineMultiSgprIndex(writer, names, sizes, align=align)


def test_maps_each_name_to_its_own_index():
    """Names are packed back-to-back; only the block start honors align."""
    writer = _Writer()
    names = ["AddressGate", "GateType", "GateStride"]

    idxVec = _define(writer, names, [2, 1, 1])

    assert idxVec[0] % 4 == 0, "block start must honor the requested alignment"
    # Packed sequentially, so interior entries are not individually aligned --
    # this is exactly why a duplicate name is dangerous.
    assert idxVec == [idxVec[0], idxVec[0] + 2, idxVec[0] + 3]
    assert {n: writer.sgprs[n] for n in names} == dict(zip(names, idxVec))


def test_rejects_duplicate_names():
    """Regression: the gate-residual args were once registered twice.

    The second registration rebound AddressGate to an odd SGPR and broke
    assembly of hipSPARSELt gate-residual kernels. Duplicates must be rejected
    up front, naming the offender.
    """
    writer = _Writer()
    names = ["AddressGate", "GateType", "GateStride",
             "AddressGate", "GateType", "GateStride"]

    with pytest.raises(AssertionError, match="AddressGate"):
        _define(writer, names, [2, 1, 1, 2, 1, 1])

    assert writer.sgprs == {}, "nothing should be bound when the input is rejected"


def test_rejects_mismatched_name_and_size_counts():
    writer = _Writer()
    with pytest.raises(AssertionError):
        _define(writer, ["AddressGate", "GateType"], [2])
