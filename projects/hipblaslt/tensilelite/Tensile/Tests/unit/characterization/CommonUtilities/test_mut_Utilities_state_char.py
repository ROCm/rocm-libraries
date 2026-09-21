################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.state``.

Pins the tuple-unpacking branch of ``state``: when an entry in a class's
``StateKeys`` is a 2-tuple ``(key, attr)``, the original code unpacks it so the
result dict is keyed by ``key`` but reads the value from attribute ``attr``.
The targeted mutant replaces ``(key, attr) = key`` with ``(key, attr) = None``,
which would raise ``TypeError`` (None is not iterable) instead of unpacking.
"""

import pytest

from Tensile.Common.Utilities import state

pytestmark = pytest.mark.unit


class _ObjWithTupleStateKeys:
    """A class whose StateKeys mixes a plain key and a (key, attr) tuple.

    For the tuple entry the output dict key ("renamed") differs from the
    source attribute ("source_attr"), so unpacking is observable.
    """

    StateKeys = ["plain", ("renamed", "source_attr")]

    def __init__(self):
        self.plain = 1
        self.source_attr = 42


def test_state_tuple_statekey_unpacks_key_and_attr():
    """Kills x_state__mutmut_16 ((key, attr) = None).

    On clean source the tuple StateKey ("renamed", "source_attr") is unpacked:
    the result is keyed by "renamed" and its value is read from the
    "source_attr" attribute. The mutant assigns None and raises TypeError.
    """
    result = state(_ObjWithTupleStateKeys())
    assert result == {"plain": 1, "renamed": 42}
    # Distinguishing detail: output key is the tuple's first element, not the
    # attribute name; the value comes from the second element's attribute.
    assert "renamed" in result
    assert "source_attr" not in result
    assert result["renamed"] == 42
