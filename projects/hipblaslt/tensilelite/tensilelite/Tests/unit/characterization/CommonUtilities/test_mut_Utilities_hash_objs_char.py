################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.hash_objs``.

These tests pin the current behavior that the returned hash is derived from the
tuple of positional arguments (``hash(tuple(objs))``), distinguishing the
original implementation from the surviving mutant that returns the constant
``hash(None)`` regardless of input.
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit


def test_hash_objs_equals_hash_of_tuple_of_args():
    """Current behavior: result equals ``hash(tuple(objs))``.

    The mutant returns ``hash(None)``; for these args ``hash((1, 2, 3))``
    differs from ``hash(None)``, so this distinguishes original from mutant.
    """
    assert U.hash_objs(1, 2, 3) == hash((1, 2, 3))


def test_hash_objs_varies_with_input():
    """Distinct argument tuples produce distinct hashes.

    The mutant ignores the arguments entirely (always ``hash(None)``), so it
    would make these two calls equal. The original distinguishes them.
    """
    assert U.hash_objs("alpha") != U.hash_objs("beta")
