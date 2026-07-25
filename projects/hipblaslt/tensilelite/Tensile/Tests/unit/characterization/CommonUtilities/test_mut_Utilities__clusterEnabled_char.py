################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.clusterEnabled``.

``clusterEnabled(clusterDim)`` returns ``(clusterDim[0] * clusterDim[1]) != 1``:
a workgroup cluster is considered enabled when the product of the two cluster
dimensions is not exactly 1 (i.e. ClusterDim is not [1, 1]).

These tests pin the ACTUAL current behavior (boolean return value, and the fact
that only indices 0 and 1 are read) so they pass on clean source and fail under
the surviving mutants:
  - mutmut_1: ``*`` -> ``/`` (product becomes float division)
  - mutmut_2: ``clusterDim[0]`` -> ``clusterDim[1]`` (index-0 dropped)
  - mutmut_3: ``clusterDim[1]`` -> ``clusterDim[2]`` (reads out-of-range index)
  - mutmut_4: ``!=`` -> ``==`` (comparison inverted)
  - mutmut_5: literal ``1`` -> ``2`` (compares product against 2)
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")
clusterEnabled = U.clusterEnabled

pytestmark = pytest.mark.unit


def test_unit_cluster_dim_is_disabled():
    # Kills mutmut_4 (`!=` -> `==`): 1 == 1 would return True.
    # Kills mutmut_5 (`1` -> `2`): 1 != 2 would return True.
    # Original: product 1*1 == 1, so `!= 1` is False.
    assert clusterEnabled([1, 1]) is False


def test_equal_nonunit_dims_enable_cluster_via_integer_product():
    # Kills mutmut_1 (`*` -> `/`): 2/2 == 1.0, and 1.0 != 1 is False.
    # Original: 2*2 == 4, and 4 != 1 is True.
    assert clusterEnabled([2, 2]) is True


def test_only_index0_and_index1_are_read_and_index0_matters():
    # Kills mutmut_2 (`clusterDim[0]` -> `clusterDim[1]`): 1*1 == 1 -> False.
    # Kills mutmut_3 (`clusterDim[1]` -> `clusterDim[2]`): reading index 2 of a
    #   two-element list raises IndexError under the mutant, so a plain return
    #   here distinguishes it.
    # Original: 2*1 == 2, and 2 != 1 is True.
    assert clusterEnabled([2, 1]) is True


def test_two_element_input_does_not_read_a_third_element():
    # Reinforces the kill of mutmut_3: the original never touches index 2, so a
    # strictly two-element sequence is accepted and returns a bool. The mutant
    # (clusterDim[2]) raises IndexError on this same input.
    result = clusterEnabled([3, 4])
    assert result is True


def test_index1_unit_with_nonunit_index0_stays_enabled():
    # Additional discriminator for mutmut_2 (index-0 dropped): with clusterDim[1]
    # == 1, mutmut_2 evaluates 1*1 == 1 -> False, while the original uses
    # clusterDim[0]: 5*1 == 5, and 5 != 1 is True.
    assert clusterEnabled([5, 1]) is True
