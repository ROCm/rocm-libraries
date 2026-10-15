################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.ProgressBar.increment``.

Pins the default step size of ``increment`` (value=1): calling it with no
argument advances ``priorValue`` by exactly one. Kills the mutant that flips
the default argument from 1 to 2.
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit


def test_increment_default_step_is_one():
    # maxValue large enough that update() never short-circuits on tick math;
    # update() always sets priorValue = value regardless of tick changes.
    pb = U.ProgressBar(maxValue=1000)
    assert pb.priorValue == 0

    pb.increment()  # default value=1
    assert pb.priorValue == 1

    pb.increment()  # default value=1 again -> 2
    assert pb.priorValue == 2
