################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Fixtures for the GlobalParameters suite: the cap map (isaInfoMap) + an
isolation fixture that snapshots and restores the process-global
``globalParameters`` (and ``validParameters["ISA"]``) around each test."""

import copy

import pytest

from tensilelite.Common.Architectures import SUPPORTED_ISA
from tensilelite.Common.Capabilities import makeIsaInfoMap
from tensilelite.Toolchain.Validators import validateToolchain


@pytest.fixture(scope="session")
def isa_info_map():
    return makeIsaInfoMap(SUPPORTED_ISA, validateToolchain("amdclang++"))


@pytest.fixture
def isolate_globals():
    from tensilelite.Common.GlobalParameters import globalParameters
    from tensilelite.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_isa = validParameters.get("ISA")
    try:
        yield globalParameters
    finally:
        globalParameters.clear()
        globalParameters.update(saved_gp)
        if saved_isa is None:
            validParameters.pop("ISA", None)
        else:
            validParameters["ISA"] = saved_isa
