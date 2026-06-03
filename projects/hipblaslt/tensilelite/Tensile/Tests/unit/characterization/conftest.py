################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Characterization-suite shared configuration.

Adds the ``_codegen`` helper directory to ``sys.path`` so any characterization
suite can ``from codegen_harness import ...`` (the CPU-only assembly-emit
harness used by the codegen coverage suites), and exposes the cached
toolchain / cap-map / data-dir as session fixtures.

This file only *adds* an import path and read-only fixtures; it changes no
existing behavior.
"""

import os
import sys

import pytest

_CODEGEN_DIR = os.path.join(os.path.dirname(__file__), "_codegen")
if _CODEGEN_DIR not in sys.path:
    sys.path.insert(0, _CODEGEN_DIR)


@pytest.fixture(scope="session")
def cg_assembler():
    """The CPU-only assembler (amdclang++); shared across codegen suites."""
    from codegen_harness import get_assembler

    return get_assembler()


@pytest.fixture(scope="session")
def cg_isa_info_map():
    """The ISA capability map; shared across codegen suites."""
    from codegen_harness import get_isa_info_map

    return get_isa_info_map()
