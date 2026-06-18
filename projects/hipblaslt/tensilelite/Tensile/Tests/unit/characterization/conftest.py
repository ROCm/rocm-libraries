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


def _foreign_path_instantiable() -> bool:
    """Return True if a Windows-flavored ``pathlib.Path`` can be built here.

    On POSIX, CPython refuses to instantiate ``WindowsPath`` (``NotImplementedError``
    through 3.12, the ``NotImplementedError`` subclass ``UnsupportedOperation`` in
    3.13+).  Characterization tests that flip ``os.name`` to ``"nt"`` and let
    production code construct a concrete ``pathlib.Path`` therefore raise inside the
    function under test *and* inside pytest's own failure reporting, which builds
    ``Path(os.getcwd())`` while ``os.name`` is still ``"nt"`` -- taking down the
    whole xdist worker with an INTERNALERROR.  Detect the capability directly
    rather than sniffing the interpreter version.
    """
    from pathlib import Path

    saved = os.name
    try:
        os.name = "nt"
        Path("probe")
        return True
    except NotImplementedError:
        return False
    finally:
        os.name = saved


def pytest_collection_modifyitems(config, items):
    """Skip ``nt_path_simulation`` tests where foreign Paths can't be instantiated.

    Tests marked ``nt_path_simulation`` flip ``os.name`` to ``"nt"`` and exercise
    production code that constructs a ``pathlib.Path``.  On any POSIX interpreter
    this cannot succeed, so they are skipped there and run only where a
    Windows-flavored ``Path`` is instantiable.
    """
    if _foreign_path_instantiable():
        return
    skip_nt = pytest.mark.skip(
        reason="WindowsPath not instantiable on this interpreter; nt-path simulation skipped"
    )
    for item in items:
        if "nt_path_simulation" in item.keywords:
            item.add_marker(skip_nt)


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
