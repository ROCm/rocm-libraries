# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Self-test of the translator, run by hand (no CMake or CI entry):

    python -m pytest tools/translator/tests      (from platform/)

The translator runs in subprocesses against the fixture root in fixtures/:
pinned Python modules (TR_LIBRARY), their facts, names and code maps
(TR_DATA) and no hand fills (TR_HOLES). The C headers come from the rocKE
tree this directory sits in. Goldens change only with --update-goldens.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trtest import CLANG_FORMAT_VERSION, FIXTURES, Tr  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens",
        action="store_true",
        help="rewrite fixtures/golden/ from the current translator output",
    )


def _clang_format() -> str | None:
    """The clang-format the translator would pick (tr.translate.find_clang_format)."""
    exe = os.environ.get("CLANG_FORMAT")
    if not exe:
        beside = os.path.join(os.path.dirname(sys.executable), "clang-format")
        exe = beside if os.access(beside, os.X_OK) else shutil.which("clang-format")
    return exe


@pytest.fixture(scope="session")
def clang_format() -> str:
    exe = _clang_format()
    if not exe:
        pytest.skip("no clang-format (install the clang-format==18.1.4 wheel)")
    out = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    if CLANG_FORMAT_VERSION not in out:
        pytest.skip(f"goldens need clang-format {CLANG_FORMAT_VERSION}, found: {out}")
    return exe


@pytest.fixture(scope="session")
def fixture_tr() -> Tr:
    return Tr(FIXTURES / "library")


@pytest.fixture(scope="session")
def generated(tmp_path_factory, fixture_tr, clang_format):
    """One translation of the fixture modules: (output dir, translate runs)."""
    out = tmp_path_factory.mktemp("gen")
    runs = fixture_tr.generate(out)
    return out, runs


@pytest.fixture
def copy_library(tmp_path) -> Path:
    """A writable copy of the fixture modules."""
    lib = tmp_path / "library"
    shutil.copytree(FIXTURES / "library", lib)
    return lib
