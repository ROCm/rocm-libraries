# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The support headers translated sources include from their own directory."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import pytest
from trtest import GOLDEN, ROCKE, TR

sys.path.insert(0, str(TR))
from tr.package import PKG, support_file  # noqa: E402

PRELUDE = TR / "tr" / "prelude" / "rocke_py.hpp"
INCLUDE = ROCKE / "platform" / "cpp" / "include"


def _compiler(name: str) -> str | None:
    exe = shutil.which(name)
    if exe is None and name == "clang++":
        rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
        beside = os.path.join(rocm, "llvm", "bin", "clang++")
        exe = beside if os.access(beside, os.X_OK) else None
    return exe


@pytest.mark.parametrize("name", ["g++", "clang++"])
def test_prelude_compiles_alone(name):
    """rocke_py.hpp needs only the installed rocKE headers. (The attention
    backward adapters header needs that package's module headers, so it is
    not compiled alone.)"""
    exe = _compiler(name)
    if exe is None:
        pytest.skip(f"no {name}")
    cmd = [exe, "-std=c++17", "-fsyntax-only", "-Wall", "-Wextra", "-Werror"]
    cmd += ["-I", str(INCLUDE), "-x", "c++", str(PRELUDE)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_prelude_not_in_installed_headers():
    """The prelude ships with the translator, not in cpp/include."""
    assert not list((INCLUDE / "rocke").glob("rocke_py*.hpp"))


def test_support_files_resolve():
    assert support_file("rocke_py.hpp") == str(PRELUDE)
    adapters = support_file("rocke_py_adapters.hpp")
    assert adapters == os.path.join(PKG.data_dir, "rocke_py_adapters.hpp")
    assert support_file("no_such_header.hpp") is None
    # the adapters include the prelude from the same directory
    text = open(adapters).read()
    assert '#include "rocke_py.hpp"' in text
    assert "rocke/rocke_py" not in text


def test_goldens_include_support_by_quoted_name():
    """Every same-directory include of a translated source has a master copy."""
    for golden in sorted(GOLDEN.glob("*.cpp.golden")):
        text = golden.read_text()
        names = re.findall(r'^#include "([^"/]+)"', text, re.M)
        assert "rocke_py.hpp" in names, golden.name
        assert "rocke/rocke_py" not in text, golden.name
        for name in names:
            assert support_file(name), f"{golden.name}: {name}"
