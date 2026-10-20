#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for cmake/hipblaslt_experimental.cmake: the experimental feature deadline and cap.

Each test configures a tiny CMake project that includes the real helper, pretends to be
some hipBLASLt version, registers features, and checks whether configure passes, warns,
or fails. No compiler or ROCm is needed; tests are skipped if cmake is not on PATH.
    python3 -m pytest tools/scripts/tests/test_experimental_registry.py
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_HELPER = _ROOT / "cmake" / "hipblaslt_experimental.cmake"
_TEMPLATE = _ROOT / "library" / "include" / "hipblaslt-experimental-features.h.in"
_REAL_FEATURES = _ROOT / "library" / "experimental" / "features.cmake"
_REAL_HEADER_DIR = _ROOT / "library" / "include" / "hipblaslt"

_CMAKE = shutil.which("cmake")
pytestmark = pytest.mark.skipif(_CMAKE is None, reason="cmake not found")


def _configure(tmp_path: Path, version: str, features: str, headers=("demo",),
               header_dir: Path = None):
    """Configure a fixture project as hipBLASLt `version`. Returns (exit code, output)."""
    src = tmp_path / "src"
    if header_dir is None:
        header_dir = src / "include" / "hipblaslt"
        header_dir.mkdir(parents=True)
        for h in headers:
            (header_dir / f"hipblaslt-experimental-{h}.h").write_text("// fragment\n")
    else:
        src.mkdir()
    (src / "features.cmake").write_text(features)
    (src / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.25)
project(fixture VERSION {version} LANGUAGES NONE)
include("{_HELPER.as_posix()}")
set(HIPBLASLT_EXPERIMENTAL_HEADER_DIR "{header_dir.as_posix()}")
include("${{CMAKE_CURRENT_SOURCE_DIR}}/features.cmake")
hipblaslt_experimental_finalize(
    TEMPLATE "{_TEMPLATE.as_posix()}"
    OUTPUT "${{CMAKE_CURRENT_BINARY_DIR}}/hipblaslt-experimental-features.h")
message(STATUS "fragments: ${{HIPBLASLT_EXPERIMENTAL_FRAGMENTS}}")
""")
    r = subprocess.run([_CMAKE, "-S", str(src), "-B", str(tmp_path / "build")],
                       capture_output=True, text=True)
    # CMake wraps long messages; collapse whitespace so tests can match whole sentences.
    return r.returncode, " ".join((r.stdout + r.stderr).split())


def _features(introduced="1.5", last="1.8", extra="", name="DEMO", header="demo",
              max_features=3):
    return f"""
set(HIPBLASLT_EXPERIMENTAL_MAX_RELEASES 4)
set(HIPBLASLT_EXPERIMENTAL_MAX_FEATURES {max_features})
hipblaslt_experimental_feature({name}
    HEADER       hipblaslt-experimental-{header}.h
    OWNER        "@someone"
    TRACKING     "https://example.com/1"
    INTRODUCED   {introduced}
    LAST_RELEASE {last}
    {extra})
"""


# --- The deadline -------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["1.5.0", "1.6.0", "1.7.3"])
def test_before_last_release_is_silent(tmp_path, version):
    rc, out = _configure(tmp_path, version, _features())
    assert rc == 0, out
    assert "last release" not in out


@pytest.mark.parametrize("version", ["1.8.0", "1.8.2"])
def test_last_release_warns(tmp_path, version):
    # Patch releases of the last minor release still ship the feature.
    rc, out = _configure(tmp_path, version, _features())
    assert rc == 0, out
    assert "1.8 is the last release that may ship it" in out


def test_after_last_release_fails(tmp_path):
    rc, out = _configure(tmp_path, "1.9.0", _features())
    assert rc != 0
    assert "introduced in 1.5 and may ship through 1.8, but this is 1.9" in out
    assert "Owner: @someone" in out


def test_generated_header_includes_registered_fragments(tmp_path):
    rc, out = _configure(tmp_path, "1.5.0", _features())
    assert rc == 0, out
    text = (tmp_path / "build" / "hipblaslt-experimental-features.h").read_text()
    assert '#include "hipblaslt/hipblaslt-experimental-demo.h"' in text
    assert "DEMO: introduced 1.5, last release 1.8, owner @someone" in text
    assert re.search(r"fragments: \S+hipblaslt-experimental-demo\.h", out)


# --- Registration rules -------------------------------------------------------------------


def test_span_over_limit_fails(tmp_path):
    rc, out = _configure(tmp_path, "1.5.0", _features(last="1.9"))
    assert rc != 0
    assert "1.5 through 1.9 is more than 4 releases" in out


def test_extension_moves_deadline_by_one(tmp_path):
    extra = 'EXTENDED_TO 1.9\n    APPROVED_BY "@lead"'
    rc, out = _configure(tmp_path, "1.9.0", _features(extra=extra))
    assert rc == 0, out
    assert "1.9 is the last release" in out
    rc, out = _configure(tmp_path / "next", "1.10.0", _features(extra=extra))
    assert rc != 0
    assert "ship through 1.9" in out


@pytest.mark.parametrize("extra", ['EXTENDED_TO 1.9', 'EXTENDED_TO 1.10\n    APPROVED_BY "@lead"'])
def test_bad_extension_fails(tmp_path, extra):
    rc, out = _configure(tmp_path, "1.5.0", _features(extra=extra))
    assert rc != 0
    assert "EXTENDED_TO must be one release" in out


def test_missing_field_fails(tmp_path):
    rc, out = _configure(tmp_path, "1.5.0", _features().replace('OWNER        "@someone"', ""))
    assert rc != 0
    assert "OWNER is required" in out


def test_unregistered_fragment_fails(tmp_path):
    rc, out = _configure(tmp_path, "1.5.0", _features(), headers=("demo", "sneaky"))
    assert rc != 0
    assert "hipblaslt-experimental-sneaky.h is not registered" in out


def test_feature_cap(tmp_path):
    two = _features() + _features(name="OTHER", header="other").split("\n", 3)[3]
    rc, out = _configure(tmp_path, "1.5.0", two, headers=("demo", "other"))
    assert rc == 0, out
    capped = two.replace("MAX_FEATURES 3", "MAX_FEATURES 1")
    rc, out = _configure(tmp_path / "capped", "1.5.0", capped, headers=("demo", "other"))
    assert rc != 0
    assert "the cap is 1" in out


# --- The real registry --------------------------------------------------------------------


def test_real_registry_configures_at_current_version(tmp_path):
    cmakelists = (_ROOT / "CMakeLists.txt").read_text()
    version = re.search(r'set\(HIPBLASLT_PROJECT_VERSION "([\d.]+)"', cmakelists).group(1)
    rc, out = _configure(tmp_path, version, _REAL_FEATURES.read_text(),
                         header_dir=_REAL_HEADER_DIR)
    assert rc == 0, out
