#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for cmake/hipblaslt_experimental.cmake: the experimental feature deadline and cap.

Each test configures a tiny CMake project that includes the real helper, pretends to target
some ROCm release, registers features, and checks whether configure passes, warns, or fails.
Deadlines are counted on an ordered list of ROCm releases, set by each test. No compiler or
ROCm is needed; tests are skipped if cmake is not on PATH.
    python3 -m pytest tools/scripts/tests/test_experimental_registry.py
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_HELPER = _ROOT / "cmake" / "hipblaslt_experimental.cmake"
_RELEASES = _ROOT / "cmake" / "rocm_feature_releases.cmake"
_TEMPLATE = _ROOT / "library" / "include" / "hipblaslt-experimental-features.h.in"
_REAL_FEATURES = _ROOT / "library" / "experimental" / "features.cmake"
_REAL_HEADER_DIR = _ROOT / "library" / "include" / "hipblaslt"

# Deliberately irregular, like real ROCm numbering: counting is by position, not arithmetic.
_TEST_RELEASES = "7.0 7.1 7.2 7.11 8.0 8.1 8.2"

_CMAKE = shutil.which("cmake")
pytestmark = pytest.mark.skipif(_CMAKE is None, reason="cmake not found")


def _configure(tmp_path: Path, release: str, features: str, headers=("demo",),
               header_dir: Path = None, releases: str = None):
    """Configure a fixture project targeting ROCm `release`. Returns (exit code, output).

    With `releases` None, the fixture uses _TEST_RELEASES; pass "real" to use
    cmake/rocm_feature_releases.cmake.
    """
    src = tmp_path / "src"
    if header_dir is None:
        header_dir = src / "include" / "hipblaslt"
        header_dir.mkdir(parents=True)
        for h in headers:
            macro = "HIPBLASLT_EXPERIMENTAL_" + h.replace("-", "_").upper()
            (header_dir / f"hipblaslt-experimental-{h}.h").write_text(f"#define {macro} 1\n")
    else:
        src.mkdir()
    if releases == "real":
        release_setup = f'include("{_RELEASES.as_posix()}")\nset(HIPBLASLT_ROCM_RELEASE {release})'
    else:
        release_setup = (f"set(ROCM_FEATURE_RELEASES {releases or _TEST_RELEASES})\n"
                         f"set(HIPBLASLT_ROCM_RELEASE {release})")
    (src / "features.cmake").write_text(features)
    (src / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.25)
project(fixture LANGUAGES NONE)
{release_setup}
include("{_HELPER.as_posix()}")
set(HIPBLASLT_EXPERIMENTAL_HEADER_DIR "{header_dir.as_posix()}")
include("${{CMAKE_CURRENT_SOURCE_DIR}}/features.cmake")
hipblaslt_experimental_finalize(
    TEMPLATE "{_TEMPLATE.as_posix()}"
    OUTPUT "${{CMAKE_CURRENT_BINARY_DIR}}/hipblaslt-experimental-features.h"
    REGISTRY "${{CMAKE_CURRENT_BINARY_DIR}}/hipblaslt-experimental-registry.inc")
message(STATUS "fragments: ${{HIPBLASLT_EXPERIMENTAL_FRAGMENTS}}")
""")
    r = subprocess.run([_CMAKE, "-S", str(src), "-B", str(tmp_path / "build")],
                       capture_output=True, text=True)
    # CMake wraps long messages; collapse whitespace so tests can match whole sentences.
    return r.returncode, " ".join((r.stdout + r.stderr).split())


def _features(introduced="7.1", last="7.11", extra="", name="DEMO", header="demo",
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


@pytest.mark.parametrize("release", ["7.1", "7.2", "7.2.3"])
def test_before_last_release_is_silent(tmp_path, release):
    rc, out = _configure(tmp_path, release, _features())
    assert rc == 0, out
    assert "last release that may ship" not in out


@pytest.mark.parametrize("release", ["7.11", "7.11.2"])
def test_last_release_warns(tmp_path, release):
    # Patch releases of the last release still ship the feature.
    rc, out = _configure(tmp_path, release, _features())
    assert rc == 0, out
    assert "ROCm 7.11 is the last release that may ship it" in out


def test_after_last_release_fails(tmp_path):
    # 8.0 follows 7.11 on the list, whatever the version numbers say.
    rc, out = _configure(tmp_path, "8.0", _features())
    assert rc != 0
    assert "introduced in ROCm 7.1 and may ship through ROCm 7.11, but this is ROCm 8.0" in out
    assert "Owner: @someone" in out


def test_release_not_on_the_list_fails(tmp_path):
    rc, out = _configure(tmp_path, "7.3", _features())
    assert rc != 0
    assert "ROCm release '7.3' is not in ROCM_FEATURE_RELEASES" in out


def test_generated_files(tmp_path):
    rc, out = _configure(tmp_path, "7.1", _features())
    assert rc == 0, out
    header = (tmp_path / "build" / "hipblaslt-experimental-features.h").read_text()
    assert '#include "hipblaslt/hipblaslt-experimental-demo.h"' in header
    assert ("DEMO: revision 1, introduced ROCm 7.1, last release ROCm 7.11, owner @someone"
            in header)
    registry = (tmp_path / "build" / "hipblaslt-experimental-registry.inc").read_text()
    assert "HIPBLASLT_EXPERIMENTAL_FEATURE(DEMO)" in registry
    assert re.search(r"fragments: \S+hipblaslt-experimental-demo\.h", out)


# --- Registration rules -------------------------------------------------------------------


def test_span_of_four_releases_is_allowed(tmp_path):
    rc, out = _configure(tmp_path, "7.1", _features(last="8.0"))
    assert rc == 0, out


def test_span_over_limit_fails(tmp_path):
    rc, out = _configure(tmp_path, "7.1", _features(last="8.1"))
    assert rc != 0
    assert "7.1 through 8.1 is more than 4 releases" in out


def test_extension_moves_deadline_by_one(tmp_path):
    extra = 'EXTENDED_TO 8.0\n    APPROVED_BY "@lead"'
    rc, out = _configure(tmp_path, "8.0", _features(extra=extra))
    assert rc == 0, out
    assert "ROCm 8.0 is the last release" in out
    rc, out = _configure(tmp_path / "next", "8.1", _features(extra=extra))
    assert rc != 0
    assert "ship through ROCm 8.0" in out


@pytest.mark.parametrize("extra", ['EXTENDED_TO 8.0', 'EXTENDED_TO 8.1\n    APPROVED_BY "@lead"'])
def test_bad_extension_fails(tmp_path, extra):
    rc, out = _configure(tmp_path, "7.1", _features(extra=extra))
    assert rc != 0
    assert "EXTENDED_TO must be one release" in out


def test_missing_field_fails(tmp_path):
    rc, out = _configure(tmp_path, "7.1", _features().replace('OWNER        "@someone"', ""))
    assert rc != 0
    assert "OWNER is required" in out


def test_header_must_match_feature_name(tmp_path):
    rc, out = _configure(tmp_path, "7.1", _features(name="GEMM_A2A", header="demo"))
    assert rc != 0
    assert "HEADER must be hipblaslt-experimental-gemm-a2a.h" in out


def test_header_must_define_revision_macro(tmp_path):
    header_dir = tmp_path / "hdr"
    header_dir.mkdir()
    (header_dir / "hipblaslt-experimental-demo.h").write_text("// no revision\n")
    rc, out = _configure(tmp_path, "7.1", _features(), header_dir=header_dir)
    assert rc != 0
    assert "must #define HIPBLASLT_EXPERIMENTAL_DEMO" in out


def test_unregistered_fragment_fails(tmp_path):
    rc, out = _configure(tmp_path, "7.1", _features(), headers=("demo", "sneaky"))
    assert rc != 0
    assert "hipblaslt-experimental-sneaky.h is not registered" in out


def test_feature_cap(tmp_path):
    two = _features() + _features(name="OTHER", header="other").split("\n", 3)[3]
    rc, out = _configure(tmp_path, "7.1", two, headers=("demo", "other"))
    assert rc == 0, out
    capped = two.replace("MAX_FEATURES 3", "MAX_FEATURES 1")
    rc, out = _configure(tmp_path / "capped", "7.1", capped, headers=("demo", "other"))
    assert rc != 0
    assert "the cap is 1" in out


# --- The real registry --------------------------------------------------------------------


def test_real_registry_configures_at_default_release(tmp_path):
    text = _RELEASES.read_text()
    release = re.search(r'set\(HIPBLASLT_ROCM_RELEASE "([\d.]+)"', text).group(1)
    rc, out = _configure(tmp_path, release, _REAL_FEATURES.read_text(),
                         header_dir=_REAL_HEADER_DIR, releases="real")
    assert rc == 0, out
