################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
################################################################################

"""Post-build physical-layout checks for installed hipBLASLt artifacts.

Companion to test_datFileIntegrity.py (which checks dat-file MAPPING
integrity). This file checks the PHYSICAL LAYOUT — that every artifact
landed under `library/<base>/` with a `_<base>` suffix and that no
flat-root files survived. The runtime probe in
projects/hipblaslt/library/src/amd_detail/rocblaslt/src/rocblaslt_auxiliary.cpp
has no fallback for misplaced files; a layout regression produces silent
hipModuleLoad failures at first dispatch, which only surface on a GPU.

These tests run against an *already-built* install tree; they do not
build anything themselves and they do not need a GPU. Set the env var
``HIPBLASLT_TEST_LIBRARY_DIR`` to the directory containing the installed
``lib/hipblaslt/library/<arch>/`` artifacts (or a build tree containing
``library/<arch>/``) to activate them. When the env var is unset
(typical local invocation), the tests skip.

The actual validation logic lives in
``projects/hipblaslt/scripts/validate_library_layout.py`` so the same
checks can be invoked from CI (``test/therock/test_hipblaslt.py`` calls
it before launching hipblaslt-test) and from a shell prompt without
requiring pytest in the test container.
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


_LIB_DIR_ENV = "HIPBLASLT_TEST_LIBRARY_DIR"

# Resolve the path to the standalone validator script.
# Layout: projects/hipblaslt/tensilelite/Tensile/Tests/unit/<this file>
#         projects/hipblaslt/scripts/validate_library_layout.py
_THIS_DIR = Path(__file__).resolve().parent
_VALIDATOR_DIR = _THIS_DIR.parents[3] / "scripts"
if str(_VALIDATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATOR_DIR))


def _libDirOrSkip() -> Path:
    raw = os.environ.get(_LIB_DIR_ENV)
    if not raw:
        pytest.skip(
            f"set {_LIB_DIR_ENV} to an installed library directory "
            "to run physical-layout checks"
        )
    p = Path(raw)
    if not p.is_dir():
        pytest.fail(f"{_LIB_DIR_ENV}={raw!r} is not a directory")
    return p


def _import_validator():
    """Import the standalone validator; fail with a clear message if the
    path resolution above broke (which would mean the test file moved
    relative to scripts/ and needs the path math updated)."""
    try:
        import validate_library_layout
    except ImportError as e:
        pytest.fail(
            f"could not import validate_library_layout from {_VALIDATOR_DIR}: {e}. "
            "If the test file was relocated, update _VALIDATOR_DIR in this file."
        )
    return validate_library_layout


# ---------------------------------------------------------------------------
# Smoke: validator runs and returns a list
# ---------------------------------------------------------------------------
def test_validator_returns_empty_list_on_clean_tree(tmp_path):
    """Build a minimal valid per-base tree on tmp_path and confirm the
    validator returns no violations. Pins the contract that a 'good'
    install does not produce false positives.
    """
    vlib = _import_validator()
    # Standard install layout under tmp_path: lib/hipblaslt/library/gfx942/
    arch_dir = tmp_path / "lib" / "hipblaslt" / "library" / "gfx942"
    arch_dir.mkdir(parents=True)
    for fname in (
        "TensileLibrary_lazy_gfx942.dat",
        "TensileLiteLibrary_lazy_gfx942_Mapping.dat",
        "hipblasltTransform_gfx942.hsaco",
        "hipblasltExtOpLibrary_gfx942.dat",
        "extop_gfx942.co",
    ):
        (arch_dir / fname).write_text("")

    violations = vlib.validate(tmp_path)
    assert violations == [], "valid tree must produce zero violations: " + "\n".join(violations)


def test_validator_flags_flat_root_TensileLibrary(tmp_path):
    """A TensileLibrary.dat at the library root (legacy flat layout) must
    surface as a violation. This is the regression class the per-base
    refactor was supposed to eliminate."""
    vlib = _import_validator()
    library_dir = tmp_path / "lib" / "hipblaslt" / "library"
    (library_dir / "gfx942").mkdir(parents=True)
    # Minimal per-base content so the dir-discovery doesn't also error.
    for fname in (
        "TensileLibrary_lazy_gfx942.dat",
        "hipblasltTransform_gfx942.hsaco",
        "hipblasltExtOpLibrary_gfx942.dat",
        "extop_gfx942.co",
    ):
        (library_dir / "gfx942" / fname).write_text("")
    # Drop the offender at the root
    (library_dir / "TensileLibrary.dat").write_text("")

    violations = vlib.validate(tmp_path)
    assert any("flat-root" in v.lower() or "library root" in v.lower() for v in violations), \
        f"expected flat-root violation, got: {violations}"


def test_validator_flags_wrong_arch_suffix_in_dir(tmp_path):
    """A file with _gfx950 suffix sitting in library/gfx942/ is a layout
    bug — the writer wrote to the wrong dir or the file is misnamed."""
    vlib = _import_validator()
    arch_dir = tmp_path / "lib" / "hipblaslt" / "library" / "gfx942"
    arch_dir.mkdir(parents=True)
    # Required files for gfx942 so dir-completeness check passes
    for fname in (
        "TensileLibrary_lazy_gfx942.dat",
        "hipblasltTransform_gfx942.hsaco",
        "hipblasltExtOpLibrary_gfx942.dat",
        "extop_gfx942.co",
    ):
        (arch_dir / fname).write_text("")
    # Drop a mis-arch file in the gfx942 dir
    (arch_dir / "TensileLibrary_lazy_gfx950.dat").write_text("")

    violations = vlib.validate(tmp_path)
    assert any("does not match dir gfx942" in v for v in violations), \
        f"expected wrong-arch-suffix violation, got: {violations}"


def test_validator_flags_missing_required_per_base_file(tmp_path):
    """Per-base subdir without hipblasltTransform_<arch>.hsaco must be
    reported — the MatrixTransform tests would silently fail at runtime
    otherwise (was the original symptom this PR is fixing)."""
    vlib = _import_validator()
    arch_dir = tmp_path / "lib" / "hipblaslt" / "library" / "gfx942"
    arch_dir.mkdir(parents=True)
    # Everything except hipblasltTransform_*.hsaco
    for fname in (
        "TensileLibrary_lazy_gfx942.dat",
        "hipblasltExtOpLibrary_gfx942.dat",
        "extop_gfx942.co",
    ):
        (arch_dir / fname).write_text("")

    violations = vlib.validate(tmp_path)
    assert any("hipblasltTransform_gfx942.hsaco" in v for v in violations), \
        f"expected missing-transform violation, got: {violations}"


def test_validator_flags_cooked_arch_in_subdir_name(tmp_path):
    """Library subdirs must be bare base archs (gfx942, gfx950, ...).
    Target features (xnack+, sramecc+) belong in filenames only — a
    `gfx942-sramecc+` subdir indicates the build-time arch split bug
    that Source.py:_archNamesFromBundlerTarget guards against."""
    vlib = _import_validator()
    bad_dir = tmp_path / "lib" / "hipblaslt" / "library" / "gfx942-sramecc+"
    bad_dir.mkdir(parents=True)

    violations = vlib.validate(tmp_path)
    assert any("target features" in v for v in violations), \
        f"expected cooked-subdir violation, got: {violations}"


# ---------------------------------------------------------------------------
# Env-driven integration: run validator against $HIPBLASLT_TEST_LIBRARY_DIR
# ---------------------------------------------------------------------------
def test_installed_library_has_zero_layout_violations():
    """The actual end-to-end check: point HIPBLASLT_TEST_LIBRARY_DIR at
    an installed tree and demand zero violations. Skipped if env unset.
    """
    libDir = _libDirOrSkip()
    vlib = _import_validator()
    violations = vlib.validate(libDir)
    if violations:
        pytest.fail(
            f"{len(violations)} layout violation(s) under {libDir}:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
