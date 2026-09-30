################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurfaceDeep characterization: the ``if os.environ.get("ROCM_PATH")``
predicate in ``Tensile/Toolchain/Validators.py`` at line 97, inside
``_posixSearchPaths``.

Branch c03b6953169e51d676e1f454d11aef7debf7110a.  The predicate tests Python
truthiness of ``os.environ.get("ROCM_PATH")``; for ``str | None`` the only truthy
case is a non-empty string.

  * TRUE branch  -> ROCM_PATH is set to a non-empty string.  The body splits the
                    value on ``os.pathsep`` and, for each component ``p``,
                    appends ``Path(p) / "bin"`` and ``Path(p) / "lib" / "llvm" / "bin"``
                    to the search-path list.
  * FALSE branch -> ROCM_PATH is absent (``get()`` returns ``None``) or is set to the
                    empty string (``get()`` returns ``""``).  The body is skipped;
                    only the DEFAULT_ROCM_* constants and PATH entries are used.

classification: runtime-dependent (branch outcome fixed by the OS environment at
process launch; both sides reachable by setting/unsetting/emptying ROCM_PATH).
CPU-only, no GPU probe.
"""

import os
from pathlib import Path
from typing import Optional

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Pure-helper: mirror the predicate at Validators.py:97
# ---------------------------------------------------------------------------


def rocm_path_branch_taken(rocm_path_get: Optional[str]) -> bool:
    """Mirror of ``if os.environ.get("ROCM_PATH"):`` at Validators.py:97.

    Argument is the value returned by ``os.environ.get("ROCM_PATH")``:
      - ``None`` when the env var is absent
      - the (possibly empty) string value otherwise

    Returns ``True`` iff the branch body executes (get() is truthy), which for
    ``str | None`` is exactly: a non-empty string.

    pre:  rocm_path_get is None or isinstance(rocm_path_get, str)
    post: __return__ == (rocm_path_get is not None and len(rocm_path_get) > 0)
    """
    return bool(rocm_path_get)


# ---------------------------------------------------------------------------
# Pure-helper tests (no I/O, no imports needed)
# ---------------------------------------------------------------------------


def test_rocm_path_none_is_false():
    """ROCM_PATH absent -> get() returns None -> predicate False (FALSE branch)."""
    assert rocm_path_branch_taken(None) is False


def test_rocm_path_empty_string_is_false():
    """ROCM_PATH='' -> get() returns '' -> predicate False (FALSE branch)."""
    assert rocm_path_branch_taken("") is False


def test_rocm_path_non_empty_is_true():
    """ROCM_PATH='/opt/rocm' -> get() returns non-empty string -> predicate True (TRUE branch)."""
    assert rocm_path_branch_taken("/opt/rocm") is True


def test_rocm_path_minimal_non_empty_is_true():
    """ROCM_PATH='A' (z3-derived minimal witness) -> predicate True (TRUE branch)."""
    assert rocm_path_branch_taken("A") is True


# ---------------------------------------------------------------------------
# Integration tests: call real _posixSearchPaths with monkeypatched ROCM_PATH
# to pin ACTUAL function behavior for each branch polarity.
# ---------------------------------------------------------------------------


def _import_posix_search_paths():
    """Import the private _posixSearchPaths function."""
    import importlib
    M = importlib.import_module("Tensile.Toolchain.Validators")
    return M._posixSearchPaths


def test_real_function_rocm_path_absent_excludes_rocm_path_entries(monkeypatch):
    """FALSE branch (ROCM_PATH unset): _posixSearchPaths omits ROCM_PATH-derived entries.

    When ROCM_PATH is absent the branch body is skipped entirely.  The result
    must contain the two DEFAULT_ROCM_* entries but must NOT contain a path
    derived from any caller-supplied ROCM_PATH value.
    """
    _posixSearchPaths = _import_posix_search_paths()
    monkeypatch.delenv("ROCM_PATH", raising=False)
    # Also clear PATH to keep the result deterministic (no PATH entries injected)
    monkeypatch.delenv("PATH", raising=False)

    result = _posixSearchPaths()

    # Must return a list of Path objects
    assert isinstance(result, list)
    assert all(isinstance(p, Path) for p in result)

    # The two defaults must be present
    assert Path("/opt/rocm/bin") in result
    assert Path("/opt/rocm/lib/llvm/bin") in result

    # No entry derived from a ROCM_PATH component should be present
    # (we confirm by checking that none of the paths contains a component that
    # would only be there if ROCM_PATH had been set to something custom)
    result_strs = [str(p) for p in result]
    assert all("/sentinel_rocm" not in s for s in result_strs)


def test_real_function_rocm_path_empty_excludes_rocm_path_entries(monkeypatch):
    """FALSE branch (ROCM_PATH=''): _posixSearchPaths omits ROCM_PATH-derived entries.

    An empty string is falsy; the branch body must be skipped exactly as for
    the absent case.
    """
    _posixSearchPaths = _import_posix_search_paths()
    monkeypatch.setenv("ROCM_PATH", "")
    monkeypatch.delenv("PATH", raising=False)

    result = _posixSearchPaths()

    assert isinstance(result, list)
    assert Path("/opt/rocm/bin") in result
    assert Path("/opt/rocm/lib/llvm/bin") in result

    # No extra paths beyond the two defaults (since PATH is also cleared)
    assert len(result) == 2


def test_real_function_rocm_path_set_prepends_bin_and_llvm_bin(monkeypatch):
    """TRUE branch (ROCM_PATH='/tmp/testrocm'): _posixSearchPaths prepends ROCM_PATH-derived entries.

    When ROCM_PATH is a non-empty string the body splits on os.pathsep and for
    each component appends Path(p)/'bin' and Path(p)/'lib'/'llvm'/'bin'.  These
    entries must appear in the result BEFORE the DEFAULT_ROCM_* constants.
    """
    _posixSearchPaths = _import_posix_search_paths()
    monkeypatch.setenv("ROCM_PATH", "/tmp/testrocm")
    monkeypatch.delenv("PATH", raising=False)

    result = _posixSearchPaths()

    assert isinstance(result, list)
    assert Path("/tmp/testrocm/bin") in result
    assert Path("/tmp/testrocm/lib/llvm/bin") in result

    # ROCM_PATH-derived entries must come before the defaults
    idx_rocm_bin = result.index(Path("/tmp/testrocm/bin"))
    idx_default_bin = result.index(Path("/opt/rocm/bin"))
    assert idx_rocm_bin < idx_default_bin


def test_real_function_rocm_path_multi_component_yields_all_entries(monkeypatch):
    """TRUE branch (ROCM_PATH with two components): all components are expanded.

    When ROCM_PATH contains multiple os.pathsep-separated entries each must
    contribute a 'bin' and a 'lib/llvm/bin' path to the result.
    """
    _posixSearchPaths = _import_posix_search_paths()
    two_paths = os.pathsep.join(["/tmp/rocm_a", "/tmp/rocm_b"])
    monkeypatch.setenv("ROCM_PATH", two_paths)
    monkeypatch.delenv("PATH", raising=False)

    result = _posixSearchPaths()

    assert Path("/tmp/rocm_a/bin") in result
    assert Path("/tmp/rocm_a/lib/llvm/bin") in result
    assert Path("/tmp/rocm_b/bin") in result
    assert Path("/tmp/rocm_b/lib/llvm/bin") in result
