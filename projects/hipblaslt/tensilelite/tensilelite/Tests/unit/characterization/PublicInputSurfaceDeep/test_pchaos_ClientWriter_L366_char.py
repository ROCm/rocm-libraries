################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurfaceDeep characterization: ``if os.name != "nt"`` at
``Tensile/ClientWriter.py:366`` inside ``writeRunScript``.

Branch 3ae422d17a07911474102c74d1a2bf4a79523ede.

The predicate checks the Python interpreter's ``os.name`` attribute, which is
set at process start from the host OS family.  The single public input is the
OS name string: ``'posix'`` on Linux/macOS, ``'nt'`` on Windows.

  * TRUE branch  (``os.name != "nt"``, i.e. posix): ``writeRunScript`` appends
    ``exit $ERR`` to the generated shell script and calls ``os.chmod(runScript,
    0o777)`` to make it executable.  On the POSIX side, the shebang header
    (``#!/bin/bash\\n\\n``) is also emitted at line 314 and the script is named
    ``run.sh``.

  * FALSE branch (``os.name == "nt"``): the ``exit $ERR`` write and ``os.chmod``
    call are both skipped.  The script is named ``run.bat``.

Domain: {``'posix'``, ``'nt'``}.  Tests are CPU-only and deterministic.  The
integration pins ACTUAL observed behavior by calling ``writeRunScript`` in a
``tmp_path`` sandbox with ``os.name`` and ``getClientExecutablePath``
monkeypatched so no filesystem client binary is required.
"""

import os
import stat

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure-helper: mirror the predicate at ClientWriter.py:366
# ---------------------------------------------------------------------------


def _smi_clock_pinning_emitted(os_name: str) -> bool:
    """Mirror the predicate ``if os.name != "nt"`` at ClientWriter.py:366.

    Returns True when the host is NOT Windows (posix/Linux/macOS),
    meaning ``exit $ERR`` and ``os.chmod`` are emitted/called.
    Returns False on Windows (``os.name == "nt"``), where they are skipped.
    """
    return os_name != "nt"


# ---------------------------------------------------------------------------
# Pure-helper tests (no I/O, deterministic)
# ---------------------------------------------------------------------------


def test_posix_predicate_is_true():
    """os.name='posix' -> os.name != 'nt' -> True (TRUE branch L366)."""
    assert _smi_clock_pinning_emitted("posix") is True


def test_nt_predicate_is_false():
    """os.name='nt' -> os.name != 'nt' -> False (FALSE branch L366)."""
    assert _smi_clock_pinning_emitted("nt") is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_writeRunScript(tmp_path, monkeypatch, os_name_value):
    """Call writeRunScript with os.name patched to os_name_value.

    Monkeypatches:
      - Tensile.ClientWriter.os.name              (the module's os reference)
      - os.name                                   (global fallback)
      - Tensile.ClientWriter.getClientExecutablePath  (returns fake path string)

    ``getClientExecutablePath`` must be patched because it calls
    ``os.path.isfile(globalParameters.get("PrebuiltClient"))`` which returns
    None in the test environment and raises TypeError.  The patched version
    simply returns a fake path string; writeRunScript only writes that string
    into the script file, it does not execute it.

    Returns the path returned by writeRunScript.
    """
    import Tensile.ClientWriter as cw

    monkeypatch.setattr(cw.os, "name", os_name_value)
    monkeypatch.setattr("os.name", os_name_value)
    monkeypatch.setattr(cw, "getClientExecutablePath", lambda: "/fake/tensile_client")

    build_dir = str(tmp_path)
    config_paths = ["/fake/ClientParameters.ini"]

    return cw.writeRunScript(
        path=build_dir,
        forBenchmark=False,
        enableTileSelection=False,
        cxxCompiler="g++",
        cCompiler="gcc",
        buildDir=build_dir,
        configPaths=config_paths,
    )


# ---------------------------------------------------------------------------
# Integration pin: TRUE branch (posix) — writeRunScript emits exit $ERR
# and makes script executable
# ---------------------------------------------------------------------------


def test_true_branch_posix_emits_exit_err(tmp_path, monkeypatch):
    """TRUE branch (os.name='posix'): run.sh contains 'exit $ERR' line.

    Monkeypatches os.name to 'posix' so the predicate at L366 fires.
    The generated script must contain 'exit $ERR'.
    """
    result = _call_writeRunScript(tmp_path, monkeypatch, "posix")
    assert result.endswith("run.sh"), f"Expected run.sh on posix, got {result}"
    content = open(result).read()
    assert "exit $ERR" in content, "TRUE branch (posix): 'exit $ERR' must be in script"


def test_true_branch_posix_script_is_executable(tmp_path, monkeypatch):
    """TRUE branch (os.name='posix'): os.chmod(0o777) is applied to run.sh.

    The generated file must have executable bits set after writeRunScript
    returns when os.name is 'posix'.
    """
    result = _call_writeRunScript(tmp_path, monkeypatch, "posix")
    file_stat = os.stat(result)
    is_executable = bool(file_stat.st_mode & stat.S_IXUSR)
    assert is_executable, "TRUE branch (posix): run.sh must be chmod +x (S_IXUSR)"


def test_true_branch_posix_has_shebang(tmp_path, monkeypatch):
    """TRUE branch (os.name='posix'): run.sh begins with '#!/bin/bash'.

    The shebang line at ClientWriter.py:314 is guarded by the same
    os.name != 'nt' predicate applied earlier in writeRunScript.
    """
    result = _call_writeRunScript(tmp_path, monkeypatch, "posix")
    content = open(result).read()
    assert content.startswith("#!/bin/bash"), "TRUE branch: shebang must be first line"


# ---------------------------------------------------------------------------
# Integration pin: FALSE branch (nt) — no exit $ERR, no chmod
# ---------------------------------------------------------------------------


def test_false_branch_nt_no_exit_err(tmp_path, monkeypatch):
    """FALSE branch (os.name='nt'): run.bat does NOT contain 'exit $ERR'.

    Monkeypatches os.name to 'nt' so the predicate at L366 is False.
    The generated Windows batch script must not contain 'exit $ERR'.
    """
    result = _call_writeRunScript(tmp_path, monkeypatch, "nt")
    assert result.endswith("run.bat"), f"Expected run.bat on nt, got {result}"
    content = open(result).read()
    assert "exit $ERR" not in content, "FALSE branch (nt): 'exit $ERR' must NOT be in script"


def test_false_branch_nt_script_not_executable(tmp_path, monkeypatch):
    """FALSE branch (os.name='nt'): os.chmod is NOT called; file lacks full 0o777 perms.

    On Windows the chmod block is skipped.  We verify that os.chmod(0o777) was
    NOT applied by checking that the file mode is NOT 0o777 (the umask on a
    POSIX host will have removed at least one bit from the open() default).
    """
    result = _call_writeRunScript(tmp_path, monkeypatch, "nt")
    file_stat = os.stat(result)
    # Without the explicit chmod 0o777 the mode cannot be 0o777 under the
    # container's default umask (typically 022 removes write bits for group/other).
    assert (file_stat.st_mode & 0o777) != 0o777, (
        "FALSE branch (nt): os.chmod(0o777) must NOT have been called"
    )
