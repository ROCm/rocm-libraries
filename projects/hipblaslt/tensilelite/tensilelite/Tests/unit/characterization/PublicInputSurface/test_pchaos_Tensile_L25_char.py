################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurface characterization: the ``__name__ == "__main__"``
module-guard in ``tensilelite/tensilelite.py`` at line 25.

Branch 2c7170bfd056c780a059b396e0cbb8a938384ecc. The predicate is a bare
string-equality module guard:

  * TRUE branch  -> ``__name__ == "__main__"`` fires only when the file is
                    executed directly (``python tensilelite/tensilelite.py``). The
                    block prints a deprecation notice and calls ``exit(1)``.
  * FALSE branch -> ``__name__`` equals something other than ``"__main__"``
                    (e.g. ``"tensilelite.tensilelite"`` when imported). The guard is
                    skipped and the module loads normally.

These tests pin ACTUAL observed behavior; they do not assert anything
aspirational.
"""

import importlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# FALSE branch: import path -- __name__ == "tensilelite.tensilelite" -> guard skipped
# ---------------------------------------------------------------------------

def test_module_guard_false_import_succeeds():
    """Importing tensilelite.tensilelite bypasses the guard; module load succeeds."""
    M = importlib.import_module("tensilelite.tensilelite")
    # The module's __name__ attribute is the dotted import name, not "__main__".
    assert M.__name__ == "tensilelite.tensilelite"


def test_module_guard_false_name_is_not_main():
    """The loaded module's __name__ is NOT '__main__', confirming the FALSE branch."""
    M = importlib.import_module("tensilelite.tensilelite")
    assert M.__name__ != "__main__"


# ---------------------------------------------------------------------------
# TRUE branch: direct execution -- __name__ == "__main__" -> print + exit(1)
# ---------------------------------------------------------------------------

def test_module_guard_true_direct_exec_exits_one():
    """Running tensilelite.py directly triggers the TRUE branch and exits with code 1."""
    import importlib.util
    spec = importlib.util.find_spec("tensilelite.tensilelite")
    tensile_py = spec.origin  # absolute path to tensilelite.py

    result = subprocess.run(
        [sys.executable, tensile_py],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        "Expected exit code 1 from direct execution, got {}".format(result.returncode)
    )


def test_module_guard_true_direct_exec_prints_deprecation():
    """Running tensilelite.py directly prints the deprecation/redirect notice to stdout."""
    import importlib.util
    spec = importlib.util.find_spec("tensilelite.tensilelite")
    tensile_py = spec.origin

    result = subprocess.run(
        [sys.executable, tensile_py],
        capture_output=True,
        text=True,
    )
    assert "python -m tensilelite run" in result.stdout, (
        "Expected redirect notice in stdout; got: {!r}".format(result.stdout)
    )
