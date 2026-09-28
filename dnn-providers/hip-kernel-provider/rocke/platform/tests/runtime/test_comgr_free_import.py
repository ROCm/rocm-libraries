# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A precompiled-HSACO launch must not need comgr to be loadable.

``KernelLauncher`` (see :mod:`rocke.runtime.launcher`) accepts a raw ``hsaco:
bytes`` blob directly -- comgr's job (LLVM IR text -> HSACO) is already done by
the time a caller reaches it. A CI node that only ever launches precompiled
kernels should therefore never need ``libamd_comgr.so`` to resolve.

``rocke.runtime`` is a single package, though: ``rocke/runtime/__init__.py``
imports every submodule (including ``comgr``) up front, so
``import rocke.runtime.launcher`` unavoidably imports the ``comgr`` *module*.
What it must NOT do is dlopen the *shared library* -- ``comgr._load_lib`` is
only ever invoked lazily, from an actual compile call. This test proves that
by making any ``ctypes.CDLL`` call touching ``amd_comgr`` raise, in a fresh
subprocess, and then importing the launch path anyway.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_PROBE = textwrap.dedent(
    """
    import ctypes

    _real_cdll = ctypes.CDLL

    def _hostile_cdll(name, *args, **kwargs):
        if "amd_comgr" in str(name):
            raise OSError(f"comgr must not be dlopen'd at import time: {name!r}")
        return _real_cdll(name, *args, **kwargs)

    ctypes.CDLL = _hostile_cdll

    # The import under test. If anything on this path eagerly dlopens comgr,
    # the patched CDLL above raises and this process exits non-zero.
    import rocke.runtime.launcher  # noqa: F401
    from rocke.runtime import comgr

    # Importing must not have resolved (let alone loaded) the library either.
    assert comgr._lib is None, "comgr._lib was populated by a bare import"

    print("OK")
    """
)


def test_import_launcher_does_not_require_comgr_loadable() -> None:
    """``import rocke.runtime.launcher`` must succeed with comgr unloadable."""
    here = __import__("pathlib").Path(__file__).resolve()
    py_root = here.parents[2] / "python"  # tests/runtime -> tests -> platform, + python
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(py_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"import of rocke.runtime.launcher failed with comgr unloadable\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout
