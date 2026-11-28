# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke_installed_smoke.py lowers with the Python engine and checks the shared
C++ engine against it.

The smoke is the installed rocke_python_import_smoke entry. It sits in
platform/tools of the source tree and beside tests/ in an installed test tree;
these tests run whichever copy is here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import rocke
from rocke.core.backend import BACKEND_ENV, CPP_STRICT_ENV
from rocke.core.native_engine import NO_SHARED_ENGINE, SHARED_ENGINE_ENV
from rocke.tests.engines import native_engine_or_skip
from rocke.tests.source_modules import load_module_from_path

_NAME = "rocke_installed_smoke.py"
# platform/ in the source tree, the CTestTestfile's directory when installed.
_ROOT = Path(__file__).resolve().parents[2]
SMOKE = next(
    (path for path in (_ROOT / "tools" / _NAME, _ROOT / _NAME) if path.is_file()),
    None,
)
if SMOKE is None:
    raise FileNotFoundError(
        f"{_NAME} is in neither {_ROOT / 'tools'} (source tree) nor {_ROOT} "
        "(installed test tree)"
    )


def _run_smoke(**environment: str) -> subprocess.CompletedProcess:
    """Run the smoke with ``environment`` added to this process's, and the
    rocke package this test imports first on its path."""
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(rocke.__file__).resolve().parents[1]),
        **environment,
    }
    return subprocess.run(
        [sys.executable, str(SMOKE)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_the_python_engine_smoke_lowers_without_the_cpp_engine():
    """With the cpp backend requested strictly and no shared engine, a lowering
    the C++ engine served would fail; the smoke still passes, so the Python
    engine lowered its kernel."""
    result = _run_smoke(
        **{BACKEND_ENV: "cpp", CPP_STRICT_ENV: "1", SHARED_ENGINE_ENV: NO_SHARED_ENGINE}
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "rocKE installed Python engine smoke: PASS",
        "rocKE installed shared C++ engine smoke: not run (none installed)",
    ]


def test_the_shared_engine_smoke_lowers_to_the_python_engines_bytes():
    engine = native_engine_or_skip()
    path = engine.loaded_shared_engine_path()
    result = _run_smoke(**{SHARED_ENGINE_ENV: str(path)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[1] == (
        f"rocKE installed shared C++ engine smoke: PASS ({path})"
    )


def test_a_lowering_that_differs_from_the_python_engines_fails_the_smoke():
    smoke = load_module_from_path("rocke_installed_smoke", SMOKE)
    with pytest.raises(RuntimeError, match="shared C\\+\\+ engine lowered"):
        smoke._require_same_ir(
            "shared C++ engine", "define void @a()", "define void @b()"
        )
    smoke._require_same_ir("shared C++ engine", "same", "same")
