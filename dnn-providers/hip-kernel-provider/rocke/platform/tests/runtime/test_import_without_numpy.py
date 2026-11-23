# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The provider's configure-time compiler probe imports rocKE with the build's
Python, which has no numpy. The modules that probe imports must load without
numpy; numpy is needed only once a kernel's data is touched.

The child process sees a ``numpy`` that fails on import, so any module-level
``import numpy`` on that path fails the test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rocke.runtime import comgr

PKG_ROOT = str(Path(comgr.__file__).resolve().parents[2])

# The imports of the compiler probe the provider configure runs (descriptor-packaging/
# cmake/HkpPackaging.cmake, outside this tree); keep the two lists in step.
_PROBE_IMPORTS = (
    "import rocke\nfrom rocke.runtime import comgr\nfrom rocke.core import lower_llvm\n"
)


def test_the_configure_probe_imports_load_without_numpy(tmp_path):
    (tmp_path / "numpy.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'numpy'\")\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", _PROBE_IMPORTS],
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), PKG_ROOT])},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
