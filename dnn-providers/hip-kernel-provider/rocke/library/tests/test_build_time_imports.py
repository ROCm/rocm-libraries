# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The provider build imports rocKE with the build's Python, which has only the
standard library: no numpy, no torch. Its table generators import ``kernels``
and ``dispatch`` modules, and its packer imports whichever ``kernels`` builder
module a kernel descriptor names, so every module of those two packages must
import with the standard library and rocKE's own Python alone. A third-party
package belongs inside the function that launches or checks host arrays.

The packages are walked, not listed, so a new module is covered the day it is
added. Each package is imported in a child interpreter whose import system
refuses every top-level module other than the standard library, ``rocke`` and
the build-time packages, so a third-party import reached through a platform
module fails like a direct one. Platform modules are covered as far as these
packages import them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import rocke

import dispatch
import kernels

BUILD_TIME_PACKAGES = (kernels, dispatch)

# Host-array adapters the build never imports: numpy is what they operate on.
# Each entry covers the module and its submodules. Keep this list honest: an
# entry that imports cleanly must be removed.
HOST_ARRAY_MODULES = {
    "kernels.common._sdpa_seqlen": "hipDNN sequence-length tensor adapter",
    "dispatch.sdpa": "hipDNN SDPA host binding",
}

# Runs in the child: argv[1] = JSON {package, roots, allowed}. Prints one JSON
# object: the modules walked and, for each module that failed, its error.
_CHILD = """
import importlib, importlib.abc, json, pkgutil, sys

args = json.loads(sys.argv[1])
package = args["package"]
sys.path[:0] = args["roots"]
allowed = set(sys.stdlib_module_names) | set(args["allowed"])


class StandardLibraryAndRocke(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] not in allowed:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, StandardLibraryAndRocke())
walked, failed = [package], {}
try:
    root = importlib.import_module(package)
except BaseException as error:
    failed[package] = repr(error)
else:
    for info in pkgutil.walk_packages(root.__path__, package + "."):
        walked.append(info.name)
        try:
            importlib.import_module(info.name)
        except BaseException as error:
            failed[info.name] = repr(error)
print(json.dumps({"walked": walked, "failed": failed}))
"""


def _child_args(package) -> str:
    """Where this process found ``rocke`` and the build-time packages (source
    tree, installed test tree or a wheel), and the names the child may import
    besides the standard library."""
    rocke_packages = (rocke, *BUILD_TIME_PACKAGES)
    roots = {str(Path(p.__file__).resolve().parents[1]) for p in rocke_packages}
    return json.dumps(
        {
            "package": package.__name__,
            "roots": sorted(roots),
            "allowed": [p.__name__ for p in rocke_packages],
        }
    )


def _is_host_array_module(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in HOST_ARRAY_MODULES)


@pytest.mark.parametrize("package", BUILD_TIME_PACKAGES, ids=lambda p: p.__name__)
def test_every_module_imports_with_the_standard_library_only(package) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", _CHILD, _child_args(package)],
        env={**os.environ, "ROCKE_BACKEND": "python"},  # as the build runs rocKE
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])

    failed = report["failed"]
    unexpected = {m: e for m, e in failed.items() if not _is_host_array_module(m)}
    assert not unexpected, (
        "the provider build's Python (standard library only) cannot import:\n"
        + "\n".join(f"  {m}: {e}" for m, e in sorted(unexpected.items()))
    )
    assert len(report["walked"]) > 1, f"walked no module of {package.__name__}"

    stale = [
        m
        for m in HOST_ARRAY_MODULES
        if m.partition(".")[0] == package.__name__ and m not in failed
    ]
    assert not stale, f"these import cleanly now; remove them: {stale}"
