# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Order of the kernel helpers returned by generateKernelHelperObjects.

The helpers are written into Kernels.cpp in the returned order, and Kernels.cpp
is compiled into Kernels.so-000-<arch>.hsaco. KernelWriterBase hashes its kernel
name (a str), and str hashes are salted per process (PYTHONHASHSEED), so any
set() on the way through makes the hsaco differ between identical builds
(AIHPBLAS-5171). These tests pin the order and check that it does not depend
on the hash seed.

The helper writers and the name callables are replaced with fakes, so the
tests need no compiler and no logic files.
"""

import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from tensilelite.KernelWriterBase import KernelWriterBase
import tensilelite.TensileCreateLibrary.Run as Run

pytestmark = pytest.mark.unit


class _FakeHelper(KernelWriterBase):
    def __init__(self, name):
        super().__init__()
        self._name = name

    def getKernelName(self):
        return self._name

    def getSourceFileString(self):
        return ""

    def getHeaderFileString(self):
        return ""


def _enumNames(solution):
    return solution["enum"]


def _otherNames(solution):
    return solution["other"]


def _fakeInit(solution, kernelHelperType, cxxCompiler, isaInfoMap):
    # Same order as the names after generateKernelHelperObjects sorts them,
    # which is what the real initHelperKernelObjects returns.
    names = solution[kernelHelperType]
    key = lambda x: ("Enum" in x, names.index(x))
    return [_FakeHelper(n) for n in sorted(names, key=key, reverse=True)]


# Several solutions that share helpers, as real logic files do. Enough
# distinct names that a hash-ordered result matching the expected order by
# chance is negligible.
_SOLUTIONS = [
    {"enum": ["ActEnumA"], "other": ["PostGSU16_VW1", "PostGSU16_VW2", "PostGSU16_VW4"]},
    {"enum": ["ActEnumA"], "other": ["PostGSU16_VW2", "BetaOnly_HB"]},
    {"enum": ["ActEnumB"], "other": ["PostGSU2_VW1", "PostGSU2_VW2", "PostGSU2_VW4"]},
    {"enum": [], "other": ["Reduction_S", "Reduction_H", "BetaOnly_HB"]},
    {"enum": ["ActEnumC"], "other": ["ActFunc_S", "ActFunc_H"]},
]

# Enum helpers first, then the rest; within each group, reverse order of
# first appearance.
_EXPECTED = [
    "ActEnumC", "ActEnumB", "ActEnumA",
    "ActFunc_S", "ActFunc_H",
    "Reduction_S", "Reduction_H",
    "PostGSU2_VW1", "PostGSU2_VW2", "PostGSU2_VW4",
    "BetaOnly_HB",
    "PostGSU16_VW1", "PostGSU16_VW2", "PostGSU16_VW4",
]


def _helperNames():
    callables = [("enum", _enumNames), ("other", _otherNames)]
    with patch.object(Run, "kernelObjectNameCallables", return_value=callables), \
         patch.object(Run, "initHelperKernelObjects", side_effect=_fakeInit):
        khos = Run.generateKernelHelperObjects(_SOLUTIONS, "unused-cxx", {})
    return [k.getKernelName() for k in khos]


def test_helper_order_is_enum_first_then_reverse_first_seen():
    assert _helperNames() == _EXPECTED


_CHILD = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("helper_order_test", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(json.dumps(mod._helperNames()))
"""


def test_helper_order_does_not_depend_on_hash_seed():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    orders = {}
    for seed in ("0", "1", "2", "3"):
        env["PYTHONHASHSEED"] = seed
        out = subprocess.run(
            [sys.executable, "-c", _CHILD, __file__],
            env=env, capture_output=True, text=True, check=True,
        ).stdout
        orders[seed] = json.loads(out.strip().splitlines()[-1])
    assert all(order == _EXPECTED for order in orders.values()), orders
