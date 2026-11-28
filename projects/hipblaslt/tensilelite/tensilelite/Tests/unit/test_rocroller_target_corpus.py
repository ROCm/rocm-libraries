# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Registered rocRoller kernels must name a bare processor in .amdgcn_target.

rocRoller writes the target of the device it generated on, for example
``amdgcn-amd-amdhsa--gfx950:sramecc+``. TensileLite's assembler retarget
replaces the processor but keeps a ``:feature`` tail, so that kernel assembles
into a code object marked SRAMECC on (and an ``:xnack`` tail likewise pins
XNACK). Such an object only loads where the feature is set that way, while the
rest of the library is built for ``any``. A bare processor assembles to
``any`` for both, with the same instructions and kernel descriptor.

Only a kernel that a logic file registers is assembled into the library, so
the check covers those.
"""

import re
from pathlib import Path

import pytest
import yaml

from logic_corpus import needs_logic_corpus
from rocroller_logic_corpus import rocroller_logic_files

pytestmark = [pytest.mark.unit, needs_logic_corpus]

_HERE = Path(__file__).resolve().parent
_KERNELS = _HERE.parents[1] / "CustomKernels" / "rocroller"
_TARGET = re.compile(r'^\s*\.amdgcn_target\s+"([^"]*)"', re.M)
_BARE = re.compile(r"amdgcn-amd-amdhsa--gfx[0-9a-f]+")


def _registered():
    names = set()
    for path in rocroller_logic_files():
        doc = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        for solution in doc["Solutions"]:
            name = (solution.get("CustomKernel") or {}).get("name", "")
            if name.startswith("RR_GEMM_"):
                names.add(name)
    return names


def test_every_registered_rocroller_kernel_names_a_bare_processor():
    names = _registered()
    assert names, "no logic file registers an RR_GEMM_ kernel"
    bad = []
    for name in sorted(names):
        path = _KERNELS / (name + ".s")
        if not path.is_file():
            bad.append(f"{path.name}: missing")
            continue
        targets = _TARGET.findall(path.read_text())
        if len(targets) != 1 or not _BARE.fullmatch(targets[0]):
            bad.append(f"{path.name}: {targets}")
    assert not bad, (
        "rocRoller kernels must name a bare processor, without :sramecc or :xnack: " + "; ".join(bad)
    )
