# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shipped rocRoller kernels must name a bare processor in .amdgcn_target.

rocRoller writes the target of the device it generated on, for example
``amdgcn-amd-amdhsa--gfx950:sramecc+``. TensileLite's assembler retarget
replaces the processor but keeps a ``:feature`` tail, so that kernel assembles
into a code object marked SRAMECC on (and an ``:xnack`` tail likewise pins
XNACK). Such an object only loads where the feature is set that way, while the
rest of the library is built for ``any``. A bare processor assembles to
``any`` for both, with the same instructions and kernel descriptor.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_HERE = Path(__file__).resolve().parent
_KERNELS = _HERE.parents[1] / "CustomKernels" / "rocroller"
_TARGET = re.compile(r'^\s*\.amdgcn_target\s+"([^"]*)"', re.M)
_BARE = re.compile(r"amdgcn-amd-amdhsa--gfx[0-9a-f]+")


def test_every_rocroller_kernel_names_a_bare_processor():
    kernels = sorted(_KERNELS.glob("*.s"))
    assert kernels, f"no kernels under {_KERNELS}"
    bad = []
    for path in kernels:
        targets = _TARGET.findall(path.read_text())
        if len(targets) != 1 or not _BARE.fullmatch(targets[0]):
            bad.append(f"{path.name}: {targets}")
    assert not bad, (
        "rocRoller kernels must name a bare processor, without :sramecc or :xnack: " + "; ".join(bad)
    )
