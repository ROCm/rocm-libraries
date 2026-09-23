################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Shared fixtures for the ``TensileLogic`` characterization suite.

The per-ISA capability map (``isaInfoMap``) is the one expensive, shared
input across these tests: ``makeIsaInfoMap`` shells out to the assembler to
derive ``asmCaps``. Building it once per session (rather than once per module
as the ``Validators`` suite does) keeps the suite fast. The caps are stable
for a given ISA + toolchain, so anything snapshotted downstream stays
reproducible in the dev container (see ``../resistance.md``).
"""

import pytest

from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Toolchain.Validators import validateToolchain


@pytest.fixture(scope="session")
def cxx_compiler():
    return validateToolchain("amdclang++")


@pytest.fixture(scope="session")
def isa_info_map(cxx_compiler):
    return makeIsaInfoMap(SUPPORTED_ISA, cxx_compiler)
