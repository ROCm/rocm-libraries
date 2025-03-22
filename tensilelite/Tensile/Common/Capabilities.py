################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

import rocisa

from typing import List, Dict

from .Types import IsaVersion, IsaInfo


def makeIsaInfoMap(targetIsas: List[IsaVersion], cxxCompiler: str) -> Dict[IsaVersion, IsaInfo]:
    """Computes the supported capabilities for requested ISAs and compiler.

    Given a list of ISAs and a compiler, the ASM, Arch, Register capabilities
    and ASM bugs are computed and stored in a map.

    Args:
        targetIsas: A list of requested ISA versions to inspect.
        cxxCompiler: A string path to a C++ compiler to use when computing capabilities.

    Returns:
        A map of ISA versions to capabilities.
    """
    isaInfoMap = {}
    ti = rocisa.rocIsa.getInstance()
    for v in targetIsas:
        ti.init(v, cxxCompiler, False)
        asmCaps = ti.getIsaInfo(v).asmCaps
        archCaps = ti.getIsaInfo(v).archCaps
        regCaps = ti.getIsaInfo(v).regCaps
        asmBugs = ti.getIsaInfo(v).asmBugs
        isaInfoMap[v] = IsaInfo(asmCaps, archCaps, regCaps, asmBugs)
    return isaInfoMap
