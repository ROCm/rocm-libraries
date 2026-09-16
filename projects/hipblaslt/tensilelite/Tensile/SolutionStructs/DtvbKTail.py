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
"""DTVB K-tail host predicate.

DTVB tail-loop global loads over-read B when K is not a multiple of DepthU.
On gfx1200 that is a page-not-present VM-fault when B sits at the end of a
2 MiB VRAM slab (PyTorch caching allocator). Isolated hipMalloc of the exact
tensor usually survives because the extra bytes stay inside the same PTE.

The generated BoundSizeMultiple=DepthU predicate (AssertSummationElementMultiple)
must apply for every transpose. A previous NN-only gate left NT (Ailk_Bjlk /
LoRA dgrad) unprotected.

No rocisa / toolchain imports: extras tests call this directly.
"""


def applyDtvbKTailAssert(state, tc):
    """Raise AssertSummationElementMultiple to DepthU for every DTVB solution.

    Args:
        state: solution dict; must contain AssertSummationElementMultiple and DepthU
        tc: "A" or "B" (DirectToVgpr side being considered)
    """
    if tc != "B":
        return
    state["AssertSummationElementMultiple"] = max(
        state["AssertSummationElementMultiple"], state["DepthU"]
    )
