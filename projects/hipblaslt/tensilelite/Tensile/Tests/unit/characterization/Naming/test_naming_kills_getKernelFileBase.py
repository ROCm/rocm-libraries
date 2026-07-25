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

"""Mutation-kill characterization tests for
``Tensile.SolutionStructs.Naming.getKernelFileBase``.

Behavioural assertions (no snapshots) that pin the current contract precisely
enough to distinguish the clean function from surviving mutants.
"""

import pytest

from Tensile.SolutionStructs.Naming import getKernelFileBase

pytestmark = pytest.mark.unit


def test_custom_kernel_name_key_is_exact():
    # Kills mutant_2/3/4: the membership test must use the exact key
    # "CustomKernelName". Any case/spelling change makes the `in` check False and
    # short-circuits to the else branch (shortenFileBase). A short custom name
    # would round-trip identically through shortenFileBase (which also honours
    # CustomKernelName), so use a name longer than MAX_FILENAME_LENGTH: the
    # direct branch returns it verbatim while the else branch hash-shortens it.
    long_name = "A" * 200
    kernel = {"CustomKernelName": long_name}
    assert getKernelFileBase(False, kernel) == long_name


def test_split_gsu_is_forwarded_to_shorten(make_state):
    # Kills mutant_14: the else branch must forward the real splitGSU flag to
    # shortenFileBase (not a hard-coded None). With splitGSU=True and
    # GlobalSplitU=4, getKernelNameMin rewrites GSU to the string "M" and then
    # compares "M" > 0 -> TypeError (a characterized current behaviour). If
    # splitGSU were replaced by None (falsy), that rewrite is skipped and no
    # TypeError is raised.
    kernel = make_state(GlobalSplitU=4)
    with pytest.raises(TypeError):
        getKernelFileBase(True, kernel)
