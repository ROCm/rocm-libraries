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

"""Mutation-killing characterization tests for
``Tensile.SolutionStructs.Naming.shortenFileBase``.

Targets two surviving mutants:
  * __mutmut_3: the ``splitGSU`` argument forwarded to ``getKernelNameMin`` is
    replaced by ``None`` — pinned by forwarding a truthy ``splitGSU`` that the
    downstream code path reacts to.
  * __mutmut_6: the length guard ``len(base) <= MAX_FILENAME_LENGTH`` is
    weakened to ``<`` — pinned at the exact boundary ``len(base) == MAX``.
"""

import pytest

from Tensile.Common.Constants import MAX_FILENAME_LENGTH
import Tensile.SolutionStructs.Naming as N

pytestmark = pytest.mark.unit


def test_shorten_file_base_forwards_split_gsu(make_state):
    # __mutmut_3: shortenFileBase must forward the real ``splitGSU`` (not None)
    # to getKernelNameMin. With splitGSU truthy and GlobalSplitU>1, the internal
    # rewrite at Naming.py:155 turns GSU into the string "M", which the
    # subsequent `"M" > 0` comparison at Naming.py:160 raises TypeError on
    # (a characterized bug). Passing None instead (the mutant) skips the
    # rewrite and returns a name without raising, so this raise distinguishes
    # original from mutant.
    state = make_state(GlobalSplitU=4)
    with pytest.raises(TypeError):
        N.shortenFileBase(True, state)


def test_shorten_file_base_boundary_returns_base_unchanged(monkeypatch):
    # __mutmut_6: at exactly MAX_FILENAME_LENGTH the base is returned verbatim
    # (`<=` branch). The mutant's `<` would fall through to the hash-truncation
    # path and return a different string.
    base = "x" * MAX_FILENAME_LENGTH
    monkeypatch.setattr(N, "getKernelNameMin", lambda kernel, splitGSU: base)
    assert N.shortenFileBase(False, {}) == base
    assert len(N.shortenFileBase(False, {})) == MAX_FILENAME_LENGTH
