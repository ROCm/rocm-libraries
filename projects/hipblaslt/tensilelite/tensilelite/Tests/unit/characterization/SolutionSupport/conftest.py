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

"""Shared fixtures for the Solution support-slice suite: a minimal-config
``ProblemType`` builder (mirroring the ProblemType suite) and an isolation
helper for the module-global type-mismatch collector in ``Solution``.
"""

import contextlib

import pytest

from Tensile.SolutionStructs.Problem import ProblemType


@pytest.fixture
def make_pt():
    def _make(**overrides):
        cfg = {"DataType": 0}
        cfg.update(overrides)
        return ProblemType(cfg, False)

    return _make


@pytest.fixture
def isolated_collector():
    """Context manager: clear Solution's module-global type-mismatch collector,
    yield it, then restore the prior contents (so the shared session is
    unaffected and snapshots see only the delta this test produced)."""

    @contextlib.contextmanager
    def _ctx():
        from Tensile.SolutionStructs.Solution import _typeMismatchCollector

        saved = {k: {"count": v["count"], "values": set(v["values"]), "files": set(v["files"])}
                 for k, v in _typeMismatchCollector.items()}
        _typeMismatchCollector.clear()
        try:
            yield _typeMismatchCollector
        finally:
            _typeMismatchCollector.clear()
            _typeMismatchCollector.update(saved)

    return _ctx
