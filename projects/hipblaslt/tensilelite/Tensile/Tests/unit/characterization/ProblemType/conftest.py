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

"""Shared helpers for the ProblemType characterization suite: a deterministic
normaliser that renders a ``ProblemType``'s live ``state`` (which holds
``DataType`` / ``ActivationType`` objects) into a JSON-friendly, snapshot-stable
form, and a ``ProblemType`` builder that overlays overrides on the default
config.
"""

import pytest
from rocisa.enum import DataTypeEnum

from Tensile.Activation import ActivationType
from Tensile.Common.DataType import DataType
from Tensile.SolutionStructs.Problem import ProblemType


def _render(v):
    if isinstance(v, DataType):
        return f"<DataType {v.toName()}>"
    if isinstance(v, ActivationType):
        return f"<ActivationType {str(v)}>"
    if isinstance(v, DataTypeEnum):
        return f"<DataTypeEnum {v.name}>"
    if isinstance(v, (list, tuple)):
        return [_render(x) for x in v]
    if isinstance(v, dict):
        return {k: _render(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    return v


def normalize_state(pt):
    """A sorted, object-free view of a ProblemType's state dict."""
    return {k: _render(pt.state[k]) for k in sorted(pt.state)}


@pytest.fixture
def norm():
    return normalize_state


@pytest.fixture
def make_pt():
    def _make(printInfo=False, **overrides):
        # Mirror real YAML ProblemType configs: a *minimal* dict with only the
        # keys explicitly set. Missing keys default via _defaultProblemType, and
        # the dtype-derivation `if "X" in config` guards inherit correctly
        # (e.g. MacDataTypeA follows DataType when not given). Starting from the
        # full default dict instead would pin MacDataTypeA/DataTypeA/B to 0 and
        # silently override any DataType change.
        cfg = {"DataType": 0}
        cfg.update(overrides)
        return ProblemType(cfg, printInfo)

    return _make
