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

"""Characterization tests for ``Tensile.Common.Types``: the ``IsaInfo``
dataclass, the ``SemanticVersion``/``IsaVersion`` NamedTuple, the
``DebugConfig`` NamedTuple, and ``makeDebugConfig`` (the config-dict -> tuple
builder with one branch per recognised key)."""

import pytest

from Tensile.Common.Types import (
    IsaInfo,
    SemanticVersion,
    IsaVersion,
    DebugConfig,
    makeDebugConfig,
)

pytestmark = pytest.mark.unit


def test_isa_info_dataclass(snapshot):
    info = IsaInfo(asmCaps={"HasMFMA": True}, archCaps={"HasWave32": False},
                   regCaps={"r": 1}, asmBugs={"b": False})
    assert {"asmCaps": info.asmCaps, "archCaps": info.archCaps,
            "regCaps": info.regCaps, "asmBugs": info.asmBugs} == snapshot


def test_semantic_version(snapshot):
    v = SemanticVersion(9, 4, 2)
    assert {"tuple": tuple(v), "major": v.major, "minor": v.minor, "patch": v.patch,
            "is_isaversion_alias": IsaVersion is SemanticVersion} == snapshot


def test_debug_config_defaults(snapshot):
    assert DebugConfig()._asdict() == snapshot


def test_make_debug_config_empty(snapshot):
    # Empty config -> every default (each `if "X" in config` false arm).
    assert makeDebugConfig({})._asdict() == snapshot


def test_make_debug_config_full(snapshot):
    # Every recognised key set -> each `if "X" in config` true arm.
    cfg = {
        "EnableAsserts": True,
        "EnableDebugA": True,
        "EnableDebugB": True,
        "EnableDebugC": True,
        "ExpectedValueC": 32.0,
        "ForceCExpectedValue": True,
        "DebugKernel": True,
        "ForceGenerateKernel": True,
        "PrintSolutionRejectionReason": True,
        "SplitGSU": True,
        "PrintIndexAssignmentInfo": True,
    }
    assert makeDebugConfig(cfg)._asdict() == snapshot


def test_make_debug_config_unknown_key_ignored(snapshot):
    # An unrecognised key is ignored (no branch).
    assert makeDebugConfig({"NotARealKey": 123})._asdict() == snapshot
