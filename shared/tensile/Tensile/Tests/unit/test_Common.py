################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from __future__ import print_function

import Tensile.Common as Common

import os

import pytest


@pytest.fixture
def clearStrictEnv():
    """Ensure the strict compiler-target env var is unset around each test."""
    Common.os.environ.pop("TENSILE_GFX1250_COMPILER_TARGET", None)
    yield
    Common.os.environ.pop("TENSILE_GFX1250_COMPILER_TARGET", None)


def test_architectureMap_strict_maps_to_gfx1250():
    assert Common.architectureMap["gfx1250-strict"] == "gfx1250"
    assert Common.getArchitectureName("gfx1250-strict") == "gfx1250"


def test_configureCompilerTarget_setsEnvForStrict(clearStrictEnv):
    Common.configureCompilerTarget("gfx1250-strict")
    assert Common.os.environ["TENSILE_GFX1250_COMPILER_TARGET"] == "gfx1250-strict"


def test_configureCompilerTarget_clearsEnvForNonStrict(clearStrictEnv):
    Common.os.environ["TENSILE_GFX1250_COMPILER_TARGET"] = "gfx1250-strict"
    Common.configureCompilerTarget("gfx1250")
    assert "TENSILE_GFX1250_COMPILER_TARGET" not in Common.os.environ


def test_configureCompilerTarget_acceptsDelimiterVariants(clearStrictEnv):
    # CMake uses `_` delimiters, the CLI uses `;`.
    Common.configureCompilerTarget("gfx942_gfx1250-strict")
    assert Common.os.environ["TENSILE_GFX1250_COMPILER_TARGET"] == "gfx1250-strict"
    Common.os.environ.pop("TENSILE_GFX1250_COMPILER_TARGET", None)
    Common.configureCompilerTarget("gfx942;gfx1250-strict")
    assert Common.os.environ["TENSILE_GFX1250_COMPILER_TARGET"] == "gfx1250-strict"


@pytest.mark.parametrize("mixed", ["gfx1250;gfx1250-strict", "all;gfx1250-strict"])
def test_configureCompilerTarget_rejectsMixingWithGfx1250(clearStrictEnv, mixed):
    with pytest.raises(ValueError):
        Common.configureCompilerTarget(mixed)


def test_compilerTarget_rewritesGfx1250OnlyWhenStrict(clearStrictEnv):
    assert Common.compilerTarget("gfx1250") == "gfx1250"
    assert Common.compilerTarget("gfx942") == "gfx942"

    Common.os.environ["TENSILE_GFX1250_COMPILER_TARGET"] = "gfx1250-strict"
    assert Common.compilerTarget("gfx1250") == "gfx1250-strict"
    # Only gfx1250 is rewritten; other targets are untouched.
    assert Common.compilerTarget("gfx942") == "gfx942"


def test_gfxName_appliesStrictCompilerTarget(clearStrictEnv):
    assert Common.gfxName((12, 5, 0)) == "gfx1250"

    Common.os.environ["TENSILE_GFX1250_COMPILER_TARGET"] = "gfx1250-strict"
    assert Common.gfxName((12, 5, 0)) == "gfx1250-strict"
    # A different ISA is not affected by the strict env var.
    assert Common.gfxName((9, 4, 2)) == "gfx942"


def test_gfxArch():
    assert Common.gfxArch('gfx9') is None

    assert Common.gfxArch('gfx803') == (8,0,3)
    assert Common.gfxArch('gfx900') == (9,0,0)
    assert Common.gfxArch('gfx906') == (9,0,6)

    assert Common.gfxArch('gfx1010') == (10,1,0)

    assert Common.gfxArch('gfx90015') == (900,1,5)

    assert Common.gfxArch('blah gfx900 stuff') == (9,0,0)

def test_isGfx12():
    assert Common.isGfx12((12, 0, 0))
    assert Common.isGfx12((12, 0, 1))
    assert Common.isGfx12((12, 5, 0))
    assert not Common.isGfx12((11, 0, 0))
    assert not Common.isGfx12((13, 0, 0))

def test_paths():
    workingPathName = os.path.join("working", "path")
    Common.globalParameters["WorkingPath"] = workingPathName
    expectedWorkingPath = os.path.join("working", "path")
    assert Common.globalParameters["WorkingPath"] == expectedWorkingPath

    recursiveWorkingPath = "next1"
    expectedRecurrsiveWorkingPath = os.path.join("working", "path", "next1")
    Common.pushWorkingPath (recursiveWorkingPath)
    assert Common.globalParameters["WorkingPath"] == expectedRecurrsiveWorkingPath
    Common.popWorkingPath()
    assert Common.globalParameters["WorkingPath"] == expectedWorkingPath

    set1WorkingPath = os.path.join("working", "path", "set1")
    expectedSet1WorkingPath = os.path.join("working", "path", "set1")
    Common.setWorkingPath (set1WorkingPath)
    assert Common.globalParameters["WorkingPath"] == expectedSet1WorkingPath
    Common.popWorkingPath()
    assert Common.globalParameters["WorkingPath"] == expectedWorkingPath
