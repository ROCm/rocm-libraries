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
################################################################################

"""Tests that writeClientConfigIni rejects the legacy flat-layout fallback.

Previously, callers that forgot to pass ``libraryFile`` got a silent
``<sourceDir>/library/TensileLibrary.{dat,yaml}`` flat path. After the
per-base layout change that path doesn't exist — the real file lives at
``<sourceDir>/library/<base>/TensileLibrary.{dat,yaml}`` — so the silent
default produced configs that pointed nowhere. writeClientConfigIni now
asserts that ``libraryFile`` is supplied; this test pins that contract.
"""

import pytest

from Tensile.ClientWriter import writeClientConfigIni

pytestmark = pytest.mark.unit


def _invoke(sourceDir, libraryFile):
    # Minimal positional args; the assert fires before any of the other
    # parameters are touched.
    return writeClientConfigIni(
        forBenchmark        = True,
        problemSizes        = None,
        biasTypeArgs        = "",
        factorDimArgs       = "",
        activationArgs      = "",
        icacheFlushArgs     = "",
        problemType         = None,
        sourceDir           = str(sourceDir),
        codeObjectFiles     = [],
        resultsFileName     = "",
        parametersFilePath  = str(sourceDir / "ClientParameters.ini"),
        deviceId            = 0,
        gfxName             = "gfx942",
        libraryFile         = libraryFile,
    )


def test_writeClientConfigIni_rejects_none_libraryFile(tmp_path):
    with pytest.raises(AssertionError, match="libraryFile is required"):
        _invoke(tmp_path, None)


def test_writeClientConfigIni_rejects_empty_libraryFile(tmp_path):
    with pytest.raises(AssertionError, match="libraryFile is required"):
        _invoke(tmp_path, "")
