################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from Tensile.Utilities.String import splitDelimitedString
from Tensile.Utilities.toFile import toFile
from pathlib import Path
import pytest
import os

def test_splitDelimitedString():
    archs = "all"
    expected = {"all"}
    result = splitDelimitedString(archs, {";", "_"})
    assert result == expected, f"arch `{archs}` should parse to {expected} but instead maps to {result}"

    archs = "gfx000;gfx803;gfx900:xnack-"
    expected = {'gfx000', 'gfx803', 'gfx900:xnack-'}
    result = splitDelimitedString(archs, {";", "_"})
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

    archs = "gfx000_gfx803_gfx900:xnack-"
    expected = {'gfx000', 'gfx803', 'gfx900:xnack-'}
    result = splitDelimitedString(archs, {";", "_"})
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

    archs = "gfx803,    gfx906_gfx942:gfx1102"
    expected = {"gfx803,    gfx906", "gfx942:gfx1102"}
    result = splitDelimitedString(archs, {";", "_"})
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

    archs = "gfx900;gfx90a:xnack+-gfx1010"
    expected = {"gfx900", "gfx90a:xnack+-gfx1010"}
    result = splitDelimitedString(archs, {";", "_"})
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

    archs = ";gfx803;gfx906;"
    expected = {"", "gfx803", "gfx906", ""}
    result = splitDelimitedString(archs, {";", "_"})
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

    archs = "_gfx803_gfx906_"
    expected = {"", "gfx803", "gfx906", ""}
    result = splitDelimitedString(archs, {";", "_"})
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

    archs = "abc;gfx90Z;all"
    expected = {"abc", "gfx90Z", "all"}
    result = splitDelimitedString(archs, {";", "_"})
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

def test_toFile():

    manifest: Path = Path.cwd() / "my-manifest.txt"
    metaData = ["mylib.yaml"]
    codeObjectFiles = ["library/foo.co", "library/bar.co"]
    sourceCodeObjectFiles = ["library/foo.hsaco", "library/bar.hsaco"]

    if manifest.is_file():
        os.remove(manifest)

    with pytest.raises(AssertionError, match="contents must be a list."):
        toFile(manifest, (1,2,3))

    with pytest.raises(AssertionError, match="contents elements must be a str."):
        toFile(manifest, [1,2,3])

    toFile(manifest, metaData + codeObjectFiles + sourceCodeObjectFiles)

    assert manifest.is_file(), "{manifest} was not generated"
    with open(manifest, "r") as f:
        result = f.readlines()

    assert len(result) == 5, "Expected five entries in manifest file."

    if manifest.is_file():
        os.remove(manifest)
