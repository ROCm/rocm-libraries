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

import pytest
import os

from Tensile.ReplacementKernels import ReplacementKernels

def replacementDir(dirname):
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(scriptDir, 'replacement', dirname)

def test_BadFile():
    with pytest.raises(RuntimeError):
        obj = ReplacementKernels(replacementDir('bad_file'), 'default')
        obj.get("asdf")

def test_DuplicateKernel():
    with pytest.raises(RuntimeError):
        obj = ReplacementKernels(replacementDir('duplicate_kernel'), 'default')
        obj.get("asdf")

goodObjs = [ReplacementKernels(replacementDir('known_kernels_v3'), "default")]

@pytest.mark.parametrize("obj", goodObjs)
def test_foo(obj):
    foo = obj.get('foo')
    assert os.path.isfile(foo)
    assert os.path.isabs(foo)
    assert foo.endswith('kernel_named_foo.txt')

@pytest.mark.parametrize("obj", goodObjs)
def test_bar(obj):
    bar = obj.get('bar')
    assert os.path.isfile(bar)
    assert os.path.isabs(bar)
    assert bar.endswith('kernel_named_bar.txt')

@pytest.mark.parametrize("obj", goodObjs)
def test_baz(obj):
    baz = obj.get('baz')
    assert os.path.isfile(baz)
    assert os.path.isabs(baz)
    assert baz.endswith('baz.s.txt')

@pytest.mark.parametrize("obj", goodObjs)
def test_unknown(obj):
    assert obj.get('asdfds') is None
