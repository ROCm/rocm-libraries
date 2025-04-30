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

import Tensile.Tensile as Tensile

def test_2sum_gsu_src(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/2sum_gsu_src.yaml"), tmpdir.strpath])

def test_2sum(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/2sum.yaml"), tmpdir.strpath])

def test_2sum_gsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/2sum_gsu.yaml"), tmpdir.strpath])

def test_3sum_gsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/3sum_gsu.yaml"), tmpdir.strpath])

def test_2sum_gsu_simple(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/2sum_gsu_simple.yaml"), tmpdir.strpath])

def test_2sum_src(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/2sum_src.yaml"), tmpdir.strpath])

def test_simple_sum2_scrambled(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("disabled/multi_sum/simple_sum2_scrambled.yaml"), tmpdir.strpath])

