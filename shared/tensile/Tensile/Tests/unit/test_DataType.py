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
import pytest
from Tensile.DataType import DataType

def test_init_single():
    expected = DataType('S')
    assert DataType('single') == expected
    assert DataType('Float') == expected
    assert DataType('tensileDataTypeFloat') == expected

def test_init_double():
    expected = DataType('D')
    assert DataType('double') == expected
    assert DataType('Double') == expected
    assert DataType('tensileDataTypeDouble') == expected

def test_init_complexSingle():
    expected = DataType('C')
    assert DataType('complexSingle') == expected
    assert DataType('complexFloat') == expected
    assert DataType('tensileDataTypeComplexFloat') == expected

def test_init_complexDouble():
    expected = DataType('Z')
    assert DataType('complexDouble') == expected
    assert DataType('complexDouble') == expected
    assert DataType('tensileDataTypeComplexDouble') == expected

def test_init_half():
    expected = DataType('H')
    assert DataType('half') == expected
    assert DataType('Half') == expected
    assert DataType('tensileDataTypeHalf') == expected

def test_init_i8():
    expected = DataType('4xi8')
    assert DataType('int8x4') == expected
    assert DataType('Int8x4') == expected
    assert DataType('tensileDataTypeInt8x4') == expected

def test_init_i32():
    expected = DataType('I')
    assert DataType('int32') == expected
    assert DataType('Int32') == expected
    assert DataType('tensileDataTypeInt32') == expected

def test_single():
    obj = DataType(0)
    assert obj.toChar() == 'S'
    assert obj.toName() == 'single'
    assert obj.toEnum() == 'Float'
    assert obj.toHIP() == 'float'
    assert obj.toDevice("") == 'float'
    assert obj.toCpp() == 'float'
    assert obj.getLibString() == 'tensileDataTypeFloat'
    assert obj.numBytes() == 4
    assert obj.isReal()

def test_double():
    obj = DataType(1)
    assert obj.toChar() == 'D'
    assert obj.toName() == 'double'
    assert obj.toEnum() == 'Double'
    assert obj.toHIP() == 'double'
    assert obj.toDevice("") == 'double'
    assert obj.toCpp() == 'double'
    assert obj.getLibString() == 'tensileDataTypeDouble'
    assert obj.numBytes() == 8
    assert obj.isReal()

def test_complexSingle():
    obj = DataType(2)
    assert obj.toChar() == 'C'
    assert obj.toName() == 'complexSingle'
    assert obj.toEnum() == 'ComplexFloat'
    assert obj.toHIP() == 'TensileComplexFloat'
    assert obj.toDevice("") == 'TensileComplexFloat'
    assert obj.toCpp() == 'TensileComplexFloat'
    assert obj.getLibString() == 'tensileDataTypeComplexFloat'
    assert obj.numBytes() == 8
    assert not obj.isReal()

def test_complexDouble():
    obj = DataType(3)
    assert obj.toChar() == 'Z'
    assert obj.toName() == 'complexDouble'
    assert obj.toEnum() == 'ComplexDouble'
    assert obj.toHIP() == 'TensileComplexDouble'
    assert obj.toDevice("") == 'TensileComplexDouble'
    assert obj.toCpp() == 'TensileComplexDouble'
    assert obj.getLibString() == 'tensileDataTypeComplexDouble'
    assert obj.numBytes() == 16
    assert not obj.isReal()

def test_half():
    obj = DataType(4)
    assert obj.toChar() == 'H'
    assert obj.toName() == 'half'
    assert obj.toEnum() == 'Half'
    assert obj.toHIP() == 'tensile_half'
    assert obj.toDevice("") == 'tensile_half'
    assert obj.toCpp() == 'TensileHalf'
    assert obj.getLibString() == 'tensileDataTypeHalf'
    assert obj.numBytes() == 2
    assert obj.isReal()

def test_int8():
    obj = DataType(5)
    assert obj.toChar() == '4xi8'
    assert obj.toName() == 'int8x4'
    assert obj.toEnum() == 'Int8x4'
    assert obj.toHIP() == 'uint32_t'
    assert obj.toDevice("") == 'uint32_t'
    assert obj.toCpp() == 'TensileInt8x4'
    assert obj.getLibString() == 'tensileDataTypeInt8x4'
    assert obj.numBytes() == 4
    assert obj.isReal()

def test_int32():
    obj = DataType(6)
    assert obj.toChar() == 'I'
    assert obj.toName() == 'int32'
    assert obj.toEnum() == 'Int32'
    assert obj.toHIP() == 'int32_t'
    assert obj.toDevice("") == 'int32_t'
    assert obj.toCpp() == 'TensileInt32'
    assert obj.getLibString() == 'tensileDataTypeInt32'
    assert obj.numBytes() == 4
    assert obj.isReal()

def test_cmp():
    assert DataType('single') == DataType('S')
    assert not DataType('S') != DataType(0)
    assert DataType('Float') < DataType('Double')
    assert not DataType('tensileDataTypeFloat') > DataType('Z')
    assert DataType('half') >= DataType('ComplexFloat')
    assert not DataType('int32') <= DataType('tensileDataTypeInt8x4')

def test_bounds():
    with pytest.raises(Exception):
        DataType(14)
