################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa.enum import DataTypeEnum
from functools import lru_cache
import functools

@lru_cache
def _is8bitFloat(value):
    return (value == DataTypeEnum.Float8.value \
            or value == DataTypeEnum.BFloat8.value \
            or value == DataTypeEnum.Float8BFloat8.value \
            or value == DataTypeEnum.BFloat8Float8.value \
            or value == DataTypeEnum.Float8_fnuz.value \
            or value == DataTypeEnum.BFloat8_fnuz.value \
            or value == DataTypeEnum.Float8BFloat8_fnuz.value \
            or value == DataTypeEnum.BFloat8Float8_fnuz.value)

@functools.total_ordering
class DataType:
    """
    Data Type (new)
    Uses a list of dictionaries to organize the DataType and Properties for the kernels
    Changed older properties list of lists to list of dictionaries
    The inner keys (char, reg, etc) correspond with the data type properties values
    Lookup table is used to store row numbers of a specific property
    """

    properties = [
        {
            'enum': DataTypeEnum.Float,
            'char': 'S',
            'nameAbbrev': 'f32',
            'miOutTypeNameAbbrev': 'f32',
            'reg': 1,
            'hip': 'float',
            'isComplex': False,
        },
        {
            'enum': DataTypeEnum.Double,
            'char': 'D',
            'nameAbbrev': 'f64',
            'miOutTypeNameAbbrev': 'f64',
            'reg': 2,
            'hip': 'double',
            'isComplex': False,
        },
        {
            'enum': DataTypeEnum.ComplexFloat,
            'char': 'C',
            'nameAbbrev': 'f32c',
            'miOutTypeNameAbbrev': 'f32',
            'reg': 2,
            'hip': 'TensileComplexFloat',
            'isComplex': True,
        },
        {
            'enum': DataTypeEnum.ComplexDouble,
            'char': 'Z',
            'nameAbbrev': 'f64c',
            'miOutTypeNameAbbrev': 'f64',
            'reg': 4,
            'hip': 'TensileComplexDouble',
            'isComplex': True,
        },
        {
            'enum': DataTypeEnum.Half,
            'char': 'H',
            'nameAbbrev': 'f16',
            'miOutTypeNameAbbrev': 'f32',
            'reg': 0.5,
            'hip': 'tensile_half',
            'isComplex': False,
        },
        {
            'enum': DataTypeEnum.Int8x4,
            'char': '4xi8',
            'nameAbbrev': 'i8',
            'miOutTypeNameAbbrev': 'i32',
            'reg': 1,
            'hip': 'uint32_t',
            'isComplex': False,
        },
        {
            'enum': DataTypeEnum.Int32,
            'char': 'I',
            'nameAbbrev': 'i32',
            'miOutTypeNameAbbrev': 'NONE', # not supported for MI
            'reg': 1,
            'hip': 'int32_t',
            'isComplex': False,
        },
        {
            'enum': DataTypeEnum.BFloat16,
            'char': 'B',
            'nameAbbrev': 'bf16',
            'miOutTypeNameAbbrev': 'f32',
            'reg': 0.5,
            'hip': 'tensile_bfloat16',
            'isComplex': False,
        },
        {
            'enum': DataTypeEnum.Int8,
            'char': 'I8',
            'nameAbbrev': 'i8',
            'miOutTypeNameAbbrev': 'i32',
            'reg': 0.25,
            'hip': 'int8_t',
            'isComplex': False,
        },
        {
            'enum': DataTypeEnum.Int64,
            'char': 'I64',
            'nameAbbrev': 'i64',
            'miOutTypeNameAbbrev': 'NONE', # not supported for MI
            'reg': 1,
            'hip': 'int64_t',
            'isComplex': False,
        },
        {
            'enum': DataTypeEnum.XFloat32,
            'char': 'X',
            'nameAbbrev': 'xf32',
            'miOutTypeNameAbbrev': 'f32',
            'reg': 1,
            'hip': 'ERROR',
            'isComplex': False,
        },
        {   # NANOO E4M3
            'enum': DataTypeEnum.Float8_fnuz,
            'char': 'F8N',
            'nameAbbrev': 'fp8_fp8',               # to match v_mfma inst
            'miOutTypeNameAbbrev': 'f32',
            'reg': 0.25,
            'hip': 'tensile_float8_fnuz',
            'isComplex': False,
        },
        {   # NANOO E5M2
            'enum': DataTypeEnum.BFloat8_fnuz,
            'char': 'B8N',
            'nameAbbrev': 'bf8_bf8',               # to match v_mfma inst
            'miOutTypeNameAbbrev': 'f32',
            'reg': 0.25,
            'hip': 'tensile_bfloat8_fnuz',
            'isComplex': False,
        },
        {   #NANOO
            'enum': DataTypeEnum.Float8BFloat8_fnuz,
            'char': 'F8B8N',
            'nameAbbrev': 'fp8_bf8',               # to match v_mfma
            'miOutTypeNameAbbrev': 'f32',
            'reg': 0.25,
            'hip': 'ERROR',
            'isComplex': False,
        },
        {   #NANOO
            'enum': DataTypeEnum.BFloat8Float8_fnuz,
            'char': 'B8F8N',
            'nameAbbrev': 'bf8_fp8',               # to match v_mfma
            'miOutTypeNameAbbrev': 'f32',
            'reg': 0.25,
            'hip': 'ERROR',
            'isComplex': False,
        },
        {   # OCP E4M3
            'enum': DataTypeEnum.Float8,
            'char': 'F8',
            'nameAbbrev': 'fp8_fp8',               # to match v_mfma inst
            'miOutTypeNameAbbrev': 'f32',
            'reg': 0.25,
            'hip': 'tensile_float8',
            'isComplex': False,
        },
        {   # OCP E5M2
            'enum': DataTypeEnum.BFloat8,
            'char': 'B8',
            'nameAbbrev': 'bf8_bf8',               # to match v_mfma inst
            'miOutTypeNameAbbrev': 'f32',
            'reg': 0.25,
            'hip': 'tensile_bfloat8',
            'isComplex': False,
        },
        {   #OCP
            'enum': DataTypeEnum.Float8BFloat8,
            'char': 'F8B8',
            'nameAbbrev': 'fp8_bf8',               # to match v_mfma
            'miOutTypeNameAbbrev': 'f32',
            'reg': 0.25,
            'hip': 'ERROR',
            'isComplex': False,
        },
        {   #OCP
            'enum': DataTypeEnum.BFloat8Float8,
            'char': 'B8F8',
            'nameAbbrev': 'bf8_fp8',               # to match v_mfma
            'miOutTypeNameAbbrev': 'f32',
            'reg': 0.25,
            'hip': 'ERROR',
            'isComplex': False,
        },
    ]
    lookup = {}

    def __init__(self, value):
        if isinstance(value, DataTypeEnum):
            self.value = value.value
        elif isinstance(value, int):
            self.value = value
        elif isinstance(value, str):
            self.value = DataType.lookup[value.lower()]
        elif isinstance(value, DataType):
            self.value = value.value
        else:
            raise RuntimeError("initializing DataType to {0} {1}".format(str(type(value)), str(value)))

        self.properties = DataType.properties[self.value]

    def toChar(self):
        return self.properties['char']
    def toName(self):
        return self.properties['enum'].name
    def toNameAbbrev(self):
        return self.properties['nameAbbrev']
    def toEnum(self):
        return self.properties['enum']
    def toDevice(self, language):
        if language == "HIP":
            return self.properties['hip']
        else:
            assert 0

    ########################################
    def zeroString(self, language, vectorWidth):
        """
        Returns a string containing the data output format, depending on programming language
        and in the case of complex numbers, the vector width.
        """
        zeroString = "("
        zeroString += self.toDevice(language)
        if vectorWidth > 1:
            zeroString += str(vectorWidth)
        zeroString += ")("

        """
        if self.value == DataTypeEnum.Half:
            single = "0"
            vectorWidth = 1
        elif self.value == DataTypeEnum.Float.value:
            single = "0.f"
        elif self.value == DataTypeEnum.Double.value:
            single = "0.0"
        elif self.value == DataTypeEnum.ComplexSingle.value:
            single = "0.f, 0.f"
        elif self.value == DataTypeEnum.ComplexDouble.value:
            single = "0.0, 0.0"
        """
        zeroString += "0"
        zeroString += ")"
        return zeroString

    def isReal(self):
        return not self.isComplex()
    def isComplex(self):
        return self.properties['isComplex']
    def isDoubleComplex(self):
        return self.value == DataTypeEnum.ComplexDouble.value
    def isSingleComplex(self):
        return self.value == DataTypeEnum.ComplexFloat.value
    def isDouble(self):
        return self.value == DataTypeEnum.Double.value
    def isSingle(self):
        return self.value == DataTypeEnum.Float.value
    def isHalf(self):
        return self.value == DataTypeEnum.Half.value
    def isInt32(self):
        return self.value == DataTypeEnum.Int32.value
    def isInt64(self):
        return self.value == DataTypeEnum.Int64.value
    def isInt8x4(self):
        return self.value == DataTypeEnum.Int8x4.value
    def isInt8(self):
        return self.value == DataTypeEnum.Int8.value
    def isBFloat16(self):
        return self.value == DataTypeEnum.BFloat16.value
    def isXFloat32(self):
        return self.value == DataTypeEnum.XFloat32.value
    def isFloat8(self):
        return self.value == DataTypeEnum.Float8.value
    def isFloat8_fnuz(self):
        return self.value == DataTypeEnum.Float8_fnuz.value
    def isAnyFloat8(self):
        return (self.value == DataTypeEnum.Float8.value \
                or self.value == DataTypeEnum.Float8_fnuz.value)
    def isBFloat8(self):
        return self.value == DataTypeEnum.BFloat8.value
    def isBFloat8_fnuz(self):
        return self.value == DataTypeEnum.BFloat8_fnuz.value
    def isAnyBFloat8(self):
        return (self.value == DataTypeEnum.BFloat8.value \
                or self.value == DataTypeEnum.BFloat8_fnuz.value)
    def isFloat8BFloat8(self):
        return self.value == DataTypeEnum.Float8BFloat8.value
    def isFloat8BFloat8_fnuz(self):
        return self.value == DataTypeEnum.Float8BFloat8_fnuz.value
    def isAnyFloat8BFloat8(self):
        return (self.value == DataTypeEnum.Float8BFloat8.value \
                or self.value == DataTypeEnum.Float8BFloat8_fnuz.value)
    def isBFloat8Float8(self):
        return self.value == DataTypeEnum.BFloat8Float8.value
    def isBFloat8Float8_fnuz(self):
        return self.value == DataTypeEnum.BFloat8Float8_fnuz.value
    def isAnyBFloat8Float8(self):
        return (self.value == DataTypeEnum.BFloat8Float8.value \
                or self.value == DataTypeEnum.BFloat8Float8_fnuz.value)
    def is8bitFloat(self):
        return _is8bitFloat(self.value)
    def isFloat8A(self):
        return (self.value == DataTypeEnum.Float8.value \
                or self.value == DataTypeEnum.Float8BFloat8.value)
    def isFloat8_fnuzA(self):
        return (self.value == DataTypeEnum.Float8_fnuz.value \
                or self.value == DataTypeEnum.Float8BFloat8_fnuz.value)
    def isAnyFloat8A(self):
        return (self.value == DataTypeEnum.Float8.value \
                or self.value == DataTypeEnum.Float8BFloat8.value \
                or self.value == DataTypeEnum.Float8_fnuz.value \
                or self.value == DataTypeEnum.Float8BFloat8_fnuz.value)
    def isBFloat8A(self):
        return (self.value == DataTypeEnum.BFloat8.value \
                or self.value == DataTypeEnum.BFloat8Float8.value)
    def isBFloat8_fnuzA(self):
        return (self.value == DataTypeEnum.BFloat8_fnuz.value \
                or self.value == DataTypeEnum.BFloat8Float8_fnuz.value)
    def isAnyBFloat8A(self):
        return (self.value == DataTypeEnum.BFloat8.value \
                or self.value == DataTypeEnum.BFloat8Float8.value \
                or self.value == DataTypeEnum.BFloat8_fnuz.value \
                or self.value == DataTypeEnum.BFloat8Float8_fnuz.value)
    def isFloat8B(self):
        return (self.value == DataTypeEnum.Float8.value \
                or self.value == DataTypeEnum.BFloat8Float8.value)
    def isFloat8_fnuzB(self):
        return (self.value == DataTypeEnum.Float8_fnuz.value \
                or self.value == DataTypeEnum.BFloat8Float8_fnuz.value)
    def isAnyFloat8B(self):
        return (self.value == DataTypeEnum.Float8.value \
                or self.value == DataTypeEnum.BFloat8Float8.value \
                or self.value == DataTypeEnum.Float8_fnuz.value \
                or self.value == DataTypeEnum.BFloat8Float8_fnuz.value)
    def isBFloat8B(self):
        return (self.value == DataTypeEnum.BFloat8.value \
                or self.value == DataTypeEnum.Float8BFloat8.value)
    def isBFloat8_fnuzB(self):
        return (self.value == DataTypeEnum.BFloat8_fnuz.value \
                or self.value == DataTypeEnum.Float8BFloat8_fnuz.value)
    def isAnyBFloat8B(self):
        return (self.value == DataTypeEnum.BFloat8.value \
                or self.value == DataTypeEnum.Float8BFloat8.value \
                or self.value == DataTypeEnum.BFloat8_fnuz.value \
                or self.value == DataTypeEnum.Float8BFloat8_fnuz.value)
    def isNone(self):
        return self.value == None

    def numRegisters(self):
        return self.properties['reg']
    def numBytes(self):
        return int(self.numRegisters() * 4)
    def MIOutputTypeNameAbbrev(self):
        return self.properties['miOutTypeNameAbbrev']
    def flopsPerMac(self):
        return 2 if self.isReal() else 8

    def state(self): return self.toName()

    def __str__(self):
        return self.toChar()
    def __repr__(self):
        return self.__str__()

    def getAttributes(self):
        return (self.value,)

    def __hash__(self):
        return hash(self.getAttributes())

    def __eq__(self, other):
        if not isinstance(other, DataType):
            return NotImplemented

        return self.getAttributes() == other.getAttributes()

    def __lt__(self, other):
        if not isinstance(other, DataType):
            return NotImplemented

        return self.getAttributes() < other.getAttributes()

    # Other operands are provided by functools.total_ordering.

def _populateLookupTable(properties,lookup):
    """
    Populates Lookup Table with the corresponding row number for each DataType. The row number
    is assigned to self.value when a DataType object is called
    """
    for i,e in enumerate(properties):
        if i != int(e['enum'].value):
            print(e['enum'].name, ":", e['enum'].value, "does not match index", i, "in properties list")
            raise RuntimeError("Enum value does not match index in properties list")
        for k in ['enum','char']:
            lookupKey = e[k].name.lower() if k == 'enum' else e[k].lower()
            if lookupKey in lookup and lookup[lookupKey] != i:
                raise RuntimeError("Duplicate key {1} in property '{0}'".format(k,lookupKey))
            lookup[lookupKey] = i

_populateLookupTable(DataType.properties,DataType.lookup)
