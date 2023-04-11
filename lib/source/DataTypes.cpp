/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <map>
#include <string>

namespace rocRoller
{
    std::string ToString(DataDirection dir)
    {
        switch(dir)
        {
        case DataDirection::ReadOnly:
            return "read_only";
        case DataDirection::WriteOnly:
            return "write_only";
        case DataDirection::ReadWrite:
            return "read_write";
        case DataDirection::Count:
        default:
            break;
        }
        throw std::runtime_error("Invalid DataDirection");
    }

    std::ostream& operator<<(std::ostream& stream, DataDirection dir)
    {
        return stream << ToString(dir);
    }

    std::map<VariableType, DataTypeInfo> DataTypeInfo::data;
    std::map<std::string, VariableType>  DataTypeInfo::typeNames;

    std::string ToString(DataType d)
    {
        switch(d)
        {
        case DataType::Float:
            return "Float";
        case DataType::Double:
            return "Double";
        case DataType::ComplexFloat:
            return "ComplexFloat";
        case DataType::ComplexDouble:
            return "ComplexDouble";
        case DataType::Half:
            return "Half";
        case DataType::Halfx2:
            return "Halfx2";
        case DataType::Int8x4:
            return "Int8x4";
        case DataType::Int8:
            return "Int8";
        case DataType::Int16:
            return "Int16";
        case DataType::Int32:
            return "Int32";
        case DataType::Int64:
            return "Int64";
        case DataType::BFloat16:
            return "BFloat16";
        case DataType::Raw32:
            return "Raw32";
        case DataType::UInt8:
            return "UInt8";
        case DataType::UInt16:
            return "UInt16";
        case DataType::UInt32:
            return "UInt32";
        case DataType::UInt64:
            return "UInt64";
        case DataType::Bool:
            return "Bool";
        case DataType::Bool32:
            return "Bool32";

        case DataType::Count:;
        }
        return "Invalid";
    }

    std::string TypeAbbrev(DataType d)
    {
        switch(d)
        {
        case DataType::Float:
            return "S";
        case DataType::Double:
            return "D";
        case DataType::ComplexFloat:
            return "C";
        case DataType::ComplexDouble:
            return "Z";
        case DataType::Half:
            return "H";
        case DataType::Halfx2:
            return "2xH";
        case DataType::Int8x4:
            return "4xi8";
        case DataType::Int8:
            return "I8";
        case DataType::Int16:
            return "I16";
        case DataType::Int32:
            return "I";
        case DataType::Int64:
            return "I64";
        case DataType::BFloat16:
            return "B";
        case DataType::Raw32:
            return "R";
        case DataType::UInt8:
            return "U8";
        case DataType::UInt16:
            return "U16";
        case DataType::UInt32:
            return "U32";
        case DataType::UInt64:
            return "U64";
        case DataType::Bool:
            return "BL";
        case DataType::Bool32:
            return "BL32";

        case DataType::Count:;
        }
        return "Invalid";
    }

    std::string ToString(LayoutType l)
    {
        switch(l)
        {

        case LayoutType::MATRIX_A:
            return "MATRIX_A";
        case LayoutType::MATRIX_B:
            return "MATRIX_B";
        case LayoutType::MATRIX_ACCUMULATOR:
            return "MATRIX_ACCUMULATOR";

        case LayoutType::Count:;
        }
        return "Invalid";
    }

    std::ostream& operator<<(std::ostream& stream, LayoutType l)
    {
        return stream << ToString(l);
    }

    std::string ToString(NaryArgument n)
    {
        switch(n)
        {

        case NaryArgument::DEST:
            return "DEST";
        case NaryArgument::LHS:
            return "LHS";
        case NaryArgument::RHS:
            return "RHS";

        case NaryArgument::Count:;
        }
        return "Invalid";
    }

    std::ostream& operator<<(std::ostream& stream, NaryArgument n)
    {
        return stream << ToString(n);
    }

    std::string ToString(PointerType const& p)
    {
        switch(p)
        {
        case PointerType::Value:
            return "Value";
        case PointerType::PointerLocal:
            return "PointerLocal";
        case PointerType::PointerGlobal:
            return "PointerGlobal";
        case PointerType::Buffer:
            return "Buffer";

        case PointerType::Count:;
        }
        return "Invalid";
    }

    std::ostream& operator<<(std::ostream& stream, PointerType const& p)
    {
        return stream << ToString(p);
    }

    std::string ToString(VariableType const& v)
    {
        return ToString(v.pointerType) + ": " + ToString(v.dataType);
    }

    std::string TypeAbbrev(VariableType const& v)
    {
        switch(v.pointerType)
        {
        case PointerType::Value:
            return TypeAbbrev(v.dataType);
        case PointerType::PointerLocal:
            return "PL";
        case PointerType::PointerGlobal:
            return "PG";
        case PointerType::Buffer:
            return "PB";

        case PointerType::Count:;
        }
        return "Invalid";
    }

    std::ostream& operator<<(std::ostream& stream, const VariableType& v)
    {
        return stream << ToString(v);
    }

    size_t VariableType::getElementSize() const
    {
        switch(pointerType)
        {
        case PointerType::Value:
            return DataTypeInfo::Get(dataType).elementSize;
        case PointerType::PointerLocal:
            return 4;
        case PointerType::PointerGlobal:
            return 8;
        case PointerType::Buffer:
            return 16;

        default:
        case PointerType::Count:
            break;
        }
        throw std::runtime_error(
            concatenate("Invalid pointer type: ", static_cast<int>(pointerType)));
    }

    VariableType VariableType::Promote(VariableType lhs, VariableType rhs)
    {
        // Two pointers: Error!
        if(lhs.pointerType != PointerType::Value && rhs.pointerType != PointerType::Value)
            Throw<FatalError>("Invalid type promotion.");

        // Same types: just return.
        if(lhs == rhs)
            return lhs;

        if(lhs.pointerType == PointerType::Value && rhs.pointerType != PointerType::Value)
            std::swap(lhs, rhs);

        auto const& lhsInfo = DataTypeInfo::Get(lhs);
        auto const& rhsInfo = DataTypeInfo::Get(rhs);

        // If there's a pointer, it's in LHS.

        if(lhs.pointerType != PointerType::Value)
        {
            AssertFatal(rhsInfo.isIntegral, "Indexing variable must be integral.", ShowValue(rhs));
            return lhs;
        }

        AssertFatal(!lhs.isPointer() && !rhs.isPointer(), "Shouldn't get here!");

        if(lhs.dataType == DataType::Raw32)
            return rhs;
        if(rhs.dataType == DataType::Raw32)
            return lhs;

        AssertFatal(lhsInfo.isIntegral == rhsInfo.isIntegral,
                    "No automatic promotion between integral and non-integral types",
                    ShowValue(lhs),
                    ShowValue(rhs));

        AssertFatal(lhsInfo.isComplex == rhsInfo.isComplex,
                    "No automatic promotion between complex and non-complex types",
                    ShowValue(lhs),
                    ShowValue(rhs));

        if(lhsInfo.elementSize > rhsInfo.elementSize)
            return lhs;

        if(lhsInfo.elementSize < rhsInfo.elementSize)
            return rhs;

        if(lhsInfo.packing < rhsInfo.packing)
            return lhs;

        if(lhsInfo.packing > rhsInfo.packing)
            return rhs;

        if(rhsInfo.isSigned)
            return rhs;

        return lhs;
    }

    template <typename T>
    void DataTypeInfo::registerTypeInfo()
    {
        using T_Info = TypeInfo<T>;

        DataTypeInfo info;

        info.variableType        = T_Info::Var;
        info.segmentVariableType = T_Info::SegmentVariableType;
        info.name                = T_Info::Name();
        info.abbrev              = T_Info::Abbrev();

        info.packing     = T_Info::Packing;
        info.elementSize = T_Info::ElementSize;
        info.segmentSize = T_Info::SegmentSize;
        info.alignment   = T_Info::Alignment;

        info.isComplex  = T_Info::IsComplex;
        info.isIntegral = T_Info::IsIntegral;
        info.isSigned   = T_Info::IsSigned;

        addInfoObject(info);
    }

    void DataTypeInfo::registerAllTypeInfo()
    {
        registerTypeInfo<BFloat16>();
        registerTypeInfo<Half>();
        registerTypeInfo<Halfx2>();
        registerTypeInfo<double>();
        registerTypeInfo<float>();
        registerTypeInfo<std::complex<double>>();
        registerTypeInfo<std::complex<float>>();

        registerTypeInfo<Int8x4>();
        registerTypeInfo<Raw32>();

        registerTypeInfo<int8_t>();
        registerTypeInfo<int16_t>();
        registerTypeInfo<int32_t>();
        registerTypeInfo<int64_t>();

        registerTypeInfo<uint8_t>();
        registerTypeInfo<uint16_t>();
        registerTypeInfo<uint32_t>();
        registerTypeInfo<uint64_t>();

        registerTypeInfo<bool>();
        registerTypeInfo<Bool32>();

        registerTypeInfo<PointerLocal>();
        registerTypeInfo<PointerGlobal>();
        registerTypeInfo<Buffer>();
    }

    void DataTypeInfo::registerAllTypeInfoOnce()
    {
        static int call_once = (registerAllTypeInfo(), 0);

        // Use the variable to quiet the compiler.
        if(call_once)
            return;
    }

    void DataTypeInfo::addInfoObject(DataTypeInfo const& info)
    {
        data[info.variableType] = info;
        typeNames[info.name]    = info.variableType;
    }

    DataTypeInfo const& DataTypeInfo::Get(int index)
    {
        return Get(static_cast<DataType>(index));
    }

    DataTypeInfo const& DataTypeInfo::Get(DataType t)
    {
        registerAllTypeInfoOnce();

        auto iter = data.find(t);
        if(iter == data.end())
            throw std::runtime_error(concatenate("Invalid data type: ", static_cast<int>(t)));

        return iter->second;
    }

    DataTypeInfo const& DataTypeInfo::Get(VariableType const& v)
    {
        registerAllTypeInfoOnce();

        auto iter = data.find(v);
        AssertFatal(iter != data.end(),
                    "Invalid variable type: ",
                    static_cast<int>(v.dataType),
                    " ",
                    static_cast<int>(v.pointerType));

        return iter->second;
    }

    DataTypeInfo const& DataTypeInfo::Get(std::string const& str)
    {
        registerAllTypeInfoOnce();

        auto iter = typeNames.find(str);
        if(iter == typeNames.end())
            throw std::runtime_error(concatenate("Invalid data type: ", str));

        return Get(iter->second);
    }

    std::ostream& operator<<(std::ostream& stream, const DataType& t)
    {
        return stream << ToString(t);
    }

    std::istream& operator>>(std::istream& stream, DataType& t)
    {
        std::string strValue;
        stream >> strValue;

        t = DataTypeInfo::Get(strValue).variableType.dataType;

        return stream;
    }

    static_assert(CCountedEnum<LayoutType>);

    static_assert(CArithmeticType<Half>);
    static_assert(CArithmeticType<float>);
    static_assert(CArithmeticType<double>);
    static_assert(CArithmeticType<int8_t>);
    static_assert(CArithmeticType<int32_t>);
    static_assert(CArithmeticType<int64_t>);
    static_assert(CArithmeticType<uint32_t>);
    static_assert(CArithmeticType<uint64_t>);
    static_assert(!CArithmeticType<uint64_t*>);
    static_assert(!CArithmeticType<std::string>);

} // namespace rocRoller
