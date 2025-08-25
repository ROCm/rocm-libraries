/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 AMD ROCm(TM) Software
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
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <map>
#include <string>

namespace rocRoller
{
    std::string toString(DataDirection dir)
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
        return stream << toString(dir);
    }

    std::string toString(DataType d)
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
        case DataType::FP8:
            return "FP8";
        case DataType::BF8:
            return "BF8";
        case DataType::FP6:
            return "FP6";
        case DataType::BF6:
            return "BF6";
        case DataType::FP4:
            return "FP4";
        case DataType::FP8x4:
            return "FP8x4";
        case DataType::BF8x4:
            return "BF8x4";
        case DataType::FP4x8:
            return "FP4x8";
        case DataType::FP6x16:
            return "FP6x16";
        case DataType::BF6x16:
            return "BF6x16";
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
        case DataType::BFloat16x2:
            return "BFloat16x2";
        case DataType::Raw32:
            return "Raw32";
        case DataType::UInt8x4:
            return "UInt8x4";
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
        case DataType::Bool64:
            return "Bool64";
        case DataType::E8M0:
            return "E8M0";
        case DataType::E8M0x4:
            return "E8M0x4";
        case DataType::None:
            return "None";
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
        case DataType::FP8:
            return "FP8";
        case DataType::FP8x4:
            return "4xFP8";
        case DataType::BF8:
            return "BF8";
        case DataType::BF8x4:
            return "4xBF8";
        case DataType::FP4x8:
            return "8xFP4";
        case DataType::FP6x16:
            return "16xFP6";
        case DataType::FP6:
            return "FP6";
        case DataType::BF6x16:
            return "16xBF6";
        case DataType::BF6:
            return "BF6";
        case DataType::FP4:
            return "FP4";
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
        case DataType::BFloat16x2:
            return "2xB";
        case DataType::Raw32:
            return "R";
        case DataType::UInt8x4:
            return "4xU8";
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
        case DataType::Bool64:
            return "BL64";
        case DataType::E8M0:
            return "E8M0";
        case DataType::E8M0x4:
            return "4xE8M0";
        case DataType::None:
            return "NA";

        case DataType::Count:;
        }
        return "Invalid";
    }

    std::string toString(MemoryType m)
    {
        switch(m)
        {
        case MemoryType::Global:
            return "Global";
        case MemoryType::LDS:
            return "LDS";
        case MemoryType::AGPR:
            return "AGPR";
        case MemoryType::VGPR:
            return "VGPR";
        case MemoryType::WAVE:
            return "WAVE";
        case MemoryType::WAVE_LDS:
            return "WAVE_LDS";
        case MemoryType::WAVE_SPLIT:
            return "WAVE_SPLIT";
        case MemoryType::WAVE_Direct2LDS:
            return "WAVE_Direct2LDS";
        case MemoryType::WAVE_SWIZZLE:
            return "WAVE_SWIZZLE";
        case MemoryType::Literal:
            return "Literal";
        case MemoryType::None:
            return "None";

        case MemoryType::Count:;
        }
        return "INVALID";
    }

    std::ostream& operator<<(std::ostream& stream, MemoryType const& m)
    {
        return stream << toString(m);
    }

    std::string toString(LayoutType l)
    {
        switch(l)
        {
        case LayoutType::SCRATCH:
            return "SCRATCH";
        case LayoutType::MATRIX_A:
            return "MATRIX_A";
        case LayoutType::MATRIX_B:
            return "MATRIX_B";
        case LayoutType::MATRIX_ACCUMULATOR:
            return "MATRIX_ACCUMULATOR";
        case LayoutType::None:
            return "None";

        case LayoutType::Count:;
        }
        return "INVALID";
    }

    std::ostream& operator<<(std::ostream& stream, LayoutType l)
    {
        return stream << toString(l);
    }

    std::string toString(NaryArgument n)
    {
        switch(n)
        {

        case NaryArgument::DEST:
            return "DEST";
        case NaryArgument::LHS:
            return "LHS";
        case NaryArgument::RHS:
            return "RHS";
        case NaryArgument::LHS_SCALE:
            return "LHS_SCALE";
        case NaryArgument::RHS_SCALE:
            return "RHS_SCALE";

        case NaryArgument::Count:;
        }
        return "Invalid";
    }

    std::ostream& operator<<(std::ostream& stream, NaryArgument n)
    {
        return stream << toString(n);
    }

    std::string toString(PointerType const& p)
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
        return stream << toString(p);
    }

    std::string toString(VariableType const& v)
    {
        return toString(v.pointerType) + ": " + toString(v.dataType);
    }

    int VariableType::registerAlignment(Register::Type         regType,
                                        int                    count,
                                        GPUArchitecture const& gpuArch)
    {

        if(this->pointerType == PointerType::Buffer)
        {
            return 4;
        }

        if(regType == Register::Type::Vector && count > 1)
        {
            return 2;
        }
        else if(regType == Register::Type::Scalar && count > 1)
        {
            return 2;
        }
        return 1;
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
        return stream << toString(v);
    }

    size_t VariableType::getElementSize() const
    {
        switch(pointerType)
        {
        case PointerType::Value:
            // TODO Audit bytes/bits
            return DataTypeInfo::Get(dataType).elementBytes;
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

        if(lhs == DataType::None || rhs == DataType::None)
            return DataType::None;

        if(lhs == DataType::Count || rhs == DataType::Count)
            return DataType::None;

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

        if(lhs.dataType == DataType::Bool32 && rhs.dataType == DataType::Bool)
            return lhs;
        if(rhs.dataType == DataType::Bool32 && lhs.dataType == DataType::Bool)
            return rhs;

        if(lhs.dataType == DataType::Bool64 && rhs.dataType == DataType::Bool)
            return lhs;
        if(rhs.dataType == DataType::Bool64 && lhs.dataType == DataType::Bool)
            return rhs;

        AssertFatal(lhsInfo.isIntegral == rhsInfo.isIntegral,
                    "No automatic promotion between integral and non-integral types",
                    ShowValue(lhs),
                    ShowValue(rhs));

        AssertFatal(lhsInfo.isComplex == rhsInfo.isComplex,
                    "No automatic promotion between complex and non-complex types",
                    ShowValue(lhs),
                    ShowValue(rhs));

        if(lhsInfo.segmentVariableType == rhsInfo.segmentVariableType
           && lhsInfo.packing < rhsInfo.packing)
            return lhs;

        if(lhsInfo.segmentVariableType == rhsInfo.segmentVariableType
           && lhsInfo.packing > rhsInfo.packing)
            return rhs;

        if(lhsInfo.elementBits > rhsInfo.elementBits)
            return lhs;

        if(lhsInfo.elementBits < rhsInfo.elementBits)
            return rhs;

        // TODO Audit bytes/bits
        // Since we promote based on bits (see above), are the proceeding two checks necessary?
        if(lhsInfo.packing < rhsInfo.packing)
            return lhs;

        if(lhsInfo.packing > rhsInfo.packing)
            return rhs;

        if(!rhsInfo.isSigned)
            return rhs;

        return lhs;
    }

    DataTypeInfo::Data::Data()
    {
        registerAllTypeInfo();
    }
    std::map<VariableType, DataTypeInfo, CompareVariableTypesPointersEqual> const&
        DataTypeInfo::Data::data() const
    {
        return m_data;
    }
    std::map<std::string, VariableType> const& DataTypeInfo::Data::typeNames() const
    {
        return m_typeNames;
    }

    template <typename T>
    void DataTypeInfo::Data::registerTypeInfo()
    {
        using T_Info = TypeInfo<T>;

        DataTypeInfo info;

        info.variableType        = T_Info::Var;
        info.segmentVariableType = T_Info::SegmentVariableType;
        info.name                = T_Info::Name();
        info.abbrev              = T_Info::Abbrev();

        info.packing       = T_Info::Packing;
        info.elementBytes  = T_Info::ElementBytes;
        info.elementBits   = T_Info::ElementBits;
        info.alignment     = T_Info::Alignment;
        info.registerCount = T_Info::RegisterCount;

        info.isComplex  = T_Info::IsComplex;
        info.isIntegral = T_Info::IsIntegral;
        info.isSigned   = T_Info::IsSigned;

        addInfoObject(info);
    }

    void DataTypeInfo::Data::registerAllTypeInfo()
    {
        registerTypeInfo<FP8>();
        registerTypeInfo<BF8>();
        registerTypeInfo<FP6>();
        registerTypeInfo<BF6>();
        registerTypeInfo<FP4>();
        registerTypeInfo<FP8x4>();
        registerTypeInfo<BF8x4>();
        registerTypeInfo<FP4x8>();
        registerTypeInfo<FP6x16>();
        registerTypeInfo<BF6x16>();
        registerTypeInfo<BFloat16>();
        registerTypeInfo<Half>();
        registerTypeInfo<Halfx2>();
        registerTypeInfo<BFloat16>();
        registerTypeInfo<BFloat16x2>();
        registerTypeInfo<double>();
        registerTypeInfo<float>();
        registerTypeInfo<std::complex<double>>();
        registerTypeInfo<std::complex<float>>();

        registerTypeInfo<Int8x4>();
        registerTypeInfo<UInt8x4>();
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
        registerTypeInfo<Bool64>();

        registerTypeInfo<PointerLocal>();
        registerTypeInfo<PointerGlobal>();
        registerTypeInfo<Buffer>();

        registerTypeInfo<E8M0>();
        registerTypeInfo<E8M0x4>();
    }

    void DataTypeInfo::Data::addInfoObject(DataTypeInfo const& info)
    {
        m_data[info.variableType] = info;
        m_typeNames[info.name]    = info.variableType;
    }

    DataTypeInfo const& DataTypeInfo::Get(int index)
    {
        return Get(static_cast<DataType>(index));
    }

    DataTypeInfo const& DataTypeInfo::Get(DataType t)
    {
        auto data = Data::getInstance();

        auto iter = data->data().find(t);
        if(iter == data->data().end())
            throw std::runtime_error(concatenate("Invalid data type: ", static_cast<int>(t)));

        return iter->second;
    }

    DataTypeInfo const& DataTypeInfo::Get(VariableType const& v)
    {
        if(v.isPointer())
        {
            VariableType genericPointer(v.pointerType);
            if(genericPointer != v)
                return Get(genericPointer);
        }

        auto data = Data::getInstance();

        auto iter = data->data().find(v);
        AssertFatal(iter != data->data().end(),
                    "Invalid variable type: ",
                    static_cast<int>(v.dataType),
                    v.dataType,
                    " ",
                    static_cast<int>(v.pointerType),
                    v.pointerType);

        return iter->second;
    }

    DataTypeInfo const& DataTypeInfo::Get(std::string const& str)
    {
        auto data = Data::getInstance();

        auto iter = data->typeNames().find(str);
        if(iter == data->typeNames().end())
            throw std::runtime_error(concatenate("Invalid data type: ", str));

        return Get(iter->second);
    }

    std::optional<VariableType> DataTypeInfo::packedVariableType() const
    {
        auto data = Data::getInstance();

        // Finds the reverse mapping
        for(auto const& [key, value] : data->data())
        {
            if(variableType == value.segmentVariableType)
            {
                // This check exists because some variables are its own segmentVariableType
                if(key != value.segmentVariableType)
                {
                    return key;
                }
            }
        }
        return {};
    }

    std::ostream& operator<<(std::ostream& stream, const DataType& t)
    {
        return stream << toString(t);
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
