/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/DataTypes.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>

namespace rocisa
{
    std::string TypeAbbrev(rocisa::DataType d)
    {
        switch(d)
        {
        case rocisa::DataType::Float:
            return "S";
        case rocisa::DataType::Double:
            return "D";
        case rocisa::DataType::ComplexFloat:
            return "C";
        case rocisa::DataType::ComplexDouble:
            return "Z";
        case rocisa::DataType::Half:
            return "H";
        case rocisa::DataType::Int8x4:
            return "4xi8";
        case rocisa::DataType::Int32:
            return "I";
        case rocisa::DataType::Int64:
            return "I64";
        case rocisa::DataType::BFloat16:
            return "B";
        case rocisa::DataType::Int8:
            return "I8";
        case rocisa::DataType::XFloat32:
            return "X";
        case rocisa::DataType::Float8:
            return "F8";
        case rocisa::DataType::BFloat8:
            return "B8";
        case rocisa::DataType::Float8_fnuz:
            return "F8N";
        case rocisa::DataType::BFloat8_fnuz:
            return "B8N";
        case rocisa::DataType::Float8BFloat8:
            return "F8B8";
        case rocisa::DataType::BFloat8Float8:
            return "B8F8";
        case rocisa::DataType::Float8BFloat8_fnuz:
            return "F8B8N";
        case rocisa::DataType::BFloat8Float8_fnuz:
            return "B8F8N";
        case rocisa::DataType::Count:;
        }
        return "Invalid";
    }

    size_t GetElementSize(rocisa::DataType d)
    {
        switch(d)
        {
        case rocisa::DataType::Float:
            return TensileLite::TypeInfo<float>::ElementSize;
        case rocisa::DataType::Double:
            return TensileLite::TypeInfo<double>::ElementSize;
        case rocisa::DataType::ComplexFloat:
            return TensileLite::TypeInfo<std::complex<float>>::ElementSize;
        case rocisa::DataType::ComplexDouble:
            return TensileLite::TypeInfo<std::complex<double>>::ElementSize;
        case rocisa::DataType::Half:
            return TensileLite::TypeInfo<TensileLite::Half>::ElementSize;
        case rocisa::DataType::Int8x4:
            return TensileLite::TypeInfo<TensileLite::Int8x4>::ElementSize;
        case rocisa::DataType::Int32:
            return TensileLite::TypeInfo<int32_t>::ElementSize;
        case rocisa::DataType::Int64:
            return TensileLite::TypeInfo<int64_t>::ElementSize;
        case rocisa::DataType::BFloat16:
            return TensileLite::TypeInfo<TensileLite::BFloat16>::ElementSize;
        case rocisa::DataType::Int8:
            return TensileLite::TypeInfo<int8_t>::ElementSize;
        case rocisa::DataType::XFloat32:
            return TensileLite::TypeInfo<TensileLite::XFloat32>::ElementSize;
        case rocisa::DataType::Float8:
            return TensileLite::TypeInfo<TensileLite::Float8>::ElementSize;
        case rocisa::DataType::BFloat8:
            return TensileLite::TypeInfo<TensileLite::BFloat8>::ElementSize;
        case rocisa::DataType::Float8_fnuz:
            return TensileLite::TypeInfo<TensileLite::Float8_fnuz>::ElementSize;
        case rocisa::DataType::BFloat8_fnuz:
            return TensileLite::TypeInfo<TensileLite::BFloat8_fnuz>::ElementSize;
        case rocisa::DataType::Float8BFloat8:
            return TensileLite::TypeInfo<Float8BFloat8>::ElementSize;
        case rocisa::DataType::BFloat8Float8:
            return TensileLite::TypeInfo<BFloat8Float8>::ElementSize;
        case rocisa::DataType::Float8BFloat8_fnuz:
            return TensileLite::TypeInfo<Float8BFloat8_fnuz>::ElementSize;
        case rocisa::DataType::BFloat8Float8_fnuz:
            return TensileLite::TypeInfo<BFloat8Float8_fnuz>::ElementSize;
        case rocisa::DataType::Count:;
        }
        return 1;
    }

    std::ostream& operator<<(std::ostream& stream, const rocisa::DataType& t)
    {
        return stream << rocisa::toString(t);
    }

    std::istream& operator>>(std::istream& stream, rocisa::DataType& t)
    {
        std::string strValue;
        stream >> strValue;

#if 1
        t = TensileLite::DataTypeInfo::Get(strValue).dataType;

#else

        if(strValue == ToString(rocisa::DataType::Float))
            t = rocisa::DataType::Float;
        else if(strValue == ToString(rocisa::DataType::Double))
            t = rocisa::DataType::Double;
        else if(strValue == ToString(rocisa::DataType::ComplexFloat))
            t = rocisa::DataType::ComplexFloat;
        else if(strValue == ToString(rocisa::DataType::ComplexDouble))
            t = rocisa::DataType::ComplexDouble;
        else if(strValue == ToString(rocisa::DataType::Half))
            t = rocisa::DataType::Half;
        else if(strValue == ToString(rocisa::DataType::Int8x4))
            t = rocisa::DataType::Int8x4;
        else if(strValue == ToString(rocisa::DataType::Int32))
            t = rocisa::DataType::Int32;
        else if(strValue == ToString(rocisa::DataType::Int64))
            t = rocisa::DataType::Int64;
        else if(strValue == ToString(rocisa::DataType::Int8))
            t = rocisa::DataType::Int8;
        else if(strValue == ToString(rocisa::DataType::XFloat32))
            t = rocisa::DataType::XFloat32;
        else if(std::all_of(strValue.begin(), strValue.end(), isdigit))
        {
            int value = atoi(strValue.c_str());
            if(value >= 0 && value < static_cast<int>(rocisa::DataType::Count))
                t = static_cast<DataType>(value);
            else
                throw std::runtime_error(concatenate("Can't convert ", strValue, " to DataType."));
        }
        else
        {
            throw std::runtime_error(concatenate("Can't convert ", strValue, " to DataType."));
        }
#endif

        return stream;
    }
}

namespace TensileLite
{
    std::string ToString(rocisa::DataType d)
    {
        return rocisa::toString(d);
    }

    std::map<rocisa::DataType, DataTypeInfo>* DataTypeInfo::getData()
    {
        static std::map<rocisa::DataType, DataTypeInfo> data;
        return &data;
    }

    std::map<std::string, rocisa::DataType>* DataTypeInfo::getTypeNames()
    {
        static std::map<std::string, rocisa::DataType> typeNames;
        return &typeNames;
    }

    template <typename T>
    void DataTypeInfo::registerTypeInfo()
    {
        using T_Info = TensileLite::TypeInfo<T>;

        DataTypeInfo info;

        info.dataType = T_Info::Enum;
        info.name     = T_Info::Name();
        info.abbrev   = T_Info::Abbrev();

        info.packing     = T_Info::Packing;
        info.elementSize = T_Info::ElementSize;
        info.segmentSize = T_Info::SegmentSize;

        info.isComplex  = T_Info::IsComplex;
        info.isIntegral = T_Info::IsIntegral;

        addInfoObject(info);
    }

    void DataTypeInfo::registerAllTypeInfo()
    {
        registerTypeInfo<float>();
        registerTypeInfo<double>();
        registerTypeInfo<std::complex<float>>();
        registerTypeInfo<std::complex<double>>();
        registerTypeInfo<Half>();
        registerTypeInfo<Int8x4>();
        registerTypeInfo<int32_t>();
        registerTypeInfo<int64_t>();
        registerTypeInfo<BFloat16>();
        registerTypeInfo<int8_t>();
        registerTypeInfo<XFloat32>();
        registerTypeInfo<Float8>();
        registerTypeInfo<BFloat8>();
        registerTypeInfo<Float8_fnuz>();
        registerTypeInfo<BFloat8_fnuz>();
        registerTypeInfo<Float8BFloat8>();
        registerTypeInfo<BFloat8Float8>();
        registerTypeInfo<Float8BFloat8_fnuz>();
        registerTypeInfo<BFloat8Float8_fnuz>();
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
        auto* data      = getData();
        auto* typeNames = getTypeNames();

        data->emplace(info.dataType, info);
        typeNames->emplace(info.name, info.dataType);
    }

    DataTypeInfo const& DataTypeInfo::Get(int index)
    {
        return Get(static_cast<rocisa::DataType>(index));
    }

    DataTypeInfo const& DataTypeInfo::Get(rocisa::DataType t)
    {
        registerAllTypeInfoOnce();

        auto* data = getData();
        auto  iter = data->find(t);
        if(iter == data->end())
            throw std::runtime_error(concatenate("Invalid data type: ", static_cast<int>(t)));

        return iter->second;
    }

    DataTypeInfo const& DataTypeInfo::Get(std::string const& str)
    {
        registerAllTypeInfoOnce();

        auto* typeNames = getTypeNames();
        auto  iter      = typeNames->find(str);
        if(iter == typeNames->end())
            throw std::runtime_error(concatenate("Invalid data type: ", str));

        return Get(iter->second);
    }

    std::string ToString(ConstantVariant d)
    {
        return std::visit(
            [](const auto& cv) {
                using T = std::decay_t<decltype(cv)>;
                if constexpr(std::is_same_v<
                                 T,
                                 std::complex<float>> || std::is_same_v<T, std::complex<double>>)
                    return "(" + std::to_string(cv.real()) + ", " + std::to_string(cv.imag()) + ")";
                else
                    return std::to_string(cv);
            },
            d);
    }

    bool CompareValue(const ConstantVariant& d, double value)
    {
        switch(d.index())
        {
        case static_cast<int>(rocisa::DataType::Float):
            return (*std::get_if<float>(&d)) == float(value);
        case static_cast<int>(rocisa::DataType::Double):
            return (*std::get_if<double>(&d)) == double(value);
        case static_cast<int>(rocisa::DataType::ComplexFloat):
            return (*std::get_if<std::complex<float>>(&d)) == std::complex<float>(value);
        case static_cast<int>(rocisa::DataType::ComplexDouble):
            return (*std::get_if<std::complex<double>>(&d)) == std::complex<double>(value);
        case static_cast<int>(rocisa::DataType::Half):
            return (*std::get_if<Half>(&d)) == Half(value);
        case static_cast<int>(rocisa::DataType::Int32):
            return (*std::get_if<int32_t>(&d)) == int32_t(value);
        case static_cast<int>(rocisa::DataType::Int64):
            return (*std::get_if<int64_t>(&d)) == int64_t(value);
        case static_cast<int>(rocisa::DataType::BFloat16):
            return (*std::get_if<BFloat16>(&d)) == BFloat16(value);
        case static_cast<int>(rocisa::DataType::Int8):
            return (*std::get_if<int8_t>(&d)) == int8_t(value);
        case static_cast<int>(rocisa::DataType::Float8):
            return (*std::get_if<Float8>(&d)) == Float8(static_cast<float>(value));
        case static_cast<int>(rocisa::DataType::BFloat8):
            return (*std::get_if<BFloat8>(&d)) == BFloat8(static_cast<float>(value));
        case static_cast<int>(rocisa::DataType::Float8_fnuz):
            return (*std::get_if<Float8_fnuz>(&d)) == Float8_fnuz(static_cast<float>(value));
        case static_cast<int>(rocisa::DataType::BFloat8_fnuz):
            return (*std::get_if<BFloat8_fnuz>(&d)) == BFloat8_fnuz(static_cast<float>(value));
        default:
            throw std::runtime_error("Unsupported variant cast type.");
        }
    }
} // namespace TensileLite
