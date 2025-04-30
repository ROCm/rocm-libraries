/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

namespace TensileLite
{
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
        case DataType::Int8x4:
            return "Int8x4";
        case DataType::Int32:
            return "Int32";
        case DataType::BFloat16:
            return "BFloat16";
        case DataType::Int8:
            return "Int8";
        case DataType::Float8:
            return "Float8";
        case DataType::BFloat8:
            return "BFloat8";
        case DataType::Float8_fnuz:
            return "Float8_fnuz";
        case DataType::BFloat8_fnuz:
            return "BFloat8_fnuz";
        case DataType::XFloat32:
            return "XFloat32";
        case DataType::Float8BFloat8:
            return "Float8BFloat8";
        case DataType::BFloat8Float8:
            return "BFloat8Float8";
        case DataType::Float8BFloat8_fnuz:
            return "Float8BFloat8_fnuz";
        case DataType::BFloat8Float8_fnuz:
            return "BFloat8Float8_fnuz";
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
        case DataType::Int8x4:
            return "4xi8";
        case DataType::Int32:
            return "I";
        case DataType::BFloat16:
            return "B";
        case DataType::Int8:
            return "I8";
        case DataType::Float8:
            return "F8";
        case DataType::BFloat8:
            return "B8";
        case DataType::Float8_fnuz:
            return "F8N";
        case DataType::BFloat8_fnuz:
            return "B8N";
        case DataType::XFloat32:
            return "X";
        case DataType::Float8BFloat8:
            return "F8B8";
        case DataType::BFloat8Float8:
            return "B8F8";
        case DataType::Float8BFloat8_fnuz:
            return "F8B8N";
        case DataType::BFloat8Float8_fnuz:
            return "B8F8N";
        case DataType::Count:;
        }
        return "Invalid";
    }

    size_t GetElementSize(DataType d)
    {
        switch(d)
        {
        case DataType::Float:
            return TypeInfo<float>::ElementSize;
        case DataType::Double:
            return TypeInfo<double>::ElementSize;
        case DataType::ComplexFloat:
            return TypeInfo<std::complex<float>>::ElementSize;
        case DataType::ComplexDouble:
            return TypeInfo<std::complex<double>>::ElementSize;
        case DataType::Half:
            return TypeInfo<Half>::ElementSize;
        case DataType::Int8x4:
            return TypeInfo<Int8x4>::ElementSize;
        case DataType::Int32:
            return TypeInfo<int32_t>::ElementSize;
        case DataType::BFloat16:
            return TypeInfo<BFloat16>::ElementSize;
        case DataType::Int8:
            return TypeInfo<int8_t>::ElementSize;
        case DataType::Float8:
            return TypeInfo<Float8>::ElementSize;
        case DataType::BFloat8:
            return TypeInfo<BFloat8>::ElementSize;
        case DataType::Float8_fnuz:
            return TypeInfo<Float8_fnuz>::ElementSize;
        case DataType::BFloat8_fnuz:
            return TypeInfo<BFloat8_fnuz>::ElementSize;
        case DataType::XFloat32:
            return TypeInfo<XFloat32>::ElementSize;
        case DataType::Float8BFloat8:
            return TypeInfo<Float8BFloat8>::ElementSize;
        case DataType::BFloat8Float8:
            return TypeInfo<BFloat8Float8>::ElementSize;
        case DataType::Float8BFloat8_fnuz:
            return TypeInfo<Float8BFloat8_fnuz>::ElementSize;
        case DataType::BFloat8Float8_fnuz:
            return TypeInfo<BFloat8Float8_fnuz>::ElementSize;
        case DataType::Count:;
        }
        return 1;
    }

    std::map<DataType, DataTypeInfo>* DataTypeInfo::getData()
    {
        static std::map<DataType, DataTypeInfo> data;
        return &data;
    }

    std::map<std::string, DataType>* DataTypeInfo::getTypeNames()
    {
        static std::map<std::string, DataType> typeNames;
        return &typeNames;
    }

    template <typename T>
    void DataTypeInfo::registerTypeInfo()
    {
        using T_Info = TypeInfo<T>;

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
        registerTypeInfo<BFloat16>();
        registerTypeInfo<int8_t>();
        registerTypeInfo<Float8>();
        registerTypeInfo<BFloat8>();
        registerTypeInfo<Float8_fnuz>();
        registerTypeInfo<BFloat8_fnuz>();
        registerTypeInfo<XFloat32>();
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
        return Get(static_cast<DataType>(index));
    }

    DataTypeInfo const& DataTypeInfo::Get(DataType t)
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

    std::ostream& operator<<(std::ostream& stream, const DataType& t)
    {
        return stream << ToString(t);
    }

    std::istream& operator>>(std::istream& stream, DataType& t)
    {
        std::string strValue;
        stream >> strValue;

#if 1
        t = DataTypeInfo::Get(strValue).dataType;

#else

        if(strValue == ToString(DataType::Float))
            t = DataType::Float;
        else if(strValue == ToString(DataType::Double))
            t = DataType::Double;
        else if(strValue == ToString(DataType::ComplexFloat))
            t = DataType::ComplexFloat;
        else if(strValue == ToString(DataType::ComplexDouble))
            t = DataType::ComplexDouble;
        else if(strValue == ToString(DataType::Half))
            t = DataType::Half;
        else if(strValue == ToString(DataType::Int8x4))
            t = DataType::Int8x4;
        else if(strValue == ToString(DataType::Int32))
            t = DataType::Int32;
        else if(strValue == ToString(DataType::Int8))
            t = DataType::Int8;
        else if(strValue == ToString(DataType::XFloat32))
            t = DataType::XFloat32;
        else if(std::all_of(strValue.begin(), strValue.end(), isdigit))
        {
            int value = atoi(strValue.c_str());
            if(value >= 0 && value < static_cast<int>(DataType::Count))
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
        case static_cast<int>(DataType::Float):
            return (*std::get_if<float>(&d)) == float(value);
        case static_cast<int>(DataType::Double):
            return (*std::get_if<double>(&d)) == double(value);
        case static_cast<int>(DataType::ComplexFloat):
            return (*std::get_if<std::complex<float>>(&d)) == std::complex<float>(value);
        case static_cast<int>(DataType::ComplexDouble):
            return (*std::get_if<std::complex<double>>(&d)) == std::complex<double>(value);
        case static_cast<int>(DataType::Half):
            return (*std::get_if<Half>(&d)) == Half(value);
        case static_cast<int>(DataType::Int32):
            return (*std::get_if<int32_t>(&d)) == int32_t(value);
        case static_cast<int>(DataType::BFloat16):
            return (*std::get_if<BFloat16>(&d)) == BFloat16(value);
        case static_cast<int>(DataType::Int8):
            return (*std::get_if<int8_t>(&d)) == int8_t(value);
        case static_cast<int>(DataType::Float8):
            return (*std::get_if<Float8>(&d)) == Float8(static_cast<float>(value));
        case static_cast<int>(DataType::BFloat8):
            return (*std::get_if<BFloat8>(&d)) == BFloat8(static_cast<float>(value));
        case static_cast<int>(DataType::Float8_fnuz):
            return (*std::get_if<Float8_fnuz>(&d)) == Float8_fnuz(static_cast<float>(value));
        case static_cast<int>(DataType::BFloat8_fnuz):
            return (*std::get_if<BFloat8_fnuz>(&d)) == BFloat8_fnuz(static_cast<float>(value));
        default:
            throw std::runtime_error("Unsupported variant cast type.");
        }
    }
} // namespace TensileLite
