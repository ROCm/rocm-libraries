/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/ArithmeticUnitTypes.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>

namespace Tensile
{
    std::map<ArithmeticUnit, ArithmeticUnitTypeInfo> ArithmeticUnitTypeInfo::data;
    std::map<std::string, ArithmeticUnit>            ArithmeticUnitTypeInfo::typeNames;

    std::string ToString(ArithmeticUnit d)
    {
        switch(d)
        {
        case ArithmeticUnit::Any:
            return "Any";
        case ArithmeticUnit::MFMA:
            return "MFMA";
        case ArithmeticUnit::VALU:
            return "VALU";

        case ArithmeticUnit::Count:
        default:;
        }
        return "Invalid";
    }

    template <ArithmeticUnit T>
    void ArithmeticUnitTypeInfo::registerTypeInfo()
    {
        using T_Info = ArithmeticUnitInfo<T>;

        ArithmeticUnitTypeInfo info;

        info.m_arithmeticUnit = T_Info::Enum;
        info.name             = T_Info::Name();

        addInfoObject(info);
    }

    void ArithmeticUnitTypeInfo::registerAllTypeInfo()
    {
        registerTypeInfo<ArithmeticUnit::Any>();
        registerTypeInfo<ArithmeticUnit::MFMA>();
        registerTypeInfo<ArithmeticUnit::VALU>();
    }

    void ArithmeticUnitTypeInfo::registerAllTypeInfoOnce()
    {
        static int call_once = (registerAllTypeInfo(), 0);

        // Use the variable to quiet the compiler.
        if(call_once)
            return;
    }

    void ArithmeticUnitTypeInfo::addInfoObject(ArithmeticUnitTypeInfo const& info)
    {
        auto toLower = [](std::string tmp) {
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
            return tmp;
        };

        data[info.m_arithmeticUnit] = info;

        // Add some flexibility to names registry. Accept
        // lower case versions of the strings
        typeNames[info.name]          = info.m_arithmeticUnit;
        typeNames[toLower(info.name)] = info.m_arithmeticUnit;
    }

    ArithmeticUnitTypeInfo const& ArithmeticUnitTypeInfo::Get(int index)
    {
        return Get(static_cast<ArithmeticUnit>(index));
    }

    ArithmeticUnitTypeInfo const& ArithmeticUnitTypeInfo::Get(ArithmeticUnit t)
    {
        registerAllTypeInfoOnce();

        auto iter = data.find(t);
        if(iter == data.end())
            throw std::runtime_error(concatenate("Invalid arithmetic unit: ", static_cast<int>(t)));

        return iter->second;
    }

    ArithmeticUnitTypeInfo const& ArithmeticUnitTypeInfo::Get(std::string const& str)
    {
        registerAllTypeInfoOnce();

        auto iter = typeNames.find(str);
        if(iter == typeNames.end())
            throw std::runtime_error(concatenate("Invalid arithmetic unit: ", str));

        return Get(iter->second);
    }

    std::ostream& operator<<(std::ostream& stream, const ArithmeticUnit& t)
    {
        return stream << ToString(t);
    }

    std::istream& operator>>(std::istream& stream, ArithmeticUnit& t)
    {
        std::string strValue;
        stream >> strValue;

        t = ArithmeticUnitTypeInfo::Get(strValue).m_arithmeticUnit;

        return stream;
    }
} // namespace Tensile
