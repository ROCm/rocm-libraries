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

#include <Tensile/ScalarValueTypes.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>

namespace TensileLite
{
    std::string ToString(ScalarValue d)
    {
        switch(d)
        {
        case ScalarValue::Any:
            return "Any";
        case ScalarValue::One:
            return "1";
        case ScalarValue::NegativeOne:
            return "-1";

        case ScalarValue::Count:
        default:;
        }
        return "Invalid";
    }

    std::map<ScalarValue, ScalarValueTypeInfo>* ScalarValueTypeInfo::getData()
    {
        static std::map<ScalarValue, ScalarValueTypeInfo> data;
        return &data;
    }

    std::map<std::string, ScalarValue>* ScalarValueTypeInfo::getTypeNames()
    {
        static std::map<std::string, ScalarValue> typeNames;
        return &typeNames;
    }

    template <ScalarValue T>
    void ScalarValueTypeInfo::registerTypeInfo()
    {
        using T_Info = ScalarValueInfo<T>;

        ScalarValueTypeInfo info;

        info.m_value = T_Info::Enum;
        info.name    = T_Info::Name();

        addInfoObject(info);
    }

    void ScalarValueTypeInfo::registerAllTypeInfo()
    {
        registerTypeInfo<ScalarValue::Any>();
        registerTypeInfo<ScalarValue::One>();
        registerTypeInfo<ScalarValue::NegativeOne>();
    }

    void ScalarValueTypeInfo::registerAllTypeInfoOnce()
    {
        static int call_once = (registerAllTypeInfo(), 0);

        // Use the variable to quiet the compiler.
        if(call_once)
            return;
    }

    void ScalarValueTypeInfo::addInfoObject(ScalarValueTypeInfo const& info)
    {
        auto toLower = [](std::string tmp) {
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
            return tmp;
        };

        auto* data      = getData();
        auto* typeNames = getTypeNames();

        data->emplace(info.m_value, info);

        // Add some flexibility to names registry. Accept
        // lower case versions of the strings
        typeNames->emplace(info.name, info.m_value);
        typeNames->emplace(toLower(info.name), info.m_value);
    }

    ScalarValueTypeInfo const& ScalarValueTypeInfo::Get(int index)
    {
        return Get(static_cast<ScalarValue>(index));
    }

    ScalarValueTypeInfo const& ScalarValueTypeInfo::Get(ScalarValue t)
    {
        registerAllTypeInfoOnce();

        auto* data = getData();
        auto  iter = data->find(t);
        if(iter == data->end())
            throw std::runtime_error(concatenate("Invalid scalar value: ", static_cast<int>(t)));

        return iter->second;
    }

    ScalarValueTypeInfo const& ScalarValueTypeInfo::Get(std::string const& str)
    {
        registerAllTypeInfoOnce();

        auto* typeNames = getTypeNames();
        auto  iter      = typeNames->find(str);
        if(iter == typeNames->end())
            throw std::runtime_error(concatenate("Invalid scalar value: ", str));

        return Get(iter->second);
    }

    std::ostream& operator<<(std::ostream& stream, const ScalarValue& t)
    {
        return stream << ToString(t);
    }

    std::istream& operator>>(std::istream& stream, ScalarValue& t)
    {
        std::string strValue;
        stream >> strValue;

        t = ScalarValueTypeInfo::Get(strValue).m_value;

        return stream;
    }
} // namespace TensileLite
