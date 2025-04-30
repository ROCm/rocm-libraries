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

#include <Tensile/PerformanceMetricTypes.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>

namespace TensileLite
{
    std::string ToString(PerformanceMetric d)
    {
        switch(d)
        {
        case PerformanceMetric::Auto:
            return "Auto";
        case PerformanceMetric::CUEfficiency:
            return "CUEfficiency";
        case PerformanceMetric::DeviceEfficiency:
            return "DeviceEfficiency";
        case PerformanceMetric::ExperimentalDTree:
            return "ExperimentalDTree";
        case PerformanceMetric::ExperimentalStreamK:
            return "ExperimentalStreamK";

        case PerformanceMetric::Count:
        default:;
        }
        return "Invalid";
    }

    std::string TypeAbbrev(PerformanceMetric d)
    {
        switch(d)
        {
        case PerformanceMetric::Auto:
            return "Auto";
        case PerformanceMetric::CUEfficiency:
            return "CUEff";
        case PerformanceMetric::DeviceEfficiency:
            return "DvEff";
        case PerformanceMetric::ExperimentalDTree:
            return "DTree";
        case PerformanceMetric::ExperimentalStreamK:
            return "StreamK";

        case PerformanceMetric::Count:
        default:;
        }
        return "Invalid";
    }

    std::map<PerformanceMetric, PerformanceMetricTypeInfo>* PerformanceMetricTypeInfo::getData()
    {
        static std::map<PerformanceMetric, PerformanceMetricTypeInfo> data;
        return &data;
    }

    std::map<std::string, PerformanceMetric>* PerformanceMetricTypeInfo::getTypeNames()
    {
        static std::map<std::string, PerformanceMetric> typeNames;
        return &typeNames;
    }

    template <PerformanceMetric T>
    void PerformanceMetricTypeInfo::registerTypeInfo()
    {
        using T_Info = PerformanceMetricInfo<T>;

        PerformanceMetricTypeInfo info;

        info.m_performanceMetric = T_Info::Enum;
        info.name                = T_Info::Name();
        info.abbrev              = T_Info::Abbrev();

        addInfoObject(info);
    }

    void PerformanceMetricTypeInfo::registerAllTypeInfo()
    {
        registerTypeInfo<PerformanceMetric::Auto>();
        registerTypeInfo<PerformanceMetric::CUEfficiency>();
        registerTypeInfo<PerformanceMetric::DeviceEfficiency>();
        registerTypeInfo<PerformanceMetric::ExperimentalDTree>();
        registerTypeInfo<PerformanceMetric::ExperimentalStreamK>();
    }

    void PerformanceMetricTypeInfo::registerAllTypeInfoOnce()
    {
        static int call_once = (registerAllTypeInfo(), 0);

        // Use the variable to quiet the compiler.
        if(call_once)
            return;
    }

    void PerformanceMetricTypeInfo::addInfoObject(PerformanceMetricTypeInfo const& info)
    {
        auto toLower = [](std::string tmp) {
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
            return tmp;
        };

        auto* data      = getData();
        auto* typeNames = getTypeNames();

        data->emplace(info.m_performanceMetric, info);

        // Add some flexibility to names registry. Accept abbreviations and
        // lower case versions of the strings
        typeNames->emplace(info.name, info.m_performanceMetric);
        typeNames->emplace(toLower(info.name), info.m_performanceMetric);
        typeNames->emplace(info.abbrev, info.m_performanceMetric);
        typeNames->emplace(toLower(info.abbrev), info.m_performanceMetric);
    }

    PerformanceMetricTypeInfo const& PerformanceMetricTypeInfo::Get(int index)
    {
        return Get(static_cast<PerformanceMetric>(index));
    }

    PerformanceMetricTypeInfo const& PerformanceMetricTypeInfo::Get(PerformanceMetric t)
    {
        registerAllTypeInfoOnce();

        auto* data = getData();
        auto  iter = data->find(t);
        if(iter == data->end())
            throw std::runtime_error(
                concatenate("Invalid performance metric: ", static_cast<int>(t)));

        return iter->second;
    }

    PerformanceMetricTypeInfo const& PerformanceMetricTypeInfo::Get(std::string const& str)
    {
        registerAllTypeInfoOnce();

        auto* typeNames = getTypeNames();
        auto  iter      = typeNames->find(str);
        if(iter == typeNames->end())
            throw std::runtime_error(concatenate("Invalid performance metric: ", str));

        return Get(iter->second);
    }

    std::ostream& operator<<(std::ostream& stream, const PerformanceMetric& t)
    {
        return stream << ToString(t);
    }

    std::istream& operator>>(std::istream& stream, PerformanceMetric& t)
    {
        std::string strValue;
        stream >> strValue;

        t = PerformanceMetricTypeInfo::Get(strValue).m_performanceMetric;

        return stream;
    }
} // namespace TensileLite
