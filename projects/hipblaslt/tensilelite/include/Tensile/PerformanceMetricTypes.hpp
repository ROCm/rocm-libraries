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

#pragma once

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace TensileLite
{
    /**
 * \ingroup Tensile
 * \defgroup PerformanceMetrics Performance metric type Info
 *
 * @brief Definitions and metadata on supported performance metric types.
 */

    /**
 * \ingroup PerformanceMetrics
 * @{
 */

    /**
 * Experimental options
 */
    enum class ExperimentalOption : int
    {
        None    = 0,
        MLP     = 1,
        StreamK = 2,
        Count
    };

    /**
 * Performance Metric
 */
    enum class PerformanceMetric : int
    {
        Auto,
        CUEfficiency,
        DeviceEfficiency,
        ExperimentalMLP,
        ExperimentalStreamK,
        Count
    };

    std::string   ToString(PerformanceMetric d);
    std::string   TypeAbbrev(PerformanceMetric d);
    std::ostream& operator<<(std::ostream& stream, PerformanceMetric const& t);
    std::istream& operator>>(std::istream& stream, PerformanceMetric& t);

    /**
 * \ingroup PerformanceMetrics
 * \brief Runtime accessible performance metric type metadata
 */
    struct PerformanceMetricTypeInfo
    {
        static PerformanceMetricTypeInfo const& Get(int index);
        static PerformanceMetricTypeInfo const& Get(PerformanceMetric t);
        static PerformanceMetricTypeInfo const& Get(std::string const& str);

        PerformanceMetric m_performanceMetric;
        std::string       name;
        std::string       abbrev;

    private:
        static void registerAllTypeInfo();
        static void registerAllTypeInfoOnce();

        template <PerformanceMetric T_Enum>
        static void registerTypeInfo();

        static void addInfoObject(PerformanceMetricTypeInfo const& info);

        static std::map<PerformanceMetric, PerformanceMetricTypeInfo>* getData();
        static std::map<std::string, PerformanceMetric>*               getTypeNames();
    };

    /**
 * \ingroup PerformanceMetrics
 * \brief Compile-time accessible performance metric type metadata.
 */
    template <PerformanceMetric T_Enum>
    struct PerformanceMetricInfo;

    template <PerformanceMetric T_Enum>
    struct BasePerformanceMetricInfo
    {
        constexpr static PerformanceMetric Enum = T_Enum;

        static inline std::string Name()
        {
            return ToString(Enum);
        }
        static inline std::string Abbrev()
        {
            return TypeAbbrev(Enum);
        }
    };

    template <PerformanceMetric T_Enum>
    constexpr PerformanceMetric BasePerformanceMetricInfo<T_Enum>::Enum;

    template <>
    struct PerformanceMetricInfo<PerformanceMetric::Auto>
        : public BasePerformanceMetricInfo<PerformanceMetric::Auto>
    {
    };
    template <>
    struct PerformanceMetricInfo<PerformanceMetric::CUEfficiency>
        : public BasePerformanceMetricInfo<PerformanceMetric::CUEfficiency>
    {
    };
    template <>
    struct PerformanceMetricInfo<PerformanceMetric::DeviceEfficiency>
        : public BasePerformanceMetricInfo<PerformanceMetric::DeviceEfficiency>
    {
    };
    template <>
    struct PerformanceMetricInfo<PerformanceMetric::ExperimentalMLP>
        : public BasePerformanceMetricInfo<PerformanceMetric::ExperimentalMLP>
    {
    };
    template <>
    struct PerformanceMetricInfo<PerformanceMetric::ExperimentalStreamK>
        : public BasePerformanceMetricInfo<PerformanceMetric::ExperimentalStreamK>
    {
    };

    /**
 * @}
 */
} // namespace TensileLite

namespace std
{
    template <>
    struct hash<TensileLite::PerformanceMetric>
    {
        inline size_t operator()(TensileLite::PerformanceMetric const& val) const
        {
            return hash<int>()(static_cast<int>(val));
        }
    };
} // namespace std
