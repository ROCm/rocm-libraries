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

#pragma once

#include "Reference.hpp"
#include <iostream>

namespace TensileLite
{
    namespace Client
    {
        template <typename T>
        struct NullComparison
        {
            inline void
                operator()(T referenceValue, T resultValue, size_t elemIndex, size_t elemNumber)
            {
            }

            template <typename... Args>
            inline void before(T value, size_t elemIndex, size_t elemCount)
            {
            }

            inline void inside(T value, size_t elemIndex, size_t elemCount) {}

            template <typename... Args>
            inline void after(T value, size_t elemIndex, size_t elemCount)
            {
            }

            void report() const {}

            bool error() const
            {
                return false;
            }
        };

        template <typename T>
        class PointwiseComparison
        {
        public:
            PointwiseComparison(bool printValids, size_t printMax, bool printReport)
                : m_printValids(printValids)
                , m_printMax(printMax)
                , m_doPrint(printMax > 0)
                , m_printReport(printReport)
            {
            }

            inline void
                operator()(T referenceValue, T resultValue, size_t elemIndex, size_t elemNumber)
            {
                m_values++;
                bool match = AlmostEqual(referenceValue, resultValue);
                if(!match)
                    m_errors++;

                if(!match || m_printValids)
                {
                    if(m_doPrint)
                    {
                        if(m_printed == 0)
                        {
                            std::cout << "Index:  Device | Reference" << std::endl;
                        }

                        if constexpr(std::is_same<int8_t, T>())
                        {
                            std::cout << "[" << (m_printed) << "] "
                                    << " elem=" << elemNumber << " idx=" << elemIndex << ": "
                                    << static_cast<int>(resultValue) << (match ? "==" : "!=")
                                    << static_cast<int>(referenceValue) << std::endl;
                        }
                        else if constexpr(std::is_same<Float8, T>() || std::is_same<BFloat8, T>())
                        {
                            std::cout << "[" << (m_printed) << "] "
                                    << " elem=" << elemNumber << " idx=" << elemIndex << ": "
                                    << static_cast<float>(resultValue) << (match ? "==" : "!=")
                                    << static_cast<float>(referenceValue) << std::endl;
                        }
                        else
                        {
                            std::cout << "[" << (m_printed) << "] "
                                    << " elem=" << elemNumber << " idx=" << elemIndex << ": "
                                    << resultValue << (match ? "==" : "!=")
                                    << referenceValue << std::endl;
                        }

                        m_printed++;

                        if(m_printMax >= 0 && m_printed >= m_printMax)
                            m_doPrint = false;
                    }
                }
            }

            void report()
            {
                if(m_errors && m_printReport)
                {
                    std::cout << "Found " << m_errors << " incorrect values in " << m_values
                              << " total values compared." << std::endl;
                    m_failed = true;
                }
                m_errors  = 0;
                m_values  = 0;
                m_doPrint = true;
                m_printed = 0;
            }

            bool error()
            {
                bool failed = m_failed;
                m_failed    = false;
                return failed;
            }

        private:
            size_t m_errors      = 0;
            size_t m_values      = 0;
            bool   m_printValids = 0;
            size_t m_printMax    = 0;
            size_t m_printed     = 0;
            bool   m_doPrint     = false;
            bool   m_printReport = false;
            bool   m_failed      = false;
        };

        template <typename T>
        struct Magnitude
        {
            inline static T abs(T val)
            {
                return std::abs(val);
            }
        };

        template <typename T>
        struct Magnitude<std::complex<T>>
        {
            inline static T abs(std::complex<T> val)
            {
                return std::abs(val);
            }
        };

        template <>
        struct Magnitude<Half>
        {
            inline static Half abs(Half val)
            {
                return static_cast<Half>(std::abs(static_cast<float>(val)));
            }
        };

        template <typename T>
        class RMSComparison
        {
        public:
            RMSComparison(double threshold, bool printReport)
                : m_threshold(threshold)
                , m_printReport(printReport)
            {
            }

            inline void
                operator()(T referenceValue, T resultValue, size_t elemIndex, size_t elemNumber)
            {
                m_values++;

                using m = Magnitude<T>;

                m_maxReference
                    = std::max(m_maxReference, static_cast<double>(m::abs(referenceValue)));
                m_maxResult = std::max(m_maxResult, static_cast<double>(m::abs(resultValue)));
                auto diff   = m::abs(referenceValue - resultValue);
                m_squareDifference += static_cast<double>(diff * diff);
            }

            inline void report() const
            {
                if(m_printReport)
                {
                    std::cout << "Max reference value: " << m_maxReference
                              << ", max result value: " << m_maxResult << " (" << m_values
                              << " values)" << std::endl;
                    std::cout << "RMS Error: " << errorValue() << " (threshold: " << m_threshold
                              << ")" << std::endl;
                }
            }

            bool error() const
            {
                auto value = errorValue();
                return value > m_threshold;
            }

            double errorValue() const
            {
                double maxMagnitude
                    = std::max({m_maxReference, m_maxResult, std::numeric_limits<double>::min()});
                double denom = std::sqrt(static_cast<double>(m_values)) * maxMagnitude;
                return std::sqrt(m_squareDifference) / denom;
            }

        private:
            size_t m_values      = 0;
            bool   m_printReport = false;

            double m_maxReference     = 0;
            double m_maxResult        = 0;
            double m_squareDifference = 0;
            double m_threshold        = 1e-7;
        };

        template <typename T>
        class InvalidComparison
        {
        public:
            InvalidComparison(size_t printMax, bool printReport)
                : m_printMax(printMax)
                , m_printReport(printReport)
                , m_doPrintBefore(printMax > 0)
                , m_doPrintInside(printMax > 0)
                , m_doPrintAfter(printMax > 0)
            {
            }

            inline void before(T value, size_t elemIndex, size_t elemCount)
            {
                m_checkedBefore++;

                if(!DataInitialization::isBadOutput(value))
                {
                    m_errorsBefore++;

                    if(m_doPrintBefore)
                    {
                        if(m_printedBefore == 0)
                        {
                            std::cout << "Value written before output buffer:" << std::endl;
                            m_printedBefore++;
                        }

                        std::cout << "Index " << elemIndex << " / " << elemCount << ": found "
                                  << value << " instead of "
                                  << DataInitialization::getValue<T, InitMode::BadOutput>()
                                  << std::endl;

                        if(m_printedBefore >= m_printMax)
                            m_doPrintBefore = false;
                    }
                }
            }

            inline void inside(T value, size_t elemIndex, size_t elemCount)
            {
                m_checkedInside++;

                if(!DataInitialization::isBadOutput(value))
                {
                    m_errorsInside++;

                    if(m_doPrintInside)
                    {
                        if(m_printedInside == 0)
                        {
                            std::cout << "Value written inside output buffer, ouside tensor:"
                                      << std::endl;
                            m_printedInside++;
                        }

                        std::cout << "Index " << elemIndex << " / " << elemCount << ": found "
                                  << value << " instead of "
                                  << DataInitialization::getValue<T, InitMode::BadOutput>()
                                  << std::endl;

                        if(m_printedInside >= m_printMax)
                            m_doPrintInside = false;
                    }
                }
            }

            inline void after(T value, size_t elemIndex, size_t elemCount)
            {
                m_checkedAfter++;

                if(!DataInitialization::isBadOutput(value))
                {
                    m_errorsAfter++;

                    if(m_doPrintAfter)
                    {
                        if(m_printedAfter == 0)
                        {
                            std::cout << "Value written after output buffer:" << std::endl;
                            m_printedAfter++;
                        }

                        std::cout << "Index " << elemIndex << " / " << elemCount << ": found "
                                  << value << " instead of "
                                  << DataInitialization::getValue<T, InitMode::BadOutput>()
                                  << std::endl;

                        if(m_printedAfter >= m_printMax)
                            m_doPrintAfter = false;
                    }
                }
            }

            void report() const
            {
                if(m_printReport
                   && (m_checkedBefore > 0 || m_checkedInside > 0 || m_checkedAfter > 0))
                {
                    std::cout << "BOUNDS CHECK:" << std::endl;
                    std::cout << "Before buffer: found " << m_errorsBefore << " errors in "
                              << m_checkedBefore << " values checked." << std::endl;
                    std::cout << "Inside buffer: found " << m_errorsInside << " errors in "
                              << m_checkedInside << " values checked." << std::endl;
                    std::cout << "After buffer: found " << m_errorsAfter << " errors in "
                              << m_checkedAfter << " values checked." << std::endl;
                }
            }

            bool error() const
            {
                return m_errorsBefore != 0 || m_errorsInside != 0 || m_errorsAfter != 0;
            }

        private:
            size_t m_printMax    = 0;
            bool   m_printReport = false;

            size_t m_checkedBefore = 0;
            size_t m_checkedInside = 0;
            size_t m_checkedAfter  = 0;

            size_t m_errorsBefore = 0;
            size_t m_errorsInside = 0;
            size_t m_errorsAfter  = 0;

            size_t m_printedBefore = 0;
            size_t m_printedInside = 0;
            size_t m_printedAfter  = 0;

            bool m_doPrintBefore = false;
            bool m_doPrintInside = false;
            bool m_doPrintAfter  = false;
        };

    } // namespace Client
} // namespace TensileLite
