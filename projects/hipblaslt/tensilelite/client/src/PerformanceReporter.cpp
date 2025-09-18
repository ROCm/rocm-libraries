/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "PerformanceReporter.hpp"
#include "Tensile/hip/HipUtils.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>

namespace TensileLite
{
    namespace Client
    {
        std::shared_ptr<PerformanceReporter>
            PerformanceReporter::Default(po::variables_map const& args)
        {
            int    deviceIndex        = args["device-idx"].as<int>();
            double l2ReadHits         = args["perf-l2-read-hits"].as<double>();
            double l2WriteHits        = args["perf-l2-write-hits"].as<double>();
            double l2ReadBwMultiplier = args["perf-l2-read-bw-mul"].as<double>();
            double readEff            = args["perf-read-efficiency"].as<double>();

            return std::make_shared<PerformanceReporter>(
                deviceIndex, l2ReadHits, l2WriteHits, l2ReadBwMultiplier, readEff);
        }

        PerformanceReporter::PerformanceReporter(int    deviceIndex,
                                                 double l2ReadHits,
                                                 double l2WriteHits,
                                                 double l2ReadBwMultiplier,
                                                 double readEff)
        {
            HIP_CHECK_EXC(hipGetDeviceProperties(&m_props, deviceIndex));
#if HIP_VERSION >= 50220730
            int hip_version;
            HIP_CHECK_EXC(hipRuntimeGetVersion(&hip_version));
            if(hip_version >= 50220730)
            {
                HIP_CHECK_EXC(hipDeviceGetAttribute(&m_props.multiProcessorCount,
                                                    hipDeviceAttributePhysicalMultiProcessorCount,
                                                    deviceIndex));
            }
#endif
            setNumCUs();
            setMemoryBusWidth();
            setPerfModel(l2ReadHits, l2WriteHits, l2ReadBwMultiplier, readEff);
            m_deviceProps = true;

            perf.l2ReadHitRate  = getL2ReadHits();
            perf.l2WriteHitRate = getL2WriteHits();
            perf.l2ReadBwMul    = getL2ReadBwMultiplier();
            perf.readEff        = getReadEff();
            perf.CUs            = getNumCUs();
        }

        void PerformanceReporter::reportValue_uint(std::string const& key, uint64_t value)
        {
            reportValue_numeric(key, value);
        }

        void PerformanceReporter::reportValue_double(std::string const& key, double value)
        {
            reportValue_numeric(key, value);
        }

        template <typename T>
        void PerformanceReporter::reportValue_numeric(std::string const& key, T value)
        {
            if(key == ResultKey::ClockRateSys && m_deviceProps)
            {
                setClockMhz(value);
            }
            if(key == ResultKey::ClockRateMem && m_deviceProps)
            {
                setMemClockMhz(value);
            }
        }

        void PerformanceReporter::setClockMhz(double value)
        {
            m_clockMhz = value;
            perf.clock = getClockMhz();
        }

        void PerformanceReporter::setMemClockMhz(double value)
        {
            m_memClockMhz = value;
            perf.memClock = getMemClockMhz();
            setMemBandwidthMBps();
        }

        void PerformanceReporter::setMemBandwidthMBps()
        {
            m_memBandwidthMBps    = m_memoryBusWidth * m_memClockMhz;
            perf.memBandwidthMBps = getMemBandwidthMBps();
        }

        void PerformanceReporter::postSolution()
        {
            m_clockMhz         = std::numeric_limits<double>::quiet_NaN();
            m_memClockMhz      = std::numeric_limits<double>::quiet_NaN();
            m_gFlops           = std::numeric_limits<double>::quiet_NaN();
            m_memBandwidthMBps = std::numeric_limits<double>::quiet_NaN();
        }

        void PerformanceReporter::setPerfModel(double l2ReadHits,
                                               double l2WriteHits,
                                               double l2ReadBwMultiplier,
                                               double readEff)
        {
            m_l2ReadHits  = l2ReadHits;
            m_l2WriteHits = l2WriteHits;
            m_l2ReadBwMul = l2ReadBwMultiplier;
            m_readEff     = readEff;
        }

        void PerformanceReporter::setNumCUs()
        {
            m_numCUs = m_props.multiProcessorCount;
        }

        void PerformanceReporter::setMemoryBusWidth()
        {
            m_memoryBusWidth = m_props.memoryBusWidth / 1024;
        }

        int PerformanceReporter::getNumCUs()
        {
            return m_numCUs;
        }
        double PerformanceReporter::getMemClockMhz()
        {
            return m_memClockMhz;
        }
        double PerformanceReporter::getClockMhz()
        {
            return m_clockMhz;
        }
        double PerformanceReporter::getL2ReadBwMultiplier()
        {
            return m_l2ReadBwMul;
        }
        double PerformanceReporter::getL2ReadHits()
        {
            return m_l2ReadHits;
        }
        double PerformanceReporter::getL2WriteHits()
        {
            return m_l2WriteHits;
        }
        double PerformanceReporter::getReadEff()
        {
            return m_readEff;
        }
        double PerformanceReporter::getMemBandwidthMBps()
        {
            return m_memBandwidthMBps;
        }

        void PerformanceReporter::reportValue_int(std::string const& key, int64_t value) {}
        void PerformanceReporter::reportValue_string(std::string const& key,
                                                     std::string const& value)
        {
        }
        void PerformanceReporter::reportValue_sizes(std::string const&         key,
                                                    std::vector<size_t> const& value)
        {
        }
        void PerformanceReporter::reportValue_vecOfSizes(
            std::string const& key, std::vector<std::vector<size_t>> const& value)
        {
        }
        void PerformanceReporter::preProblem(ContractionProblem* const problem) {}
        void PerformanceReporter::preSolution(ContractionSolution const& solution) {}
        void PerformanceReporter::finalizeReport() {}

    } // namespace Client
} // namespace TensileLite
