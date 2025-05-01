/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/ContractionSolution.hpp>

#include "ResultReporter.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include <boost/program_options.hpp>
#include <hip/hip_runtime.h>

namespace Tensile
{
    namespace Client
    {
        namespace po = boost::program_options;

        class PerformanceReporter : public ResultReporter
        {
        public:
            static std::shared_ptr<PerformanceReporter> Default(po::variables_map const& args);

            PerformanceReporter(int    deviceIndex,
                                double l2ReadHits,
                                double l2WriteHits,
                                double l2ReadBwMultiplier,
                                double readEff);

            virtual void reportValue_int(std::string const& key, int64_t value) override;

            virtual void reportValue_uint(std::string const& key, uint64_t value) override;

            virtual void reportValue_double(std::string const& key, double value) override;

            virtual void preSolution(ContractionSolution const& solution) override;

            virtual void preProblem(ContractionProblem const& problem) override;

            virtual void postSolution() override;

            void setPerfModel(double l2ReadHits,
                              double l2WriteHits,
                              double l2ReadBwMul,
                              double readEff);
            void setNumCUs();
            void setMemoryBusWidth();
            void setClockMhz(double value);
            void setMemClockMhz(double value);
            void setMemBandwidthMBps();

            int    getNumCUs();
            int    getMagicNum();
            double getMemClockMhz();
            double getClockMhz();
            double getL2ReadBwMultiplier();
            double getL2ReadHits();
            double getL2WriteHits();
            double getReadEff();
            double getMemBandwidthMBps();

            template <typename T>
            void         reportValue_numeric(std::string const& key, T value);
            virtual void reportValue_string(std::string const& key,
                                            std::string const& value) override;
            virtual void reportValue_sizes(std::string const&         key,
                                           std::vector<size_t> const& value) override;
            void         finalizeReport() override;

        protected:
            hipDeviceProp_t m_props;
            double          m_clockMhz    = std::numeric_limits<double>::quiet_NaN();
            double          m_memClockMhz = std::numeric_limits<double>::quiet_NaN();
            double          m_gFlops      = std::numeric_limits<double>::quiet_NaN();
            int             m_numCUs;
            int             m_memoryBusWidth;
            bool            m_deviceProps = false;
            double          m_l2ReadHits;
            double          m_l2WriteHits;
            double          m_l2ReadBwMul;
            double          m_readEff;
            double          m_memBandwidthMBps;
        };
    } // namespace Client
} // namespace Tensile
