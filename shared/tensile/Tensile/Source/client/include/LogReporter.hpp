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

#pragma once

#include "CSVStackFile.hpp"
#include "ResultReporter.hpp"

#include <cstddef>
#include <string>
#include <unordered_set>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace Tensile
{
    namespace Client
    {
        class LogReporter : public ResultReporter
        {
        public:
            LogReporter(LogLevel                           level,
                        std::initializer_list<const char*> keys,
                        std::ostream&                      stream,
                        bool                               dumpTensors)
                : m_level(level)
                , m_stream(stream)
                , m_csvOutput(stream, ",")
                , m_dumpTensors(dumpTensors)
            {
                for(auto const& key : keys)
                    m_csvOutput.setHeaderForKey(key, key);
            }

            LogReporter(LogLevel                           level,
                        std::initializer_list<std::string> keys,
                        std::ostream&                      stream,
                        bool                               dumpTensors)
                : m_level(level)
                , m_stream(stream)
                , m_csvOutput(stream, ",")
                , m_dumpTensors(dumpTensors)
            {
                for(auto const& key : keys)
                    m_csvOutput.setHeaderForKey(key, key);
            }

            LogReporter(LogLevel                           level,
                        std::initializer_list<std::string> keys,
                        std::shared_ptr<std::ostream>      stream,
                        bool                               dumpTensors)
                : m_level(level)
                , m_stream(*stream)
                , m_ownedStream(stream)
                , m_csvOutput(stream, ",")
                , m_dumpTensors(dumpTensors)
            {
                for(auto const& key : keys)
                    m_csvOutput.setHeaderForKey(key, key);
            }

            template <typename Stream>
            static std::shared_ptr<LogReporter> Default(po::variables_map const& args,
                                                        Stream&                  stream,
                                                        LogLevel level = LogLevel::Count)
            {
                bool dumpTensors = args["dump-tensors"].as<bool>();
                using namespace ResultKey;
                if(level == LogLevel::Count)
                    level = args["log-level"].as<LogLevel>();
                std::cout << "Log level: " << level << std::endl;

                PerformanceMetric metric = args["performance-metric"].as<PerformanceMetric>();
                // Default to 'DeviceEfficiency' benchmarking if CUEfficiency not specified
                const std::string perfUnit
                    = (metric == PerformanceMetric::CUEfficiency ? SpeedGFlopsPerCu : SpeedGFlops);

                return std::shared_ptr<LogReporter>(new LogReporter(level,
                                                                    {BenchmarkRunNumber,
                                                                     ProblemProgress,
                                                                     SolutionProgress,
                                                                     OperationIdentifier,
                                                                     ProblemSizes,
                                                                     SolutionName,
                                                                     Validation,
                                                                     TimeUS,
                                                                     perfUnit,
                                                                     Empty,
                                                                     TotalGranularity,
                                                                     TilesPerCu,
                                                                     NumCus,
                                                                     Tile0Granularity,
                                                                     Tile1Granularity,
                                                                     CuGranularity,
                                                                     WaveGranularity,
                                                                     MemReadBytes,
                                                                     MemWriteBytes,
                                                                     TempEdge,
                                                                     ClockRateSys,
                                                                     ClockRateSOC,
                                                                     ClockRateMem,
                                                                     FanSpeedRPMs,
                                                                     HardwareSampleCount,
                                                                     GfxFrequency,
                                                                     Power,
                                                                     TemperatureHot,
                                                                     EnqueueTime},
                                                                    stream,
                                                                    dumpTensors));
            }

            static std::shared_ptr<LogReporter> Default(po::variables_map const& args)
            {
                return Default(args, std::cout);
            }

            virtual void reportValue_string(std::string const& key,
                                            std::string const& value) override
            {
                if(key == ResultKey::Validation)
                    acceptValidation(value);

                m_csvOutput.setValueForKey(key, value);
            }

            virtual void reportValue_uint(std::string const& key, uint64_t value) override
            {
                m_csvOutput.setValueForKey(key, value);
            }

            virtual void reportValue_int(std::string const& key, int64_t value) override
            {
                m_csvOutput.setValueForKey(key, value);
            }

            virtual void reportValue_double(std::string const& key, double value) override
            {
                m_csvOutput.setValueForKey(key, value);
            }

            virtual void reportValue_sizes(std::string const&         key,
                                           std::vector<size_t> const& value) override
            {
                std::ostringstream msg;
                msg << "(";
                streamJoin(msg, value, ",");
                msg << ")";
                reportValue_string(key, msg.str());
            }

            void acceptValidation(std::string const& value)
            {
                if(value == "PASSED" || value == "NO_CHECK")
                    m_rowLevel = LogLevel::Normal;
                else if(value == "FAILED" || value == "FAILED_CONV")
                    m_rowLevel = LogLevel::Error;
                else if(value == "WRONG_HARDWARE")
                    m_rowLevel = LogLevel::Terse;
                else if(value == "DID_NOT_SATISFY_ASSERTS")
                    m_rowLevel = LogLevel::Terse;
                else if(value == "INVALID")
                    m_rowLevel = LogLevel::Error;
            }

            virtual bool logAtLevel(LogLevel level) override
            {
                return level <= m_level;
            }

            virtual void logMessage(LogLevel level, std::string const& message) override
            {
                if(logAtLevel(level))
                {
                    m_stream << message;
                    m_stream.flush();
                }
            }

            template <typename T>
            void logTensorTyped(LogLevel                level,
                                std::string const&      name,
                                T const*                data,
                                TensorDescriptor const& tensor,
                                T const*                ptrVal)
            {
                if(logAtLevel(level))
                {
                    if(m_dumpTensors)
                    {
                        std::string   fname = concatenate("tensor_", name, ".bin");
                        std::ofstream ofile(fname.c_str());
                        ofile.write(reinterpret_cast<const char*>(data),
                                    tensor.totalAllocatedBytes());

                        m_stream << "Dumped tensor to file " << fname << std::endl;
                    }
                    else
                    {
                        m_stream << name << ": " << tensor << std::endl;
                        WriteTensor(m_stream, data, tensor, ptrVal);
                    }
                }
            }

            virtual void logTensor(LogLevel                level,
                                   std::string const&      name,
                                   void const*             data,
                                   TensorDescriptor const& tensor,
                                   void const*             ptrVal) override
            {
                if(logAtLevel(level))
                {
                    if(tensor.dataType() == DataType::Float)
                        logTensorTyped(level,
                                       name,
                                       reinterpret_cast<float const*>(data),
                                       tensor,
                                       reinterpret_cast<float const*>(ptrVal));
                    else if(tensor.dataType() == DataType::Half)
                        logTensorTyped(level,
                                       name,
                                       reinterpret_cast<Half const*>(data),
                                       tensor,
                                       reinterpret_cast<Half const*>(ptrVal));
                    else if(tensor.dataType() == DataType::ComplexFloat)
                        logTensorTyped(level,
                                       name,
                                       reinterpret_cast<std::complex<float> const*>(data),
                                       tensor,
                                       reinterpret_cast<std::complex<float> const*>(ptrVal));
                    else if(tensor.dataType() == DataType::Double)
                        logTensorTyped(level,
                                       name,
                                       reinterpret_cast<double const*>(data),
                                       tensor,
                                       reinterpret_cast<double const*>(ptrVal));
                    else if(tensor.dataType() == DataType::Int32)
                        logTensorTyped(level,
                                       name,
                                       reinterpret_cast<int32_t const*>(data),
                                       tensor,
                                       reinterpret_cast<int32_t const*>(ptrVal));
                    else if(tensor.dataType() == DataType::Int8)
                        logTensorTyped(level,
                                       name,
                                       reinterpret_cast<Int8 const*>(data),
                                       tensor,
                                       reinterpret_cast<Int8 const*>(ptrVal));
                    else if(tensor.dataType() == DataType::BFloat16)
                        logTensorTyped(level,
                                       name,
                                       reinterpret_cast<BFloat16 const*>(data),
                                       tensor,
                                       reinterpret_cast<BFloat16 const*>(ptrVal));
                    else if(tensor.dataType() == DataType::Float8)
                        logTensorTyped(level,
                                       name,
                                       reinterpret_cast<Float8 const*>(data),
                                       tensor,
                                       reinterpret_cast<Float8 const*>(ptrVal));
                    else if(tensor.dataType() == DataType::BFloat8)
                        logTensorTyped(level,
                                       name,
                                       reinterpret_cast<BFloat8 const*>(data),
                                       tensor,
                                       reinterpret_cast<BFloat8 const*>(ptrVal));
                    else
                        throw std::runtime_error(
                            concatenate("Can't log tensor of type ", tensor.dataType()));
                }
            }

            /// RunListener interface functions

            virtual void setReporter(std::shared_ptr<ResultReporter> reporter) override {}

            virtual void preProblem(ContractionProblem const& problem) override
            {
                m_csvOutput.push();
            }

            virtual void preSolution(ContractionSolution const& solution) override
            {
                m_csvOutput.push();
                m_rowLevel = LogLevel::Normal;
            }

            virtual void postSolution() override
            {
                if(m_rowLevel <= m_level)
                    m_csvOutput.writeCurrentRow();
                m_csvOutput.pop();
            }

            virtual void postProblem() override
            {
                m_csvOutput.pop();
            }

            virtual void finalizeReport() override {}

        private:
            LogLevel m_level;

            std::ostream&                 m_stream;
            std::shared_ptr<std::ostream> m_ownedStream;

            bool m_firstRun    = true;
            bool m_inSolution  = false;
            bool m_dumpTensors = false;

            LogLevel m_rowLevel;

            CSVStackFile m_csvOutput;
        };
    } // namespace Client
} // namespace Tensile
