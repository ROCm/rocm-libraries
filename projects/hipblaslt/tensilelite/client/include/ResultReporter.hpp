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

#include "RunListener.hpp"

#include <cstddef>
#include <string>

namespace TensileLite
{
    namespace Client
    {
        enum class LogLevel
        {
            Error = 0,
            Terse,
            Normal,
            Verbose,
            Debug,
            Count
        };

        std::string   ToString(LogLevel level);
        std::ostream& operator<<(std::ostream& stream, LogLevel level);
        std::istream& operator>>(std::istream& stream, LogLevel& level);

        namespace ResultKey
        {
            const std::string BenchmarkRunNumber = "run";

            // Problem definition
            const std::string ProblemIndex        = "problem-index";
            const std::string ProblemCount        = "problem-count";
            const std::string ProblemProgress     = "problem-progress";
            const std::string OperationIdentifier = "operation";

            const std::string ASizes = "a-sizes";
            const std::string BSizes = "b-sizes";
            const std::string CSizes = "c-sizes";
            const std::string DSizes = "d-sizes";

            const std::string AStrides = "a-strides";
            const std::string BStrides = "b-strides";
            const std::string CStrides = "c-strides";
            const std::string DStrides = "d-strides";

            const std::string LDA = "lda";
            const std::string LDB = "ldb";
            const std::string LDC = "ldc";
            const std::string LDD = "ldd";

            const std::string TotalFlops   = "total-flops";
            const std::string ProblemSizes = "problem-sizes";

            const std::string BiasType       = "bias-type";
            const std::string FactorDim      = "factor-dim";
            const std::string ActivationType = "activation-type";

            // Solution information
            const std::string SolutionName         = "solution";
            const std::string SolutionIndex        = "solution-index";
            const std::string SolutionLibraryIndex = "solution-library-index";
            const std::string SolutionProgress     = "solution-progress";
            const std::string SolutionWinnerIdx    = "solution-winner-idx";
            const std::string SolutionWinner       = "solution-winner";

            // Performance-related
            const std::string Validation       = "validation";
            const std::string TimeUS           = "time-us";
            const std::string SpeedGFlops      = "gflops";
            const std::string SpeedGFlopsPerCu = "gflops-per-cu";
            const std::string EnqueueTime      = "enqueue-time";
            const std::string FastestGFlops    = "fastest-gflops";

            // Performance estimation and granularity
            const std::string Tile0Granularity = "tile0-gran";
            const std::string Tile1Granularity = "tile1-gran";
            const std::string CuGranularity    = "cu-gran";
            const std::string WaveGranularity  = "wave-gran";
            const std::string TotalGranularity = "total-gran";
            const std::string TilesPerCu       = "tiles-per-cu";

            const std::string MemReadBytes    = "mem-read-bytes";
            const std::string MemWriteBytes   = "mem-write-bytes";
            const std::string MemGlobalReads  = "mem-global-reads";
            const std::string MemGlobalWrites = "mem-global-writes";

            const std::string L2ReadHits      = "perf-l2-read-hits";
            const std::string L2WriteHits     = "perf-l2-write-hits";
            const std::string L2ReadBwMul     = "perf-l2-read-bw-mul";
            const std::string L2BandwidthMBps = "perf-l2-bandwidth-mbps";

            const std::string Empty      = "empty";
            const std::string Efficiency = "efficiency";
            const std::string NumCus     = "num-cus";
            const std::string PeakGFlops = "peak-gflops";

            // Hardware monitoring
            const std::string TempEdge            = "temp-edge";
            const std::string ClockRateSys        = "clock-sys"; // GPU clock in Mhz
            const std::string ClockRateSOC        = "clock-soc"; // Soc clock in Mhz
            const std::string ClockRateMem        = "clock-mem"; // Mem clock in Mhz
            const std::string DeviceIndex         = "device-idx";
            const std::string FanSpeedRPMs        = "fan-rpm";
            const std::string HardwareSampleCount = "hardware-samples";
            const std::string GfxFrequency        = "gfx-frequency(maximum)"; // GPU freq in Mhz
        }; // namespace ResultKey

        class ResultReporter : public RunListener
        {
        public:
            /**
   * Reports the value for a key, related to the current state of the run.
   */
            void report(std::string const& key, std::string const& value)
            {
                reportValue_string(key, value);
            }

            void report(std::string const& key, uint64_t value)
            {
                reportValue_uint(key, value);
            }

            void report(std::string const& key, int value)
            {
                reportValue_int(key, value);
            }

            void report(std::string const& key, int64_t value)
            {
                reportValue_int(key, value);
            }

            void report(std::string const& key, double value)
            {
                reportValue_double(key, value);
            }

            void report(std::string const& key, std::vector<size_t> const& value)
            {
                reportValue_sizes(key, value);
            }

            void report(std::string const& key, std::vector<std::vector<size_t>> const& value)
            {
                reportValue_vecOfSizes(key, value);
            }

            virtual void reportValue_string(std::string const& key, std::string const& value) = 0;
            virtual void reportValue_uint(std::string const& key, uint64_t value)             = 0;
            virtual void reportValue_int(std::string const& key, int64_t value)               = 0;
            virtual void reportValue_double(std::string const& key, double value)             = 0;
            virtual void reportValue_sizes(std::string const& key, std::vector<size_t> const& value)
                = 0;
            virtual void reportValue_vecOfSizes(std::string const&                      key,
                                                std::vector<std::vector<size_t>> const& value)
                = 0;

            virtual bool logAtLevel(LogLevel level)
            {
                return false;
            };

            /**
   * Records an informative message.  This may or may not actually get printed
   * anywhere depending on settings.
   */
            template <typename T>
            void log(LogLevel level, T const& object)
            {
                if(logAtLevel(level))
                {
                    std::ostringstream msg;
                    msg << object;
                    logMessage(level, msg.str());
                }
            }

            virtual void logMessage(LogLevel level, std::string const& message) {}
            virtual void logTensor(LogLevel                level,
                                   std::string const&      name,
                                   void const*             data,
                                   TensorDescriptor const& tensor,
                                   void const*             ptrVal)
            {
            }

            /// RunListener interface functions

            virtual void setReporter(std::shared_ptr<ResultReporter> reporter) override {}

            virtual bool needMoreBenchmarkRuns() const override
            {
                return false;
            }
            virtual void preBenchmarkRun() override {}
            virtual void postBenchmarkRun() override {}

            virtual void preProblem(ContractionProblem* const problem) override {}
            virtual void postProblem() override {}

            virtual void preSolution(ContractionSolution const& solution) override {}
            virtual void postSolution() override {}

            virtual bool needMoreRunsInSolution() const override
            {
                return false;
            }

            virtual size_t numWarmupRuns() override
            {
                return 0;
            }
            virtual void setNumWarmupRuns(size_t count) override {}
            virtual void preWarmup() override {}
            virtual void postWarmup(TimingEvents const& startEvents,
                                    TimingEvents const& stopEvents,
                                    hipStream_t const&  stream) override
            {
            }
            virtual void validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                         TimingEvents const&            startEvents,
                                         TimingEvents const&            stopEvents) override
            {
            }

            virtual size_t numSyncs() override
            {
                return 0;
            }
            virtual void setNumSyncs(size_t count) override {}
            virtual void preSyncs() override {}
            virtual void postSyncs() override {}

            virtual size_t numEnqueuesPerSync() override
            {
                return 0;
            }
            virtual void setNumEnqueuesPerSync(size_t count) override {}
            virtual void preEnqueues(hipStream_t const& stream) override {}
            virtual void postEnqueues(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents,
                                      hipStream_t const&  stream) override
            {
            }
            virtual void validateEnqueues(std::shared_ptr<ProblemInputs> inputs,
                                          TimingEvents const&            startEvents,
                                          TimingEvents const&            stopEvents) override
            {
            }

            // finalizeReport() deliberately left out of here to force it to be
            // implemented in subclasses.

            virtual int error() const override
            {
                return 0;
            }
        };

    } // namespace Client
} // namespace TensileLite
