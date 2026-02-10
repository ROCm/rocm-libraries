// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace TensileLite
{
    namespace Client
    {
        // Global flag to enable/disable timing instrumentation output
        // Set via command line: --timing-instrumentation
        inline bool g_timingInstrumentationEnabled = false;

        // Simple RAII timer that prints timing on destruction
        // Output format: TIMING:<category>:<duration_ms>
        // This format is easily parseable by post-processing scripts
        class ScopedTimer
        {
        public:
            using clock = std::chrono::high_resolution_clock;

            ScopedTimer(const std::string& category)
                : m_category(category)
                , m_start(clock::now())
            {
            }

            ~ScopedTimer()
            {
                if(g_timingInstrumentationEnabled)
                {
                    auto end      = clock::now();
                    auto duration = std::chrono::duration<double, std::milli>(end - m_start);
                    std::cerr << "TIMING:" << m_category << ":" << duration.count() << std::endl;
                }
            }

            // Get elapsed time without stopping
            double elapsedMs() const
            {
                auto now      = clock::now();
                auto duration = std::chrono::duration<double, std::milli>(now - m_start);
                return duration.count();
            }

        private:
            std::string                        m_category;
            std::chrono::time_point<clock> m_start;
        };

        // Report a timing value directly (for GPU timings already measured)
        inline void reportTiming(const std::string& category, double ms)
        {
            if(g_timingInstrumentationEnabled)
            {
                std::cerr << "TIMING:" << category << ":" << ms << std::endl;
            }
        }

        // Report problem context for correlation (single GEMM)
        inline void reportProblemContext(size_t M, size_t N, size_t K, size_t batchCount,
                                         const std::string& typeA, const std::string& typeD)
        {
            if(g_timingInstrumentationEnabled)
            {
                std::cerr << "TIMING_CONTEXT:M=" << M << ",N=" << N << ",K=" << K
                          << ",batch=" << batchCount << ",typeA=" << typeA << ",typeD=" << typeD
                          << std::endl;
            }
        }

        // Report problem context for grouped GEMM (multiple GEMMs batched together)
        inline void reportGroupedProblemContext(size_t index, size_t totalGemms,
                                                size_t M, size_t N, size_t K, size_t batchCount,
                                                const std::string& typeA, const std::string& typeD)
        {
            if(g_timingInstrumentationEnabled)
            {
                std::cerr << "TIMING_CONTEXT_GROUPED:index=" << index
                          << ",total=" << totalGemms
                          << ",M=" << M << ",N=" << N << ",K=" << K
                          << ",batch=" << batchCount << ",typeA=" << typeA << ",typeD=" << typeD
                          << std::endl;
            }
        }

    } // namespace Client
} // namespace TensileLite
