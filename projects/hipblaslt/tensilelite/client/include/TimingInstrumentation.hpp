// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

namespace TensileLite
{
    namespace Client
    {
        // Global flag to enable/disable timing instrumentation output
        // Set via command line: --timing-instrumentation
        inline bool g_timingInstrumentationEnabled = false;

        // Fast formatters — to_chars into a caller-supplied buffer
        inline char* fmtOne(char* p, char* end, const char* s)
        {
            auto len = std::strlen(s);
            auto avail = static_cast<size_t>(end - p);
            if(len > avail)
                len = avail;
            std::memcpy(p, s, len);
            return p + len;
        }

        inline char* fmtOne(char* p, char* end, const std::string& s)
        {
            return fmtOne(p, end, s.c_str());
        }

        inline char* fmtOne(char* p, char* end, size_t v)
        {
            auto result = std::to_chars(p, end, v);
            return result.ptr;
        }

        inline char* fmtOne(char* p, char* end, double v)
        {
            auto result = std::to_chars(p, end, v);
            return result.ptr;
        }

        // Format args into a stack buffer and write as a single line to std::clog
        template<typename... Args>
        inline void writeLine(Args&&... args)
        {
            char buf[256];
            char*       p   = buf;
            char* const end = buf + sizeof(buf) - 1;
            ((p = fmtOne(p, end, args)), ...);
            *p++ = '\n';
            std::clog.write(buf, p - buf);
        }

        inline void flushTimingBuffer()
        {
            std::clog.flush();
        }

        // Simple RAII timer that records timing on destruction
        // Output format: TIMING:<category>:<duration_ms>
        // This format is easily parseable by post-processing scripts
        //
        // Timing records are written to std::clog (buffered stderr).
        // Call flushTimingBuffer() to ensure all records are flushed.
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
                    writeLine("TIMING:", m_category, ":", duration.count());
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
            std::string                    m_category;
            std::chrono::time_point<clock> m_start;
        };

        // Report a timing value directly (for GPU timings already measured)
        inline void reportTiming(const std::string& category, double ms)
        {
            if(g_timingInstrumentationEnabled)
            {
                writeLine("TIMING:", category, ":", ms);
            }
        }

        // Report problem context for correlation (single GEMM)
        inline void reportProblemContext(size_t M, size_t N, size_t K, size_t batchCount,
                                         const std::string& typeA, const std::string& typeD)
        {
            if(g_timingInstrumentationEnabled)
            {
                writeLine("TIMING_CONTEXT:M=", M, ",N=", N, ",K=", K,
                          ",batch=", batchCount, ",typeA=", typeA, ",typeD=", typeD);
            }
        }

        // Report problem context for grouped GEMM (multiple GEMMs batched together)
        inline void reportGroupedProblemContext(size_t index, size_t totalGemms,
                                                size_t M, size_t N, size_t K, size_t batchCount,
                                                const std::string& typeA, const std::string& typeD)
        {
            if(g_timingInstrumentationEnabled)
            {
                writeLine("TIMING_CONTEXT_GROUPED:index=", index, ",total=", totalGemms,
                          ",M=", M, ",N=", N, ",K=", K,
                          ",batch=", batchCount, ",typeA=", typeA, ",typeD=", typeD);
            }
        }

    } // namespace Client
} // namespace TensileLite
