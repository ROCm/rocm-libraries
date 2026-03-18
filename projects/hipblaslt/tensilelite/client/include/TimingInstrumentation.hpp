// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // Global flag to enable/disable timing instrumentation output
        // Set via command line: --timing-instrumentation
        inline bool g_timingInstrumentationEnabled = false;

        // ---- Deferred I/O buffer ------------------------------------------------
        //
        // During measurement, timing data is captured as raw structs with no string
        // formatting or I/O. All formatting and writing happens in flushTimingBuffer().
        //
        // Note: single-threaded. The benchmark loop in main.cpp is sequential.

        struct TimingRec
        {
            const char* category;
            double      durationMs;
        };

        struct ContextRec
        {
            size_t      M, N, K, batchCount;
            std::string typeA, typeD;
        };

        struct GroupedContextRec
        {
            size_t      index, totalGemms;
            size_t      M, N, K, batchCount;
            std::string typeA, typeD;
        };

        using TimingRecord = std::variant<TimingRec, ContextRec, GroupedContextRec>;
        inline std::vector<TimingRecord> g_timingBuffer;

        // Reserve buffer capacity upfront. Real workloads produce ~2M+ records.
        inline void initTimingBuffer(size_t capacity = 2'500'000)
        {
            g_timingBuffer.reserve(capacity);
        }

        // ---- Formatting helpers (used only during flush) ------------------------

        inline char* fmtOne(char* p, char* end, const char* s)
        {
            auto len   = std::strlen(s);
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
            char        buf[256];
            char*       p   = buf;
            char* const end = buf + sizeof(buf) - 1;
            ((p = fmtOne(p, end, args)), ...);
            *p++ = '\n';
            std::clog.write(buf, p - buf);
        }

        // Format and write all buffered records, then clear the buffer.
        inline void flushTimingBuffer()
        {
            for(auto& rec : g_timingBuffer)
            {
                std::visit(
                    [](auto& r)
                    {
                        using T = std::decay_t<decltype(r)>;
                        if constexpr(std::is_same_v<T, TimingRec>)
                            writeLine("TIMING:", r.category, ":", r.durationMs);
                        else if constexpr(std::is_same_v<T, ContextRec>)
                            writeLine("TIMING_CONTEXT:M=", r.M, ",N=", r.N, ",K=", r.K,
                                      ",batch=", r.batchCount,
                                      ",typeA=", r.typeA, ",typeD=", r.typeD);
                        else
                            writeLine("TIMING_CONTEXT_GROUPED:index=", r.index,
                                      ",total=", r.totalGemms,
                                      ",M=", r.M, ",N=", r.N, ",K=", r.K,
                                      ",batch=", r.batchCount,
                                      ",typeA=", r.typeA, ",typeD=", r.typeD);
                    },
                    rec);
            }
            g_timingBuffer.clear();
            std::clog.flush();
        }

        using TimingClock = std::chrono::high_resolution_clock;

        // Simple RAII timer that records timing on destruction.
        // Output format: TIMING:<category>:<duration_ms>
        //
        // Timing records are buffered in g_timingBuffer during measurement.
        // Call flushTimingBuffer() to format and write all records to std::clog.
        class ScopedTimer
        {
        public:
            using clock = TimingClock;

            ScopedTimer(const char* category)
            {
                if(g_timingInstrumentationEnabled)
                {
                    m_category = category;
                    m_start    = clock::now();
                }
            }

            ~ScopedTimer()
            {
                if(g_timingInstrumentationEnabled)
                {
                    auto   end        = clock::now();
                    double durationMs
                        = std::chrono::duration<double, std::milli>(end - m_start).count();
                    g_timingBuffer.push_back(TimingRec{m_category, durationMs});
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
            const char*                    m_category = nullptr;
            std::chrono::time_point<clock> m_start;
        };

        // Report a timing value directly (for GPU timings already measured)
        inline void reportTiming(const char* category, double ms)
        {
            if(g_timingInstrumentationEnabled)
            {
                g_timingBuffer.push_back(TimingRec{category, ms});
            }
        }

        // Report problem context for correlation (single GEMM)
        inline void reportProblemContext(size_t M, size_t N, size_t K, size_t batchCount,
                                         const std::string& typeA, const std::string& typeD)
        {
            if(g_timingInstrumentationEnabled)
            {
                g_timingBuffer.push_back(ContextRec{M, N, K, batchCount, typeA, typeD});
            }
        }

        // Report problem context for grouped GEMM (multiple GEMMs batched together)
        inline void reportGroupedProblemContext(size_t index, size_t totalGemms,
                                                size_t M, size_t N, size_t K, size_t batchCount,
                                                const std::string& typeA, const std::string& typeD)
        {
            if(g_timingInstrumentationEnabled)
            {
                g_timingBuffer.push_back(
                    GroupedContextRec{index, totalGemms, M, N, K, batchCount, typeA, typeD});
            }
        }

    } // namespace Client
} // namespace TensileLite
