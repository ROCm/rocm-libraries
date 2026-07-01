// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // Global flag to enable/disable timing instrumentation output
        // Set via command line: --timing-instrumentation
        inline bool g_timingInstrumentationEnabled = false;

        // Per-call overhead measured once at startup via calibrateTimingOverhead().
        // Single-threaded — no atomic needed.
        inline double g_calibratedPerCallOverheadMs = 0;

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

        // Context buffers: separate from the main timing buffer.
        // posInTimingBuffer records g_timingBuffer.size() at push time so that
        // flushTimingBuffer() can merge context lines into the correct position.
        struct ContextEntry
        {
            size_t      posInTimingBuffer;
            size_t      M, N, K, batchCount;
            std::string typeA, typeD;
        };

        struct GroupedContextEntry
        {
            size_t      posInTimingBuffer;
            size_t      index, totalGemms;
            size_t      M, N, K, batchCount;
            std::string typeA, typeD;
        };

        // Main buffer: timing records only (hot path, ~2.5M entries, 16 bytes each).
        inline std::vector<TimingRec> g_timingBuffer;

        // Context buffers: cold path, few thousand entries each.
        inline std::vector<ContextEntry>        g_contextBuffer;
        inline std::vector<GroupedContextEntry>  g_groupedContextBuffer;

        // Reserve buffer capacity upfront. Real workloads produce ~2M+ records.
        // Only allocates when timing is enabled to avoid penalizing normal runs.
        inline void initTimingBuffer(size_t capacity = 2'500'000)
        {
            if(g_timingInstrumentationEnabled)
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

        // Format and write all buffered records, then clear the buffers.
        // Context entries are merged into the output at their recorded positions
        // to preserve the interleaving expected by analyze_timing.py.
        inline void flushTimingBuffer()
        {
            // Emit calibrated instrumentation overhead as a timing record.
            // g_timingBuffer.size() now correctly counts only TimingRec entries.
            if(g_timingInstrumentationEnabled && g_calibratedPerCallOverheadMs > 0)
            {
                double overheadMs = g_calibratedPerCallOverheadMs * g_timingBuffer.size();
                g_timingBuffer.push_back(TimingRec{"timing_overhead", overheadMs});
            }

            // Two-pointer merge: emit context lines at their recorded positions.
            size_t ctxIdx = 0;
            size_t grpIdx = 0;
            for(size_t i = 0; i < g_timingBuffer.size(); i++)
            {
                while(ctxIdx < g_contextBuffer.size()
                      && g_contextBuffer[ctxIdx].posInTimingBuffer <= i)
                {
                    auto& c = g_contextBuffer[ctxIdx++];
                    writeLine("TIMING_CONTEXT:M=", c.M, ",N=", c.N, ",K=", c.K,
                              ",batch=", c.batchCount,
                              ",typeA=", c.typeA, ",typeD=", c.typeD);
                }
                while(grpIdx < g_groupedContextBuffer.size()
                      && g_groupedContextBuffer[grpIdx].posInTimingBuffer <= i)
                {
                    auto& g = g_groupedContextBuffer[grpIdx++];
                    writeLine("TIMING_CONTEXT_GROUPED:index=", g.index,
                              ",total=", g.totalGemms,
                              ",M=", g.M, ",N=", g.N, ",K=", g.K,
                              ",batch=", g.batchCount,
                              ",typeA=", g.typeA, ",typeD=", g.typeD);
                }
                writeLine("TIMING:", g_timingBuffer[i].category, ":",
                          g_timingBuffer[i].durationMs);
            }
            // Emit any trailing context entries (after all timing records).
            while(ctxIdx < g_contextBuffer.size())
            {
                auto& c = g_contextBuffer[ctxIdx++];
                writeLine("TIMING_CONTEXT:M=", c.M, ",N=", c.N, ",K=", c.K,
                          ",batch=", c.batchCount,
                          ",typeA=", c.typeA, ",typeD=", c.typeD);
            }
            while(grpIdx < g_groupedContextBuffer.size())
            {
                auto& g = g_groupedContextBuffer[grpIdx++];
                writeLine("TIMING_CONTEXT_GROUPED:index=", g.index,
                          ",total=", g.totalGemms,
                          ",M=", g.M, ",N=", g.N, ",K=", g.K,
                          ",batch=", g.batchCount,
                          ",typeA=", g.typeA, ",typeD=", g.typeD);
            }

            g_timingBuffer.clear();
            g_contextBuffer.clear();
            g_groupedContextBuffer.clear();
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

            // category must be a string literal or have static storage duration
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

        // Measure per-call ScopedTimer overhead once at startup.
        // Must be called after initTimingBuffer().
        inline void calibrateTimingOverhead()
        {
            if(!g_timingInstrumentationEnabled)
                return;

            // Pre-touch all reserved pages so calibration runs on faulted memory,
            // matching the conditions of real workloads.
            size_t cap = g_timingBuffer.capacity();
            for(size_t i = 0; i < cap; i++)
                g_timingBuffer.push_back(TimingRec{"_warmup", 0});
            g_timingBuffer.clear();

            constexpr int kWarmup = 1000;
            constexpr int kIter   = 100000;
            for(int i = 0; i < kWarmup; i++)
            { ScopedTimer t("_calibrate"); }
            g_timingBuffer.clear();
            auto t0 = TimingClock::now();
            for(int i = 0; i < kIter; i++)
            { ScopedTimer t("_calibrate"); }
            auto t1 = TimingClock::now();
            g_calibratedPerCallOverheadMs
                = std::chrono::duration<double, std::milli>(t1 - t0).count() / kIter;
            g_timingBuffer.clear();
        }

        // Report a timing value directly (for GPU timings already measured)
        inline void reportTiming(const char* category, double ms)
        {
            if(g_timingInstrumentationEnabled)
            {
                g_timingBuffer.push_back(TimingRec{category, ms});
            }
        }

        // Report problem context for correlation (single GEMM).
        // Buffered separately from timing records; merged during flush.
        inline void reportProblemContext(size_t M, size_t N, size_t K, size_t batchCount,
                                         const std::string& typeA, const std::string& typeD)
        {
            if(g_timingInstrumentationEnabled)
            {
                g_contextBuffer.push_back(
                    ContextEntry{g_timingBuffer.size(), M, N, K, batchCount, typeA, typeD});
            }
        }

        // Report problem context for grouped GEMM (multiple GEMMs batched together).
        // Buffered separately from timing records; merged during flush.
        inline void reportGroupedProblemContext(size_t index, size_t totalGemms,
                                                size_t M, size_t N, size_t K, size_t batchCount,
                                                const std::string& typeA, const std::string& typeD)
        {
            if(g_timingInstrumentationEnabled)
            {
                g_groupedContextBuffer.push_back(GroupedContextEntry{
                    g_timingBuffer.size(), index, totalGemms, M, N, K, batchCount, typeA, typeD});
            }
        }

    } // namespace Client
} // namespace TensileLite
