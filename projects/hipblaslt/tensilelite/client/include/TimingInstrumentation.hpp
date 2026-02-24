// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace TensileLite
{
    namespace Client
    {
        // Global flag to enable/disable timing instrumentation output
        // Set via command line: --timing-instrumentation
        inline bool g_timingInstrumentationEnabled = false;

        // Buffer for timing output to avoid per-event stderr writes.
        // Accumulates into a single ostringstream and flushes to stderr
        // when flush() is called or the buffer exceeds a size threshold.
        class TimingBuffer
        {
        public:
            static TimingBuffer& instance()
            {
                static TimingBuffer buf;
                return buf;
            }

            void append(const char* data, size_t len)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_stream.write(data, len);
                m_stream.put('\n');
                m_size += len + 1;
                if(m_size >= FlushThreshold)
                    flushLocked();
            }

            void flush()
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                flushLocked();
            }

        private:
            static constexpr size_t FlushThreshold = 1 << 20; // 1 MB

            void flushLocked()
            {
                if(m_size == 0)
                    return;
                std::cerr << m_stream.str();
                m_stream.str(std::string());
                m_stream.clear();
                m_size = 0;
            }

            TimingBuffer() = default;
            std::mutex         m_mutex;
            std::ostringstream m_stream;
            size_t             m_size = 0;
        };

        inline void flushTimingBuffer()
        {
            TimingBuffer::instance().flush();
        }

        // Simple RAII timer that records timing on destruction
        // Output format: TIMING:<category>:<duration_ms>
        // This format is easily parseable by post-processing scripts
        //
        // Timing records are buffered in memory and flushed periodically
        // (every ~1 MB) or via flushTimingBuffer() to avoid per-event
        // stderr syscall overhead.
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
                    char buf[256];
                    int  n = snprintf(buf, sizeof(buf), "TIMING:%s:%.6f",
                                      m_category.c_str(), duration.count());
                    if(n > 0)
                        TimingBuffer::instance().append(buf, n);
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
                char buf[256];
                int  n = snprintf(buf, sizeof(buf), "TIMING:%s:%.6f",
                                  category.c_str(), ms);
                if(n > 0)
                    TimingBuffer::instance().append(buf, n);
            }
        }

        // Report problem context for correlation (single GEMM)
        inline void reportProblemContext(size_t M, size_t N, size_t K, size_t batchCount,
                                         const std::string& typeA, const std::string& typeD)
        {
            if(g_timingInstrumentationEnabled)
            {
                char buf[256];
                int  n = snprintf(buf, sizeof(buf),
                                  "TIMING_CONTEXT:M=%zu,N=%zu,K=%zu,batch=%zu,typeA=%s,typeD=%s",
                                  M, N, K, batchCount, typeA.c_str(), typeD.c_str());
                if(n > 0)
                    TimingBuffer::instance().append(buf, n);
            }
        }

        // Report problem context for grouped GEMM (multiple GEMMs batched together)
        inline void reportGroupedProblemContext(size_t index, size_t totalGemms,
                                                size_t M, size_t N, size_t K, size_t batchCount,
                                                const std::string& typeA, const std::string& typeD)
        {
            if(g_timingInstrumentationEnabled)
            {
                char buf[256];
                int  n = snprintf(buf, sizeof(buf),
                                  "TIMING_CONTEXT_GROUPED:index=%zu,total=%zu,"
                                  "M=%zu,N=%zu,K=%zu,batch=%zu,typeA=%s,typeD=%s",
                                  index, totalGemms, M, N, K, batchCount,
                                  typeA.c_str(), typeD.c_str());
                if(n > 0)
                    TimingBuffer::instance().append(buf, n);
            }
        }

    } // namespace Client
} // namespace TensileLite
