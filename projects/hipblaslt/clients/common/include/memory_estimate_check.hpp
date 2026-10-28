// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef GOOGLE_TEST

#include "d_vector.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

// The host-memory estimate and use of the last case MemoryEstimateCheck measured, in bytes.
struct MemoryEstimateMeasurement
{
    bool   measured = false;
    size_t estimate = 0;
    size_t used     = 0;
};

inline MemoryEstimateMeasurement& last_memory_estimate_measurement()
{
    static MemoryEstimateMeasurement measurement;
    return measurement;
}

// With HIPBLASLT_TEST_CHECK_MEMORY_ESTIMATE=1, a matmul case whose host-memory estimate is at
// least kMinBytes measures the host memory it uses, as the rise in the process's peak resident
// memory, which counts pinned and pageable memory alike. The case fails when it used more than
// its estimate plus a margin for memory the estimate leaves out, such as code objects loaded for
// a new kernel. Linux only; elsewhere, or when the peak cannot be reset, nothing is checked.
class MemoryEstimateCheck
{
public:
    static constexpr size_t kMinBytes = size_t(256) << 20;

    static size_t allowed(size_t estimate)
    {
        return estimate + estimate / 10 + (size_t(256) << 20);
    }

    explicit MemoryEstimateCheck(size_t estimate)
    {
#ifdef __linux__
        // The first matmul case in a process also loads the kernel library, which raised the
        // peak by about 0.6 GiB on gfx90a, so the check starts with the second case.
        static bool first_case = true;
        const bool  first      = std::exchange(first_case, false);
        const char* on         = std::getenv("HIPBLASLT_TEST_CHECK_MEMORY_ESTIMATE");
        if(first || !on || !*on || !strcmp(on, "0") || estimate < kMinBytes)
            return;
        // Buffers an earlier test left in the pool would be reused without raising the peak.
        memory_pool<h_memory>::ReleaseCached();
        // Writing 5 resets the peak (VmHWM) to the current resident memory (VmRSS).
        std::ofstream("/proc/self/clear_refs") << "5";
        size_t hwm = 0;
        if(!read_status("VmRSS:", m_baseline) || !read_status("VmHWM:", hwm)
           || hwm > m_baseline + (size_t(64) << 20))
            return;
        m_estimate                                  = estimate;
        m_active                                    = true;
        last_memory_estimate_measurement().measured = false;
#else
        (void)estimate;
#endif
    }

    ~MemoryEstimateCheck()
    {
        size_t hwm = 0;
        if(!m_active || !read_status("VmHWM:", hwm))
            return;
        const size_t used                  = hwm > m_baseline ? hwm - m_baseline : 0;
        last_memory_estimate_measurement() = {true, m_estimate, used};
        if(used > allowed(m_estimate))
            ADD_FAILURE() << "The memory preflight estimated " << gib(m_estimate)
                          << " of host memory for this case, but its peak resident memory rose "
                          << gib(used) << ". Update the estimate in testing_matmul.hpp.";
    }

    MemoryEstimateCheck(const MemoryEstimateCheck&)            = delete;
    MemoryEstimateCheck& operator=(const MemoryEstimateCheck&) = delete;

private:
    static bool read_status(const char* key, size_t& bytes)
    {
        std::ifstream status("/proc/self/status");
        std::string   name;
        size_t        kib = 0;
        for(std::string line; std::getline(status, line);)
        {
            std::istringstream fields(line);
            if(fields >> name >> kib && name == key)
            {
                bytes = kib * 1024;
                return true;
            }
        }
        return false;
    }

    static std::string gib(size_t bytes)
    {
        std::ostringstream s;
        s.precision(3);
        s << double(bytes) / double(size_t(1) << 30) << " GiB";
        return s.str();
    }

    bool   m_active   = false;
    size_t m_estimate = 0;
    size_t m_baseline = 0;
};

#endif
