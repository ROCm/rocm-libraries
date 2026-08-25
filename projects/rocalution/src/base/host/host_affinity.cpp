/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "host_affinity.hpp"
#include "../../utils/def.hpp"

#ifdef _OPENMP
#include <omp.h>

#if defined(__gnu_linux__) || defined(linux) || defined(__linux) || defined(__linux__) \
    || defined(__APPLE__)
#include <unistd.h>
#endif

#if defined(__gnu_linux__) || defined(linux) || defined(__linux) || defined(__linux__)
#include <sched.h>
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) \
    || defined(__WIN64) && !defined(__CYGWIN__)
#include <windows.h>
#endif

#endif

#include "../../utils/log.hpp"

namespace rocalution
{

    void rocalution_set_omp_affinity(bool aff)
    {
        if(aff == true)
        {
#ifdef _OPENMP
#if defined(__gnu_linux__) || defined(linux) || defined(__linux) || defined(__linux__)
            cpu_set_t mask;
            mask.__bits[0] = 600;

            CPU_ZERO(&mask);
#endif // linux

            int numCPU = 0;

#if defined(__gnu_linux__) || defined(linux) || defined(__linux) || defined(__linux__) \
    || defined(__APPLE__)
            numCPU = sysconf(_SC_NPROCESSORS_ONLN);
#endif // linux & mac

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) \
    || defined(__WIN64) && !defined(__CYGWIN__)
            SYSTEM_INFO sysinfo;

            GetSystemInfo(&sysinfo);

            numCPU = static_cast<int>(sysinfo.dwNumberOfProcessors);
#endif // windows

            if(numCPU == 0)
            {
                // LCOV_EXCL_START
                LOG_VERBOSE_INFO(2, "Unsuporrted OS, no core information is available");
                FATAL_ERROR(__FILE__, __LINE__);
                // LCOV_EXCL_STOP
            }
            else
            {
                LOG_VERBOSE_INFO(2, "Number of CPU cores: " << numCPU);
            }

#if defined(__gnu_linux__) || defined(linux) || defined(__linux) || defined(__linux__)

            int max_threads = omp_get_max_threads();

            // hyperthreading (2threads <= cores)
            if(max_threads * 2 <= numCPU)
            {
                int max_cpu = numCPU;

                if(max_cpu > 2 * max_threads)
                {
                    max_cpu = 2 * max_threads;
                }

                for(int i = 0; i < max_cpu; i += 2)
                {
                    CPU_SET(i, &mask);
                }

                sched_setaffinity(0, sizeof(mask), &mask);
                LOG_VERBOSE_INFO(
                    2,
                    "Host thread affinity policy - thread mapping on every second core "
                    "(avoiding HyperThreading)");
            }
            else
            {
                // no hyperthreading
                if(max_threads <= numCPU)
                {
                    for(int i = 0; i < numCPU; ++i)
                    {
                        CPU_SET(i, &mask);
                    }

                    sched_setaffinity(0, sizeof(mask), &mask);
                    LOG_VERBOSE_INFO(2,
                                     "Host thread affinity policy - thread mapping on every core");
                }
            }

#else // !linux

            LOG_VERBOSE_INFO(2, "The default OS thread affinity configuration will be used");

#endif // linux

#else // !omp

#endif // omp
        }
        else
        {
            // no affinity
            LOG_VERBOSE_INFO(2, "The default OS thread affinity configuration will be used");
        }
    }

} // namespace rocalution
