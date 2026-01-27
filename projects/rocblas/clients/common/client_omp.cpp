/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include <thread>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#else
#ifndef __HIP_DEVICE_COMPILE__
#pragma GCC warning "_OPENMP not defined so client build can not utilize OPENMP."
#endif
#endif

#include "client_omp.hpp"
#include "rocblas_ostream.hpp"

// Constant to reduce threads to avoid performance degradation when using all logical cores
// Increased from 2 to 4 based on AOCL performance recommendations
static constexpr int c_thread_reducer = 4;

client_omp_manager::client_omp_manager(size_t std_thread_count)
    : m_original_omp_threads(1)
    , m_active(false)
{
#ifdef _OPENMP
    if(std_thread_count > 1)
    {
        // OPENMP behaviour not defined in std::threads so reduce potential for over threading
        const int processor_count     = std::thread::hardware_concurrency();
        m_original_omp_threads        = omp_get_max_threads();
        const int omp_current_threads = m_original_omp_threads;

        if(omp_current_threads * std_thread_count > processor_count - c_thread_reducer)
        {
            int omp_limit_threads = omp_current_threads / std_thread_count;
            omp_limit_threads     = std::max(1, omp_limit_threads);

            if(omp_limit_threads != m_original_omp_threads)
            {
                m_active = true;
                omp_set_num_threads(omp_limit_threads);
                static int once
                    = (rocblas_cout << "rocBLAS info: client (OPENMP) multi-thread reducing "
                                       "omp_set_num_threads from "
                                    << m_original_omp_threads << " to " << omp_limit_threads
                                    << " per thread." << std::endl,
                       1);
            }
        }
    }
#endif
}

client_omp_manager::~client_omp_manager()
{
#ifdef _OPENMP
    if(m_active)
    {
        omp_set_num_threads(m_original_omp_threads);
    }
#endif
}

void client_omp_manager::limit_by_processor_count()
{
    // Limit OMP usage to avoid performance degradation in reference library at high thread counts
    // See: rocBLAS Programmer's Guide - AOCL threading recommendations
#ifdef _OPENMP
    const int omp_default_threads = omp_get_max_threads();
    if(omp_default_threads <= 0)
    {
        return;  // Sanity check - should not happen if OpenMP is working
    }
    
    // If user explicitly set OMP_NUM_THREADS, respect their choice
    const char* env_omp_threads = std::getenv("OMP_NUM_THREADS");
    if(env_omp_threads != nullptr)
    {
        rocblas_cout << "rocBLAS info: Found OMP_NUM_THREADS environment variable set to "
                     << env_omp_threads << std::endl;
        return;
    }

    // Preserve c_thread_reducer cores free to avoid AOCL performance degradation at high
    // thread counts. On small systems, use single-threaded mode to avoid contention entirely
    int safe_thread_count = std::max(1, omp_default_threads - c_thread_reducer);
    
    omp_set_num_threads(safe_thread_count);

    rocblas_cout << "rocBLAS info: OMP_NUM_THREADS not set, using "
                 << safe_thread_count << " threads (system default "
                 << omp_default_threads << " minus " << c_thread_reducer 
                 << " to optimize AOCL performance)" << std::endl;
#endif
}
