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

#ifdef _OPENMP
#include <omp.h>
#else
#ifndef __HIP_DEVICE_COMPILE__
#pragma GCC warning "_OPENMP not defined so client build can not utilize OPENMP."
#endif
#endif

#include "client_omp.hpp"
#include "rocblas_ostream.hpp"

// constant to reduce threads to avoid potential deadlocks when using all logical cores
static constexpr int c_thread_reducer = 2;

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
    // limit OMP usage as deadlock issues seen in reference library
#ifdef _OPENMP
    const int processor_count = std::thread::hardware_concurrency();
    if(processor_count > 0)
    {
        const int omp_current_threads = omp_get_max_threads();
        if(omp_current_threads >= processor_count)
        {
            int omp_limit_threads
                = processor_count > 4 ? processor_count - c_thread_reducer : processor_count;
            omp_limit_threads = std::max(1, omp_limit_threads);

            if(omp_limit_threads != omp_current_threads)
            {
                omp_set_num_threads(omp_limit_threads);

                rocblas_cout << "rocBLAS info: client (OPENMP) reduced omp_set_num_threads to "
                             << omp_limit_threads << std::endl;
            }
        }
    }
#endif
}
