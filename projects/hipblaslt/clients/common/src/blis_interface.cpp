/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
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

#include "blis.h"
#include "omp.h"
#include <algorithm>
#include <cstdlib>
#include <thread>

static int available_processor_count()
{
    const unsigned hardware = std::thread::hardware_concurrency();
    return hardware ? static_cast<int>(hardware) : 1;
}

static void hipblaslt_set_default_blis_threads()
{
    // BLIS reads BLIS_NUM_THREADS, falling back to OMP_NUM_THREADS, on every
    // simple-interface call; don't override an explicit choice.
    const char* omp_requested = std::getenv("OMP_NUM_THREADS");
    if(omp_requested && *omp_requested)
        return;

    const char* blis_requested = std::getenv("BLIS_NUM_THREADS");
    if(blis_requested && *blis_requested)
        return;

    constexpr int max_default_threads = 8;
    const int    threads = std::clamp(available_processor_count(), 1, max_default_threads);

    bli_thread_set_num_threads(threads);
}

void setup_blis()
{
#ifndef _WIN32
    bli_init();
    hipblaslt_set_default_blis_threads();
#endif
}
