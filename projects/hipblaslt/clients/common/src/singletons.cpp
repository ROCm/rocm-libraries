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
#include "singletons.hpp"
#include <mutex>

// global for device memory padding see d_vector.hpp
// RUN_TEST_ON_THREADS_STREAMS writes this from every worker; the mutex keeps
// that update from racing the reads in the d_vector constructors.
static std::mutex g_pad_mutex;
size_t            g_DVEC_PAD = 4096;

void d_vector_set_pad_length(size_t pad)
{
    std::lock_guard<std::mutex> lock(g_pad_mutex);
    g_DVEC_PAD = pad;
}

size_t d_vector_get_pad_length()
{
    std::lock_guard<std::mutex> lock(g_pad_mutex);
    return g_DVEC_PAD;
}
