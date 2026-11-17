// Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <algorithm>
#include <charconv>
#include <limits>
#include <thread>

#include "environment.h"

#ifndef _WIN32
#include <sched.h>
#endif

// We temporarily add a limit on OMP_NUM_THREADS in order to un-block
// theRock CI, which is using OMP_NUM_THREADS in order to reduce
// CPU over-subscription when running multiple tests on the same node.
// If the environment variable is set incorrectly, return the max int value.
// Also, floor the value at 1.
static int getenv_OMP_NUM_THREADS()
{
    const auto env_str = rocfft_getenv("OMP_NUM_THREADS");
    if(env_str != "")
    {
        int ompnumthreads = std::numeric_limits<int>::max();
        auto [ptr, ec]
            = std::from_chars(env_str.data(), env_str.data() + env_str.size(), ompnumthreads);
        if(ec == std::errc())
        {
            return std::max<int>(1, ompnumthreads);
        }
    }
    return std::numeric_limits<int>::max();
}

// Work out how many parallel tasks to run, based on available
// resources.  On Linux, this will look at the cpu affinity mask (if
// available) which might be restricted in a container.  Otherwise,
// return std::thread::hardware_concurrency().
static unsigned int rocfft_concurrency()
{
#ifndef _WIN32
    cpu_set_t cpuset;
    if(sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0)
    {
        return std::min(CPU_COUNT(&cpuset), getenv_OMP_NUM_THREADS());
    }
#endif
    return std::min<unsigned int>(std::thread::hardware_concurrency(), getenv_OMP_NUM_THREADS());
}
