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

#include <iostream>
#include <thread>

#include <iostream>

#ifndef _WIN32
#include <sched.h>
#endif

// work out how many parallel tasks to run, based on available
// resources.  on Linux, this will look at the cpu affinity mask (if
// available) which might be restricted in a container.  otherwise,
// return std::thread::hardware_concurrency().

// We temporarily add a limit on OMP_NUM_THREADS in order to un-block
// theRock CI, which is using OMP_NUM_THREADS in order to reduce
// CPU over-subscription when running multiple tests on the same node.
static int getenv_OMP_NUM_THREADS()
{
    const char* env_raw = std::getenv("OMP_NUM_THREADS");
    int ompnumthreads = std::numeric_limits<int>::max();
    if (env_raw != nullptr)
    {
        try
        {
            ompnumthreads = std::stoi(env_raw);
        }
        catch (const std::invalid_argument& e)
        {
            std::cerr << "Error: OMP_NUM_THREADS is not a valid number.\n";
        }
        catch (const std::out_of_range& e)
        {
            std::cerr << "Error: OMP_NUM_THREADS is too large to fit into an int.\n";
        }
    }
    return ompnumthreads;
}

static unsigned int rocfft_concurrency()
{
#ifndef _WIN32
    cpu_set_t cpuset;
    if(sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0)
    {
        return std::min(CPU_COUNT(&cpuset), getenv_OMP_NUM_THREADS());
    }
#endif

    return std::min<unsigned int>(std::thread::hardware_concurrency(),
                                  getenv_OMP_NUM_THREADS());
}
