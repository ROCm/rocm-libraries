
/* ************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 * ************************************************************************/
#pragma once

#include <hip/hip_runtime_api.h>
#include <vector>

// Samples the GPU engine (per XCD) and memory clocks with AMD SMI while a timing loop runs.
// ROCBLAS_BENCH_FREQ reports the slowest XCD, ROCBLAS_BENCH_FREQ_ALL every XCD, and
// ROCBLAS_BENCH_PERF adds the GEMM efficiency against the peak at the measured clock.
// Without any of them, and on Windows, the monitor does nothing. Clocks are in MHz.
class FrequencyMonitor
{
public:
    virtual ~FrequencyMonitor() = default;

    virtual bool enabled()          = 0;
    virtual bool detailedReport()   = 0;
    virtual bool efficiencyReport() = 0;

    // Optional: start() binds to the calling thread's current HIP device
    virtual void set_device_id(int deviceId) = 0;

    // start(stream) samples once the work queued on stream so far has finished, so a timed loop
    // queued behind cold calls is sampled only while it runs (event timing does not synchronize).
    // start() samples from now, for timing loops of host-synchronous calls.
    virtual void start()                   = 0;
    virtual void start(hipStream_t stream) = 0;
    virtual void stop()                    = 0;

    virtual double              getLowestAverageSYSCLK() = 0;
    virtual double              getLowestMedianSYSCLK()  = 0;
    virtual std::vector<double> getAllAverageSYSCLK()    = 0;
    virtual std::vector<double> getAllMedianSYSCLK()     = 0;
    virtual double              getAverageMEMCLK()       = 0;
    virtual double              getMedianMEMCLK()        = 0;
    virtual double              getCuCount()             = 0;
};

// One monitor per calling thread, so parallel device runs do not share one
FrequencyMonitor& getFrequencyMonitor();
void              freeFrequencyMonitor();
