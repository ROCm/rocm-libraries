// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocfft/rocfft.h"

// declare things that RTC needs to link a standalone executable
// without the rest of rocFFT
int log_trace_fd    = -1;
int log_bench_fd    = -1;
int log_profile_fd  = -1;
int log_plan_fd     = -1;
int log_kernelio_fd = -1;
int log_rtc_fd      = -1;
int log_tuning_fd   = -1;

#ifndef ROCFFT_BUILD_OFFLINE_TUNER
extern "C" rocfft_status rocfft_plan_create(rocfft_plan*                  plan,
                                            rocfft_result_placement       placement,
                                            rocfft_transform_type         transform_type,
                                            rocfft_precision              precision,
                                            size_t                        dimensions,
                                            const size_t*                 lengths,
                                            size_t                        number_of_transforms,
                                            const rocfft_plan_description description)
{
    return rocfft_status_failure;
}
#endif