// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_TEST_CALLBACKS_H
#define ROCFFT_TEST_CALLBACKS_H

#include "fft_params.h"
#include <functional>

// Helpers to work with device functions passed as callbacks, to be
// used by unit tests

// Data used by the test callback functions
struct callback_test_data
{
    // scalar to modify the input/output with
    double scalar;
};

// Get a pointer (on the host) to a load callback device function for
// the current device
void* get_load_callback_host(fft_array_type itype,
                             fft_precision  precision,
                             bool           round_trip_inverse = false);

// Get a pointer (on the host) to a store callback device function for
// the current device
void* get_store_callback_host(fft_array_type otype,
                              fft_precision  precision,
                              bool           round_trip_inverse = false);

// Collect load callback function and data pointers for the given
// params.  We'd expect N pointers for N input bricks on the current
// multi-processing rank.
//
// Data structs are allocated on the device in all_cb_data.
void get_rank_load_callbacks(const fft_params&                          params,
                             std::vector<void*>&                        load_cb_func,
                             std::vector<void*>&                        load_cb_data,
                             bool                                       round_trip_inverse,
                             std::vector<gpubuf_t<callback_test_data>>& all_cb_data);

// Collect store callback function and data pointers for the given
// params.  We'd expect N pointers for N output bricks on the current
// multi-processing rank.
//
// Data structs are allocated on the device in all_cb_data.
void get_rank_store_callbacks(const fft_params&                          params,
                              std::vector<void*>&                        store_cb_func,
                              std::vector<void*>&                        store_cb_data,
                              bool                                       round_trip_inverse,
                              std::vector<gpubuf_t<callback_test_data>>& all_cb_data);

// Execute the load/store callback function on a host buffer, to
// ensure that the reference host FFT is comparable to a device FFT
// that would run the same callbacks.
void apply_load_callback(const fft_params& params, std::vector<hostbuf>& input);
void apply_store_callback(const fft_params& params, std::vector<hostbuf>& output);

#endif
