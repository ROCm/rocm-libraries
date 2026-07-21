// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <vector>

// Helpers to work with device functions passed as callbacks, to be
// used by unit tests

// Data used by the test callback functions
struct callback_test_data
{
    // scalar to modify the input/output with
    double scalar;
};

inline constexpr const char* callback_test_data_jit = R"(
struct callback_test_data
{
    // scalar to modify the input/output with
    double scalar;
};
)";

// Get a function pointer (on the host) to a load callback device
// function on the current device
void* get_load_callback_funcptr(fft_array_type itype,
                                fft_precision  precision,
                                bool           round_trip_inverse = false);

// Get a function pointer (on the host) to a store callback device
// function on the current device
void* get_store_callback_funcptr(fft_array_type otype,
                                 fft_precision  precision,
                                 bool           round_trip_inverse = false);

// Get compiled SPIR-V for a load JIT callback device function
std::vector<char> get_load_callback_jit(fft_array_type itype,
                                        fft_precision  precision,
                                        bool           round_trip_inverse = false);

// Get compiled SPIR-V for a store JIT callback device function
std::vector<char> get_store_callback_jit(fft_array_type otype,
                                         fft_precision  precision,
                                         bool           round_trip_inverse = false);

// Collect load callback function pointers and data pointers for the
// given params.  We'd expect N pointers for N input bricks on the
// current multi-processing rank.
//
// Data structs are allocated on the device in all_cb_data.
void get_rank_load_callbacks_funcptr(const fft_params&                          params,
                                     std::vector<void*>&                        load_cb_func,
                                     std::vector<void*>&                        load_cb_data,
                                     bool                                       round_trip_inverse,
                                     std::vector<gpubuf_t<callback_test_data>>& all_cb_data);

// Collect store callback function pointers and data pointers for the
// given params.  We'd expect N pointers for N output bricks on the
// current multi-processing rank.
//
// Data structs are allocated on the device in all_cb_data.
void get_rank_store_callbacks_funcptr(const fft_params&                          params,
                                      std::vector<void*>&                        store_cb_func,
                                      std::vector<void*>&                        store_cb_data,
                                      bool                                       round_trip_inverse,
                                      std::vector<gpubuf_t<callback_test_data>>& all_cb_data);

// For the current rank, get a JIT callback and a vector of callback
// data pointers.  cb_data has an element for each visible HIP
// device, though pointers are only set for devices that could
// participate in the param's transform.
enum class jit_callback_op
{
    LOAD,
    STORE,
};
std::shared_ptr<fft_params::jit_cb_state_t> get_rank_jit_state(const fft_params& params,
                                                               const char*       symbol,
                                                               bool              round_trip_inverse,
                                                               jit_callback_op   type);

// Execute the load/store callback function on a host buffer, to
// ensure that the reference host FFT is comparable to a device FFT
// that would run the same callbacks.
void apply_load_callback(const fft_params& params, std::vector<hostbuf>& input);
void apply_store_callback(const fft_params& params, std::vector<hostbuf>& output);

// Compile a string of source code for use as a JIT callback.  In
// HIP, this results in a SPIR-V bitcode while on CUDA this
// produces LTO-IR data.
//
// Throws hiprtc_runtime_error when hiprtc APIs fail.
std::vector<char> compile_jit_callback(const std::string& src);

#endif
