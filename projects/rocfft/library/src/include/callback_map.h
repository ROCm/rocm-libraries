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

#ifndef ROCFFT_CALLBACK_MAP_H
#define ROCFFT_CALLBACK_MAP_H

#include <map>

// Load/store callbacks for a device, intended to be stored in a
// std::map, so that we can derive callback info from a device
// ID.
struct device_callback_t
{
    void* load_fn    = nullptr;
    void* load_data  = nullptr;
    void* store_fn   = nullptr;
    void* store_data = nullptr;
};

// Build map of callback structs, so we can know which callback
// functions run on which devices.  Index in the map is HIP device
// ID.
//
// We do this because callbacks need to be specified in the execution
// info in the order of the bricks specified in the transform, which
// might not be in any sensible order.
struct rocfft_execution_info_internal;
struct rocfft_plan_description_t;
std::map<int, device_callback_t> DeviceCallbackMap(const rocfft_execution_info_internal& exec_info,
                                                   const rocfft_plan_description_t&      desc,
                                                   int local_comm_rank);
#endif
