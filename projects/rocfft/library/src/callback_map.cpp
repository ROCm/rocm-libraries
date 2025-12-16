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

#include "callback_map.h"
#include "plan.h"
#include "transform.h"

std::map<int, device_callback_t> DeviceCallbackMap(const rocfft_execution_info_t*   info,
                                                   const rocfft_plan_description_t& desc)
{
    // tolerate user not providing an execution_info
    rocfft_execution_info_t exec_info;
    if(info)
        exec_info = *info;

    int local_device = 0;
    if(hipGetDevice(&local_device) != hipSuccess)
        throw std::runtime_error("failed to get device");

    std::map<int, device_callback_t> callbacks;

    auto set_field_callback = [&callbacks](const std::vector<rocfft_field_t>& fields,
                                           void**                             src_fn,
                                           void**                             src_data,
                                           bool                               load) {
        size_t src_idx = 0;
        for(const auto& f : fields)
        {
            for(const auto& b : f.bricks)
            {
                int device_id = b.location.device;

                if(load)
                {
                    if(src_fn && src_fn[src_idx] != nullptr)
                    {
                        // don't overwrite existing callbacks
                        if(callbacks[device_id].load_fn != nullptr)
                        {
                            throw std::runtime_error("Conflicting load callbacks for device "
                                                     + std::to_string(device_id));
                        }
                        callbacks[device_id].load_fn = src_fn[src_idx];
                    }

                    if(src_data && src_data[src_idx] != nullptr)
                    {
                        callbacks[device_id].load_data = src_data[src_idx];
                    }
                }
                else
                {
                    if(src_fn && src_fn[src_idx] != nullptr)
                    {
                        // don't overwrite existing callbacks
                        if(callbacks[device_id].store_fn != nullptr)
                        {
                            throw std::runtime_error("Conflicting store callbacks for device "
                                                     + std::to_string(device_id));
                        }
                        callbacks[device_id].store_fn = src_fn[src_idx];
                    }

                    if(src_data && src_data[src_idx] != nullptr)
                    {
                        callbacks[device_id].store_data = src_data[src_idx];
                    }
                }
                ++src_idx;
            }
        }
    };

    if(desc.inFields.empty())
    {
        // we have at most one load callback
        if(exec_info.load_cb_fns)
        {
            callbacks[local_device].load_fn = exec_info.load_cb_fns[0];
            if(exec_info.load_cb_data)
                callbacks[local_device].load_data = exec_info.load_cb_data[0];
        }
    }
    else
    {
        set_field_callback(desc.inFields, exec_info.load_cb_fns, exec_info.load_cb_data, true);
    }

    if(desc.outFields.empty())
    {
        // we have at most one store callback
        if(exec_info.store_cb_fns)
        {
            callbacks[local_device].store_fn = exec_info.store_cb_fns[0];
            if(exec_info.store_cb_data)
                callbacks[local_device].store_data = exec_info.store_cb_data[0];
        }
    }
    else
    {
        set_field_callback(desc.outFields, exec_info.store_cb_fns, exec_info.store_cb_data, false);
    }

    return callbacks;
}