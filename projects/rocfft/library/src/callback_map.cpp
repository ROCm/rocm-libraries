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
#include "exec_info.h"
#include "plan.h"

std::map<int, device_callback_t> DeviceCallbackMap(const rocfft_execution_info_internal& exec_info,
                                                   const rocfft_plan_description_t&      desc,
                                                   int local_comm_rank)
{
    int local_device = 0;
    if(hipGetDevice(&local_device) != hipSuccess)
        throw std::runtime_error("failed to get device");

    std::map<int, device_callback_t> callbacks;

    auto set_field_callback = [=, &callbacks](const std::vector<rocfft_field_t>& fields,
                                              void**                             src_fn,
                                              void**                             src_data,
                                              bool                               load) {
        size_t src_idx = 0;
        for(const auto& f : fields)
        {
            for(const auto& b : f.bricks)
            {
                int device_id = b.location.device;
                if(b.location.comm_rank != local_comm_rank)
                    continue;

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

    auto load_cb_fns   = exec_info.get_load_cb_fns();
    auto load_cb_data  = exec_info.get_load_cb_data();
    auto store_cb_fns  = exec_info.get_store_cb_fns();
    auto store_cb_data = exec_info.get_store_cb_data();

    if(desc.inFields.empty())
    {
        // we have at most one load callback
        if(load_cb_fns)
        {
            callbacks[local_device].load_fn = load_cb_fns[0];
            if(load_cb_data)
                callbacks[local_device].load_data = load_cb_data[0];
        }
    }
    else
    {
        set_field_callback(desc.inFields, load_cb_fns, load_cb_data, true);
    }

    if(desc.outFields.empty())
    {
        // we have at most one store callback
        if(store_cb_fns)
        {
            callbacks[local_device].store_fn = store_cb_fns[0];
            if(store_cb_data)
                callbacks[local_device].store_data = store_cb_data[0];
        }
    }
    else
    {
        set_field_callback(desc.outFields, store_cb_fns, store_cb_data, false);
    }

    return callbacks;
}
