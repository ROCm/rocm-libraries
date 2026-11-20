// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "exec_info.h"
#include "../../shared/rocfft_hip.h"
#include "logging.h"
#include "plan.h"

rocfft_execution_info_t::rocfft_execution_info_t()
{
    int deviceCount = rocfft_scoped_device::device_count();
    workBuffers.resize(deviceCount);
    rocfft_streams.resize(deviceCount);
}

rocfft_execution_info_internal::rocfft_execution_info_internal(
    const rocfft_execution_info_t* user_info, const rocfft_plan_t& plan)
    : user_info(user_info)
{
    validate_jit_data_ptr_count(plan);

    if(user_info)
    {
        execWorkBuffers.resize(user_info->workBuffers.size());
    }
    else
    {
        int deviceCount = rocfft_scoped_device::device_count();
        execWorkBuffers.resize(deviceCount);
    }

    // now ensure that each of the work buffers is big enough,
    // allocating if necessary
    bool have_multigpu_plan
        = std::any_of(plan.multiPlan.begin(), plan.multiPlan.end(), [](const auto& item) {
              auto exec = dynamic_cast<const ExecPlan*>(item.get());
              return exec && exec->mgpuPlan;
          });
    ensure_work_buffer_size(plan.tempBufferUserAllocs, have_multigpu_plan);

    // now that all of the work buffers is big enough, assign
    // concrete pointers to each of this rank's temp buffers
    for(const auto& tempBuf : plan.tempBuffers)
    {
        const auto& location = tempBuf->get_location();
        if(location.comm_rank != plan.desc.get_local_comm_rank())
            continue;
        tempBufferPtrs.emplace(
            tempBuf.get(),
            get_work_buffer(location.device).data_offset(tempBuf->get_offset_bytes()));
    }
}

void rocfft_execution_info_internal::ensure_work_buffer_size(
    const std::vector<size_t>& sizes_bytes_per_device, bool have_multigpu_plan)
{
    // assert that all our vectors are the same length
    if(execWorkBuffers.size() != sizes_bytes_per_device.size()
       || (user_info && execWorkBuffers.size() != user_info->workBuffers.size()))
        throw std::logic_error("unexpected work buf lengths");

    // go over each device's memory requirements
    for(size_t device = 0; device < execWorkBuffers.size(); ++device)
    {
        const size_t size_bytes = sizes_bytes_per_device[device];
        if(size_bytes == 0)
            continue;

        // check if user gave us a buffer
        if(user_info)
        {
            auto& buf = user_info->workBuffers[device];
            if(buf)
            {
                if(buf.size() >= size_bytes)
                {
                    continue;
                }
                else
                {
                    // not big enough
                    if(LOG_TRACE_ENABLED())
                        (*LogSingleton::GetInstance().GetTraceOS())
                            << "user work buffer too small" << std::endl;
                    throw rocfft_status_invalid_work_buffer;
                }
            }
        }

        // no user work buffer was specified, allocate one
        if(LOG_TRACE_ENABLED())
            (*LogSingleton::GetInstance().GetTraceOS())
                << "user work buffer not specified for device " << device << ", allocating size "
                << size_bytes << std::endl;
        rocfft_scoped_device dev(device);

        // Async stream-ordered allocation is preferred as it reduces
        // host blocking, but it also requires device support.  So
        // query that before attempting it and fall back to regular
        // alloc if support is not there
        int pool_supported = 0;

        // NOTE: multi-GPU plans don't always respect the
        // user-provided yet, so avoid stream-ordered allocs in that
        // case.

        // If query fails, assume pool is not supported
        if(!have_multigpu_plan
           && hipDeviceGetAttribute(&pool_supported, hipDeviceAttributeMemoryPoolsSupported, device)
                  == hipSuccess
           && pool_supported)
        {
            // Stream-ordered allocation can also work on the null stream
            hipStream_t user_stream = user_info ? user_info->rocfft_streams[device] : nullptr;
            if(execWorkBuffers[device].alloc_async(size_bytes, user_stream) != hipSuccess)
                throw std::runtime_error("stream-ordered work buffer allocation failure");
        }
        else
        {
            if(execWorkBuffers[device].alloc(size_bytes) != hipSuccess)
                throw std::runtime_error("work buffer allocation failure");
        }
    }
}

const gpubuf& rocfft_execution_info_internal::get_work_buffer(int device) const
{
    if(user_info && user_info->workBuffers[device])
        return user_info->workBuffers[device];
    return execWorkBuffers[device];
}

void** rocfft_execution_info_internal::get_load_cb_fns() const
{
    if(user_info)
        return user_info->load_cb_fns;
    return nullptr;
}

void** rocfft_execution_info_internal::get_load_cb_data() const
{
    if(user_info)
        return user_info->load_cb_data;
    return nullptr;
}

size_t rocfft_execution_info_internal::get_load_cb_lds_bytes() const
{
    if(user_info)
        return user_info->load_cb_lds_bytes;
    return 0;
}

void** rocfft_execution_info_internal::get_store_cb_fns() const
{
    if(user_info)
        return user_info->store_cb_fns;
    return nullptr;
}

void** rocfft_execution_info_internal::get_store_cb_data() const
{
    if(user_info)
        return user_info->store_cb_data;
    return nullptr;
}

size_t rocfft_execution_info_internal::get_store_cb_lds_bytes() const
{
    if(user_info)
        return user_info->store_cb_lds_bytes;
    return 0;
}

hipStream_t rocfft_execution_info_internal::get_user_stream(int device) const
{
    if(user_info)
        return user_info->rocfft_streams[device];
    return nullptr;
}

void rocfft_execution_info_internal::validate_jit_data_ptr_count(const rocfft_plan_t& plan) const
{
    if(!user_info)
        return;

    // Check load cb data against input pointers
    auto local_comm_rank = plan.desc.get_local_comm_rank();
    if(!user_info->load_cb_data_jit.empty()
       && user_info->load_cb_data_jit.size()
              != plan.desc.count_pointers(
                  plan.desc.inFields, plan.desc.inArrayType, local_comm_rank))
        throw std::runtime_error("load JIT callback data count doesn't match input brick count");
    // Check store cb data against output pointers
    if(!user_info->store_cb_data_jit.empty()
       && user_info->store_cb_data_jit.size()
              != plan.desc.count_pointers(
                  plan.desc.outFields, plan.desc.outArrayType, local_comm_rank))
        throw std::runtime_error("store JIT callback data count doesn't match output brick count");
}
