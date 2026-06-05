// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <miopen/env.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/kernel_tuning_mode.hpp>
#include <miopen/logger.hpp>
#include <miopen/oclkernel.hpp>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)

static std::string DimToFormattedString(const size_t* dims, size_t count)
{
    std::stringstream ss;
    ss << '{';
    for(size_t i = 0; i < count; ++i)
    {
        if(i > 0)
            ss << ", ";
        else
            ss << ' ';
        ss << dims[i];
    }
    ss << " }";
    return ss.str();
}

void OCLKernelInvoke::run() const
{
    MIOPEN_LOG_I2("kernel_name = "
                  << GetName() << ", work_dim = " << work_dim << ", global_work_offset = "
                  << DimToFormattedString(global_work_offset.data(), work_dim)
                  << ", global_work_dim = " << DimToFormattedString(gdims.data(), work_dim)
                  << ", local_work_dim = " << DimToFormattedString(ldims.data(), work_dim));

    MIOPEN_HANDLE_LOCK

    const auto& arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
    {
        MIOPEN_THROW("MIOPEN_DEVICE_ARCH used, escaping launching kernel");
    }

    cl_event ev;
    /* way to run OCL group larger than 256
     * hack to ensure local_size == 0, just checking that the 1st dim is 0
     * may want to use a better solution*/
    cl_int status = clEnqueueNDRangeKernel(queue,
                                           kernel.get(),
                                           work_dim,
                                           ((work_dim == 0) ? nullptr : global_work_offset.data()),
                                           gdims.data(),
                                           ((ldims[0] == 0) ? nullptr : ldims.data()),
                                           0,
                                           nullptr,
                                           callback ? &ev : nullptr);

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "Running kernel failed: ");
    }
    // If profiling is enable, then callback exists.
    if(callback)
    {
        clWaitForEvents(1, &ev);
        // Profiling is enable when Performance Logs are see `Run` in handleocl.cpp so this is
        // nested in callback Log to JSON accumulator
        if(IsLoggingKernel())
        {
            // Match handleocl.cpp `SetProfilingResult`
            size_t st, end;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(size_t), &st, nullptr);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(size_t), &end, nullptr);
            float elapsed_time = static_cast<float>(end - st) * 1.0e-6; // NOLINT

            auto kernel_name        = GetName();
            const bool is_transpose = IsTransposeOrTransformKernel(kernel_name);
            AddKernelToJsonAccumulator(kernel_name, elapsed_time, is_transpose);
        }
        callback(ev);
    }
}

std::string OCLKernelInvoke::GetName() const
{
    std::array<char, 200> buffer{};

    cl_int status =
        clGetKernelInfo(kernel.get(), CL_KERNEL_FUNCTION_NAME, 200, buffer.data(), nullptr);

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "Error getting kernel name");
    }
    return buffer.data();
}

OCLKernelInvoke OCLKernel::Invoke(cl_command_queue q, std::function<void(cl_event&)> callback) const
{
#ifndef NDEBUG
    MIOPEN_LOG_I(GetName());
#endif
    OCLKernelInvoke result{q, kernel, gdims.size(), {}, {}, {}, callback};
    std::copy(gdims.begin(), gdims.end(), result.gdims.begin());
    std::copy(ldims.begin(), ldims.end(), result.ldims.begin());
    return result;
}

std::string OCLKernel::GetName() const
{
    std::array<char, 200> buffer{};

    cl_int status =
        clGetKernelInfo(kernel.get(), CL_KERNEL_FUNCTION_NAME, 200, buffer.data(), nullptr);

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "Error getting kernel name");
    }
    return buffer.data();
}

} // namespace miopen
