// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/detail/GpuRefValidatorHelpers.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefLaunch.hpp>

#include <string>

namespace hipdnn_gpu_ref
{
namespace detail
{

std::vector<std::string> buildValidatorDefines(const char* dataType, const char* computeType)
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DDATA_TYPE=") + dataType);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + computeType);
    return defines;
}

void launchValidatorKernel(hipFunction_t function, int64_t totalElements, ValidatorArgs& args)
{
    launchKernelForElements(function, totalElements, &args, sizeof(args), "launchValidatorKernel");
}

} // namespace detail
} // namespace hipdnn_gpu_ref
