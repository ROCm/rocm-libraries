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
    defines.emplace_back(std::string("-DLOCAL_SIZE=") + std::to_string(VALIDATOR_BLOCK_SIZE));
    return defines;
}

void launchValidatorKernel(hipFunction_t function, int64_t totalElements, ValidatorArgs& args)
{
    launchKernelForElements(
        function, totalElements, &args, sizeof(ValidatorArgs), VALIDATOR_BLOCK_SIZE);
}

void launchValidatorKernel(hipFunction_t function, int64_t totalElements, RmsValidatorArgs& args)
{
    launchKernelForElements(
        function, totalElements, &args, sizeof(RmsValidatorArgs), VALIDATOR_BLOCK_SIZE);
}

} // namespace detail
} // namespace hipdnn_gpu_ref
