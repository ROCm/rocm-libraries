// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/detail/GpuRefValidatorHelpers.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>

#include <limits>
#include <stdexcept>
#include <string>

namespace hipdnn_gpu_ref
{
namespace detail
{

namespace
{

void launch(hipFunction_t function, int64_t totalElements, void* args, size_t argsSize)
{
    auto gridSize = (totalElements + VALIDATOR_BLOCK_SIZE - 1) / VALIDATOR_BLOCK_SIZE;

    if(gridSize > static_cast<int64_t>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::runtime_error("Grid size exceeds hipModuleLaunchKernel limit");
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    throwOnHipError(hipModuleLaunchKernel(function,
                                          static_cast<unsigned int>(gridSize),
                                          1,
                                          1,
                                          static_cast<unsigned int>(VALIDATOR_BLOCK_SIZE),
                                          1,
                                          1,
                                          0,
                                          nullptr,
                                          nullptr,
                                          config),
                    "launchValidatorKernel: hipModuleLaunchKernel failed");

    throwOnHipError(hipDeviceSynchronize(), "launchValidatorKernel: hipDeviceSynchronize failed");
}

} // namespace

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
    launch(function, totalElements, &args, sizeof(ValidatorArgs));
}

void launchValidatorKernel(hipFunction_t function, int64_t totalElements, RmsValidatorArgs& args)
{
    launch(function, totalElements, &args, sizeof(RmsValidatorArgs));
}

} // namespace detail
} // namespace hipdnn_gpu_ref
