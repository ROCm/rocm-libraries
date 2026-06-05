// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_gpu_ref/detail/GpuRefValidatorHelpers.hpp>

#include <hipdnn_gpu_ref/detail/GpuRefHipError.hpp>

#include <limits>
#include <stdexcept>
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
    const int64_t blockSize = 256;
    auto gridSize = (totalElements + blockSize - 1) / blockSize;

    if(gridSize > static_cast<int64_t>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::runtime_error("Grid size exceeds hipModuleLaunchKernel limit");
    }

    auto argsSize = sizeof(ValidatorArgs);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    throwOnHipError(hipModuleLaunchKernel(function,
                                          static_cast<unsigned int>(gridSize),
                                          1,
                                          1,
                                          static_cast<unsigned int>(blockSize),
                                          1,
                                          1,
                                          0,
                                          nullptr,
                                          nullptr,
                                          config),
                    "launchValidatorKernel: hipModuleLaunchKernel failed");

    throwOnHipError(hipDeviceSynchronize(), "launchValidatorKernel: hipDeviceSynchronize failed");
}

} // namespace detail
} // namespace hipdnn_gpu_ref
