// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/detail/GpuRefValidatorHelpers.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefLaunch.hpp>

#include <algorithm>
#include <functional>
#include <numeric>
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

void setRaggedLayout(ValidatorArgs& args,
                     const hipdnn_data_sdk::utilities::RaggedIterationInfo& layout,
                     const std::vector<int64_t>& dims,
                     hipdnn_data_sdk::utilities::MigratableMemory<long long>& rowOffsets)
{
    if(args.ndim == 0)
    {
        throw std::logic_error("setRaggedLayout requires the strided layout set first");
    }

    rowOffsets.resize(layout.rowOffsets.size());
    std::copy(layout.rowOffsets.begin(), layout.rowOffsets.end(), rowOffsets.hostData());

    args.rowOffsets = static_cast<const long long*>(rowOffsets.deviceData());
    args.seqAxis = layout.seqAxis;
    args.seqStride = static_cast<long long>(layout.seqStride);
    args.totalElements = std::accumulate(
        dims.begin(), dims.end(), static_cast<long long>(1), std::multiplies<long long>());
}

} // namespace detail
} // namespace hipdnn_gpu_ref
