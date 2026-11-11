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
                     hipdnn_data_sdk::utilities::MigratableMemory<long long>& batchTable)
{
    if(args.ndim == 0)
    {
        throw std::logic_error("setRaggedLayout requires the strided layout set first");
    }
    if(dims.empty() || layout.origins.size() != layout.lengths.size()
       || static_cast<int64_t>(layout.origins.size()) != dims[0])
    {
        throw std::invalid_argument("setRaggedLayout needs one origin and one length per batch");
    }

    const auto batchCount = layout.origins.size();
    batchTable.resize(2 * batchCount);
    auto* host = batchTable.hostData();
    std::copy(layout.origins.begin(), layout.origins.end(), host);
    std::copy(layout.lengths.begin(), layout.lengths.end(), host + batchCount);

    const auto* device = static_cast<const long long*>(batchTable.deviceData());
    args.origins = device;
    args.lengths = device + batchCount;
    args.seqAxis = layout.seqAxis;
    args.totalElements
        = std::accumulate(dims.begin(), dims.end(), static_cast<long long>(1), std::multiplies<>());
}

} // namespace detail
} // namespace hipdnn_gpu_ref
