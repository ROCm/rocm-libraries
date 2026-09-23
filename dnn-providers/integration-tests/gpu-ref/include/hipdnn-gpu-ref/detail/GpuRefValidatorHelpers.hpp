// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{
namespace detail
{

// Shared argument struct — single definition used by both host and device (HipRTC).
#include <GpuRefValidatorArgs.h> // NOLINT(misc-include-cleaner)

// Threads per block for every validator launch. Also compiled into the kernels as
// LOCAL_SIZE, which validateRms sizes its shared-memory reduction by.
constexpr int64_t VALIDATOR_BLOCK_SIZE = 256;

std::vector<std::string> buildValidatorDefines(const char* dataType, const char* computeType);

void launchValidatorKernel(hipFunction_t function, int64_t totalElements, ValidatorArgs& args);
void launchValidatorKernel(hipFunction_t function, int64_t totalElements, RmsValidatorArgs& args);

// Fills the strided-layout fields of a validator's args. Leaves ndim at 0 — the packed
// fast path — when both tensors are packed.
template <class Args>
void setStridedLayout(Args& args,
                      const hipdnn_data_sdk::utilities::ITensor& reference,
                      const hipdnn_data_sdk::utilities::ITensor& implementation)
{
    if(reference.isPacked() && implementation.isPacked())
    {
        return;
    }

    const auto& refStrides = reference.strides();
    const auto& implStrides = implementation.strides();
    const auto& dims = reference.dims();
    const auto ndim = dims.size();
    if(ndim > 8)
    {
        throw std::runtime_error("GPU validator supports up to 8 dimensions, got "
                                 + std::to_string(ndim));
    }
    args.ndim = static_cast<int>(ndim);
    for(size_t d = 0; d < ndim; ++d)
    {
        args.refStrides[d] = static_cast<long long>(refStrides[d]);
        args.implStrides[d] = static_cast<long long>(implStrides[d]);
        args.dims[d] = static_cast<long long>(dims[d]);
    }
}

} // namespace detail
} // namespace hipdnn_gpu_ref
