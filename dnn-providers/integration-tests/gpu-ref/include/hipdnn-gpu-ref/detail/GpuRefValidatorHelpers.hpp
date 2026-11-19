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
// LOCAL_SIZE, which validateRms sizes its shared-memory reduction by. That reduction
// halves its stride each step, so anything but a power of two drops elements.
constexpr int64_t VALIDATOR_BLOCK_SIZE = 256;
static_assert((VALIDATOR_BLOCK_SIZE & (VALIDATOR_BLOCK_SIZE - 1)) == 0,
              "VALIDATOR_BLOCK_SIZE must be a power of two");

std::vector<std::string> buildValidatorDefines(const char* dataType, const char* computeType);

void launchValidatorKernel(hipFunction_t function, int64_t totalElements, ValidatorArgs& args);
void launchValidatorKernel(hipFunction_t function, int64_t totalElements, RmsValidatorArgs& args);

// The kernels index every logical position of the padded dims, which for a ragged
// tensor aliases other batches' rows or runs past the end of its buffer.
inline void throwIfRagged(const hipdnn_data_sdk::utilities::ITensor& reference,
                          const hipdnn_data_sdk::utilities::ITensor& implementation)
{
    if(reference.raggedIterationInfo().has_value()
       || implementation.raggedIterationInfo().has_value())
    {
        throw std::invalid_argument("ragged tensors are not supported by the device validator");
    }
}

// Fills the strided-layout fields of a validator's args. Leaves ndim at 0 — the linear
// fast path — only when both tensors are packed in the same stride order. Packed alone
// is not enough: an NCHW-packed and an NHWC-packed tensor are both packed, and indexing
// them by the same linear offset pairs up different logical elements.
template <class Args>
void setStridedLayout(Args& args,
                      const hipdnn_data_sdk::utilities::ITensor& reference,
                      const hipdnn_data_sdk::utilities::ITensor& implementation)
{
    throwIfRagged(reference, implementation);

    const auto& refStrides = reference.strides();
    const auto& implStrides = implementation.strides();
    if(reference.isPacked() && implementation.isPacked() && refStrides == implStrides)
    {
        return;
    }

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
