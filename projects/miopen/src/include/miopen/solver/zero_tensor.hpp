// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

namespace miopen {
namespace solver {

inline void ZeroTensor(const Handle& handle, const TensorDescriptor& tensorDesc, Data_t tensorData)
{
    // SetTensor is required for non-packed tensors, but is also slower.
    // Use faster clear if possible.
    if(tensorDesc.IsPacked())
    {
        HipEventProfiler pfr(handle);

        auto status = hipMemsetAsync(tensorData, 0, tensorDesc.GetNumBytes(), handle.GetStream());
        if(status != hipSuccess)
        {
            MIOPEN_THROW_HIP_STATUS(status, "hipMemsetAsync() failed");
        }
    }
    else
    {
        auto zero = 0.0f;
        SetTensor(handle, tensorDesc, tensorData, &zero);
    }
}

} // namespace solver
} // namespace miopen
