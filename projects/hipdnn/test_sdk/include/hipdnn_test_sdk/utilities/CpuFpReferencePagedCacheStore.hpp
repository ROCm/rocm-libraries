// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_test_sdk::utilities
{

/// CPU reference for the paged KV-cache store (paged_cache_store_attributes.fbs).
///
/// The container uses the SDPA paged layout `[num_blocks, H, block_size, D]`. Two token
/// layouts are accepted:
///  - token-packed: `x` `[T, H, D]` with `slotMapping` `[T]`;
///  - SDPA-native:  `x` `[B, H, S, D]` (any strides) with `slotMapping` `[B, S]`.
/// For every token with `slot = slotMapping[token] >= 0`:
///   `container[slot / block_size, h, slot % block_size, :] = cast(x[token, h, :] * scale)`
/// Tokens with a negative slot are skipped, and every other container element keeps its
/// contents. A slot at or past `num_blocks * block_size`, or two tokens addressing the same
/// slot, is rejected. The scale is a FLOAT scalar; absent, it is 1. The product is formed in
/// `ComputeDataType` and cast once to the container type (FLOAT, HALF, BFLOAT16 or an FP8
/// type).
class CpuFpReferencePagedCacheStore
{
public:
    template <class XDataType,
              class SlotDataType,
              class ContainerDataType,
              class ComputeDataType = float>
    static void store(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                      const hipdnn_data_sdk::utilities::TensorBase<SlotDataType>& slotMapping,
                      hipdnn_data_sdk::utilities::TensorBase<ContainerDataType>& container,
                      const hipdnn_data_sdk::utilities::TensorBase<float>* scale = nullptr)
    {
        validate(x, slotMapping, container, scale);

        const auto& xDims = x.dims();
        const auto& cDims = container.dims();
        const bool tokenPacked = xDims.size() == 3;
        const int64_t batch = tokenPacked ? 1 : xDims[0];
        const int64_t seqLen = tokenPacked ? xDims[0] : xDims[2];
        const int64_t heads = xDims[1];
        const int64_t headDim = xDims.back();
        const int64_t blockSize = cDims[2];
        const int64_t slotCount = cDims[0] * blockSize;

        auto scaleValue = static_cast<ComputeDataType>(1.0f);
        if(scale != nullptr)
        {
            const std::vector<int64_t> origin(scale->dims().size(), 0);
            scaleValue = static_cast<ComputeDataType>(scale->getHostValue(origin));
        }

        std::vector<bool> written(static_cast<size_t>(slotCount), false);
        for(int64_t b = 0; b < batch; ++b)
        {
            for(int64_t s = 0; s < seqLen; ++s)
            {
                const auto slot = static_cast<int64_t>(
                    tokenPacked ? slotMapping.getHostValue(s) : slotMapping.getHostValue(b, s));
                if(slot < 0)
                {
                    continue;
                }
                if(slot >= slotCount)
                {
                    throw std::out_of_range("CpuFpReferencePagedCacheStore: slot "
                                            + std::to_string(slot) + " is outside the "
                                            + std::to_string(slotCount) + " container slots");
                }
                const auto slotIndex = static_cast<size_t>(slot);
                if(written[slotIndex])
                {
                    throw std::invalid_argument("CpuFpReferencePagedCacheStore: slot "
                                                + std::to_string(slot)
                                                + " is addressed by two tokens");
                }
                written[slotIndex] = true;

                const int64_t block = slot / blockSize;
                const int64_t offset = slot % blockSize;
                for(int64_t h = 0; h < heads; ++h)
                {
                    for(int64_t d = 0; d < headDim; ++d)
                    {
                        const auto value
                            = tokenPacked ? x.getHostValue(s, h, d) : x.getHostValue(b, h, s, d);
                        const auto scaled = static_cast<ComputeDataType>(value) * scaleValue;
                        container.setHostValue(
                            static_cast<ContainerDataType>(scaled), block, h, offset, d);
                    }
                }
            }
        }
        container.memory().markHostModified();
    }

private:
    template <class XDataType, class SlotDataType, class ContainerDataType>
    static void validate(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                         const hipdnn_data_sdk::utilities::TensorBase<SlotDataType>& slotMapping,
                         const hipdnn_data_sdk::utilities::TensorBase<ContainerDataType>& container,
                         const hipdnn_data_sdk::utilities::TensorBase<float>* scale)
    {
        const auto& xDims = x.dims();
        const auto& sDims = slotMapping.dims();
        const auto& cDims = container.dims();
        if(xDims.size() == 3)
        {
            if(sDims.size() != 1 || sDims[0] != xDims[0])
            {
                throw std::invalid_argument(
                    "CpuFpReferencePagedCacheStore: slot_mapping must be [T] for x [T, H, D]");
            }
        }
        else if(xDims.size() == 4)
        {
            if(sDims.size() != 2 || sDims[0] != xDims[0] || sDims[1] != xDims[2])
            {
                throw std::invalid_argument(
                    "CpuFpReferencePagedCacheStore: slot_mapping must be [B, S] for x "
                    "[B, H, S, D]");
            }
        }
        else
        {
            throw std::invalid_argument(
                "CpuFpReferencePagedCacheStore: x must be [T, H, D] or [B, H, S, D]");
        }
        if(cDims.size() != 4 || cDims[0] <= 0 || cDims[2] <= 0 || cDims[1] != xDims[1]
           || cDims[3] != xDims.back())
        {
            throw std::invalid_argument("CpuFpReferencePagedCacheStore: container must be "
                                        "[num_blocks, H, block_size, D] matching x");
        }
        if(scale != nullptr && scale->elementCount() != 1)
        {
            throw std::invalid_argument(
                "CpuFpReferencePagedCacheStore: scale must have exactly one element");
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
