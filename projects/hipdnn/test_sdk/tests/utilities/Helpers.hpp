// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace hipdnn_data_sdk::helpers
{

template <typename T>
utilities::MigratableMemory<T> createBuffer(size_t size, T mult)
{
    utilities::MigratableMemory<T> buffer(size);

    T* data = buffer.hostData();

    for(size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<T>(static_cast<float>(i)) * mult;
    }

    return buffer;
}

template <typename T>
utilities::MigratableMemory<T> createConstantBuffer(size_t size, T value)
{
    utilities::MigratableMemory<T> buffer(size);

    T* data = buffer.hostData();

    for(size_t i = 0; i < size; ++i)
    {
        data[i] = value;
    }

    return buffer;
}

// Logical SDPA [B=2, H=2, S_max=4, D=2] with BSHD strides, so one sequence row is H*D = 4
// elements. S_max exceeds every batch's row count, so each batch has out-of-block positions,
// and those of the last batch run past the end of the buffer.
inline const std::vector<int64_t> RAGGED_SDPA_DIMS = {2, 2, 4, 2};
inline const std::vector<int64_t> RAGGED_SDPA_STRIDES = {16, 2, 4, 1};
inline const std::vector<int32_t> RAGGED_SDPA_ROW_OFFSETS = {0, 2, 5};
inline constexpr int64_t RAGGED_SDPA_ROW_ELEMENTS = 4;

template <typename T>
utilities::RaggedTensor<T> createRaggedSdpaTensor(float fillValue,
                                                  const std::vector<int32_t>& rowOffsets
                                                  = RAGGED_SDPA_ROW_OFFSETS)
{
    auto offsets = std::make_shared<utilities::Tensor<int32_t>>(
        std::vector<int64_t>{static_cast<int64_t>(rowOffsets.size()), 1, 1, 1});
    for(size_t i = 0; i < rowOffsets.size(); ++i)
    {
        offsets->setHostValue(rowOffsets[i], static_cast<int64_t>(i), 0, 0, 0);
    }

    utilities::RaggedTensor<T> tensor(RAGGED_SDPA_DIMS,
                                      RAGGED_SDPA_STRIDES,
                                      utilities::SDPA_SEQ_AXIS,
                                      std::move(offsets),
                                      std::nullopt,
                                      RAGGED_SDPA_ROW_ELEMENTS);
    tensor.fillTensorWithValue(fillValue);
    return tensor;
}

} // namespace hipdnn_data_sdk::helpers
