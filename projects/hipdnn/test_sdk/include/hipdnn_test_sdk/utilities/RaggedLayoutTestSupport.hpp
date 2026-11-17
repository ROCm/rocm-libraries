// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

// Test support for ragged (RFC 0014) tensors: the layout table of RFC 0014 §4.5.1 and INT32 aux
// tensors (ragged_offset, seq_len) at a batch stride.
namespace hipdnn_test_sdk::utilities::ragged_layout_test
{

// Value of the slots between strided aux values; any read of one is out of range for every layout.
inline constexpr int32_t STRIDED_AUX_POISON = 1 << 20;

// `values` laid out at batch stride `stride` (value i at i * stride), the slots between them
// holding STRIDED_AUX_POISON.
inline std::vector<int32_t> stridedAuxBuffer(const std::vector<int32_t>& values, int64_t stride)
{
    std::vector<int32_t> buffer((values.size() - 1) * static_cast<size_t>(stride) + 1,
                                STRIDED_AUX_POISON);
    for(size_t i = 0; i < values.size(); ++i)
    {
        buffer[i * static_cast<size_t>(stride)] = values[i];
    }
    return buffer;
}

// INT32 [count, 1, 1, 1] aux tensor holding stridedAuxBuffer(values, stride) at batch stride
// `stride`.
inline std::shared_ptr<hipdnn_data_sdk::utilities::Tensor<int32_t>>
    makeInt32Aux(const std::vector<int32_t>& values, int64_t stride = 1)
{
    auto aux = std::make_shared<hipdnn_data_sdk::utilities::Tensor<int32_t>>(
        std::vector<int64_t>{static_cast<int64_t>(values.size()), 1, 1, 1},
        std::vector<int64_t>{stride, 1, 1, 1});
    const auto buffer = stridedAuxBuffer(values, stride);
    std::copy(buffer.begin(), buffer.end(), aux->memory().hostData());
    aux->memory().markHostModified();
    return aux;
}

// A row of RFC 0014 §4.5.1: logical SDPA dims [B, H, S, D] (sequence axis 2), strides, the stored
// ragged_offset values (B + 1, times `multiplier` they are element-unit origins), an optional
// seq_len, and the lengths, element count and element space RFC 0014 §4.5 gives it. `rejected`
// rows are aliasing layouts.
struct RaggedLayoutCase
{
    std::string id;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::vector<int32_t> offsets;
    int64_t multiplier = 1;
    std::vector<int32_t> seqLen; // empty: lengths come from the offset blocks
    std::vector<int64_t> lengths;
    size_t count = 0;
    size_t space = 0;
    bool rejected = false;

    int64_t batch() const
    {
        return dims[0];
    }

    // Element-unit origin of batch b.
    int64_t origin(int64_t b) const
    {
        return static_cast<int64_t>(offsets[static_cast<size_t>(b)]) * multiplier;
    }

    // Element address of logical (b, h, s, d).
    int64_t address(int64_t b, int64_t h, int64_t s, int64_t d) const
    {
        return origin(b) + h * strides[1] + s * strides[2] + d * strides[3];
    }

    // The offset table as an INT32 [B + 1, 1, 1, 1] aux.
    std::shared_ptr<hipdnn_data_sdk::utilities::ITensor> offsetTensor() const
    {
        return makeInt32Aux(offsets);
    }

    // The seq_len as an INT32 [B, 1, 1, 1] aux, or null without one.
    std::shared_ptr<hipdnn_data_sdk::utilities::ITensor> seqLenTensor() const
    {
        return seqLen.empty() ? nullptr : makeInt32Aux(seqLen);
    }
};

// Rows L1-L15 of RFC 0014 §4.5.1 plus L5s, L7a-d, L9m and L7bm (offsets and seq_len are element
// units unless noted).
inline const std::vector<RaggedLayoutCase>& raggedLayoutCases()
{
    static const std::vector<RaggedLayoutCase> s_cases = {
        // id, dims, strides, offsets, multiplier, seqLen, lengths, count, space, rejected
        {"L1", {2, 2, 4, 2}, {16, 2, 4, 1}, {0, 8, 20}, 1, {}, {2, 3}, 20, 20},
        {"L2", {2, 4, 3, 8}, {288, 8, 96, 1}, {0, 192, 480}, 1, {}, {2, 3}, 160, 416},
        {"L3", {2, 4, 3, 8}, {288, 8, 96, 1}, {0, 0, 0}, 1, {}, {0, 0}, 0, 0},
        {"L4", {2, 4, 8, 1}, {32, 1, 4, 1}, {0, 20, 40}, 1, {}, {5, 5}, 40, 40},
        {"L5", {2, 4, 5, 8}, {160, 8, 32, 1}, {0, 256, 512}, 1, {}, {5, 5}, 320, 416},
        {"L5s", {2, 4, 5, 8}, {160, 8, 32, 1}, {0, 256, 512}, 1, {3, 5}, {3, 5}, 256, 416},
        {"L6", {2, 2, 5, 2}, {20, 10, 2, 1}, {0, 4, 10}, 1, {}, {2, 3}, 20, 20},
        {"L7a", {2, 4, 3, 1}, {3, 5, 1, 1}, {0, 2, 5}, 1, {}, {2, 3}, 20, 20},
        {"L7b", {2, 4, 3, 1}, {12, 1, 4, 1}, {0, 8, 20}, 1, {}, {2, 3}, 20, 20},
        {"L7c", {2, 4, 3, 1}, {3, 10, 1, 1}, {0, 2, 5}, 1, {}, {2, 3}, 20, 35},
        {"L7d", {2, 2, 3, 1}, {3, 7, 1, 1}, {0, 4, 7}, 1, {2, 3}, {2, 3}, 10, 14},
        {"L8", {2, 1, 4, 2}, {8, 2, 2, 1}, {4, 8, 14}, 1, {}, {2, 3}, 10, 14},
        {"L9", {2, 1, 4, 2}, {8, 2, 2, 1}, {10, 0, 0}, 1, {2, 3}, {2, 3}, 10, 14},
        {"L9m", {2, 1, 4, 2}, {8, 2, 2, 1}, {5, 0, 0}, 2, {2, 3}, {2, 3}, 10, 14},
        {"L7bm", {2, 4, 3, 1}, {12, 1, 4, 1}, {0, 2, 5}, 4, {}, {2, 3}, 20, 20},
        {"L10", {2, 1, 4, 2}, {8, 2, 2, 1}, {0, 2, 0}, 1, {3, 3}, {}, 0, 0, true},
        {"L11", {2, 2, 4, 2}, {16, 8, 2, 1}, {0, 4, 10}, 1, {}, {}, 0, 0, true},
        {"L12", {2, 2, 3, 2}, {6, 0, 2, 1}, {0, 4, 10}, 1, {}, {2, 3}, 20, 10},
        {"L13", {2, 1, 4, 2}, {8, 2, 2, 1}, {0, 0, 6}, 1, {}, {0, 3}, 6, 6},
        {"L14", {2, 1, 4, 2}, {8, 2, 2, 1}, {0, 4, 12}, 1, {2, 3}, {2, 3}, 10, 10},
        {"L15", {2, 1, 4, 2}, {8, 2, 2, 1}, {6, 0, 0}, 1, {0, 3}, {0, 3}, 6, 6},
    };
    return s_cases;
}

inline const RaggedLayoutCase& raggedLayoutCase(const std::string& id)
{
    for(const auto& layout : raggedLayoutCases())
    {
        if(layout.id == id)
        {
            return layout;
        }
    }
    throw std::invalid_argument("unknown ragged layout " + id);
}

} // namespace hipdnn_test_sdk::utilities::ragged_layout_test
