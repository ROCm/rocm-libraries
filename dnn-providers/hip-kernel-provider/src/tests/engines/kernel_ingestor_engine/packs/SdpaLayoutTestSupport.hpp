// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <vector>

/**
 * @file SdpaLayoutTestSupport.hpp
 * @brief The memory layouts an SDPA unit-test graph gives a (B, H, S, D) operand, and their
 *        strides: one home for the dense, unified and FlyDSL attention suites.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// Physical arrangement of a (B, H, S, D) logical tensor.
enum class SdpaLayout
{
    /// Token-major, head varying fastest: [B, S, H, D] in memory.
    BSHD,
    /// Head-major: [B, H, S, D] in memory.
    BHSD,
    /// BSHD with each token row padded past H * D: token-major, but not gap-free.
    PADDED_BSHD,
    /// Token-major with D outer to H: [B, S, D, H] in memory, so H has the unit stride.
    BSDH
};

/// Element strides of a (B, H, S, D) logical tensor laid out as @p layout.
inline std::vector<int64_t>
    sdpaStrides(SdpaLayout layout, int64_t heads, int64_t sequence, int64_t headSize)
{
    switch(layout)
    {
    case SdpaLayout::BHSD:
        return {heads * sequence * headSize, sequence * headSize, headSize, 1};
    case SdpaLayout::PADDED_BSHD:
    {
        // Elements of slack a real allocator leaves between token rows.
        constexpr int64_t ROW_PAD = 8;
        const int64_t rowStride = heads * headSize + ROW_PAD;
        return {sequence * rowStride, headSize, rowStride, 1};
    }
    case SdpaLayout::BSDH:
        return {sequence * headSize * heads, 1, headSize * heads, heads};
    case SdpaLayout::BSHD:
    default:
        return {sequence * heads * headSize, headSize, heads * headSize, 1};
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
