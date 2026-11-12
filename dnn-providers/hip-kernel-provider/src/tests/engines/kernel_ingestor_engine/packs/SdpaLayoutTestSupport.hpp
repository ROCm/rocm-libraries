// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <stdexcept>
#include <vector>

/**
 * @file SdpaLayoutTestSupport.hpp
 * @brief The memory layouts an SDPA test graph gives a (B, H, S, D) operand, and their
 *        strides: one home for the dense, unified and FlyDSL unit suites and
 *        IngestorIntegrationSupport.hpp.
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
    /// BSHD with each token row padded from H * D to H * D + 8 elements: token-major, but
    /// not gap-free.
    PADDED_BSHD,
    /// BHSD with each head row padded from D to D + 8 elements: head-major, but not gap-free.
    PADDED_BHSD,
    /// Token-major with D outer to H: [B, S, D, H] in memory, so H has the unit stride.
    BSDH
};

/// Element strides of a (B, H, S, D) logical tensor laid out as @p layout. Throws
/// std::invalid_argument for a value that names no SdpaLayout.
inline std::vector<int64_t>
    sdpaStrides(SdpaLayout layout, int64_t heads, int64_t sequence, int64_t headSize)
{
    // Elements of slack a real allocator leaves after each row of a padded layout.
    constexpr int64_t ROW_PAD = 8;
    switch(layout)
    {
    case SdpaLayout::BHSD:
        return {heads * sequence * headSize, sequence * headSize, headSize, 1};
    case SdpaLayout::PADDED_BSHD:
    {
        const int64_t rowStride = heads * headSize + ROW_PAD;
        return {sequence * rowStride, headSize, rowStride, 1};
    }
    case SdpaLayout::PADDED_BHSD:
    {
        const int64_t rowStride = headSize + ROW_PAD;
        return {heads * sequence * rowStride, sequence * rowStride, rowStride, 1};
    }
    case SdpaLayout::BSDH:
        return {sequence * headSize * heads, 1, headSize * heads, heads};
    case SdpaLayout::BSHD:
        return {sequence * heads * headSize, headSize, heads * headSize, 1};
    default:
        throw std::invalid_argument("sdpaStrides: unhandled SdpaLayout value");
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
