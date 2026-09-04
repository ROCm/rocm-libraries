// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <vector>

namespace roc::host_numerics::amd_gpu_layout {
enum class MxScaleStorageLayout {
    Natural,
    Gfx950,
    Gfx1250,
};

// Maps an AMD GPU architecture name, including feature suffixes such as
// "gfx950:sramecc+:xnack-", to its required physical MX scale layout.
MxScaleStorageLayout mxScaleStorageLayoutForArchitectureName(std::string_view architectureName);

// Copies a natural [slow, fast] scale tensor into the requested GPU physical
// layout. Architecture-specific layouts add their required zero padding.
std::vector<std::byte> copyMxScaleStorageToPhysicalLayout(
    const std::byte* naturalScaleStorage, size_t naturalScaleByteCount,
    std::array<size_t, 2> slowThenFastDimensions, size_t blockSize, MxScaleStorageLayout layout);

namespace detail {

// Compiled bridges for the type-independent validation, layout planning, and
// copying used by the public templates below. A null output requests only
// validation and the required output element count.
size_t preSwizzleBytes(const std::byte* input, size_t inputElementCount, size_t elementSize,
                       const std::vector<size_t>& sizes, const std::vector<size_t>& preSwizzleSize,
                       const std::vector<size_t>& preTileSize, std::byte* output);

}  // namespace detail

/**
 * @brief Converts a contiguous two-dimensional host tensor to an AMD GPU
 *        operand layout.
 *
 * Dimension 0 of `sizes` is contiguous in `input`. `preSwizzleSize`, when
 * present, is `{tileMN, tileK, subTileK}` and selects the lane/VGPR ordering
 * consumed by the kernel. `preTileSize`, when present, is
 * `{tileDimension0, tileDimension1}` and makes each physical tile contiguous.
 * Supplying neither configuration returns an owning copy of `input`.
 *
 * This is a physical storage conversion for GPU operands. It is not a matrix
 * transpose for cache locality: logical coordinates are unchanged and only
 * their linear storage offsets are permuted.
 *
 * `T` must be trivially copyable. The returned vector owns storage independent
 * of `input`.
 *
 * @throws std::runtime_error if the shape is not two-dimensional, a
 * configuration has the wrong length or unsupported values, the input length
 * differs from the shape product, or a tensor dimension is not divisible by
 * its configured tile.
 * @throws std::overflow_error if a shape, stride, or storage-size calculation
 * overflows `size_t`.
 */
template <typename T>
inline std::vector<T> preSwizzle(const std::vector<T>& input, const std::vector<size_t>& sizes,
                                 const std::vector<size_t>& preSwizzleSize,
                                 const std::vector<size_t>& preTileSize) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "AMD GPU physical-layout conversions require trivially copyable elements");

    const auto* inputBytes = reinterpret_cast<const std::byte*>(input.data());
    const size_t outputElementCount = detail::preSwizzleBytes(
        inputBytes, input.size(), sizeof(T), sizes, preSwizzleSize, preTileSize, nullptr);
    std::vector<T> output(outputElementCount);
    detail::preSwizzleBytes(inputBytes, input.size(), sizeof(T), sizes, preSwizzleSize, preTileSize,
                            reinterpret_cast<std::byte*>(output.data()));
    return output;
}

}  // namespace roc::host_numerics::amd_gpu_layout
