// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

namespace roc::host_validation::amd_gpu_layout {
namespace detail {

// Compiled bridges for the type-independent validation, layout planning, and
// copying used by the public templates below. A null output requests only
// validation and the required output element count.
size_t preSwizzleBytes(const std::byte* input, size_t inputElementCount, size_t elementSize,
                       const std::vector<size_t>& sizes, const std::vector<size_t>& preSwizzleSize,
                       const std::vector<size_t>& preTileSize, std::byte* output);
size_t preSwizzleScalesGFX950Bytes(const std::byte* input, size_t inputElementCount,
                                   size_t elementSize, const std::vector<size_t>& sizes,
                                   std::byte* output);
size_t preSwizzleScalesGFX1250Bytes(const std::byte* input, size_t inputElementCount,
                                    size_t elementSize, size_t slowDimension, size_t fastDimension,
                                    size_t mxBlock, std::byte* output);

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

/**
 * @brief Converts a natural MX scale matrix to the physical scale layout
 *        consumed by gfx950-class kernels.
 *
 * `sizes` is `{rowCount, columnCount}` for row-major `input`. Rows are padded
 * to a multiple of 32 and columns to a multiple of 8 with `T{}`. The padded
 * matrix is viewed as
 * `{paddedRows / 32, 2, 16, paddedColumns / 8, 2, 4}` and permuted to
 * physical dimension order `{0, 3, 5, 2, 4, 1}`.
 *
 * This conversion matches the kernel's global-memory scale addressing. It is
 * not a cache transpose.
 *
 * `T` must be trivially copyable. The returned vector owns the padded,
 * converted storage.
 *
 * @throws std::runtime_error if `sizes` does not contain two dimensions or
 * the input length differs from their product.
 * @throws std::overflow_error if padding, shape, stride, or storage-size
 * calculations overflow `size_t`.
 */
template <typename T>
inline std::vector<T> preSwizzleScalesGFX950(const std::vector<T>& input,
                                             const std::vector<size_t>& sizes) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "AMD GPU physical-layout conversions require trivially copyable elements");

    const auto* inputBytes = reinterpret_cast<const std::byte*>(input.data());
    const size_t outputElementCount =
        detail::preSwizzleScalesGFX950Bytes(inputBytes, input.size(), sizeof(T), sizes, nullptr);
    std::vector<T> output(outputElementCount, T{});
    detail::preSwizzleScalesGFX950Bytes(inputBytes, input.size(), sizeof(T), sizes,
                                        reinterpret_cast<std::byte*>(output.data()));
    return output;
}

/**
 * @brief Converts natural MX scale storage to the physical scale layout
 *        consumed by gfx1250-class kernels.
 *
 * Natural input offset `(slow, fast)` is
 * `slow * fastDimension + fast`. With `dimK = 128 / mxBlock`, the output
 * offset is
 * `(fast / dimK) * slowDimension * dimK + slow * dimK + fast % dimK`.
 * `fastDimension` is padded to a multiple of `dimK` with `T{}`.
 *
 * For matrix A, `(slowDimension, fastDimension)` is `(M, K / mxBlock)` when A
 * is transposed and `(K / mxBlock, M)` otherwise. For matrix B it is
 * `(N, K / mxBlock)` when B is not transposed and `(K / mxBlock, N)`
 * otherwise.
 *
 * This conversion matches the kernel's global-memory scale addressing. It is
 * not a cache transpose.
 *
 * `T` must be trivially copyable. The returned vector owns the padded,
 * converted storage.
 *
 * @throws std::runtime_error if `mxBlock` is not 16 or 32, or if the input
 * length differs from `slowDimension * fastDimension`.
 * @throws std::overflow_error if shape, padding, or storage-size calculations
 * overflow `size_t`.
 */
template <typename T>
inline std::vector<T> preSwizzleScalesGFX1250(const std::vector<T>& input, size_t slowDimension,
                                              size_t fastDimension, size_t mxBlock) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "AMD GPU physical-layout conversions require trivially copyable elements");

    const auto* inputBytes = reinterpret_cast<const std::byte*>(input.data());
    const size_t outputElementCount = detail::preSwizzleScalesGFX1250Bytes(
        inputBytes, input.size(), sizeof(T), slowDimension, fastDimension, mxBlock, nullptr);
    std::vector<T> output(outputElementCount, T{});
    detail::preSwizzleScalesGFX1250Bytes(inputBytes, input.size(), sizeof(T), slowDimension,
                                         fastDimension, mxBlock,
                                         reinterpret_cast<std::byte*>(output.data()));
    return output;
}

}  // namespace roc::host_validation::amd_gpu_layout
