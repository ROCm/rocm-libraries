// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <roc/hostnumerics/reshape_permutation.hpp>
#include <vector>

namespace roc::hostnumerics::amd_gpu_layout {

// Presets for the global-memory storage a kernel expects. A matrix instruction
// fixes the association of scales with operand values in registers; its kernel
// chooses a memory layout that reduces the rearrangement needed after loads.
// These names identify current storage conventions, not GPU architectures.
enum class MxScaleStorageLayout {
    Natural,

    // Pad [rows, K blocks] to multiples of [32, 8], reshape to
    // [rows/32, 2, 16, blocks/8, 2, 4], then permute (0, 3, 5, 2, 4, 1).
    // Each 32 x 8 tile is contiguous, with row-half changing fastest.
    Tile32x8,

    // With q = 128 / blockSize, pad the fast dimension to q, reshape to
    // [slow, fast/q, q], then permute (1, 0, 2). Each group of q scales is
    // contiguous. Current kernels use blocks of 16 or 32 data elements.
    FastDimensionTiles,
};

// Storage sizes for converting one natural rank-two MX scale tensor, with one
// byte per scale, to the byte layout an AMD GPU kernel reads.
class MxScaleStoragePlan {
   public:
    // For one scale per row and block of K values, naturalShape is
    // [rows, ceil(K / blockSize)]. Callers first put their scales in the
    // [slow, fast] order the selected kernel expects. blockSize is the number of data
    // elements sharing each scale and must be nonzero. FastDimensionTiles groups
    // 128 / blockSize scales along the fast dimension and requires a block size
    // of 16 or 32; the Natural and Tile32x8 byte layouts do not depend on it.
    // Throws std::runtime_error for a zero or unsupported block size, an invalid
    // layout, or shapes whose sizes or strides overflow the descriptor's limits.
    MxScaleStoragePlan(std::array<size_t, 2> naturalShape, size_t blockSize,
                       MxScaleStorageLayout layout);

    MxScaleStorageLayout layout() const {
        return m_layout;
    }
    std::array<size_t, 2> naturalShape() const {
        return {m_transform.naturalShape()[0], m_transform.naturalShape()[1]};
    }
    size_t blockSize() const {
        return m_blockSize;
    }
    size_t naturalByteCount() const {
        return m_transform.naturalElementCount();
    }
    // Includes any zero padding the layout requires.
    size_t physicalByteCount() const {
        return m_transform.physicalElementCount();
    }

    // The complete padding, reshape and axis order, in scale-element units.
    const ReshapePermutation& transform() const {
        return m_transform;
    }

   private:
    MxScaleStorageLayout m_layout = MxScaleStorageLayout::Natural;
    size_t m_blockSize = 0;
    ReshapePermutation m_transform;
};

// Copies natural [slow, fast] scale bytes into a plan's physical layout and
// returns physicalByteCount() bytes, with zero bytes in any padding. Throws
// std::runtime_error if the source byte count differs from the plan's
// naturalByteCount() or if the source is null with a nonzero byte count.
std::vector<std::byte> copyMxScaleStorageToPhysicalLayout(const std::byte* naturalScaleStorage,
                                                          size_t naturalScaleByteCount,
                                                          const MxScaleStoragePlan& plan);

}  // namespace roc::hostnumerics::amd_gpu_layout
