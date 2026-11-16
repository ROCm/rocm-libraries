// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <roc/hostnumerics/reshape_permutation.hpp>
#include <vector>

namespace roc::hostnumerics::amd_gpu_layout {

// Plans the global-memory layout of a rank-two MX scale tensor with one byte
// per scale. A scale accompanies a block of operand values along the reduction
// dimension: an M x K operand with blockSize values per scale has natural scale
// shape [M, ceil(K / blockSize)]. These plans take extents in scales, after that
// division; they do not describe operand quantization or store blockSize.
//
// Matrix instructions specify the association of scales and operand values in
// registers. Kernels choose a global-memory layout that reduces rearrangement
// after loading them. The product adapter or kernel author chooses a preset
// and first orients the scales in the [slow, fast] order that kernel expects.
// The complete padding, reshape and permutation are available through transform().
// All factories throw std::runtime_error if shapes, padding or strides overflow
// the ReshapePermutation limits.
// Assign a new plan before querying or copying from a moved-from plan.
class MxScaleStoragePlan {
   public:
    // Keeps the input shape and order, with no padding.
    static MxScaleStoragePlan natural(std::array<size_t, 2> naturalShape);

    // Pad [rows, K blocks] to [R, S], multiples of [32, 8], reshape to
    // [R/32, 2, 16, S/8, 2, 4], then permute (0, 3, 5, 2, 4, 1), slowest first.
    // Each 32 x 8 tile is contiguous, with row-half changing fastest. This is
    // the convention used by current gfx950 kernels and AITER's e8m0_shuffle.
    // For example, natural shape [2, 3] needs 256 bytes for six natural scales.
    static MxScaleStoragePlan tile32x8(std::array<size_t, 2> naturalShape);

    // Pad the fast extent to F, a multiple of q = scalesPerTile, reshape to
    // [slow, F/q, q], then permute (1, 0, 2), slowest first. Each group of q
    // scales is contiguous. q must be nonzero; it counts scales, not data values.
    // Current gfx1250 kernels use q = 128 / blockSize: four scales for blocks
    // of 32 data values, or eight for blocks of 16. The adapter translates that
    // kernel convention into a scale-tile width before calling this factory.
    //
    // For example, natural shape [2, 5] with four scales per tile transforms
    // [1,2,3,4,5, 6,7,8,9,10] to [1,2,3,4,6,7,8,9, 5,0,0,0,10,0,0,0].
    static MxScaleStoragePlan fastDimensionTiles(std::array<size_t, 2> naturalShape,
                                                 size_t scalesPerTile);

    std::array<size_t, 2> naturalShape() const {
        return {m_transform.naturalShape()[0], m_transform.naturalShape()[1]};
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
    explicit MxScaleStoragePlan(ReshapePermutation transform);

    ReshapePermutation m_transform;
};

// Copies natural [slow, fast] scale bytes into a plan's physical layout and
// returns physicalByteCount() bytes, with zero bytes in any padding. Throws
// std::runtime_error if the source byte count differs from the plan's
// naturalByteCount() or if the source is null with a nonzero byte count.
// Containers of one-byte integers can supply a byte view without copying:
//   bytes = read-only byte view of the natural scales
//   physical = copy those bytes using the plan
// See copyScales in amd_gpu_mx_layout_test.cpp for the C++ byte-view conversion.
std::vector<std::byte> copyMxScaleStorageToPhysicalLayout(const std::byte* naturalScaleStorage,
                                                          size_t naturalScaleByteCount,
                                                          const MxScaleStoragePlan& plan);

}  // namespace roc::hostnumerics::amd_gpu_layout
