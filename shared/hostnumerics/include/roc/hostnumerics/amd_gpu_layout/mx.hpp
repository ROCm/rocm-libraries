// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

namespace roc::hostnumerics::amd_gpu_layout {
enum class MxScaleStorageLayout {
    // Contiguous [slow, fast] scale bytes with no architecture-specific padding or permutation.
    Natural,

    // GFX950 block-scale ABI: pad the natural dimensions to the required tile and permute the
    // bytes into the kernel's tiled scale order.
    Gfx950,

    // GFX1250 block-scale ABI: tile the padded fast dimension in groups of 128 / blockSize,
    // ordered as [fast tile, slow coordinate, coordinate within the tile].
    Gfx1250,
};

// Complete storage contract for converting one natural rank-two MX scale
// tensor to the byte layout consumed by an AMD GPU kernel.
struct MxScaleStoragePlan {
    MxScaleStorageLayout layout = MxScaleStorageLayout::Natural;
    std::array<size_t, 2> naturalShape{};  // [slow, fast] dimensions.
    size_t blockSize = 0;                  // Data elements sharing each scale.
    size_t naturalByteCount = 0;
    size_t physicalByteCount = 0;
};

// Maps an AMD GPU architecture name to its required physical MX scale layout.
// The name must be exactly "gfx950", "gfx1250" or "gfx1250-strict", optionally
// followed by ':' and feature flags such as "gfx950:sramecc+:xnack-". Any other
// name, including a longer name that merely contains one of these, maps to
// Natural.
MxScaleStorageLayout mxScaleStorageLayoutForArchitectureName(std::string_view architectureName);

// Computes the natural and padded physical storage sizes of one-byte scales.
// `blockSize` is the number of data elements sharing each scale and must be
// nonzero. Gfx1250 groups 128 / blockSize scales along the fast dimension and
// requires a block size of 16 or 32; the Natural and Gfx950 byte layouts do not
// depend on it. Throws std::runtime_error for a zero or unsupported block size,
// an invalid layout, or a byte count that overflows size_t.
MxScaleStoragePlan planMxScaleStorage(std::array<size_t, 2> naturalShape, size_t blockSize,
                                      MxScaleStorageLayout layout);

// Copies natural [slow, fast] scale bytes into a plan's physical layout and
// returns physicalByteCount bytes, with zero bytes in any padding. Throws
// std::runtime_error if the plan differs from planMxScaleStorage's result for
// its shape, block size and layout, if the source byte count differs from
// naturalByteCount, or if the source is null with a nonzero byte count.
std::vector<std::byte> copyMxScaleStorageToPhysicalLayout(const std::byte* naturalScaleStorage,
                                                          size_t naturalScaleByteCount,
                                                          const MxScaleStoragePlan& plan);

}  // namespace roc::hostnumerics::amd_gpu_layout
