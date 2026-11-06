// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string_view>

/// @file Gfx950ConvGeometryCommon.hpp
/// What the gfx950 conv AOT engines share, whatever their direction: launch geometry,
/// ceiling and magic-number division, the tensor-size and buffer-overlap checks, the
/// windowed depthwise kernel's limits and the forward direct kernels' argument block.
/// Pure helpers with no dependency on one engine's problem shape; each engine's geometry
/// header (Gfx950ConvDgradGeometry.hpp) brings them into its own namespace.
namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv
{

/// One kernel launch: its grid and its workgroup's x extent.
struct LaunchGeometry
{
    unsigned int gridX;
    unsigned int gridY;
    unsigned int gridZ;
    unsigned int blockX;
};

/// ceil(a / b) for a >= 0, b > 0, without the overflow of a + b - 1.
inline int64_t ceilDiv(int64_t a, int64_t b)
{
    return a == 0 ? 0 : 1 + (a - 1) / b;
}

/// One (multiplier, shift) pair of rocKE's calculate_magic_numbers in
/// rocke/platform/python/rocke/helpers/transforms.py, a port of CK Tile's
/// magic_division32_bit_range. The kernel reads the multiplier with an unsigned mul-hi,
/// so it is passed as the i32 with the same bit pattern (conv_args._magic_as_i32).
struct MagicDivision
{
    int32_t multiplier;
    int32_t shift;
};

inline std::optional<MagicDivision> magicDivision(int64_t divisor)
{
    if(divisor < 1 || divisor > std::numeric_limits<int32_t>::max())
    {
        return std::nullopt;
    }
    int32_t shift = 0;
    while((int64_t{1} << shift) < divisor)
    {
        ++shift;
    }
    // 2^shift - divisor < divisor <= 2^31, so the shifted value fits in 63 bits and the
    // quotient plus one stays below 2^32.
    const auto d = static_cast<uint64_t>(divisor);
    const uint64_t multiplier = ((((uint64_t{1} << shift) - d) << 32) / d) + 1;
    return MagicDivision{static_cast<int32_t>(static_cast<uint32_t>(multiplier)), shift};
}

/// conv_direct_grouped: the windowed depthwise kernel's sentinel addressing range and the
/// channels one MFMA wave of it owns.
constexpr int64_t DW_DGRAD_WIN_MAX_TENSOR_BYTES = (int64_t{1} << 30) - 1;
constexpr int64_t DW_DGRAD_MFMA_CH = 8;

/// The product of @p factors when it is below 2^62, else std::nullopt (nonnegative
/// factors): the byte counts and offsets of a whole tensor.
inline std::optional<int64_t> wideProduct(std::initializer_list<int64_t> factors)
{
    constexpr int64_t LIMIT = int64_t{1} << 62;
    int64_t product = 1;
    for(const auto factor : factors)
    {
        if(factor < 0 || (factor != 0 && product >= LIMIT / factor))
        {
            return std::nullopt;
        }
        product *= factor;
    }
    return product;
}

/// The compiled pointer arguments are noalias. Check complete storage ranges, including
/// partially overlapping views, before passing them to the kernel.
inline bool bufferRangesDoNotOverlap(const std::array<uintptr_t, 3>& addresses,
                                     const std::array<int64_t, 3>& bytes)
{
    for(size_t i = 0; i < addresses.size(); ++i)
    {
        if(bytes[i] <= 0
           || addresses[i]
                  > std::numeric_limits<uintptr_t>::max() - static_cast<uintptr_t>(bytes[i]))
        {
            return false;
        }
        for(size_t j = 0; j < i; ++j)
        {
            // Subtract ordered addresses so even untrusted pointers cannot overflow.
            if(addresses[i] >= addresses[j]
                   ? addresses[i] - addresses[j] < static_cast<uintptr_t>(bytes[j])
                   : addresses[j] - addresses[i] < static_cast<uintptr_t>(bytes[i]))
            {
                return false;
            }
        }
    }
    return true;
}

/// The forward direct ABI (A, D strides) the fused, 4c and MFMA compute passes take on the
/// transposed problem: conv_direct_arg_names(direction="fwd") in conv_abi.py. Filter
/// extents, stride and padding are compiled into the direct kernels, so they are not
/// arguments.
constexpr std::array<std::string_view, 14> DIRECT_KERNARG_NAMES{"p_N",
                                                                "p_Hi",
                                                                "p_Wi",
                                                                "p_Ho",
                                                                "p_Wo",
                                                                "p_groups",
                                                                "p_total_c",
                                                                "p_total_k",
                                                                "p_A_stride_n",
                                                                "p_A_stride_hi",
                                                                "p_A_stride_wi",
                                                                "p_D_stride_n",
                                                                "p_D_stride_ho",
                                                                "p_D_stride_wo"};

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv
