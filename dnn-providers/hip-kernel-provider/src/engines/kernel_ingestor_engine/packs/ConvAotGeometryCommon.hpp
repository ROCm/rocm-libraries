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

/// @file ConvAotGeometryCommon.hpp
/// What the conv AOT engines (gfx950 and gfx1250, forward and dgrad) share: launch geometry,
/// ceiling and magic-number division, the tensor-size and buffer-overlap checks, the
/// windowed depthwise kernel's limits and the forward direct kernels' argument block.
/// Pure helpers with no dependency on one engine's problem shape; each engine's geometry
/// header (ConvDgradAotGeometry.hpp, Gfx950ConvFwdGeometry.hpp, Gfx1250ConvFwdTypes.hpp)
/// brings them into its own namespace.
namespace hip_kernel_provider::kernel_ingestor_engine::conv_aot
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

constexpr int64_t I32_MAX = std::numeric_limits<int32_t>::max();
constexpr int64_t I32_MIN = std::numeric_limits<int32_t>::min();
/// The hardware cap of grid y and z, and of every axis of the DirectConvSpec compute grids.
constexpr int64_t MAX_GRID_DIM = 65535;
/// A dispatch's x extent in work-items is 32-bit.
constexpr int64_t MAX_DISPATCH_ITEMS = (int64_t{1} << 32) - 1;
constexpr int64_t MAX_BLOCK_THREADS = 1024;

/// One architecture's packaged conv engine: its wave size and LDS per workgroup
/// (ArchTarget.lds_capacity_bytes, arch_specs.json).
struct Target
{
    std::string_view arch;
    int64_t waveSize;
    int64_t ldsCapacityBytes;
};

inline constexpr Target GFX950{"gfx950", 64, 163840};
inline constexpr Target GFX1250{"gfx1250", 32, 327680};

/// The forward implicit-GEMM ABI's leading pointer/size block, then IMPLICIT_GEMM_KERNARG_NAMES.
inline constexpr std::array<std::string_view, 6> FWD_LEADING_KERNARG_NAMES{
    "A", "B", "D", "A_bytes", "B_bytes", "D_bytes"};

/// The AOT problem block that follows (A*, B*, D*, A_bytes, B_bytes, D_bytes) in the 2D
/// forward implicit-GEMM ABI, in order: _fwd_arg_names in
/// rocke/library/kernels/common/conv_abi.py. The kernel packs these positionally, so the
/// names exist to be compared against the packaged symbol's recorded arguments.
constexpr std::array<std::string_view, 39> IMPLICIT_GEMM_KERNARG_NAMES{"p_N",
                                                                       "p_Hi",
                                                                       "p_Wi",
                                                                       "p_C",
                                                                       "p_K",
                                                                       "p_Y",
                                                                       "p_X",
                                                                       "p_sH",
                                                                       "p_sW",
                                                                       "p_pH",
                                                                       "p_pW",
                                                                       "p_dH",
                                                                       "p_dW",
                                                                       "p_groups",
                                                                       "p_Ho",
                                                                       "p_Wo",
                                                                       "p_cpg",
                                                                       "p_kpg",
                                                                       "p_K_gemm",
                                                                       "p_M",
                                                                       "p_A_stride_n",
                                                                       "p_A_stride_hi",
                                                                       "p_A_stride_wi",
                                                                       "p_B_stride_k",
                                                                       "p_B_stride_y",
                                                                       "p_B_stride_x",
                                                                       "p_D_stride_n",
                                                                       "p_D_stride_ho",
                                                                       "p_D_stride_wo",
                                                                       "p_magic_m_Ho_mult",
                                                                       "p_magic_m_Ho_shift",
                                                                       "p_magic_m_Wo_mult",
                                                                       "p_magic_m_Wo_shift",
                                                                       "p_magic_k_X_mult",
                                                                       "p_magic_k_X_shift",
                                                                       "p_magic_k_cpg_mult",
                                                                       "p_magic_k_cpg_shift",
                                                                       "p_num_pid_m",
                                                                       "p_num_pid_n"};

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

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_aot
