// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd
{

/// Logical NCHW/KCYX dimensions. The kernel stores A as NHWC, B as KYXC, D as NHWK.
struct Problem
{
    int64_t n;
    int64_t c;
    int64_t k;
    int64_t hi;
    int64_t wi;
    int64_t y;
    int64_t x;
    int64_t strideH;
    int64_t strideW;
    int64_t padH;
    int64_t padW;
    int64_t dilationH;
    int64_t dilationW;
};

struct Geometry
{
    int64_t ho;
    int64_t wo;
    int64_t gemmM;
    std::array<int32_t, 3> tensorBytes; // A, B, D, in the rocKE launch ABI order.
};

struct LaunchGeometry
{
    unsigned int gridX;
    unsigned int gridY;
    unsigned int gridZ;
    unsigned int blockX;
};

/// The compiled pointer arguments are noalias. Check complete storage ranges,
/// including partially overlapping views, before passing them to the kernel.
inline bool bufferRangesDoNotOverlap(const std::array<uintptr_t, 3>& addresses,
                                     const std::array<int32_t, 3>& bytes)
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

/// The builder's A_bytes/B_bytes/D_bytes arguments and indexing use signed i32.
inline std::optional<int32_t> tensorByteSize(const std::array<int64_t, 4>& dims)
{
    int64_t bytes = 2; // Both supported storage types, fp16 and bf16, occupy two bytes.
    for(const auto extent : dims)
    {
        if(extent <= 0 || extent > std::numeric_limits<int32_t>::max() / bytes)
        {
            return std::nullopt;
        }
        bytes *= extent;
    }
    return static_cast<int32_t>(bytes);
}

inline std::optional<int64_t>
    outputExtent(int64_t input, int64_t filter, int64_t stride, int64_t pad, int64_t dilation)
{
    constexpr auto MAX_INDEX = std::numeric_limits<int32_t>::max();
    if(input <= 0 || input > MAX_INDEX || filter <= 0 || filter > MAX_INDEX || stride <= 0
       || stride > MAX_INDEX || pad < 0 || pad > MAX_INDEX || dilation <= 0 || dilation > MAX_INDEX)
    {
        return std::nullopt;
    }

    // Check before multiplying: both the host arithmetic and the builder's i32
    // coordinate transforms must represent the padded and dilated extents.
    if(pad > (MAX_INDEX - input) / 2 || filter - 1 > (MAX_INDEX - 1) / dilation)
    {
        return std::nullopt;
    }
    const auto padded = input + 2 * pad;
    const auto effectiveFilter = (filter - 1) * dilation + 1;
    if(padded < effectiveFilter)
    {
        return std::nullopt;
    }
    return (padded - effectiveFilter) / stride + 1;
}

inline std::optional<Geometry> deriveGeometry(const Problem& problem)
{
    const auto aBytes = tensorByteSize({problem.n, problem.c, problem.hi, problem.wi});
    const auto bBytes = tensorByteSize({problem.k, problem.c, problem.y, problem.x});
    const auto ho
        = outputExtent(problem.hi, problem.y, problem.strideH, problem.padH, problem.dilationH);
    const auto wo
        = outputExtent(problem.wi, problem.x, problem.strideW, problem.padW, problem.dilationW);
    if(!aBytes || !bBytes || !ho || !wo)
    {
        return std::nullopt;
    }
    const auto dBytes = tensorByteSize({problem.n, problem.k, *ho, *wo});
    if(!dBytes)
    {
        return std::nullopt;
    }

    // dBytes already bounded this product, including its extra factor of K * 2.
    return Geometry{*ho, *wo, problem.n * *ho * *wo, {*aBytes, *bBytes, *dBytes}};
}

inline std::optional<LaunchGeometry> launchGeometry(const Problem& problem,
                                                    const Geometry& geometry,
                                                    int64_t tileM,
                                                    int64_t tileN,
                                                    int64_t warpM,
                                                    int64_t warpN,
                                                    int64_t waveSize)
{
    if(tileM <= 0 || tileN <= 0 || warpM <= 0 || warpN <= 0 || waveSize != 64
       || warpM > 1024 / waveSize || warpN > 1024 / (waveSize * warpM) || problem.k <= 0
       || geometry.gemmM <= 0)
    {
        return std::nullopt;
    }

    // Mirrors implicit_gemm_conv_grid and ImplicitGemmConvSpec.launch_block_size
    // in rocke/instances/common/conv_implicit_gemm.py (mem pipeline, groups=1).
    // Subtract before division to avoid overflowing even for an untrusted tile size.
    const auto gridX = 1 + (problem.k - 1) / tileN;
    const auto gridY = 1 + (geometry.gemmM - 1) / tileM;
    // Same grid bound as rocKE's is_valid_spec_for_problem.
    if(gridX > 65535 || gridY > 65535)
    {
        return std::nullopt;
    }
    return LaunchGeometry{static_cast<unsigned int>(gridX),
                          static_cast<unsigned int>(gridY),
                          1,
                          static_cast<unsigned int>(warpM * warpN * waveSize)};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd
