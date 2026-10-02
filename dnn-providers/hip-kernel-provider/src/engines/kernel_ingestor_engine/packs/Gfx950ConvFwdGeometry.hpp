// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd
{

/// Logical NCHW/KCYX dimensions. The kernel stores A as NHWC, B as KYXC, D as NHWK.
/// C and K are totals across all groups; B holds C / groups channels per filter.
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
    int64_t groups = 1; // Last, so the groups=1 aggregate initializers stay valid.
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
    if(problem.groups <= 0 || problem.c % problem.groups != 0 || problem.k % problem.groups != 0)
    {
        return std::nullopt;
    }
    const auto aBytes = tensorByteSize({problem.n, problem.c, problem.hi, problem.wi});
    // B is [K, Y, X, C / groups]. This size becomes B's buffer-resource num_records, so a
    // total-C count would let a grouped kernel read past the filter allocation.
    const auto bBytes
        = tensorByteSize({problem.k, problem.c / problem.groups, problem.y, problem.x});
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
       || problem.groups <= 0 || problem.k % problem.groups != 0 || geometry.gemmM <= 0)
    {
        return std::nullopt;
    }

    // Mirrors implicit_gemm_conv_grid and ImplicitGemmConvSpec.launch_block_size in
    // rocke/library/kernels/common/conv_implicit_gemm.py: N-tiles over the per-group
    // GEMM N (K / groups), M-tiles, and one z slice per group.
    // Subtract before division to avoid overflowing even for an untrusted tile size.
    const auto gridX = 1 + (problem.k / problem.groups - 1) / tileN;
    const auto gridY = 1 + (geometry.gemmM - 1) / tileM;
    // Same x/y bound as rocKE's is_valid_spec_for_problem; z has the same hardware cap.
    if(gridX > 65535 || gridY > 65535 || problem.groups > 65535)
    {
        return std::nullopt;
    }
    return LaunchGeometry{static_cast<unsigned int>(gridX),
                          static_cast<unsigned int>(gridY),
                          static_cast<unsigned int>(problem.groups),
                          static_cast<unsigned int>(warpM * warpN * waveSize)};
}

/// The kernel_family metadata value, also the engine knob that forces a family.
/// Mirrors KERNEL_FAMILY_* in rocke/library/builders/common/convolution_forward.py.
enum class KernelFamily : int64_t
{
    IMPLICIT_GEMM = 0,
    DIRECT_DEPTHWISE = 1,
};

/// The direct_variant metadata value: "none" for implicit GEMM, "std" for
/// DirectDepthwiseSpec and "spatial" for DirectDepthwiseSpatialSpec.
enum class DirectVariant
{
    NONE,
    STD,
    SPATIAL,
};

/// A closed value set: any other integer, including a future family this build does not
/// know how to launch, is refused.
inline std::optional<KernelFamily> parseKernelFamily(int64_t value)
{
    switch(value)
    {
    case static_cast<int64_t>(KernelFamily::IMPLICIT_GEMM):
        return KernelFamily::IMPLICIT_GEMM;
    case static_cast<int64_t>(KernelFamily::DIRECT_DEPTHWISE):
        return KernelFamily::DIRECT_DEPTHWISE;
    default:
        return std::nullopt;
    }
}

inline std::optional<DirectVariant> parseDirectVariant(std::string_view value)
{
    if(value == "none")
    {
        return DirectVariant::NONE;
    }
    if(value == "std")
    {
        return DirectVariant::STD;
    }
    if(value == "spatial")
    {
        return DirectVariant::SPATIAL;
    }
    return std::nullopt;
}

/// rocKE's direct depthwise kernels launch 64-lane waves only.
constexpr int64_t DIRECT_WAVE_SIZE = 64;
constexpr int64_t DIRECT_MAX_BLOCK_WAVES = 16;
/// rocKE's grid bound for the direct kernels, applied to every axis.
constexpr int64_t DIRECT_MAX_GRID_DIM = 65535;
/// The default direct arm, gfx950_conv_fwd_direct_spec_for_request with no overrides:
/// spatial with one wave below 64 groups, otherwise std with this block width.
constexpr int64_t DEFAULT_DIRECT_BLOCK_W = 4;
constexpr int64_t DEFAULT_DIRECT_BLOCK_WAVES = 1;

/// Mirrors _direct_rows_covered in convolution_forward.py for one spatial axis (dilation
/// 1). Both direct kernels stream input rows and flush output row floor(p / stride) only
/// for input rows p < extent, so the last row they write is floor((extent - 1) / stride).
/// When the output has fewer rows than that, the runtime H loop writes into the next
/// image; when it has more, the trailing rows are never written. Either is a wrong
/// answer, so a direct kernel is accepted only where the two agree.
inline bool directRowsCovered(int64_t extent, int64_t pad, int64_t filter, int64_t stride)
{
    const auto output = outputExtent(extent, filter, stride, pad, 1);
    // outputExtent already bounded every operand, so the division below is safe.
    return output.has_value() && (extent - 1) / stride == *output - 1;
}

/// Both axes of @p problem; the direct kernels compile one stride and one padding, which
/// kernelFits requires to be equal across H and W before calling this.
inline bool directRowsCovered(const Problem& problem)
{
    return directRowsCovered(problem.hi, problem.padH, problem.y, problem.strideH)
           && directRowsCovered(problem.wi, problem.padW, problem.x, problem.strideW);
}

/// The problem-side half of _direct_error in convolution_forward.py: pure depthwise
/// (cpg == kpg == 1), one compiled stride and padding (the kernels read neither sW nor
/// pW), no dilation, and both axes covered by the input stream. Everything this declines
/// stays on implicit GEMM.
inline bool directProblemSupported(const Problem& problem)
{
    return problem.groups == problem.c && problem.c == problem.k
           && problem.strideH == problem.strideW && problem.padH == problem.padW
           && problem.dilationH == 1 && problem.dilationW == 1 && directRowsCovered(problem);
}

/// Grid and block for a direct depthwise kernel. Mirrors Gfx950ConvFwdSpec.grid()/block()
/// in convolution_forward.py, which follow the kernel spec properties in
/// rocke/library/kernels/common/conv_direct_grouped.py (block_ch, n_w_per_wave,
/// threads_per_block) and the launch in benchmarks/common/benchmark_direct_conv.py:
///   std:     (ceil(Wo / block_w), ceil(groups / (64 * block_waves)), N)
///   spatial: (ceil(Wo / (block_waves * (64 / groups))), 1, N), groups < 64, block_w 0
/// Both launch 64 * block_waves threads. Refuses any axis above 65535.
inline std::optional<LaunchGeometry> directLaunchGeometry(const Problem& problem,
                                                          const Geometry& geometry,
                                                          DirectVariant variant,
                                                          int64_t blockW,
                                                          int64_t blockWaves,
                                                          int64_t waveSize)
{
    if(waveSize != DIRECT_WAVE_SIZE || blockWaves < 1 || blockWaves > DIRECT_MAX_BLOCK_WAVES
       || problem.groups <= 0 || problem.n <= 0 || geometry.wo <= 0)
    {
        return std::nullopt;
    }

    int64_t columnsPerBlock = 0;
    int64_t gridY = 1;
    switch(variant)
    {
    case DirectVariant::STD:
        if(blockW <= 0 || blockW > std::numeric_limits<int32_t>::max())
        {
            return std::nullopt;
        }
        columnsPerBlock = blockW;
        // Subtract before division so the bounded operands cannot overflow.
        gridY = 1 + (problem.groups - 1) / (blockWaves * waveSize);
        break;
    case DirectVariant::SPATIAL:
        // The spatial kernel derives its block width; at groups == 64 a wave would cover
        // one column, which rocKE's validator admits but its own benchmark never runs.
        if(blockW != 0 || problem.groups >= waveSize)
        {
            return std::nullopt;
        }
        columnsPerBlock = blockWaves * (waveSize / problem.groups);
        break;
    case DirectVariant::NONE:
    default:
        return std::nullopt;
    }

    const auto gridX = 1 + (geometry.wo - 1) / columnsPerBlock;
    if(gridX > DIRECT_MAX_GRID_DIM || gridY > DIRECT_MAX_GRID_DIM
       || problem.n > DIRECT_MAX_GRID_DIM)
    {
        return std::nullopt;
    }
    return LaunchGeometry{static_cast<unsigned int>(gridX),
                          static_cast<unsigned int>(gridY),
                          static_cast<unsigned int>(problem.n),
                          static_cast<unsigned int>(blockWaves * waveSize)};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd
