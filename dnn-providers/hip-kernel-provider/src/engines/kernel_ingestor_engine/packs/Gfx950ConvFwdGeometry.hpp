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

/// The forward implicit-GEMM kernels decode the reduction index with 24-bit multiplies;
/// rocKE's ConvArgs refuses a reduction extent Y * X * C / groups at or above this
/// (conv_args.MUL24_REDUCTION_LIMIT).
constexpr int64_t MUL24_REDUCTION_LIMIT = int64_t{1} << 23;

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

using ImplicitGemmKernargs = std::array<int32_t, IMPLICIT_GEMM_KERNARG_NAMES.size()>;

/// The values for IMPLICIT_GEMM_KERNARG_NAMES. Mirrors ConvArgs.to_launch_values in
/// rocke/library/kernels/common/conv_args.py for a 2D forward problem: packed NHWC, KYXC
/// with C / groups channels, and NHWK strides in elements, one magic pair per unmerge
/// divisor, and the tile counts @p launch was derived from. std::nullopt when the
/// reduction extent breaks the kernel's 24-bit address products.
inline std::optional<ImplicitGemmKernargs> implicitGemmKernargs(const Problem& problem,
                                                                const Geometry& geometry,
                                                                const LaunchGeometry& launch)
{
    const auto cpg = problem.c / problem.groups;
    const auto kpg = problem.k / problem.groups;
    // B's byte count bounds this product, so it cannot overflow.
    const auto kGemm = problem.y * problem.x * cpg;
    if(kGemm >= MUL24_REDUCTION_LIMIT)
    {
        return std::nullopt;
    }
    const auto ho = magicDivision(geometry.ho);
    const auto wo = magicDivision(geometry.wo);
    const auto x = magicDivision(problem.x);
    const auto c = magicDivision(cpg);
    if(!ho || !wo || !x || !c)
    {
        return std::nullopt;
    }
    // Every value below is an extent, attribute, or a stride or product bounded by a
    // tensor's element count, all of which deriveGeometry kept within i32.
    const auto i32 = [](int64_t value) { return static_cast<int32_t>(value); };
    return ImplicitGemmKernargs{i32(problem.n),
                                i32(problem.hi),
                                i32(problem.wi),
                                i32(problem.c),
                                i32(problem.k),
                                i32(problem.y),
                                i32(problem.x),
                                i32(problem.strideH),
                                i32(problem.strideW),
                                i32(problem.padH),
                                i32(problem.padW),
                                i32(problem.dilationH),
                                i32(problem.dilationW),
                                i32(problem.groups),
                                i32(geometry.ho),
                                i32(geometry.wo),
                                i32(cpg),
                                i32(kpg),
                                i32(kGemm),
                                i32(geometry.gemmM),
                                i32(problem.hi * problem.wi * problem.c),
                                i32(problem.wi * problem.c),
                                i32(problem.c),
                                i32(kGemm),
                                i32(problem.x * cpg),
                                i32(cpg),
                                i32(geometry.ho * geometry.wo * problem.k),
                                i32(geometry.wo * problem.k),
                                i32(problem.k),
                                ho->multiplier,
                                ho->shift,
                                wo->multiplier,
                                wo->shift,
                                x->multiplier,
                                x->shift,
                                c->multiplier,
                                c->shift,
                                static_cast<int32_t>(launch.gridY),
                                static_cast<int32_t>(launch.gridX)};
}

/// The runtime block that follows the six leading arguments in the forward direct-conv
/// ABI, in order: conv_direct_arg_names(direction="fwd") in conv_abi.py. Filter extents,
/// stride and padding are compiled into the direct kernels, so they are not arguments.
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

using DirectKernargs = std::array<int32_t, DIRECT_KERNARG_NAMES.size()>;

/// The values for DIRECT_KERNARG_NAMES; mirrors ConvArgs.to_launch_values for a
/// DirectConvProblem. Every value is bounded as in implicitGemmKernargs.
inline DirectKernargs directKernargs(const Problem& problem, const Geometry& geometry)
{
    const auto i32 = [](int64_t value) { return static_cast<int32_t>(value); };
    return DirectKernargs{i32(problem.n),
                          i32(problem.hi),
                          i32(problem.wi),
                          i32(geometry.ho),
                          i32(geometry.wo),
                          i32(problem.groups),
                          i32(problem.c),
                          i32(problem.k),
                          i32(problem.hi * problem.wi * problem.c),
                          i32(problem.wi * problem.c),
                          i32(problem.c),
                          i32(geometry.ho * geometry.wo * problem.k),
                          i32(geometry.wo * problem.k),
                          i32(problem.k)};
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

/// Mirrors forward_padding_reason in rocke/library/kernels/common/conv_direct_grouped.py,
/// which rocKE's direct depthwise validators apply: odd filter extents and "same" padding
/// (KH - 1) / 2. rocKE refuses every other forward shape before building it.
inline bool directPaddingSupported(const Problem& problem)
{
    return problem.y % 2 == 1 && problem.x % 2 == 1 && problem.padH == (problem.y - 1) / 2;
}

/// The problem-side half of _direct_error in convolution_forward.py, plus rocKE's own
/// padding rule: pure depthwise (cpg == kpg == 1), one compiled stride and padding (the
/// kernels read neither sW nor pW), no dilation, "same" padding, and both axes covered by
/// the input stream. Everything this declines stays on implicit GEMM.
inline bool directProblemSupported(const Problem& problem)
{
    return problem.groups == problem.c && problem.c == problem.k
           && problem.strideH == problem.strideW && problem.padH == problem.padW
           && problem.dilationH == 1 && problem.dilationW == 1 && directPaddingSupported(problem)
           && directRowsCovered(problem);
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
