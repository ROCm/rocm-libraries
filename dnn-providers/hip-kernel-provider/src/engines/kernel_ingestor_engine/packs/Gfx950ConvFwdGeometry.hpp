// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
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

/// ConvProblem.is_pointwise: a 1x1, unit-stride, unpadded convolution, which rocKE builds
/// as a flat GEMM instead of through the coordinate transforms.
inline bool isPointwise(const Problem& problem)
{
    return problem.y == 1 && problem.x == 1 && problem.strideH == 1 && problem.strideW == 1
           && problem.padH == 0 && problem.padW == 0;
}

/// The widest of 8, 4, 2, 1 that divides @p extent: rocKE's
/// ImplicitGemmConvSpec.default_vector_sizes for 2-byte elements.
inline int64_t defaultVectorWidth(int64_t extent)
{
    for(const int64_t width : {8, 4, 2})
    {
        if(extent % width == 0)
        {
            return width;
        }
    }
    return 1;
}

/// rocke.helpers.spec.choose_load_vec for 2-byte elements: the widest width up to 8 that
/// divides tile_k and spreads both tiles' loads evenly over the block's threads.
inline std::optional<int64_t>
    tileLoadWidth(int64_t tileM, int64_t tileN, int64_t tileK, int64_t blockSize)
{
    if(tileM <= 0 || tileN <= 0 || tileK <= 0 || blockSize <= 0)
    {
        return std::nullopt;
    }
    for(int64_t width = 8; width >= 1; width /= 2)
    {
        if(tileK % width != 0)
        {
            continue;
        }
        const auto aLoads = tileM * tileK / width;
        const auto bLoads = tileN * tileK / width;
        if(aLoads >= blockSize && bLoads >= blockSize && aLoads % blockSize == 0
           && bLoads % blockSize == 0)
        {
            return width;
        }
    }
    return std::nullopt;
}

/// What an AOT implicit-GEMM binary keeps from the problem it was compiled for. Every
/// extent, stride, padding, dilation and channel count is a kernel argument; these are
/// the only build-time decisions that bound which problems the binary computes.
struct ImplicitGemmCapabilities
{
    /// A and B load width: the builder's default for the build C / groups, clamped by the
    /// tile (ImplicitGemmConvSpec._sync_load_vecs). Both run along C / groups.
    int64_t loadWidth;
    /// The cshuffle store width (default_vector_sizes for the build K / groups); 1 for
    /// the default epilogue, which stores scalars.
    int64_t storeWidth;
    /// Built as the flat pointwise GEMM, which computes only pointwise problems.
    bool pointwise;
    /// Built on the grouped code path, which computes groups == 1 as well.
    bool grouped;
};

/// The capabilities of a binary compiled for @p build with the given tile, or
/// std::nullopt when no load width fits the tile (rocKE would not have built it).
inline std::optional<ImplicitGemmCapabilities> implicitGemmCapabilities(const Problem& build,
                                                                        int64_t tileM,
                                                                        int64_t tileN,
                                                                        int64_t tileK,
                                                                        int64_t warpM,
                                                                        int64_t warpN,
                                                                        bool cshuffleEpilogue)
{
    if(build.groups <= 0 || build.c % build.groups != 0 || build.k % build.groups != 0)
    {
        return std::nullopt;
    }
    const auto tileWidth = tileLoadWidth(tileM, tileN, tileK, warpM * warpN * 64);
    if(!tileWidth)
    {
        return std::nullopt;
    }
    return ImplicitGemmCapabilities{
        std::min(defaultVectorWidth(build.c / build.groups), *tileWidth),
        cshuffleEpilogue ? defaultVectorWidth(build.k / build.groups) : 1,
        isPointwise(build),
        build.groups > 1};
}

/// Whether a binary with @p capabilities computes @p problem. The forward subset of
/// rocKE's KernelCache.supports_problem: the baked load width divides C / groups and the
/// store width K / groups (a wider access would straddle a group's slab), a grouped
/// problem needs the grouped code path, and the pointwise GEMM serves only pointwise
/// problems. A non-pointwise binary serves pointwise problems through its descriptors.
inline bool implicitGemmServes(const ImplicitGemmCapabilities& capabilities, const Problem& problem)
{
    if(problem.groups <= 0 || problem.c % problem.groups != 0 || problem.k % problem.groups != 0)
    {
        return false;
    }
    return (problem.groups == 1 || capabilities.grouped)
           && (!capabilities.pointwise || isPointwise(problem))
           && (problem.c / problem.groups) % capabilities.loadWidth == 0
           && (problem.k / problem.groups) % capabilities.storeWidth == 0;
}

/// One shape-generic AOT instance of hipkernel:Gfx950ConvFwdAot: the build-time fields
/// of a rocKE forward KernelIdentity that bound which problems its binary computes.
/// Mirrors Gfx950ConvFwdAotSpec in rocke/library/builders/common/convolution_forward.py.
struct AotInstance
{
    int64_t tileM;
    int64_t tileN;
    int64_t warpM;
    int64_t warpN;
    int64_t vectorA; ///< A load width; A runs along C / groups.
    int64_t vectorB; ///< B load width; B runs along C / groups.
    int64_t vectorC; ///< D store width; D runs along K / groups.
    bool asyncDma;
    int64_t asyncChunkA; ///< Elements per async A chunk; 0 without asyncDma.
    int64_t asyncChunkB;
    bool grouped; ///< Built with the grouped code path, which also serves groups == 1.
};

/// The launch of @p instance on @p problem, or std::nullopt when the binary cannot
/// compute it. Mirrors gfx950_conv_fwd_aot_serves in convolution_forward.py, the forward
/// subset of rocKE's KernelCache.supports_problem: implicitGemmServes with the widths the
/// identity records (A and B both run along C / groups, so both divide it exactly when
/// their lcm does; the 3x3 probe is never the pointwise GEMM), an async chunk divides
/// C / groups (it would otherwise read across filter positions), and launchGeometry's
/// 65535 grid bound. implicitGemmKernargs adds the 24-bit reduction bound.
inline std::optional<LaunchGeometry>
    aotInstanceLaunch(const Problem& problem, const Geometry& geometry, const AotInstance& instance)
{
    if(instance.vectorA <= 0 || instance.vectorB <= 0 || instance.vectorC <= 0)
    {
        return std::nullopt;
    }
    const ImplicitGemmCapabilities capabilities{
        std::lcm(instance.vectorA, instance.vectorB), instance.vectorC, false, instance.grouped};
    if(!implicitGemmServes(capabilities, problem))
    {
        return std::nullopt;
    }
    const auto cpg = problem.c / problem.groups;
    if(instance.asyncDma
       && (instance.asyncChunkA <= 0 || instance.asyncChunkB <= 0 || cpg % instance.asyncChunkA != 0
           || cpg % instance.asyncChunkB != 0))
    {
        return std::nullopt;
    }
    return launchGeometry(
        problem, geometry, instance.tileM, instance.tileN, instance.warpM, instance.warpN, 64);
}

/// Deterministic order among the AOT instances that serve one problem, used only when
/// nothing is benchmarked or recorded. Higher is better: the fraction of the launched
/// tile area that is real output (padding waste costs work), times how well the grid
/// fills a device of CU_COUNT compute units, with wider vectors breaking ties. A
/// placeholder until rocKE's dispatcher ranks instances; BenchmarkPlan replaces it.
inline double aotInstanceScore(const Problem& problem,
                               const Geometry& geometry,
                               const AotInstance& instance,
                               const LaunchGeometry& launch)
{
    constexpr double CU_COUNT = 256.0;
    const auto kpg = static_cast<double>(problem.k / problem.groups);
    const auto useful = static_cast<double>(geometry.gemmM) * kpg;
    const auto launched = static_cast<double>(launch.gridY) * static_cast<double>(instance.tileM)
                          * static_cast<double>(launch.gridX) * static_cast<double>(instance.tileN);
    const auto tiles = static_cast<double>(launch.gridX) * static_cast<double>(launch.gridY)
                       * static_cast<double>(launch.gridZ);
    const auto fill = tiles >= CU_COUNT ? 1.0 : tiles / CU_COUNT;
    const auto widths
        = static_cast<double>(instance.vectorA + instance.vectorB + instance.vectorC) / 24.0;
    return (useful / launched) * fill + 1e-3 * widths;
}

/// The kernel_family metadata value, also the engine knob that forces a family.
/// Mirrors KERNEL_FAMILY_* in rocke/library/builders/common/convolution_forward.py.
enum class KernelFamily : int64_t
{
    IMPLICIT_GEMM = 0,
    DIRECT_DEPTHWISE = 1,
};

/// The direct_variant metadata value: "none" for implicit GEMM, "std" for
/// DirectDepthwiseSpec and "spatial" for DirectDepthwiseSpatialSpec. The other values are
/// direct kernels only the AOT-instance engine packages; the exact-shape engine has std
/// and spatial only.
enum class DirectVariant
{
    NONE,
    STD,
    SPATIAL,
    TILED, ///< "tiled": DirectDepthwiseTiledSpec.
    COL, ///< "col": DirectDepthwiseColSpec.
    GROUPED, ///< "grouped": DirectConvSpec.
    NONGROUPED, ///< "nongrouped": DirectNongroupedConvSpec.
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
    if(value == "tiled")
    {
        return DirectVariant::TILED;
    }
    if(value == "col")
    {
        return DirectVariant::COL;
    }
    if(value == "grouped")
    {
        return DirectVariant::GROUPED;
    }
    if(value == "nongrouped")
    {
        return DirectVariant::NONGROUPED;
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

/// What a direct depthwise binary compiles in: the filter, stride and padding shape the
/// unrolled tap loop, and the kernels take no dilation. N, Hi, Wi and the group count are
/// kernel arguments for both variants (the spatial one derives its 64 / groups columns
/// per wave at runtime), so they are not capabilities.
struct DirectCapabilities
{
    int64_t y;
    int64_t x;
    int64_t strideH;
    int64_t strideW;
    int64_t padH;
    int64_t padW;
    int64_t dilationH;
    int64_t dilationW;
};

/// The capabilities of a direct binary compiled for @p build.
inline DirectCapabilities directCapabilities(const Problem& build)
{
    return DirectCapabilities{build.y,
                              build.x,
                              build.strideH,
                              build.strideW,
                              build.padH,
                              build.padW,
                              build.dilationH,
                              build.dilationW};
}

/// Whether a direct depthwise binary with @p compiled computes @p problem, which
/// directProblemSupported already admitted: every compiled value must be the graph's.
inline bool directServes(const DirectCapabilities& compiled, const Problem& problem)
{
    return compiled.y == problem.y && compiled.x == problem.x && compiled.strideH == problem.strideH
           && compiled.strideW == problem.strideW && compiled.padH == problem.padH
           && compiled.padW == problem.padW && compiled.dilationH == problem.dilationH
           && compiled.dilationW == problem.dilationW;
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

/// One direct binary as both engines describe it: what it compiled in, and the variant and
/// block knobs its grid follows. Mirrors the direct fields of Gfx950ConvFwdAotSpec in
/// rocke/library/builders/common/convolution_forward.py; the exact-shape engine reads the
/// compiled values from its build problem instead, and has std and spatial only. Each
/// variant reads only its own knobs; the rest stay 0.
struct DirectInstance
{
    DirectCapabilities compiled;
    DirectVariant variant;
    /// Output columns per block for std, tiled and col; 0 for spatial, which derives it.
    int64_t blockW;
    /// Waves per block, one channel per lane, for the depthwise variants.
    int64_t blockWaves;
    /// Output rows per block (tiled, col); the grouped kernel's row tile over the input.
    int64_t blockH = 0;
    /// grouped: output columns and groups per block, and the compiled channel counts.
    int64_t blockQ = 0;
    int64_t blockGroups = 0;
    int64_t cpg = 0;
    int64_t kpg = 0;
    /// nongrouped: output rows, columns and channels per block, input channels per LDS
    /// stage, the waves along channels and rows, and the MFMA atom's M == N.
    int64_t tileH = 0;
    int64_t tileW = 0;
    int64_t tileKOut = 0;
    int64_t ck = 0;
    int64_t wavesM = 0;
    int64_t wavesN = 0;
    int64_t atomTile = 0;
};

/// The output tile of DirectNongroupedConvSpec's MFMA atom (_ATOMS in
/// rocke/library/kernels/common/conv_direct_nongrouped.py); std::nullopt for any other.
inline std::optional<int64_t> nongroupedAtomTile(std::string_view atom)
{
    if(atom == "32x32x16" || atom == "32x32x8")
    {
        return 32;
    }
    if(atom == "16x16x32" || atom == "16x16x16")
    {
        return 16;
    }
    return std::nullopt;
}

/// rocKE's DirectConvProblem takes one stride and one padding, and the instance's compiled
/// filter, stride, padding and dilation must all be the graph's.
inline bool directCompiledMatches(const DirectCapabilities& compiled, const Problem& problem)
{
    return problem.strideH == problem.strideW && problem.padH == problem.padW
           && directServes(compiled, problem);
}

/// ceil(a / b) for a >= 0, b > 0, without the overflow of a + b - 1.
inline int64_t ceilDiv(int64_t a, int64_t b)
{
    return a == 0 ? 0 : 1 + (a - 1) / b;
}

/// The (x, y, z) grid with every axis at most 65535 and at least 1, else std::nullopt.
inline std::optional<LaunchGeometry>
    boundedDirectGrid(int64_t gridX, int64_t gridY, int64_t gridZ, int64_t threads)
{
    for(const auto extent : {gridX, gridY, gridZ})
    {
        if(extent < 1 || extent > DIRECT_MAX_GRID_DIM)
        {
            return std::nullopt;
        }
    }
    return LaunchGeometry{static_cast<unsigned int>(gridX),
                          static_cast<unsigned int>(gridY),
                          static_cast<unsigned int>(gridZ),
                          static_cast<unsigned int>(threads)};
}

/// rocKE's depthwise filter bound (_DW_MAX_KH, _DW_MAX_KW).
constexpr int64_t DIRECT_DEPTHWISE_MAX_FILTER = 32;

/// The tiled and col depthwise kernels: is_valid_depthwise_tiled_spec and
/// is_valid_depthwise_col_spec in conv_direct_grouped.py on the runtime problem, then
/// direct_launch_geometry's (ceil(Wo / block_w), ceil(groups / (64 * block_waves)),
/// N * ceil(Ho / block_h)) grid of 64 * block_waves threads. Both index outputs directly
/// and bound their rows by the runtime Ho, so neither is tied to "same" padding. col
/// also requires Ho <= Hi, a padding below the filter and a block no wider than Wo.
inline std::optional<LaunchGeometry> depthwiseTileLaunch(const Problem& problem,
                                                         const Geometry& geometry,
                                                         const DirectInstance& instance)
{
    const auto& compiled = instance.compiled;
    if(!(problem.groups == problem.c && problem.c == problem.k)
       || !directCompiledMatches(compiled, problem) || compiled.dilationH != 1
       || compiled.dilationW != 1 || compiled.y < 1 || compiled.y > DIRECT_DEPTHWISE_MAX_FILTER
       || compiled.x < 1 || compiled.x > DIRECT_DEPTHWISE_MAX_FILTER || instance.blockW < 1
       || instance.blockH < 1 || instance.blockWaves < 1
       || instance.blockWaves > DIRECT_MAX_BLOCK_WAVES || geometry.ho < 1 || geometry.wo < 1)
    {
        return std::nullopt;
    }
    if(instance.variant == DirectVariant::TILED)
    {
        if(compiled.strideH > compiled.y)
        {
            return std::nullopt;
        }
    }
    else if(geometry.ho > problem.hi || compiled.padH >= compiled.y || compiled.padH >= compiled.x
            || instance.blockW > geometry.wo)
    {
        return std::nullopt;
    }
    const auto threads = instance.blockWaves * DIRECT_WAVE_SIZE;
    if(problem.n > DIRECT_MAX_GRID_DIM)
    {
        return std::nullopt;
    }
    return boundedDirectGrid(ceilDiv(geometry.wo, instance.blockW),
                             ceilDiv(problem.groups, threads),
                             problem.n * ceilDiv(geometry.ho, instance.blockH),
                             threads);
}

/// The grouped direct kernel: DirectConvSpec.validate and is_valid_spec in
/// conv_direct_grouped.py on the runtime problem (its compiled channel counts, "same"
/// padding with an odd filter and no dilation, stride 1, block_groups dividing the group
/// count), then direct_launch_geometry's (ceil(Wo / block_q), groups / block_groups,
/// N * ceil(Hi / block_h)) grid of 64 * block_groups threads: one wave per group, the row
/// tile over the input height, as the kernel decodes it.
inline std::optional<LaunchGeometry> groupedDirectLaunch(const Problem& problem,
                                                         const Geometry& geometry,
                                                         const DirectInstance& instance)
{
    const auto& compiled = instance.compiled;
    if(problem.groups <= 0 || problem.c % problem.groups != 0 || problem.k % problem.groups != 0
       || problem.c / problem.groups != instance.cpg || problem.k / problem.groups != instance.kpg
       || !directCompiledMatches(compiled, problem) || compiled.dilationH != 1
       || compiled.dilationW != 1 || compiled.y % 2 != 1 || compiled.x % 2 != 1
       || compiled.padH != (compiled.y - 1) / 2 || compiled.strideH != 1 || instance.cpg < 4
       || instance.cpg % 4 != 0 || instance.kpg < 1 || instance.blockGroups < 1
       || instance.blockGroups > DIRECT_MAX_BLOCK_WAVES || instance.blockQ < 16
       || instance.blockQ % 16 != 0 || instance.blockH < 1
       || problem.groups % instance.blockGroups != 0 || geometry.wo < 1
       || problem.n > DIRECT_MAX_GRID_DIM)
    {
        return std::nullopt;
    }
    return boundedDirectGrid(ceilDiv(geometry.wo, instance.blockQ),
                             problem.groups / instance.blockGroups,
                             problem.n * ceilDiv(problem.hi, instance.blockH),
                             instance.blockGroups * DIRECT_WAVE_SIZE);
}

/// tile_w_candidates in conv_direct_nongrouped.py: the multiples of @p atomTile up to six
/// atoms that tile @p wo exactly, or if none does, the one that wastes the fewest columns
/// (the first of them). rocKE's direct_plans offers a non-grouped binary only when its
/// tile width is one of these for the problem's Wo.
inline bool nongroupedTileWidthFits(int64_t tileW, int64_t wo, int64_t atomTile)
{
    constexpr int64_t MAX_MULTIPLE = 6;
    bool anyExact = false;
    int64_t leastWaste = 0;
    int64_t leastWasteWidth = 0;
    for(int64_t multiple = 1; multiple <= MAX_MULTIPLE; ++multiple)
    {
        const auto width = atomTile * multiple;
        if(wo % width == 0)
        {
            anyExact = true;
            if(width == tileW)
            {
                return true;
            }
        }
        const auto covered = ceilDiv(wo, width) * width;
        if(multiple == 1 || covered < leastWaste)
        {
            leastWaste = covered;
            leastWasteWidth = width;
        }
    }
    return !anyExact && tileW == leastWasteWidth;
}

/// The byte offset every masked-out access of the non-grouped kernel is redirected to
/// (_OOB_BASE in conv_direct_nongrouped.py): each tensor must end at or below it.
constexpr int64_t NONGROUPED_OOB_BASE = 0x7F000000;
/// The non-grouped kernel's input loads are dwordx4 of 8 halves (_X_LOAD_VEC).
constexpr int64_t NONGROUPED_LOAD_VECTOR = 8;

/// The non-grouped direct kernel (groups == 1, C and K kernel arguments):
/// DirectNongroupedConvSpec.validate and is_valid_nongrouped_spec in
/// conv_direct_nongrouped.py on the runtime problem (C a multiple of ck and of 8, K a
/// multiple of the atom tile, every tensor below the masked-access offset, stride 1 or
/// 2), the tile widths direct_plans tries for Wo (nongroupedTileWidthFits), then its
/// flat grid of n_w_tiles * n_h_tiles * N * n_k_tiles blocks of waves_m * waves_n * 64
/// threads. The block id is decoded in i32 and a dispatch's x extent in work-items is
/// 32-bit, so both bound the grid.
inline std::optional<LaunchGeometry> nongroupedDirectLaunch(const Problem& problem,
                                                            const Geometry& geometry,
                                                            const DirectInstance& instance)
{
    const auto& compiled = instance.compiled;
    if(problem.groups != 1 || !directCompiledMatches(compiled, problem) || compiled.dilationH < 1
       || compiled.dilationW < 1 || (compiled.strideH != 1 && compiled.strideH != 2)
       || instance.tileH < 1 || instance.tileW < 1 || instance.tileKOut < 1 || instance.ck < 1
       || instance.wavesM < 1 || instance.wavesN < 1 || instance.atomTile < 1
       || instance.wavesM * instance.wavesN > 1024 / DIRECT_WAVE_SIZE || problem.c <= 0
       || problem.k <= 0 || problem.c % instance.ck != 0 || problem.c % NONGROUPED_LOAD_VECTOR != 0
       || problem.k % instance.atomTile != 0 || geometry.ho < 1 || geometry.wo < 1)
    {
        return std::nullopt;
    }
    for(const auto bytes : geometry.tensorBytes)
    {
        if(bytes > NONGROUPED_OOB_BASE)
        {
            return std::nullopt;
        }
    }
    // The last prefetch reads one chunk past C from the masked-access offset, in i32.
    constexpr int64_t I32_MAX = std::numeric_limits<int32_t>::max();
    if(problem.c + instance.ck > (I32_MAX - NONGROUPED_OOB_BASE) / 2
       || !nongroupedTileWidthFits(instance.tileW, geometry.wo, instance.atomTile))
    {
        return std::nullopt;
    }
    // D's byte count bounds N * Ho * Wo * K, and so this product of tile counts.
    const auto blocks = ceilDiv(geometry.wo, instance.tileW) * ceilDiv(geometry.ho, instance.tileH)
                        * problem.n * ceilDiv(problem.k, instance.tileKOut);
    const auto threads = instance.wavesM * instance.wavesN * DIRECT_WAVE_SIZE;
    if(blocks > I32_MAX || blocks * threads > int64_t{std::numeric_limits<uint32_t>::max()})
    {
        return std::nullopt;
    }
    return LaunchGeometry{
        static_cast<unsigned int>(blocks), 1, 1, static_cast<unsigned int>(threads)};
}

/// The launch of @p instance on @p problem, or std::nullopt when the binary cannot compute
/// it. std and spatial mirror _direct_serves in convolution_forward.py: the direct guard
/// (directProblemSupported), the compiled filter, stride and padding (directServes), and
/// directLaunchGeometry's variant rules and 65535 grid bound. The other variants mirror
/// _aot_direct_serves, rocKE's own offer (KernelCache.supports_problem, then plan_for's
/// re-validation and launch geometry). directKernargs then holds no further refusal.
inline std::optional<LaunchGeometry> directInstanceLaunch(const Problem& problem,
                                                          const Geometry& geometry,
                                                          const DirectInstance& instance)
{
    switch(instance.variant)
    {
    case DirectVariant::STD:
    case DirectVariant::SPATIAL:
        if(!directProblemSupported(problem) || !directServes(instance.compiled, problem))
        {
            return std::nullopt;
        }
        return directLaunchGeometry(problem,
                                    geometry,
                                    instance.variant,
                                    instance.blockW,
                                    instance.blockWaves,
                                    DIRECT_WAVE_SIZE);
    case DirectVariant::TILED:
    case DirectVariant::COL:
        return depthwiseTileLaunch(problem, geometry, instance);
    case DirectVariant::GROUPED:
        return groupedDirectLaunch(problem, geometry, instance);
    case DirectVariant::NONGROUPED:
        return nongroupedDirectLaunch(problem, geometry, instance);
    case DirectVariant::NONE:
    default:
        return std::nullopt;
    }
}

/// The variants that compute only pure depthwise problems; grouped and non-grouped
/// compete with implicit GEMM on ordinary graphs.
inline bool isDepthwiseDirectVariant(DirectVariant variant)
{
    return variant == DirectVariant::STD || variant == DirectVariant::SPATIAL
           || variant == DirectVariant::TILED || variant == DirectVariant::COL;
}

/// Whether @p variant, @p blockW and @p blockWaves are gfx950_conv_fwd_direct_spec_for_request's
/// choice with no overrides for @p groups: spatial with one wave below 64 groups, otherwise
/// std with block_w 4 and one wave.
inline bool
    isDefaultDirectArm(int64_t groups, DirectVariant variant, int64_t blockW, int64_t blockWaves)
{
    if(groups < DIRECT_WAVE_SIZE)
    {
        return variant == DirectVariant::SPATIAL && blockW == 0
               && blockWaves == DEFAULT_DIRECT_BLOCK_WAVES;
    }
    return variant == DirectVariant::STD && blockW == DEFAULT_DIRECT_BLOCK_W
           && blockWaves == DEFAULT_DIRECT_BLOCK_WAVES;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd
