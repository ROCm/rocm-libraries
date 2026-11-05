// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>

#include <numeric>
#include <string>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdConvArgs.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdInstance.hpp"

/**
 * @file Gfx1250ConvFwdImplicitGemm.hpp
 * @brief Admission and launch of the gfx1250 forward implicit-GEMM instances.
 *
 * A port of rocKE's dispatcher for `implicit_gemm` (dispatch/grouped_convolution.py):
 * admission is kernels/conv/kernel_identity.identity_supports_problem on the width-folded
 * problem (`_gemm_problem`), the grid ConvArgs.grid() over group-merged groups, the block
 * the AOT job's spec block (spec_launch_block_size), the arguments ConvArgs.
 *
 * Only what the shipped instances use is ported: every one is grouped and 2-D, with patch,
 * split-K, async DMA, stream-K and pointwise off. The packaging adapter
 * (builders/gfx1250/conv_fwd_aot._CONSTANT) refuses an instance that turns one of those on,
 * and parse() refuses an ungrouped instance.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
{

struct ImplicitGemmKernel
{
    std::string label;
    int64_t tileM, tileN, warpM, warpN, waveSize;
    int64_t vectorA, vectorB, vectorC;
    int64_t groupMerge;
    bool wavelet;
    BakedShape baked;

    static ImplicitGemmKernel parse(const InstanceGeometry& g)
    {
        // `grouped` is a KMD field the adapter carries: the port covers the grouped kernel.
        if(!g.flag("grouped"))
        {
            throw std::invalid_argument(g.label() + ": unported ungrouped implicit-GEMM instance");
        }
        return {g.label(),
                g.integer("tile_m"),
                g.integer("tile_n"),
                g.integer("warp_m"),
                g.integer("warp_n"),
                g.integer("wave_size"),
                g.integer("vector_size_a"),
                g.integer("vector_size_b"),
                g.integer("vector_size_c"),
                std::max<int64_t>(1, g.integer("group_merge")),
                g.str("pipeline") == "wavelet",
                BakedShape::parse(g)};
    }
};

/// Load waves a wavelet kernel appends to its workgroup: ImplicitGemmConvSpec.num_load_waves'
/// default, which the AOT job keeps (unchecked agreement, pinned by the golden vectors).
constexpr int64_t WAVELET_LOAD_WAVES = 4;
/// The hardware cap on a grid axis (kernel_identity._MAX_GRID_DIM).
constexpr int64_t MAX_GRID_DIM = 65535;
/// The forward loaders' 24-bit reduction-index products (conv_args.MUL24_REDUCTION_LIMIT).
constexpr int64_t MUL24_REDUCTION_LIMIT = int64_t{1} << 23;

/// The problem the implicit-GEMM instances run (conv_implicit_gemm_dgrad.fold_width): a 2-D
/// ungrouped convolution whose width folds into its channels runs folded.
inline ConvGeometry implicitGemmProblem(const Problem& p)
{
    auto g = ConvGeometry::of(p);
    if(g.groups != 1 || g.dW != 1)
    {
        return g;
    }
    const auto f = std::gcd(std::gcd(g.x, g.sW), std::gcd(g.wi, g.pW));
    if(f <= 1)
    {
        return g;
    }
    g.wi /= f;
    g.c *= f;
    g.x /= f;
    g.sW /= f;
    g.pW /= f;
    return g;
}

inline Refusal refuse(const ImplicitGemmKernel& kernel, const Problem& problem)
{
    // rocKE checks the baked fields against the problem the kernel runs: the width-folded one.
    const auto g = implicitGemmProblem(problem);
    if(auto why
       = kernel.baked.mismatch({g.y, g.x, g.sH, g.sW, g.dH, g.dW, g.pH, g.pW, g.cpg(), g.kpg()}))
    {
        return why;
    }
    const auto cpg = g.cpg();
    const auto kpg = g.kpg();
    const auto gm = kernel.groupMerge;
    std::array<int64_t, 3> extents = {cpg, cpg, kpg};
    std::array<const char*, 3> extentNames = {"cpg", "cpg", "kpg"};
    if(gm > 1)
    {
        if(cpg != 1 || kpg != 1)
        {
            return "group-merged kernel needs depthwise (cpg=kpg=1)";
        }
        if(g.groups % gm != 0)
        {
            return "group_merge does not divide groups";
        }
        if(kpg * gm > kernel.tileN)
        {
            return "merged GEMM-N exceeds tile_n";
        }
        extents[0] = cpg * gm;
        extents[2] = kpg * gm;
        extentNames[0] = "cpg*Gm";
        extentNames[2] = "kpg*Gm";
    }
    const std::array<int64_t, 3> vectors = {kernel.vectorA, kernel.vectorB, kernel.vectorC};
    for(size_t i = 0; i < vectors.size(); ++i)
    {
        if(vectors[i] != 0 && extents[i] != 0 && extents[i] % vectors[i] != 0)
        {
            return std::string(extentNames[i]) + " not divisible by the vector width";
        }
    }
    if(ceilDiv(g.gemmM(), kernel.tileM) > MAX_GRID_DIM
       || ceilDiv(kpg * gm, kernel.tileN) > MAX_GRID_DIM || g.groups / gm > MAX_GRID_DIM)
    {
        return "grid exceeds the hardware cap";
    }
    if(g.gemmK() >= MUL24_REDUCTION_LIMIT)
    {
        return "reduction extent Y*X*cpg reaches 2**23 (24-bit address products)";
    }
    return std::nullopt;
}

inline Plan plan(const ImplicitGemmKernel& kernel, const Problem& problem)
{
    const auto g = implicitGemmProblem(problem);
    auto merged = g;
    merged.groups = g.groups / kernel.groupMerge;
    const auto threads = kernel.warpM * kernel.warpN * kernel.waveSize
                         + (kernel.wavelet ? WAVELET_LOAD_WAVES * kernel.waveSize : 0);
    Step step;
    step.kernel = kernel.label;
    step.grid = {
        ceilDiv(merged.kpg(), kernel.tileN), ceilDiv(merged.gemmM(), kernel.tileM), merged.groups};
    step.block = {threads, 1, 1};
    step.args = implicitFwdArgs(g, kernel.tileM, kernel.tileN);
    return Plan{{std::move(step)}, {}};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
