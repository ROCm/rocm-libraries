// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvDgradGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx950ConvDgradTestVectors.hpp"

/**
 * @file TestGfx950ConvDgradGeometry.cpp
 * @brief The hipkernel:Gfx950ConvDgradAot launch contract against the rocKE adapter's own
 *        numbers: the sub-GEMM table packer, the 44 implicit-GEMM kernel arguments and
 *        flat grid, the depthwise launch, the MFMA pipeline's plan, and the served
 *        predicates, each compared with golden vectors (Gfx950ConvDgradTestVectors.hpp)
 *        that convolution_backward_data.py produced. Host-only.
 */
namespace
{

namespace dgrad = hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_dgrad;
namespace vectors = hip_kernel_provider::kernel_ingestor_engine::dgrad_test_vectors;

dgrad::Problem toProblem(const vectors::ProblemValues& v)
{
    return dgrad::Problem{
        v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13]};
}

dgrad::Geometry geometryOf(const dgrad::Problem& problem)
{
    const auto geometry = dgrad::deriveGeometry(problem);
    EXPECT_TRUE(geometry);
    return geometry.value_or(dgrad::Geometry{});
}

/// The golden cases' instance: the tile under test, a 1x1 warp grid, scalar widths, the
/// grouped code path and the catalog's 1024-record table.
dgrad::ImplicitInstance goldenInstance(int64_t tileM, int64_t tileN, int64_t tileK)
{
    return dgrad::ImplicitInstance{tileM, tileN, tileK, 1, 1, 1, 1, 1, 1024, true};
}

const vectors::Instance& catalogInstance(std::string_view id)
{
    const auto& instances = vectors::catalogInstances();
    const auto it = std::find_if(instances.begin(), instances.end(), [&](const auto& instance) {
        return id == instance.instanceId;
    });
    if(it == instances.end())
    {
        throw std::runtime_error("no catalog instance " + std::string(id));
    }
    return *it;
}

dgrad::DirectInstance directOf(const vectors::Instance& i)
{
    const auto variant = dgrad::parseDirectVariant(i.direct_variant);
    dgrad::DirectInstance instance{variant.value_or(dgrad::DirectVariant::NONE),
                                   {i.filter_h,
                                    i.filter_w,
                                    i.stride_h,
                                    i.stride_w,
                                    i.pad_h,
                                    i.pad_w,
                                    i.dilation_h,
                                    i.dilation_w,
                                    i.cpg,
                                    i.kpg}};
    instance.blockW = i.block_w;
    instance.blockWaves = i.block_waves;
    instance.blockH = i.block_h;
    instance.blockQ = i.block_q;
    instance.blockGroups = i.block_groups;
    instance.foldK32 = i.fold_k32;
    instance.wavesK = i.waves_k;
    instance.wavesQ = i.waves_q;
    instance.persistentGrid = i.persistent_grid;
    instance.runtimeKLoop = i.runtime_k_loop;
    return instance;
}

dgrad::ImplicitInstance implicitOf(const vectors::Instance& i)
{
    return dgrad::ImplicitInstance{i.tile_m,
                                   i.tile_n,
                                   i.tile_k,
                                   i.warp_m,
                                   i.warp_n,
                                   i.vector_size_a,
                                   i.vector_size_b,
                                   i.vector_size_c,
                                   i.max_sub_gemms,
                                   i.grouped == 1};
}

std::vector<int64_t> gridOf(const dgrad::LaunchGeometry& launch)
{
    return {launch.gridX, launch.gridY, launch.gridZ};
}

std::vector<int64_t> asVector(const std::array<int64_t, 3>& values)
{
    return {values.begin(), values.end()};
}

TEST(TestGfx950ConvDgradGeometry, SubGemmTableMatchesTheAdaptersPacker)
{
    ASSERT_EQ(vectors::subGemmCases().size(), 24U);
    for(const auto& c : vectors::subGemmCases())
    {
        SCOPED_TRACE(std::string(c.name) + " tile " + std::to_string(c.tileM) + "x"
                     + std::to_string(c.tileN) + "x" + std::to_string(c.tileK));
        const auto problem = toProblem(c.problem);
        const auto geometry = geometryOf(problem);
        const auto subGemms
            = dgrad::enumerateSubGemms(problem, geometry, c.tileM, c.tileN, c.tileK);
        ASSERT_TRUE(subGemms);
        EXPECT_EQ(static_cast<int64_t>(subGemms->size()), c.numSubGemms);
        const auto buffer = dgrad::packSubGemmBuffer(*subGemms, c.tileM, c.tileN);
        ASSERT_TRUE(buffer);
        EXPECT_EQ(*buffer, c.buffer);
        // Records first, then the magic entries: never interleaved.
        EXPECT_EQ(static_cast<int64_t>(buffer->size()),
                  c.numSubGemms * (dgrad::SUB_GEMM_RECORD_FIELDS + dgrad::SUB_GEMM_MAGIC_FIELDS));
    }
}

TEST(TestGfx950ConvDgradGeometry, ImplicitLaunchAndKernargsMatchTheAdapter)
{
    for(const auto& c : vectors::subGemmCases())
    {
        SCOPED_TRACE(std::string(c.name) + " tile " + std::to_string(c.tileM) + "x"
                     + std::to_string(c.tileN) + "x" + std::to_string(c.tileK));
        const auto problem = toProblem(c.problem);
        const auto geometry = geometryOf(problem);
        const auto launch = dgrad::implicitInstanceLaunch(
            problem, geometry, goldenInstance(c.tileM, c.tileN, c.tileK));
        ASSERT_TRUE(launch);
        EXPECT_EQ(gridOf(launch->geometry), asVector(c.grid));
        EXPECT_EQ(launch->geometry.blockX, 64U);
        EXPECT_EQ(launch->numSubGemms, c.numSubGemms);
        EXPECT_EQ(static_cast<int64_t>(launch->workspaceBytes), c.workspaceBytes);
        EXPECT_EQ(launch->zeroDx, c.zeroDx);
        EXPECT_EQ(launch->table, c.buffer);

        // The ABI: dY, W, dX, their byte sizes, the 36 by-value values, sub_gemm_buf (a
        // device pointer), num_sub_gemms.
        EXPECT_EQ(c.kernargs[3], geometry.tensorBytes[0]);
        EXPECT_EQ(c.kernargs[4], geometry.tensorBytes[1]);
        EXPECT_EQ(c.kernargs[5], geometry.tensorBytes[2]);
        const auto kernargs = dgrad::implicitGemmKernargs(problem, geometry, c.tileM, c.tileN);
        ASSERT_TRUE(kernargs);
        for(size_t i = 0; i < kernargs->size(); ++i)
        {
            SCOPED_TRACE(dgrad::IMPLICIT_GEMM_KERNARG_NAMES[i]);
            EXPECT_EQ(dgrad::IMPLICIT_GEMM_KERNARG_NAMES[i],
                      std::string_view(vectors::IMPLICIT_KERNARG_NAMES[6 + i]));
            EXPECT_EQ((*kernargs)[i], c.kernargs[6 + i]);
        }
        EXPECT_EQ(std::string_view(vectors::IMPLICIT_KERNARG_NAMES[42]), "sub_gemm_buf");
        EXPECT_EQ(std::string_view(vectors::IMPLICIT_KERNARG_NAMES[43]), "num_sub_gemms");
        EXPECT_EQ(c.kernargs[43], launch->numSubGemms);
    }
}

TEST(TestGfx950ConvDgradGeometry, TheTableIsPerGroupNotPerTotalChannels)
{
    // rocKE's C++ rocke_enumerate_sub_gemms reads the total C and K; the port follows the
    // Python reference: gemm_n = C / groups and gemm_k counts K / groups per tap.
    const dgrad::Problem problem{2, 96, 192, 15, 15, 3, 3, 2, 2, 1, 1, 1, 1, 3};
    const auto geometry = geometryOf(problem);
    const auto subGemms = dgrad::enumerateSubGemms(problem, geometry, 64, 16, 32);
    ASSERT_TRUE(subGemms);
    ASSERT_EQ(subGemms->size(), 4U);
    for(const auto& sg : *subGemms)
    {
        EXPECT_EQ(sg.gemmN, 32);
        EXPECT_EQ(sg.gemmK, sg.yDotSlice * sg.xDotSlice * 64);
    }
    const auto buffer = dgrad::packSubGemmBuffer(*subGemms, 64, 16);
    ASSERT_TRUE(buffer);
    // Record field 2 is ceil(cpg / tile_n), and the first magic divisor is kpg.
    EXPECT_EQ((*buffer)[2], 2);
    const auto kpg = dgrad::magicDivision(64);
    EXPECT_EQ((*buffer)[4 * dgrad::SUB_GEMM_RECORD_FIELDS], kpg->multiplier);
    EXPECT_EQ((*buffer)[4 * dgrad::SUB_GEMM_RECORD_FIELDS + 1], kpg->shift);
}

TEST(TestGfx950ConvDgradGeometry, TheContractsMagicExample)
{
    // CONTRACT §6: stride3x2_dil2x1, tile 64x64x32.
    const dgrad::Problem problem{1, 16, 24, 17, 13, 3, 3, 3, 2, 2, 1, 2, 1, 1};
    const auto geometry = geometryOf(problem);
    const auto kernargs = dgrad::implicitGemmKernargs(problem, geometry, 64, 64);
    ASSERT_TRUE(kernargs);
    EXPECT_EQ((*kernargs)[30], -505290270);
    EXPECT_EQ((*kernargs)[31], 5);
    EXPECT_EQ((*kernargs)[32], 991146300);
    EXPECT_EQ((*kernargs)[33], 4);
    const auto launch
        = dgrad::implicitInstanceLaunch(problem, geometry, goldenInstance(64, 64, 32));
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->numSubGemms, 6);
    EXPECT_EQ(gridOf(launch->geometry), (std::vector<int64_t>{6, 1, 1}));
    EXPECT_EQ(launch->workspaceBytes, 720U);
}

TEST(TestGfx950ConvDgradGeometry, OutputExtentRefusesANegativeNumeratorBeforeDividing)
{
    // Hi = 1, Y = 2, p = 0, s = 2: (1 - 1 - 1) / 2 truncates to 0 in C++, so a naive port
    // computes Ho = 1 where Python's floor gives 0 and refuses.
    EXPECT_FALSE(dgrad::dgradOutputExtent(1, 2, 2, 0, 1));
    EXPECT_FALSE(dgrad::deriveGeometry({1, 16, 16, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 1}));
    EXPECT_FALSE(dgrad::dgradOutputExtent(3, 3, 1, 0, 2));
    // A zero numerator is one output row.
    EXPECT_EQ(dgrad::dgradOutputExtent(2, 2, 2, 0, 1), 1);
    EXPECT_EQ(dgrad::dgradOutputExtent(5, 3, 1, 0, 2), 1);
    EXPECT_EQ(dgrad::dgradOutputExtent(14, 3, 2, 1, 1), 7);
    EXPECT_EQ(dgrad::dgradOutputExtent(15, 3, 2, 1, 1), 8);
    // Out of signed 32-bit descriptor arithmetic.
    EXPECT_FALSE(dgrad::dgradOutputExtent(int64_t{1} << 31, 1, 1, 0, 1));
    EXPECT_FALSE(dgrad::dgradOutputExtent(8, 1, 1, int64_t{1} << 30, 1));
    EXPECT_FALSE(dgrad::dgradOutputExtent(8, 0, 1, 0, 1));
    EXPECT_FALSE(dgrad::dgradOutputExtent(8, 3, 0, 0, 1));
}

TEST(TestGfx950ConvDgradGeometry, GeometryRefusesOutOfContractProblems)
{
    const dgrad::Problem base{2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(dgrad::deriveGeometry(base));
    auto groups = base;
    groups.groups = 3;
    EXPECT_FALSE(dgrad::deriveGeometry(groups));
    groups.groups = 65536;
    groups.c = groups.k = 65536;
    EXPECT_FALSE(dgrad::deriveGeometry(groups));
    groups.groups = 0;
    EXPECT_FALSE(dgrad::deriveGeometry(groups));
    // dX's byte count must fit the signed 32-bit ABI.
    auto large = base;
    large.n = 1 << 14;
    large.hi = large.wi = 64;
    EXPECT_FALSE(dgrad::deriveGeometry(large));
    // W holds C / groups channels.
    auto grouped = base;
    grouped.groups = 4;
    const auto geometry = dgrad::deriveGeometry(grouped);
    ASSERT_TRUE(geometry);
    EXPECT_EQ(geometry->cpg, 16);
    EXPECT_EQ(geometry->kpg, 16);
    EXPECT_EQ(geometry->tensorBytes[1], 64 * 3 * 3 * 16 * 2);
}

TEST(TestGfx950ConvDgradGeometry, ImplicitApplicabilityFollowsTheAdaptersOrder)
{
    const auto serves = [](const dgrad::Problem& problem, const dgrad::ImplicitInstance& instance) {
        const auto geometry = dgrad::deriveGeometry(problem);
        return geometry && dgrad::implicitInstanceLaunch(problem, *geometry, instance).has_value();
    };
    const dgrad::ImplicitInstance wide{64, 64, 32, 2, 2, 8, 8, 8, 1024, true};
    // 1. The dY width divides K / groups; the W and dX widths C / groups.
    EXPECT_TRUE(serves({1, 16, 16, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, wide));
    EXPECT_FALSE(serves({1, 16, 12, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, wide));
    EXPECT_FALSE(serves({1, 12, 16, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, wide));
    auto aOnly = wide;
    aOnly.vectorB = aOnly.vectorC = 1;
    EXPECT_TRUE(serves({1, 12, 16, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, aOnly));
    EXPECT_FALSE(serves({1, 16, 12, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, aOnly));
    // 2. A grouped problem needs a grouped binary; a grouped binary serves groups == 1.
    auto dense = wide;
    dense.grouped = false;
    EXPECT_TRUE(serves({1, 16, 16, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, dense));
    EXPECT_FALSE(serves({1, 32, 32, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 2}, dense));
    EXPECT_TRUE(serves({1, 32, 32, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 2}, wide));
    // 3. The non-empty sub-GEMM count fits max_sub_gemms: a 32-stride problem has 1024,
    // a 33-stride one 1089.
    const dgrad::ImplicitInstance scalar{64, 64, 32, 2, 2, 1, 1, 1, 1024, true};
    EXPECT_TRUE(serves({1, 8, 8, 64, 64, 32, 32, 32, 32, 0, 0, 1, 1, 1}, scalar));
    EXPECT_FALSE(serves({1, 8, 8, 66, 66, 33, 33, 33, 33, 0, 0, 1, 1, 1}, scalar));
    auto small = scalar;
    small.maxSubGemms = 3;
    EXPECT_FALSE(serves({1, 16, 16, 15, 15, 3, 3, 2, 2, 1, 1, 1, 1, 1}, small));
    small.maxSubGemms = 4;
    EXPECT_TRUE(serves({1, 16, 16, 15, 15, 3, 3, 2, 2, 1, 1, 1, 1, 1}, small));
    // 4. Grouped depthwise (cpg == 1) is the direct kernels' only.
    EXPECT_FALSE(serves({1, 32, 32, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 32}, scalar));
    EXPECT_FALSE(serves({1, 32, 64, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 32}, scalar));
    EXPECT_TRUE(serves({1, 64, 64, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 32}, scalar));
    EXPECT_TRUE(serves({1, 1, 1, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, scalar));
    // 6. The flat tile grid fits one dispatch.
    const dgrad::ImplicitInstance tiny{1, 1, 32, 16, 1, 1, 1, 1, 1024, true};
    // 2^19 dX pixels x 1024 channels in 1x1 tiles of 1024 threads: 2^39 work-items.
    EXPECT_FALSE(serves({1, 1024, 1, 1024, 512, 1, 1, 1, 1, 0, 0, 1, 1, 1}, tiny));
    EXPECT_TRUE(serves({1, 1, 1, 8, 8, 1, 1, 1, 1, 0, 0, 1, 1, 1}, tiny));
}

TEST(TestGfx950ConvDgradGeometry, StridedProblemsZeroFillDxAndUnitStrideOnesDoNot)
{
    const auto zeroFill = [](const dgrad::Problem& problem) {
        const auto geometry = dgrad::deriveGeometry(problem);
        const auto launch
            = dgrad::implicitInstanceLaunch(problem, *geometry, goldenInstance(64, 64, 32));
        EXPECT_TRUE(launch);
        return launch && launch->zeroDx;
    };
    EXPECT_FALSE(zeroFill({1, 8, 8, 9, 9, 3, 3, 1, 1, 2, 2, 2, 2, 1}));
    EXPECT_TRUE(zeroFill({1, 8, 8, 9, 9, 3, 3, 2, 1, 1, 1, 1, 1, 1}));
    EXPECT_TRUE(zeroFill({1, 8, 8, 9, 9, 3, 3, 1, 2, 1, 1, 1, 1, 1}));
}

TEST(TestGfx950ConvDgradGeometry, DirectLaunchesMatchTheAdapter)
{
    ASSERT_EQ(vectors::directLaunchCases().size(), 56U);
    for(const auto& c : vectors::directLaunchCases())
    {
        SCOPED_TRACE(std::string(c.problemName) + " " + c.instanceId);
        const auto problem = toProblem(c.problem);
        const auto geometry = geometryOf(problem);
        const auto instance = directOf(catalogInstance(c.instanceId));
        const auto& dy = geometry.tensorBytes[0];
        const auto& w = geometry.tensorBytes[1];
        const auto& dx = geometry.tensorBytes[2];
        if(instance.variant == dgrad::DirectVariant::DEPTHWISE)
        {
            ASSERT_EQ(c.steps.size(), 1U);
            const auto launch = dgrad::depthwiseLaunch(problem, geometry, instance);
            ASSERT_TRUE(launch);
            const auto direct = dgrad::directInstanceLaunch(problem, geometry, instance);
            ASSERT_TRUE(direct);
            EXPECT_EQ(gridOf(*direct), gridOf(*launch));
            EXPECT_EQ(gridOf(*launch), asVector(c.steps[0].grid));
            EXPECT_EQ(launch->blockX, c.steps[0].block);
            const auto& args = c.steps[0].kernargs;
            ASSERT_EQ(args.size(), 20U);
            EXPECT_EQ((std::vector<int64_t>{args[3], args[4], args[5]}),
                      (std::vector<int64_t>{dy, w, dx}));
            const auto kernargs = dgrad::depthwiseKernargs(problem, geometry);
            EXPECT_EQ(std::vector<int64_t>(kernargs.begin(), kernargs.end()),
                      std::vector<int64_t>(args.begin() + 6, args.end()));
            continue;
        }
        ASSERT_EQ(instance.variant, dgrad::DirectVariant::MFMA);
        const auto plan = dgrad::mfmaLaunchPlan(problem, geometry, instance);
        ASSERT_TRUE(plan);
        EXPECT_EQ(plan->workspaceBytes, c.workspaceBytes);
        ASSERT_EQ(c.steps.size(), plan->reorganize ? 3U : 2U);
        EXPECT_EQ(plan->reorganize ? plan->wsCoaOffset : -1, c.wsCoaOffset);
        EXPECT_EQ(plan->reorganize.has_value(), dgrad::usesCoalescedWeights(instance));
        constexpr int64_t WS = 0x50000;
        constexpr int64_t DY = 0x10000;
        constexpr int64_t W = 0x20000;
        constexpr int64_t DX = 0x30000;
        // Transpose: W -> ws_t.
        EXPECT_EQ(std::string_view(c.steps[0].kernel), "transpose");
        EXPECT_EQ(gridOf(plan->transpose), asVector(c.steps[0].grid));
        EXPECT_EQ(plan->transpose.blockX, c.steps[0].block);
        EXPECT_EQ(c.steps[0].kernargs, (std::vector<int64_t>{W, WS, w, plan->wsTBytes}));
        int64_t weights = WS;
        int64_t weightBytes = plan->wsTBytes;
        if(plan->reorganize)
        {
            EXPECT_EQ(std::string_view(c.steps[1].kernel), "reorganize");
            EXPECT_EQ(gridOf(*plan->reorganize), asVector(c.steps[1].grid));
            EXPECT_EQ(plan->reorganize->blockX, c.steps[1].block);
            EXPECT_EQ(c.steps[1].kernargs,
                      (std::vector<int64_t>{
                          WS, WS + plan->wsCoaOffset, plan->wsTBytes, plan->wsCoaBytes}));
            weights = WS + plan->wsCoaOffset;
            weightBytes = plan->wsCoaBytes;
            EXPECT_EQ(plan->workspaceBytes, plan->wsCoaOffset + plan->wsCoaBytes);
            EXPECT_EQ(plan->wsCoaOffset % dgrad::MFMA_WORKSPACE_ALIGNMENT, 0);
        }
        else
        {
            EXPECT_EQ(plan->workspaceBytes, plan->wsTBytes);
        }
        // Compute: the forward direct ABI on the transposed problem.
        const auto& compute = c.steps.back();
        EXPECT_EQ(std::string_view(compute.kernel), "compute");
        EXPECT_EQ(gridOf(plan->compute), asVector(compute.grid));
        EXPECT_EQ(plan->compute.blockX, compute.block);
        const auto direct = dgrad::directInstanceLaunch(problem, geometry, instance);
        ASSERT_TRUE(direct);
        EXPECT_EQ(gridOf(*direct), gridOf(plan->compute));
        ASSERT_EQ(compute.kernargs.size(), 20U);
        EXPECT_EQ((std::vector<int64_t>(compute.kernargs.begin(), compute.kernargs.begin() + 6)),
                  (std::vector<int64_t>{DY, weights, DX, dy, weightBytes, dx}));
        const auto kernargs = dgrad::mfmaComputeKernargs(problem, geometry, *plan);
        EXPECT_EQ(std::vector<int64_t>(kernargs.begin(), kernargs.end()),
                  std::vector<int64_t>(compute.kernargs.begin() + 6, compute.kernargs.end()));
    }
}

TEST(TestGfx950ConvDgradGeometry, TheContractsMfmaExample)
{
    // CONTRACT §9: mfma_c48k192, fold_k32 = 1, waves_k = 6.
    dgrad::DirectInstance instance{dgrad::DirectVariant::MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 48, 192}};
    instance.blockH = 8;
    instance.blockQ = 32;
    instance.blockGroups = 1;
    instance.foldK32 = 1;
    instance.wavesK = 6;
    instance.wavesQ = 1;
    const dgrad::Problem problem{2, 48, 192, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1};
    const auto plan = dgrad::mfmaLaunchPlan(problem, geometryOf(problem), instance);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->wsTBytes, 165888);
    EXPECT_EQ(plan->wsCoaOffset, 165888);
    EXPECT_EQ(plan->wsCoaBytes, 165888);
    EXPECT_EQ(gridOf(plan->transpose), (std::vector<int64_t>{9, 3, 48}));
    ASSERT_TRUE(plan->reorganize);
    EXPECT_EQ(gridOf(*plan->reorganize), (std::vector<int64_t>{162, 1, 1}));
    EXPECT_EQ(gridOf(plan->compute), (std::vector<int64_t>{1, 1, 4}));
    EXPECT_EQ(plan->compute.blockX, 384U);
}

TEST(TestGfx950ConvDgradGeometry, CoalescedWeightInstancesRefuseGroupedProblems)
{
    ASSERT_EQ(vectors::refusalCases().size(), 9U);
    for(const auto& c : vectors::refusalCases())
    {
        SCOPED_TRACE(std::string(c.problemName) + " " + c.instanceId);
        const auto problem = toProblem(c.problem);
        ASSERT_GT(problem.groups, 1);
        const auto geometry = geometryOf(problem);
        const auto instance = directOf(catalogInstance(c.instanceId));
        ASSERT_TRUE(dgrad::usesCoalescedWeights(instance));
        EXPECT_FALSE(dgrad::mfmaLaunchPlan(problem, geometry, instance));
        EXPECT_FALSE(dgrad::directInstanceLaunch(problem, geometry, instance));
        // The same instance serves one group of the same problem.
        auto dense = problem;
        dense.c /= problem.groups;
        dense.k /= problem.groups;
        dense.groups = 1;
        EXPECT_TRUE(dgrad::mfmaLaunchPlan(dense, geometryOf(dense), instance));
        // And a waves_k == 1 build of its channels per group would serve the grouped one.
        auto single = instance;
        single.wavesK = 1;
        single.runtimeKLoop = 0;
        if(dgrad::mfmaKnobsFit(single, geometry.kpg))
        {
            EXPECT_TRUE(dgrad::mfmaLaunchPlan(problem, geometry, single));
        }
    }
}

TEST(TestGfx950ConvDgradGeometry, MfmaApplicabilityRules)
{
    dgrad::DirectInstance base{dgrad::DirectVariant::MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 16, 16}};
    base.blockH = 16;
    base.blockQ = 32;
    base.blockGroups = 4;
    base.wavesK = 1;
    base.wavesQ = 1;
    const auto serves = [](const dgrad::Problem& problem, const dgrad::DirectInstance& instance) {
        const auto geometry = dgrad::deriveGeometry(problem);
        return geometry && dgrad::mfmaLaunchPlan(problem, *geometry, instance).has_value();
    };
    const dgrad::Problem problem{2, 128, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 8};
    EXPECT_TRUE(serves(problem, base));
    // 1. Exact capability.
    for(const auto& change :
        std::vector<dgrad::Problem>{{2, 128, 128, 14, 14, 3, 3, 2, 2, 1, 1, 1, 1, 8},
                                    {2, 128, 128, 14, 14, 3, 3, 1, 1, 0, 0, 1, 1, 8},
                                    {2, 128, 128, 14, 14, 3, 3, 1, 1, 2, 2, 2, 2, 8},
                                    {2, 128, 128, 14, 14, 3, 3, 1, 1, 1, 2, 1, 1, 8},
                                    {2, 128, 256, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 8},
                                    {2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 8},
                                    {2, 128, 128, 14, 14, 5, 5, 1, 1, 2, 2, 1, 1, 8}})
    {
        EXPECT_FALSE(serves(change, base));
    }
    // 2. _mfma_knobs_reason: fold_k32 needs kpg % 32 == 0; the atoms split over waves_k.
    auto fold = base;
    fold.foldK32 = 1;
    EXPECT_FALSE(serves(problem, fold));
    auto waves = base;
    waves.compiled.cpg = 4;
    waves.compiled.kpg = 320;
    waves.wavesK = 4;
    waves.blockGroups = 1;
    EXPECT_TRUE(serves({1, 4, 320, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, waves));
    waves.wavesK = 3;
    EXPECT_FALSE(serves({1, 4, 320, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, waves));
    waves.foldK32 = 1;
    waves.wavesK = 2;
    EXPECT_TRUE(serves({1, 4, 320, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, waves));
    // 3. Coalesced weights serve groups == 1 only.
    EXPECT_FALSE(serves({1, 8, 640, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 2}, waves));
    auto runtimeLoop = base;
    runtimeLoop.runtimeKLoop = 1;
    EXPECT_FALSE(serves(problem, runtimeLoop));
    // 4. block_groups divides the group count.
    EXPECT_FALSE(serves({2, 96, 96, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 6}, base));
    EXPECT_TRUE(serves({2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 4}, base));
    // 5. The compute grid's z axis is N * ceil(Ho' / block_h).
    EXPECT_TRUE(serves({65535, 128, 128, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 8}, base));
    EXPECT_FALSE(serves({65536, 128, 128, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 8}, base));
    EXPECT_TRUE(serves({2, 128, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 8}, base));
}

TEST(TestGfx950ConvDgradGeometry, DepthwiseApplicabilityRules)
{
    dgrad::DirectInstance base{dgrad::DirectVariant::DEPTHWISE, {3, 3, 2, 2, 1, 1, 1, 1, 1, 1}};
    base.blockW = 4;
    base.blockWaves = 2;
    const auto launchOf = [&](const dgrad::Problem& problem) {
        const auto geometry = dgrad::deriveGeometry(problem);
        return geometry ? dgrad::depthwiseLaunch(problem, *geometry, base) : std::nullopt;
    };
    const auto launch = launchOf({2, 300, 300, 29, 27, 3, 3, 2, 2, 1, 1, 1, 1, 300});
    ASSERT_TRUE(launch);
    EXPECT_EQ(gridOf(*launch), (std::vector<int64_t>{7, 3, 2}));
    EXPECT_EQ(launch->blockX, 128U);
    // C == K == groups, one stride and padding, compiled exactly.
    EXPECT_FALSE(launchOf({2, 300, 600, 29, 27, 3, 3, 2, 2, 1, 1, 1, 1, 300}));
    EXPECT_FALSE(launchOf({2, 300, 300, 29, 27, 3, 3, 1, 1, 1, 1, 1, 1, 300}));
    EXPECT_FALSE(launchOf({2, 300, 300, 29, 27, 3, 3, 2, 2, 0, 0, 1, 1, 300}));
    EXPECT_FALSE(launchOf({2, 300, 300, 29, 27, 3, 3, 2, 1, 1, 1, 1, 1, 300}));
    EXPECT_FALSE(launchOf({2, 300, 300, 29, 27, 3, 3, 2, 2, 1, 1, 2, 2, 300}));
    EXPECT_FALSE(launchOf({2, 300, 300, 29, 27, 5, 5, 2, 2, 1, 1, 1, 1, 300}));
    EXPECT_FALSE(launchOf({2, 300, 300, 29, 27, 3, 3, 2, 2, 1, 1, 1, 1, 150}));
    // Grid z carries N.
    EXPECT_TRUE(launchOf({65535, 8, 8, 2, 2, 3, 3, 2, 2, 1, 1, 1, 1, 8}));
    EXPECT_FALSE(launchOf({65536, 8, 8, 2, 2, 3, 3, 2, 2, 1, 1, 1, 1, 8}));
}

TEST(TestGfx950ConvDgradGeometry, ServedPredicatesMatchTheAdapterOverTheCatalog)
{
    // gfx950_conv_dgrad_aot_serves over every fp16 catalog instance and 258 problems,
    // here through the geometry header alone (the matcher test runs the same matrix
    // through the kernel matcher).
    const auto& instances = vectors::catalogInstances();
    size_t served = 0;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto problem = toProblem(row.problem);
        const auto geometry = dgrad::deriveGeometry(problem);
        ASSERT_EQ(std::string_view(row.served).size(), instances.size());
        for(size_t i = 0; i < instances.size(); ++i)
        {
            const auto& instance = instances[i];
            bool actual = false;
            if(geometry && instance.kernel_family == 0)
            {
                const auto gemm = implicitOf(instance);
                actual = dgrad::implicitInstanceLaunch(problem, *geometry, gemm).has_value()
                         && dgrad::implicitGemmKernargs(problem, *geometry, gemm.tileM, gemm.tileN)
                                .has_value();
            }
            else if(geometry && instance.kernel_family == 1)
            {
                actual = dgrad::directInstanceLaunch(problem, *geometry, directOf(instance))
                             .has_value();
            }
            EXPECT_EQ(actual, row.served[i] == '1')
                << instance.instanceId << " on {" << problem.n << ", " << problem.c << ", "
                << problem.k << ", " << problem.hi << ", " << problem.wi << ", " << problem.y
                << ", " << problem.x << ", s" << problem.strideH << "x" << problem.strideW << ", p"
                << problem.padH << "x" << problem.padW << ", d" << problem.dilationH << "x"
                << problem.dilationW << ", g" << problem.groups << "}";
            served += actual ? 1 : 0;
        }
    }
    EXPECT_GT(served, 1000U);
}

TEST(TestGfx950ConvDgradGeometry, FallbackScorePrefersLessTilePadding)
{
    const dgrad::Problem problem{4, 32, 32, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1};
    const auto geometry = geometryOf(problem);
    const dgrad::ImplicitInstance snug{64, 32, 32, 2, 2, 8, 8, 8, 1024, true};
    const dgrad::ImplicitInstance padded{64, 128, 32, 2, 2, 8, 8, 8, 1024, true};
    const auto snugLaunch = dgrad::implicitInstanceLaunch(problem, geometry, snug);
    const auto paddedLaunch = dgrad::implicitInstanceLaunch(problem, geometry, padded);
    ASSERT_TRUE(snugLaunch);
    ASSERT_TRUE(paddedLaunch);
    const auto snugScore = dgrad::aotInstanceScore(problem, geometry, snug, *snugLaunch);
    const auto paddedScore = dgrad::aotInstanceScore(problem, geometry, padded, *paddedLaunch);
    EXPECT_GT(snugScore, paddedScore);
    EXPECT_GT(paddedScore, 0.0);
    EXPECT_LT(snugScore, 1.01);
}

} // namespace
