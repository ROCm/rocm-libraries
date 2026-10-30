// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvDgradGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx950ConvDgradTestCatalog.hpp"

/**
 * @file TestGfx950ConvDgradGeometry.cpp
 * @brief The hipkernel:Gfx950ConvDgradAot launch plan against the rocKE adapter's own:
 *        every shipped catalog instance on every golden problem, compared plan for plan by
 *        digest (and a few in full), then the rules the plan encodes one at a time. The
 *        golden vectors are Gfx950ConvDgradTestVectors.hpp, which
 *        tools/make_conv_dgrad_aot_test_vectors.py wrote from convolution_backward_data.py.
 *        Host-only.
 */
namespace
{

namespace dgrad = hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_dgrad;
namespace vectors = hip_kernel_provider::kernel_ingestor_engine::dgrad_test_vectors;
namespace catalog = hip_kernel_provider::kernel_ingestor_engine::dgrad_test_catalog;

std::string describe(const dgrad::Problem& p)
{
    return "{N" + std::to_string(p.n) + " C" + std::to_string(p.c) + " K" + std::to_string(p.k)
           + " " + std::to_string(p.hi) + "x" + std::to_string(p.wi) + " f" + std::to_string(p.y)
           + "x" + std::to_string(p.x) + " s" + std::to_string(p.strideH) + "x"
           + std::to_string(p.strideW) + " p" + std::to_string(p.padH) + "x"
           + std::to_string(p.padW) + " d" + std::to_string(p.dilationH) + "x"
           + std::to_string(p.dilationW) + " g" + std::to_string(p.groups) + "}";
}

/// A dense-path implicit-GEMM instance with the tile under test: a 2x2 warp grid, scalar
/// widths, the grouped code path and the catalog's 1024-record table.
dgrad::ImplicitInstance scalarInstance(int64_t tileM = 64, int64_t tileN = 64, int64_t tileK = 32)
{
    return dgrad::ImplicitInstance{tileM, tileN, tileK, 2, 2, 1, 1, 1, 1024, true};
}

bool serves(const dgrad::Problem& problem, const dgrad::ImplicitInstance& instance)
{
    return dgrad::implicitPlan(problem, instance).has_value();
}

TEST(TestGfx950ConvDgradGeometry, EveryCatalogInstanceParses)
{
    for(const auto* dtype : {"fp16", "bf16"})
    {
        size_t helpers = 0;
        for(const auto& entry : catalog::catalog(dtype))
        {
            ASSERT_TRUE(entry.instance) << entry.text("instance_id");
            EXPECT_EQ(entry.instance->implicit.has_value(), entry.integer("kernel_family") == 0);
            helpers += entry.isHelper() ? 1 : 0;
        }
        EXPECT_EQ(helpers, 4U) << dtype;
        EXPECT_GT(catalog::catalog(dtype).size(), 300U) << dtype;
    }
}

TEST(TestGfx950ConvDgradGeometry, PlansMatchTheAdapterOverTheCatalog)
{
    // gfx950_conv_dgrad_aot_plan over every catalog instance of the problem's dtype: the same
    // instances serve, and their plans fold to the adapter's digest. On a digest mismatch,
    // make_conv_dgrad_aot_test_vectors.py's render() names each serving plan's digest.
    std::map<std::string, size_t> servedByVariant;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto problem = catalog::toProblem(row.problem);
        const auto& entries = catalog::catalog(row.dtype);
        ASSERT_EQ(std::string_view(row.served).size(), entries.size());
        uint64_t folded = 0xCBF29CE484222325ULL;
        for(size_t i = 0; i < entries.size(); ++i)
        {
            const auto& entry = entries[i];
            const bool expected = row.served[i] == '1';
            const auto plan
                = entry.isHelper() ? std::nullopt : dgrad::instancePlan(problem, *entry.instance);
            EXPECT_EQ(plan.has_value(), expected)
                << entry.text("direct_variant") << " " << entry.text("instance_id") << " on "
                << describe(problem) << " " << row.dtype;
            if(!plan)
            {
                continue;
            }
            const auto digest = dgrad::planDigest(*plan);
            for(int byte = 0; byte < 8; ++byte)
            {
                folded ^= (digest >> (8 * byte)) & 0xFFU;
                folded *= 0x100000001B3ULL;
            }
            ++servedByVariant[entry.text("direct_variant")];
        }
        EXPECT_EQ(folded, row.digest) << describe(problem) << " " << row.dtype;
    }
    // Every family is reached.
    for(const auto* variant : {"none",
                               "depthwise_dgrad",
                               "depthwise_dgrad_win",
                               "grouped_dgrad_fused",
                               "grouped_dgrad_4c",
                               "grouped_dgrad_mfma",
                               "dgrad_mfma_rtc"})
    {
        EXPECT_GT(servedByVariant[variant], 0U) << variant;
    }
}

TEST(TestGfx950ConvDgradGeometry, FullPlansMatchTheAdapter)
{
    ASSERT_GT(vectors::fullPlans().size(), 8U);
    for(const auto& full : vectors::fullPlans())
    {
        const auto& row = vectors::servedMatrix()[full.row];
        const auto problem = catalog::toProblem(row.problem);
        const auto& entry = catalog::catalog(row.dtype)[full.instance];
        SCOPED_TRACE(entry.text("direct_variant") + " " + entry.text("instance_id") + " on "
                     + describe(problem));
        const auto plan = dgrad::instancePlan(problem, *entry.instance);
        ASSERT_TRUE(plan);
        EXPECT_EQ(dgrad::serializePlan(*plan), full.serialized);
    }
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
    // A batch past the 32-bit ABI runs as batch slices: one image has to fit.
    auto large = base;
    large.n = 1 << 14;
    large.hi = large.wi = 64;
    const auto largeGeometry = dgrad::deriveGeometry(large);
    ASSERT_TRUE(largeGeometry);
    EXPECT_GT(largeGeometry->tensorBytes[2], dgrad::I32_MAX);
    auto huge = base;
    huge.hi = huge.wi = 8192;
    huge.c = 32;
    EXPECT_FALSE(dgrad::deriveGeometry(huge));
    // W holds C / groups channels.
    auto grouped = base;
    grouped.groups = 4;
    const auto geometry = dgrad::deriveGeometry(grouped);
    ASSERT_TRUE(geometry);
    EXPECT_EQ(geometry->cpg, 16);
    EXPECT_EQ(geometry->kpg, 16);
    EXPECT_EQ(geometry->tensorBytes[1], 64 * 3 * 3 * 16 * 2);
}

TEST(TestGfx950ConvDgradGeometry, TheTableIsPerGroupNotPerTotalChannels)
{
    // rocKE's C++ rocke_enumerate_sub_gemms reads the total C and K; the port follows the
    // Python reference: gemm_n = C / groups and gemm_k counts K / groups per tap.
    const dgrad::Problem problem{2, 96, 192, 15, 15, 3, 3, 2, 2, 1, 1, 1, 1, 3};
    const auto geometry = dgrad::deriveGeometry(problem);
    ASSERT_TRUE(geometry);
    const auto subGemms = dgrad::enumerateSubGemms(problem, *geometry, 64, 16, 32);
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
    // stride3x2_dil2x1, tile 64x64x32.
    const dgrad::Problem problem{1, 16, 24, 17, 13, 3, 3, 3, 2, 2, 1, 2, 1, 1};
    const auto plan = dgrad::implicitPlan(problem, scalarInstance());
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan->steps.size(), 1U);
    const auto& step = plan->steps[0];
    EXPECT_EQ(step.values[30], -505290270);
    EXPECT_EQ(step.values[31], 5);
    EXPECT_EQ(step.values[32], 991146300);
    EXPECT_EQ(step.values[33], 4);
    EXPECT_EQ(step.values.back(), 6); // num_sub_gemms
    EXPECT_EQ(step.grid, (std::array<int64_t, 3>{6, 1, 1}));
    EXPECT_EQ(plan->workspaceBytes(), 720);
}

TEST(TestGfx950ConvDgradGeometry, ImplicitApplicability)
{
    const dgrad::ImplicitInstance wide{64, 64, 32, 2, 2, 8, 8, 8, 1024, true};
    // The dY width divides K / groups; the W and dX widths C / groups.
    EXPECT_TRUE(serves({1, 16, 16, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, wide));
    EXPECT_FALSE(serves({1, 16, 12, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, wide));
    EXPECT_FALSE(serves({1, 12, 16, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, wide));
    // A grouped problem needs a grouped binary; a grouped binary serves groups == 1.
    auto dense = wide;
    dense.grouped = false;
    EXPECT_TRUE(serves({1, 16, 16, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, dense));
    EXPECT_FALSE(serves({1, 32, 32, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 2}, dense));
    EXPECT_TRUE(serves({1, 32, 32, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 2}, wide));
    // The non-empty sub-GEMM count fits max_sub_gemms.
    auto small = scalarInstance();
    small.maxSubGemms = 3;
    EXPECT_FALSE(serves({1, 16, 16, 15, 15, 3, 3, 2, 2, 1, 1, 1, 1, 1}, small));
    small.maxSubGemms = 4;
    EXPECT_TRUE(serves({1, 16, 16, 15, 15, 3, 3, 2, 2, 1, 1, 1, 1, 1}, small));
    // Grouped depthwise (cpg == 1) is the direct kernels' only.
    EXPECT_FALSE(serves({1, 32, 32, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 32}, scalarInstance()));
    EXPECT_TRUE(serves({1, 64, 64, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 32}, scalarInstance()));
    // A binary compiled for one filter and stride serves only those.
    auto baked = scalarInstance();
    baked.filterH = baked.filterW = 3;
    baked.strideH = baked.strideW = 1;
    EXPECT_TRUE(serves({1, 16, 16, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, baked));
    EXPECT_FALSE(serves({1, 16, 16, 8, 8, 5, 5, 1, 1, 2, 2, 1, 1, 1}, baked));
    EXPECT_FALSE(serves({1, 16, 16, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 1}, baked));
    // The flat tile grid fits one dispatch.
    const dgrad::ImplicitInstance tiny{1, 1, 32, 16, 1, 1, 1, 1, 1024, true};
    EXPECT_FALSE(serves({1, 1024, 1, 1024, 512, 1, 1, 1, 1, 0, 0, 1, 1, 1}, tiny));
    EXPECT_TRUE(serves({1, 1, 1, 8, 8, 1, 1, 1, 1, 0, 0, 1, 1, 1}, tiny));
}

TEST(TestGfx950ConvDgradGeometry, TapOuterKAndHaloServeOnlyWhatTheyCompute)
{
    auto tapk = scalarInstance(64, 64, 32);
    tapk.tapOuterK = true;
    // Stride 1 and dilation 1, and every K tile inside one tap.
    EXPECT_TRUE(serves({1, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, tapk));
    EXPECT_FALSE(serves({1, 64, 48, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, tapk));
    EXPECT_FALSE(serves({1, 64, 64, 14, 14, 3, 3, 2, 2, 1, 1, 1, 1, 1}, tapk));
    EXPECT_FALSE(serves({1, 64, 64, 14, 14, 3, 3, 1, 1, 2, 2, 2, 2, 1}, tapk));
    // The dY halo: the output is the input's size and at most halo_max_wo wide.
    auto halo = tapk;
    halo.dyHalo = 2;
    halo.haloMaxWo = 32;
    EXPECT_TRUE(serves({1, 64, 64, 14, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}, halo));
    EXPECT_FALSE(serves({1, 64, 64, 14, 33, 3, 3, 1, 1, 1, 1, 1, 1, 1}, halo));
    EXPECT_FALSE(serves({1, 64, 64, 14, 14, 3, 3, 1, 1, 0, 0, 1, 1, 1}, halo));
}

TEST(TestGfx950ConvDgradGeometry, TheWidthFoldsIntoTheChannels)
{
    // 2x2 at stride 2: f = gcd(X, sW, Wi, pW) = 2 folds X 2 -> 1, sW 2 -> 1, Wi 32 -> 16 and
    // C 64 -> 128.
    const dgrad::Problem patchlike{2, 64, 128, 32, 32, 2, 2, 2, 2, 0, 0, 1, 1, 1};
    EXPECT_EQ(dgrad::widthFold(patchlike), 2);
    const auto folded = dgrad::implicitLaunchProblem(patchlike);
    EXPECT_EQ(folded.c, 128);
    EXPECT_EQ(folded.wi, 16);
    EXPECT_EQ(folded.x, 1);
    EXPECT_EQ(folded.strideW, 1);
    const auto plan = dgrad::implicitPlan(patchlike, scalarInstance());
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->launched.c, 128);
    // The same bytes in all three tensors.
    EXPECT_EQ(plan->geometry.tensorBytes, dgrad::deriveGeometry(patchlike)->tensorBytes);
    // Grouped, dilated-along-W or padded problems that the gcd does not divide stay.
    EXPECT_EQ(dgrad::widthFold({2, 64, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}), 1);
    EXPECT_EQ(dgrad::widthFold({2, 64, 128, 32, 32, 2, 2, 2, 2, 0, 0, 1, 1, 2}), 1);
    EXPECT_EQ(dgrad::widthFold({2, 64, 128, 32, 32, 2, 2, 2, 2, 0, 0, 1, 2, 1}), 1);
}

TEST(TestGfx950ConvDgradGeometry, APatchKernelServesPatchEmbeddingsAsOneGemm)
{
    // dX is stored along a filter row's run of X * C channels (12 here), so 4 wide.
    auto patch = dgrad::ImplicitInstance{64, 64, 64, 2, 2, 8, 8, 4, 1024, false};
    patch.patch = true;
    // 4x4 stride 4 over 3 channels: folded by 4 to a 4x1 filter over 12 channels; one
    // record over the filter's Y * X * C = 48 channels.
    const dgrad::Problem embedding{2, 3, 64, 64, 64, 4, 4, 4, 4, 0, 0, 1, 1, 1};
    const auto plan = dgrad::implicitPlan(embedding, patch);
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan->steps.size(), 1U);
    EXPECT_EQ(plan->steps[0].values.back(), 1);
    EXPECT_FALSE(plan->zeroDx);
    // Not a patch embedding: padded, overlapping or grouped.
    EXPECT_FALSE(dgrad::implicitPlan({2, 8, 64, 64, 64, 4, 4, 4, 4, 1, 1, 1, 1, 1}, patch));
    EXPECT_FALSE(dgrad::implicitPlan({2, 8, 64, 64, 64, 4, 4, 2, 2, 0, 0, 1, 1, 1}, patch));
    // A stride past the filter leaves gaps between the patches: dX is zero-filled.
    const auto gapped = dgrad::implicitPlan({2, 8, 64, 30, 30, 4, 4, 5, 5, 0, 0, 1, 1, 1}, patch);
    ASSERT_TRUE(gapped);
    EXPECT_TRUE(gapped->zeroDx);
}

TEST(TestGfx950ConvDgradGeometry, DxIsZeroFilledOnlyWhereTheRecordsLeaveItUnwritten)
{
    const auto zeroFill = [](const dgrad::Problem& problem) {
        const auto plan = dgrad::implicitPlan(problem, scalarInstance());
        EXPECT_TRUE(plan) << describe(problem);
        return plan && plan->zeroDx;
    };
    // Stride 1, dilated or not: every pixel has a tap.
    EXPECT_FALSE(zeroFill({1, 8, 8, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_FALSE(zeroFill({1, 8, 8, 9, 9, 3, 3, 1, 1, 2, 2, 2, 2, 1}));
    // A 3x3 filter at stride 2 reaches every residue.
    EXPECT_FALSE(zeroFill({1, 8, 8, 9, 9, 3, 3, 2, 2, 1, 1, 1, 1, 1}));
    EXPECT_FALSE(zeroFill({1, 8, 8, 14, 14, 3, 3, 2, 2, 1, 1, 1, 1, 1}));
    // A 1x1 filter at stride 2 misses the odd rows and columns.
    EXPECT_TRUE(zeroFill({1, 8, 8, 14, 14, 1, 1, 2, 2, 0, 0, 1, 1, 1}));
    EXPECT_TRUE(zeroFill({1, 8, 8, 14, 14, 1, 1, 1, 2, 0, 0, 1, 1, 1}));
    // So does a 3x3 at stride 4.
    EXPECT_TRUE(zeroFill({1, 8, 8, 16, 16, 3, 3, 4, 4, 1, 1, 1, 1, 1}));
    // A row past the last output's reach is still inside its residue's tilde slice, and
    // stores the zero its out-of-range taps sum to.
    EXPECT_FALSE(zeroFill({1, 8, 8, 15, 15, 2, 2, 2, 2, 0, 0, 1, 1, 1}));
}

TEST(TestGfx950ConvDgradGeometry, ABatchPastTheAbiRunsAsSlices)
{
    // bf16 16 x 256 x 768 x 768: 4.8 GB of dX, 302 MB an image.
    const dgrad::Problem big{16, 256, 256, 768, 768, 3, 3, 1, 1, 1, 1, 1, 1, 1};
    const auto geometry = dgrad::deriveGeometry(big);
    ASSERT_TRUE(geometry);
    const auto chunks = dgrad::batchChunks(big, *geometry);
    ASSERT_TRUE(chunks);
    ASSERT_EQ(chunks->size(), 3U);
    EXPECT_EQ((*chunks)[0], (std::pair<int64_t, int64_t>{0, 6}));
    EXPECT_EQ((*chunks)[2], (std::pair<int64_t, int64_t>{12, 4}));
    const auto plan = dgrad::implicitPlan(big, scalarInstance(256, 128, 64));
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan->steps.size(), 3U);
    int64_t tables = 0;
    for(size_t i = 0; i < plan->steps.size(); ++i)
    {
        const auto& step = plan->steps[i];
        const auto [n0, n] = (*chunks)[i];
        ASSERT_TRUE(step.views[0]);
        ASSERT_TRUE(step.views[2]);
        EXPECT_FALSE(step.views[1]);
        EXPECT_EQ(step.views[2]->offset, n0 * geometry->dxImageBytes);
        EXPECT_EQ(step.views[2]->bytes, n * geometry->dxImageBytes);
        EXPECT_LE(step.views[2]->bytes, dgrad::I32_MAX);
        EXPECT_EQ(step.values[0], n);
        EXPECT_EQ(step.tableOffset, tables);
        tables += static_cast<int64_t>(step.table.size() * sizeof(int32_t));
    }
    EXPECT_EQ(plan->workspaceBytes(), tables);
    // The direct kernels take the whole tensors.
    dgrad::DirectInstance depthwise{dgrad::DirectVariant::DEPTHWISE,
                                    {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}};
    depthwise.knobs.blockW = 8;
    depthwise.knobs.blockWaves = 1;
    EXPECT_TRUE(dgrad::directPlan({2, 64, 64, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 64}, depthwise));
    EXPECT_FALSE(
        dgrad::directPlan({16, 2048, 2048, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, 2048}, depthwise));
}

TEST(TestGfx950ConvDgradGeometry, DirectPipelinesLaunchAsRockesPlanFor)
{
    // The windowed depthwise MFMA form: 8 channels a wave, w_fold images per 32-column tile.
    dgrad::DirectInstance win{dgrad::DirectVariant::DEPTHWISE_WIN, {7, 7, 1, 1, 3, 3, 1, 1, 1, 1}};
    win.knobs.blockH = 8;
    win.knobs.blockWaves = 4;
    win.knobs.mfma = 1;
    win.knobs.prefetchRows = 2;
    win.knobs.wFold = 4;
    const auto winPlan
        = dgrad::directPlan({128, 1024, 1024, 7, 7, 7, 7, 1, 1, 3, 3, 1, 1, 1024}, win);
    ASSERT_TRUE(winPlan);
    ASSERT_EQ(winPlan->steps.size(), 1U);
    EXPECT_EQ(winPlan->steps[0].abi, dgrad::Abi::DIRECT_DGRAD);
    EXPECT_EQ(winPlan->steps[0].grid, (std::array<int64_t, 3>{1, 32, 32}));
    EXPECT_EQ(winPlan->steps[0].block, 256);
    EXPECT_EQ(winPlan->workspaceBytes(), 0);
    // The MFMA form needs whole 8-channel groups.
    EXPECT_FALSE(dgrad::directPlan({128, 1020, 1020, 7, 7, 7, 7, 1, 1, 3, 3, 1, 1, 1020}, win));

    // The fused pipeline: one forward launch over the transposed problem, on W itself.
    dgrad::DirectInstance fused{dgrad::DirectVariant::FUSED, {3, 3, 1, 1, 1, 1, 1, 1, 16, 16}};
    fused.knobs.blockGroups = 2;
    fused.knobs.blockH = 16;
    fused.knobs.blockQ = 16;
    fused.knobs.dgradWeightsLds = 1;
    fused.knobs.xcdTiles = 1;
    const auto fusedPlan
        = dgrad::directPlan({16, 512, 512, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32}, fused);
    ASSERT_TRUE(fusedPlan);
    ASSERT_EQ(fusedPlan->steps.size(), 1U);
    const auto& step = fusedPlan->steps[0];
    EXPECT_EQ(step.abi, dgrad::Abi::DIRECT_FWD);
    EXPECT_EQ(
        step.buffers,
        (std::array<dgrad::Buffer, 3>{dgrad::Buffer::DY, dgrad::Buffer::W, dgrad::Buffer::DX}));
    EXPECT_EQ(step.grid, (std::array<int64_t, 3>{2, 16, 32}));
    EXPECT_EQ(step.block, 128);
    // N, H = Ho, W = Wo, Ho', Wo', groups, total_c = K, total_k = C, then strides.
    EXPECT_EQ((std::vector<int32_t>(step.values.begin(), step.values.begin() + 8)),
              (std::vector<int32_t>{16, 28, 28, 28, 28, 32, 512, 512}));
    // groups % block_groups, and xcd_tiles needs eight row bands.
    EXPECT_FALSE(dgrad::directPlan({16, 528, 528, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 33}, fused));
    EXPECT_FALSE(dgrad::directPlan({1, 512, 512, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32}, fused));

    // The 4c pipeline: 16 groups a wave.
    dgrad::DirectInstance fourC{dgrad::DirectVariant::FOUR_C, {3, 3, 1, 1, 1, 1, 1, 1, 4, 4}};
    fourC.knobs.blockGroups = 32;
    fourC.knobs.blockQ = 4;
    fourC.knobs.dgradWeightsLds = 1;
    fourC.knobs.stageRows = 1;
    const auto fourCPlan
        = dgrad::directPlan({8, 256, 256, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 64}, fourC);
    ASSERT_TRUE(fourCPlan);
    EXPECT_EQ(fourCPlan->steps[0].grid, (std::array<int64_t, 3>{8, 2, 8}));
    EXPECT_EQ(fourCPlan->steps[0].block, 128);
    EXPECT_FALSE(dgrad::directPlan({8, 192, 192, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 48}, fourC));
}

TEST(TestGfx950ConvDgradGeometry, TheMfmaPipelinesTransformTheWeightsFirst)
{
    // The dense c4k320 row: waves_k 2 reads the reorganized (coalesced) weights.
    dgrad::DirectInstance mfma{dgrad::DirectVariant::MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 4, 320}};
    mfma.knobs.blockGroups = 1;
    mfma.knobs.blockH = 16;
    mfma.knobs.blockQ = 16;
    mfma.knobs.foldK32 = 1;
    mfma.knobs.wavesK = 2;
    mfma.knobs.wavesQ = 1;
    const dgrad::Problem dense{1, 4, 320, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1};
    const auto plan = dgrad::directPlan(dense, mfma);
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan->steps.size(), 3U);
    EXPECT_EQ(plan->steps[0].kernel, dgrad::PlanKernel::TRANSPOSE);
    EXPECT_EQ(plan->steps[0].grid, (std::array<int64_t, 3>{9, 5, 4}));
    EXPECT_EQ(plan->steps[1].kernel, dgrad::PlanKernel::REORGANIZE);
    EXPECT_EQ(plan->steps[1].grid, (std::array<int64_t, 3>{90, 1, 1}));
    EXPECT_EQ(plan->steps[2].kernel, dgrad::PlanKernel::SELF);
    EXPECT_EQ(plan->steps[2].buffers[1], dgrad::Buffer::WS_COA);
    EXPECT_EQ(plan->steps[2].grid, (std::array<int64_t, 3>{4, 1, 4}));
    // ws_t: C * KH * KW * kpg * 2; ws_coa at ws_t rounded up to 256.
    ASSERT_EQ(plan->workspaces.size(), 2U);
    EXPECT_EQ(plan->workspaces[0].bytes, 4 * 9 * 320 * 2);
    EXPECT_EQ(plan->workspaces[1].offset, 23040);
    EXPECT_EQ(plan->workspaces[1].bytes, 9 * 10 * 1 * 64 * 8 * 2);
    // Coalesced weights serve groups == 1 only (the reorganize helper's group offset).
    EXPECT_FALSE(dgrad::directPlan({1, 8, 640, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 2}, mfma));

    // The runtime-channel pipeline takes cpg and kpg from the problem, within its tiles.
    auto rtc = mfma;
    rtc.variant = dgrad::DirectVariant::MFMA_RTC;
    rtc.compiled.cpg = rtc.compiled.kpg = 0;
    rtc.knobs.kAtoms = 10;
    rtc.knobs.mTiles = 1;
    const auto rtcPlan = dgrad::directPlan({1, 12, 320, 20, 24, 3, 3, 1, 1, 1, 1, 1, 1, 1}, rtc);
    ASSERT_TRUE(rtcPlan);
    EXPECT_EQ(rtcPlan->steps[0].kernel, dgrad::PlanKernel::TRANSPOSE_RTC);
    EXPECT_EQ(rtcPlan->steps[0].abi, dgrad::Abi::TRANSFORM_RTC);
    EXPECT_EQ(rtcPlan->steps[0].values, (std::vector<int32_t>{12, 320}));
    // 288 channels are 9 atoms, 20 channels two M-tiles: not this binary's.
    EXPECT_FALSE(dgrad::directPlan({1, 12, 288, 20, 24, 3, 3, 1, 1, 1, 1, 1, 1, 1}, rtc));
    EXPECT_FALSE(dgrad::directPlan({1, 20, 320, 20, 24, 3, 3, 1, 1, 1, 1, 1, 1, 1}, rtc));
}

TEST(TestGfx950ConvDgradGeometry, UnknownOrIncompleteMetadataIsRefused)
{
    const auto& implicit = catalog::firstEntry(
        "fp16", [](const catalog::Entry& e) { return e.integer("kernel_family") == 0; });
    const auto& fused = catalog::firstEntry("fp16", [](const catalog::Entry& e) {
        return e.text("direct_variant") == "grouped_dgrad_fused";
    });
    for(const auto* entry : {&implicit, &fused})
    {
        ASSERT_TRUE(dgrad::parseAotInstance(catalog::FieldsMetadata{entry->fields}));
        for(const auto& [field, value] : std::vector<std::pair<std::string, catalog::FieldValue>>{
                {"direct_variant", std::string("bogus")},
                {"kernel_family", int64_t{3}},
                {"layout", std::string("NCHW")},
                {"wave_size", int64_t{32}},
                {"halo_max_wo", int64_t{7}},
                {"block_w", int64_t{3}},
                {"dot2", int64_t{2}}})
        {
            auto fields = entry->fields;
            fields[field] = value;
            EXPECT_FALSE(dgrad::parseAotInstance(catalog::FieldsMetadata{fields}))
                << entry->text("direct_variant") << " " << field;
        }
        for(const auto& [field, value] : entry->fields)
        {
            if(field == "instance_id" || field == "dtype")
            {
                continue;
            }
            auto fields = entry->fields;
            fields.erase(field);
            EXPECT_FALSE(dgrad::parseAotInstance(catalog::FieldsMetadata{fields}))
                << entry->text("direct_variant") << " without " << field;
        }
    }
}

} // namespace
