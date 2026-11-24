// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvDgradAotGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvDgradTestCatalog.hpp"

/**
 * @file TestGfx1250ConvDgradGeometry.cpp
 * @brief The hipkernel:Gfx1250ConvDgradAot launch plan against the rocKE adapter's own:
 *        every shipped catalog instance on every golden problem, compared plan for plan by
 *        digest (and a few in full), then the rules gfx1250's own pipelines add. The
 *        golden vectors are Gfx1250ConvDgradTestVectors.hpp, which
 *        tools/make_conv_dgrad_aot_test_vectors.py wrote from convolution_backward_data.py.
 *        Host-only.
 */
namespace
{

namespace dgrad = hip_kernel_provider::kernel_ingestor_engine::conv_dgrad_aot;
namespace vectors = hip_kernel_provider::kernel_ingestor_engine::gfx1250_dgrad_test_vectors;
namespace catalog = hip_kernel_provider::kernel_ingestor_engine::dgrad_test_catalog;

using catalog::describe;

const std::vector<catalog::Entry>& entries(const std::string& dtype)
{
    return catalog::catalog(dtype, dgrad::GFX1250);
}

const catalog::Entry& firstOf(const std::string& variant)
{
    return catalog::firstEntry(
        "fp16",
        [&](const catalog::Entry& e) { return e.text("direct_variant") == variant; },
        dgrad::GFX1250);
}

dgrad::Problem problem(int64_t n,
                       int64_t c,
                       int64_t k,
                       int64_t hi,
                       int64_t wi,
                       int64_t filter,
                       int64_t stride,
                       int64_t pad,
                       int64_t groups = 1)
{
    return dgrad::Problem{n, c, k, hi, wi, filter, filter, stride, stride, pad, pad, 1, 1, groups};
}

TEST(TestGfx1250ConvDgradGeometry, EveryCatalogInstanceParses)
{
    for(const auto* dtype : {"fp16", "bf16"})
    {
        std::map<std::string, size_t> byVariant;
        for(const auto& entry : entries(dtype))
        {
            ASSERT_TRUE(entry.instance) << entry.text("instance_id");
            EXPECT_EQ(entry.instance->implicit.has_value(), entry.integer("kernel_family") == 0);
            EXPECT_FALSE(entry.isHelper());
            ++byVariant[entry.text("direct_variant")];
        }
        // The gfx1250 dgrad instance list: 109 instances per dtype, no helpers.
        EXPECT_EQ(entries(dtype).size(), 109U) << dtype;
        EXPECT_EQ(byVariant.size(), 6U) << dtype;
        // A gfx1250 kernel's metadata does not describe a gfx950 instance.
        EXPECT_FALSE(dgrad::parseAotInstance(catalog::FieldsMetadata{entries(dtype)[0].fields},
                                             dgrad::GFX950));
    }
}

TEST(TestGfx1250ConvDgradGeometry, PlansMatchTheAdapterOverTheCatalog)
{
    // conv_dgrad_aot_plan over every catalog instance of the problem's dtype: the same
    // instances serve, and their plans fold to the adapter's digest.
    std::map<std::string, size_t> servedByVariant;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto p = catalog::toProblem(row.problem);
        const auto& dtypeEntries = entries(row.dtype);
        ASSERT_EQ(std::string_view(row.served).size(), dtypeEntries.size());
        uint64_t folded = catalog::FNV_OFFSET;
        for(size_t i = 0; i < dtypeEntries.size(); ++i)
        {
            const auto& entry = dtypeEntries[i];
            const bool expected = row.served[i] == '1';
            const auto plan = dgrad::instancePlan(p, *entry.instance);
            EXPECT_EQ(plan.has_value(), expected)
                << entry.text("direct_variant") << " " << entry.text("instance_id") << " on "
                << describe(p) << " " << row.dtype;
            if(!plan)
            {
                continue;
            }
            const auto digest = dgrad::planDigest(*plan);
            catalog::foldPlanDigest(folded, digest);
            ++servedByVariant[entry.text("direct_variant")];
        }
        EXPECT_EQ(folded, row.digest) << describe(p) << " " << row.dtype;
    }
    // Every family is reached.
    for(const auto* variant : {"none",
                               "depthwise_dgrad",
                               "depthwise_dgrad_win",
                               "dgrad_wmma",
                               "dgrad_smallc_wmma",
                               "dgrad_patch_wmma"})
    {
        EXPECT_GT(servedByVariant[variant], 0U) << variant;
    }
}

TEST(TestGfx1250ConvDgradGeometry, FullPlansMatchTheAdapter)
{
    ASSERT_GT(vectors::fullPlans().size(), 5U);
    for(const auto& full : vectors::fullPlans())
    {
        const auto& row = vectors::servedMatrix()[full.row];
        const auto p = catalog::toProblem(row.problem);
        const auto& entry = entries(row.dtype)[full.instance];
        SCOPED_TRACE(entry.text("direct_variant") + " " + entry.text("instance_id") + " on "
                     + describe(p));
        const auto plan = dgrad::instancePlan(p, *entry.instance);
        ASSERT_TRUE(plan);
        EXPECT_EQ(dgrad::serializePlan(*plan), full.serialized);
    }
}

TEST(TestGfx1250ConvDgradGeometry, AWaveletTileLaunchesItsLoadWaves)
{
    for(const auto& entry : entries("fp16"))
    {
        if(!entry.instance->implicit)
        {
            continue;
        }
        const auto& gemm = *entry.instance->implicit;
        const bool wavelet = entry.text("pipeline") == "wavelet";
        EXPECT_EQ(gemm.waveSize, 32);
        EXPECT_EQ(gemm.threads(),
                  (gemm.warpM * gemm.warpN + (wavelet ? dgrad::WAVELET_LOAD_WAVES : 0)) * 32)
            << entry.text("instance_id");
    }
}

TEST(TestGfx1250ConvDgradGeometry, TheStreamingDepthwiseLanesOwnCplChannels)
{
    const auto& entry = catalog::firstEntry(
        "fp16",
        [](const catalog::Entry& e) {
            return e.text("direct_variant") == "depthwise_dgrad" && e.integer("cpl") > 1;
        },
        dgrad::GFX1250);
    const auto& direct = *entry.instance->direct;
    const auto cpl = direct.knobs.cpl;
    const auto f = direct.compiled.filterH;
    const auto s = direct.compiled.strideH;
    const auto pad = direct.compiled.padH;
    const auto channels = 64 * cpl;
    const auto plan = dgrad::instancePlan(
        problem(2, channels, channels, 28, 28, f, s, pad, channels), *entry.instance);
    ASSERT_TRUE(plan);
    const auto threads = direct.knobs.blockWaves * 64;
    EXPECT_EQ(plan->steps.back().block, threads);
    EXPECT_EQ(plan->steps.back().grid[1], dgrad::ceilDiv(channels, threads * cpl));
    // Channels that are not a multiple of cpl are refused.
    EXPECT_FALSE(dgrad::instancePlan(
        problem(2, channels + 1, channels + 1, 28, 28, f, s, pad, channels + 1), *entry.instance));
}

TEST(TestGfx1250ConvDgradGeometry, TheWmmaPipelinesServeOnlyHalfUsefulTileGrids)
{
    // A dense WMMA tile on an image far smaller than its pixel tile is mostly padding.
    const auto& dense = catalog::firstEntry(
        "fp16",
        [](const catalog::Entry& e) {
            return e.text("direct_variant") == "dgrad_wmma" && e.integer("cpg") == 0
                   && e.integer("stride_h") == 1 && e.integer("dilation_h") == 1
                   && e.integer("pad_h") == 1 && e.integer("filter_h") == 3;
        },
        dgrad::GFX1250);
    ASSERT_TRUE(dgrad::instancePlan(problem(2, 256, 256, 64, 64, 3, 1, 1), *dense.instance));
    EXPECT_FALSE(dgrad::instancePlan(problem(2, 256, 256, 2, 2, 3, 1, 1), *dense.instance));
    // Channels not a multiple of 8 are no 16-byte rows.
    EXPECT_FALSE(dgrad::instancePlan(problem(2, 252, 256, 64, 64, 3, 1, 1), *dense.instance));

    // The small-C and patch stems serve ungrouped problems with K a multiple of 8 only.
    const auto& smallc = firstOf("dgrad_smallc_wmma");
    const auto& sc = smallc.instance->direct->compiled;
    const auto stem = problem(1, sc.cpg, 320, 128, 128, sc.filterH, sc.strideH, sc.padH);
    ASSERT_TRUE(dgrad::instancePlan(stem, *smallc.instance)) << describe(stem);
    auto grouped = stem;
    grouped.c *= 2;
    grouped.k *= 2;
    grouped.groups = 2;
    EXPECT_FALSE(dgrad::instancePlan(grouped, *smallc.instance));
    auto oddK = stem;
    oddK.k = 324;
    EXPECT_FALSE(dgrad::instancePlan(oddK, *smallc.instance));

    const auto& patch = firstOf("dgrad_patch_wmma");
    const auto& pc = patch.instance->direct->compiled;
    const auto embedding = problem(2, pc.cpg, 768, 224, 224, pc.filterH, pc.strideH, 0);
    ASSERT_TRUE(dgrad::instancePlan(embedding, *patch.instance)) << describe(embedding);
    auto oddEmbedding = embedding;
    oddEmbedding.k = 772;
    EXPECT_FALSE(dgrad::instancePlan(oddEmbedding, *patch.instance));
}

TEST(TestGfx1250ConvDgradGeometry, UnknownOrIncompleteMetadataIsRefused)
{
    for(const auto* variant : {"none", "dgrad_wmma", "dgrad_smallc_wmma", "depthwise_dgrad"})
    {
        const auto& entry = catalog::firstEntry(
            "fp16",
            [&](const catalog::Entry& e) { return e.text("direct_variant") == variant; },
            dgrad::GFX1250);
        ASSERT_TRUE(dgrad::parseAotInstance(catalog::FieldsMetadata{entry.fields}, dgrad::GFX1250));
        for(const auto& [field, value] : std::vector<std::pair<std::string, catalog::FieldValue>>{
                {"direct_variant", std::string("grouped_dgrad_mfma")},
                {"wave_size", int64_t{64}},
                {"pipeline", std::string("compv3")},
                {"sched", std::string("x")},
                {"fold_k32", int64_t{1}},
                {"r_blk", int64_t{2}},
                {"prefetch", int64_t{2}}})
        {
            if(entry.fields.count(field) != 0 && entry.fields.at(field) == value)
            {
                continue;
            }
            auto fields = entry.fields;
            fields[field] = value;
            // A WMMA pipeline takes a schedule and prefetch, the patch stem r_blk.
            const bool accepted = (field == "sched" && variant != std::string_view("none")
                                   && variant != std::string_view("depthwise_dgrad"));
            EXPECT_EQ(dgrad::parseAotInstance(catalog::FieldsMetadata{fields}, dgrad::GFX1250)
                          .has_value(),
                      accepted)
                << variant << " " << field;
        }
        for(const auto& [field, value] : entry.fields)
        {
            if(field == "instance_id" || field == "dtype")
            {
                continue;
            }
            auto fields = entry.fields;
            fields.erase(field);
            EXPECT_FALSE(dgrad::parseAotInstance(catalog::FieldsMetadata{fields}, dgrad::GFX1250))
                << variant << " without " << field;
        }
    }
}

} // namespace
