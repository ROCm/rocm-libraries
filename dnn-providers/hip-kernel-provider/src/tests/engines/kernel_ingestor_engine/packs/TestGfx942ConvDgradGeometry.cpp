// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvDgradAotGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvDgradTestCatalog.hpp"

/**
 * @file TestGfx942ConvDgradGeometry.cpp
 * @brief The hipkernel:Gfx942ConvDgradAot launch plan against the rocKE adapter's own:
 *        every shipped catalog instance on every golden problem, compared plan for plan by
 *        digest (and a few in full), then what gfx942 changes from gfx950, whose pipelines
 *        it ships: its KMD (gfx950's knobs plus the depthwise cpl), the streaming depthwise
 *        lanes owning cpl channels, and the 64 KiB LDS the MFMA pipelines' forward pass
 *        stages into. The golden vectors are Gfx942ConvDgradTestVectors.hpp, which
 *        tools/make_conv_dgrad_aot_test_vectors.py wrote from convolution_backward_data.py.
 *        TestGfx950ConvDgradGeometry.cpp covers the plan rules the engines share.
 *        Host-only.
 */
namespace
{

namespace dgrad = hip_kernel_provider::kernel_ingestor_engine::conv_dgrad_aot;
namespace vectors = hip_kernel_provider::kernel_ingestor_engine::gfx942_dgrad_test_vectors;
namespace catalog = hip_kernel_provider::kernel_ingestor_engine::dgrad_test_catalog;

using catalog::describe;

const std::vector<catalog::Entry>& entries(const std::string& dtype)
{
    return catalog::catalog(dtype, dgrad::GFX942);
}

bool isMfma(const catalog::Entry& entry)
{
    const auto& variant = entry.text("direct_variant");
    return variant == "grouped_dgrad_mfma" || variant == "dgrad_mfma_rtc";
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

TEST(TestGfx942ConvDgradGeometry, EveryCatalogInstanceParses)
{
    // gfx950's variants: gfx942 ships the same pipelines (none of gfx1250's WMMA ones).
    std::set<std::string> gfx950Variants;
    for(const auto& entry : catalog::catalog("fp16", dgrad::GFX950))
    {
        gfx950Variants.insert(entry.text("direct_variant"));
    }
    for(const auto* dtype : {"fp16", "bf16"})
    {
        size_t helpers = 0;
        std::set<std::string> variants;
        for(const auto& entry : entries(dtype))
        {
            ASSERT_TRUE(entry.instance) << entry.text("instance_id");
            EXPECT_EQ(entry.instance->implicit.has_value(), entry.integer("kernel_family") == 0);
            helpers += entry.isHelper() ? 1 : 0;
            variants.insert(entry.text("direct_variant"));
            if(entry.instance->implicit)
            {
                EXPECT_EQ(entry.instance->implicit->waveSize, dgrad::GFX942.waveSize);
                EXPECT_EQ(entry.instance->implicit->loadWaves, 0) << "gfx942 builds no wavelet";
            }
            else
            {
                EXPECT_EQ(entry.instance->direct->waveSize, dgrad::GFX942.waveSize);
                EXPECT_EQ(entry.instance->direct->ldsCapacityBytes, dgrad::GFX942.ldsCapacityBytes);
            }
        }
        // The MFMA pipelines ship with their weight-transform helpers.
        EXPECT_GT(helpers, 0U) << dtype;
        EXPECT_GT(entries(dtype).size(), helpers) << dtype;
        for(const auto& variant : variants)
        {
            EXPECT_EQ(gfx950Variants.count(variant), 1U) << variant << " is not a gfx950 variant";
        }
        // A gfx942 kernel's metadata (it states cpl) describes neither a gfx950 instance
        // (whose KMD has no cpl) nor a gfx1250 one (wave32, with a schedule).
        for(const auto& entry : entries(dtype))
        {
            EXPECT_FALSE(
                dgrad::parseAotInstance(catalog::FieldsMetadata{entry.fields}, dgrad::GFX950))
                << entry.text("instance_id");
            EXPECT_FALSE(
                dgrad::parseAotInstance(catalog::FieldsMetadata{entry.fields}, dgrad::GFX1250))
                << entry.text("instance_id");
        }
    }
    // ... and a gfx950 kernel's metadata (no cpl) does not describe a gfx942 instance.
    for(const auto& entry : catalog::catalog("fp16", dgrad::GFX950))
    {
        EXPECT_FALSE(dgrad::parseAotInstance(catalog::FieldsMetadata{entry.fields}, dgrad::GFX942))
            << entry.text("instance_id");
    }
}

TEST(TestGfx942ConvDgradGeometry, PlansMatchTheAdapterOverTheCatalog)
{
    // conv_dgrad_aot_plan over every catalog instance of the problem's dtype: the same
    // instances serve, and their plans fold to the adapter's digest. On a digest mismatch,
    // make_conv_dgrad_aot_test_vectors.py's render() names each serving plan's digest.
    std::map<std::string, size_t> servedByVariant;
    std::set<std::string> catalogVariants;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto p = catalog::toProblem(row.problem);
        const auto& dtypeEntries = entries(row.dtype);
        ASSERT_EQ(std::string_view(row.served).size(), dtypeEntries.size());
        uint64_t folded = catalog::FNV_OFFSET;
        for(size_t i = 0; i < dtypeEntries.size(); ++i)
        {
            const auto& entry = dtypeEntries[i];
            if(!entry.isHelper())
            {
                catalogVariants.insert(entry.text("direct_variant"));
            }
            const bool expected = row.served[i] == '1';
            const auto plan
                = entry.isHelper() ? std::nullopt : dgrad::instancePlan(p, *entry.instance);
            EXPECT_EQ(plan.has_value(), expected)
                << entry.text("direct_variant") << " " << entry.text("instance_id") << " on "
                << describe(p) << " " << row.dtype;
            if(!plan)
            {
                continue;
            }
            catalog::foldPlanDigest(folded, dgrad::planDigest(*plan));
            ++servedByVariant[entry.text("direct_variant")];
        }
        EXPECT_EQ(folded, row.digest) << describe(p) << " " << row.dtype;
    }
    // Every selectable family of the catalog is reached.
    ASSERT_FALSE(catalogVariants.empty());
    for(const auto& variant : catalogVariants)
    {
        EXPECT_GT(servedByVariant[variant], 0U) << variant;
    }
}

TEST(TestGfx942ConvDgradGeometry, FullPlansMatchTheAdapter)
{
    ASSERT_FALSE(vectors::fullPlans().empty());
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

TEST(TestGfx942ConvDgradGeometry, TheStreamingDepthwiseLanesOwnCplChannels)
{
    const auto& entry = catalog::firstEntry(
        "fp16",
        [](const catalog::Entry& e) {
            return e.text("direct_variant") == "depthwise_dgrad" && e.integer("cpl") > 1;
        },
        dgrad::GFX942);
    const auto& direct = *entry.instance->direct;
    const auto cpl = direct.knobs.cpl;
    ASSERT_GT(cpl, 1);
    const auto f = direct.compiled.filterH;
    const auto s = direct.compiled.strideH;
    const auto pad = direct.compiled.padH;
    const auto channels = 64 * cpl;
    const auto plan = dgrad::instancePlan(
        problem(2, channels, channels, 28, 28, f, s, pad, channels), *entry.instance);
    ASSERT_TRUE(plan);
    const auto threads = direct.knobs.blockWaves * dgrad::GFX942.waveSize;
    EXPECT_EQ(plan->steps.back().block, threads);
    EXPECT_EQ(plan->steps.back().grid[1], dgrad::ceilDiv(channels, threads * cpl));
    // Channels that are not a multiple of cpl are refused.
    EXPECT_FALSE(dgrad::instancePlan(
        problem(2, channels + 1, channels + 1, 28, 28, f, s, pad, channels + 1), *entry.instance));
    // The same binary at cpl 1 serves them: only the lane ownership refuses.
    auto single = *entry.instance;
    single.direct->knobs.cpl = 1;
    EXPECT_TRUE(dgrad::instancePlan(
        problem(2, channels + 1, channels + 1, 28, 28, f, s, pad, channels + 1), single));
}

TEST(TestGfx942ConvDgradGeometry, TheMfmaForwardPassStagesWithinGfx942sLds)
{
    // The MFMA pipelines' forward pass on the transposed problem stages dY in LDS
    // (directConvSpecLaunch): about (block_q + KW - 1) * K/groups * 4 bytes, double-buffered.
    // gfx942 has 64 KiB where gfx950 has 160 KiB. The shipped c4k320 row, rebuilt as a
    // single-wave binary baking K/groups = 1024 (an instance the shipped list does not
    // carry), stages 72 KiB: gfx950's budget serves it, gfx942's refuses it, and only the
    // budget differs -- the same metadata read for either target (gfx950's KMD has no cpl).
    ASSERT_LT(dgrad::GFX942.ldsCapacityBytes, dgrad::GFX950.ldsCapacityBytes);
    const auto& c4k320 = catalog::firstEntry(
        "fp16",
        [](const catalog::Entry& e) {
            return e.text("direct_variant") == "grouped_dgrad_mfma" && e.integer("cpg") == 4
                   && e.integer("kpg") == 320;
        },
        dgrad::GFX942);
    auto fields = c4k320.fields;
    fields["kpg"] = int64_t{1024};
    fields["waves_k"] = int64_t{1};
    fields["waves_q"] = int64_t{1};
    fields["block_q"] = int64_t{16};
    fields["block_groups"] = int64_t{1};
    const auto onGfx942 = dgrad::parseAotInstance(catalog::FieldsMetadata{fields}, dgrad::GFX942);
    ASSERT_TRUE(onGfx942);
    fields.erase("cpl");
    const auto onGfx950 = dgrad::parseAotInstance(catalog::FieldsMetadata{fields}, dgrad::GFX950);
    ASSERT_TRUE(onGfx950);
    EXPECT_EQ(onGfx942->direct->ldsCapacityBytes, dgrad::GFX942.ldsCapacityBytes);
    EXPECT_EQ(onGfx950->direct->ldsCapacityBytes, dgrad::GFX950.ldsCapacityBytes);
    const auto wide = problem(1, 4, 1024, 32, 32, 3, 1, 1);
    EXPECT_TRUE(dgrad::instancePlan(wide, *onGfx950)) << describe(wide);
    EXPECT_FALSE(dgrad::instancePlan(wide, *onGfx942)) << describe(wide);
    auto widened = *onGfx942;
    widened.direct->ldsCapacityBytes = dgrad::GFX950.ldsCapacityBytes;
    EXPECT_TRUE(dgrad::instancePlan(wide, widened)) << describe(wide);
    // K/groups = 768 stages 54 KiB, within gfx942's budget.
    fields["kpg"] = int64_t{768};
    fields["cpl"] = int64_t{0};
    const auto narrower = dgrad::parseAotInstance(catalog::FieldsMetadata{fields}, dgrad::GFX942);
    ASSERT_TRUE(narrower);
    EXPECT_TRUE(dgrad::instancePlan(problem(1, 4, 768, 32, 32, 3, 1, 1), *narrower));
}

TEST(TestGfx942ConvDgradGeometry, OnlyTheMfmaPlansDependOnTheLdsBudget)
{
    // Over the golden problems, every shipped direct instance's plan with gfx950's budget:
    // a larger budget never refuses what gfx942's serves, and nothing but an MFMA pipeline
    // may change its answer. No shipped gfx942 MFMA binary stages more than 40 KiB (its
    // baked K/groups are at most 320, its runtime ones at most 16 * k_atoms = 512), so
    // none is expected to: the constructed instance above is what reaches the bound.
    for(const auto& row : vectors::servedMatrix())
    {
        const auto p = catalog::toProblem(row.problem);
        const auto& dtypeEntries = entries(row.dtype);
        for(size_t i = 0; i < dtypeEntries.size(); ++i)
        {
            const auto& entry = dtypeEntries[i];
            if(entry.isHelper() || entry.instance->implicit)
            {
                continue;
            }
            auto wider = *entry.instance;
            wider.direct->ldsCapacityBytes = dgrad::GFX950.ldsCapacityBytes;
            const bool servedWider = dgrad::instancePlan(p, wider).has_value();
            const bool served = row.served[i] == '1';
            EXPECT_TRUE(servedWider || !served)
                << entry.text("instance_id") << " on " << describe(p);
            EXPECT_TRUE(served == servedWider || isMfma(entry))
                << entry.text("direct_variant") << " " << entry.text("instance_id") << " on "
                << describe(p);
        }
    }
}

TEST(TestGfx942ConvDgradGeometry, TheKmdHoldsCplButNoneOfGfx1250sKnobs)
{
    const auto& implicit = catalog::firstEntry(
        "fp16",
        [](const catalog::Entry& e) { return e.integer("kernel_family") == 0; },
        dgrad::GFX942);
    const auto& depthwise = catalog::firstEntry(
        "fp16",
        [](const catalog::Entry& e) {
            return e.text("direct_variant") == "depthwise_dgrad" && e.integer("cpl") > 1;
        },
        dgrad::GFX942);
    const auto& fused = catalog::firstEntry(
        "fp16",
        [](const catalog::Entry& e) { return e.text("direct_variant") == "grouped_dgrad_fused"; },
        dgrad::GFX942);
    for(const auto* entry : {&implicit, &depthwise, &fused})
    {
        SCOPED_TRACE(entry->text("direct_variant"));
        ASSERT_TRUE(dgrad::parseAotInstance(catalog::FieldsMetadata{entry->fields}, dgrad::GFX942));
        // cpl is a field of every gfx942 kernel, 0 where the variant does not use it.
        ASSERT_EQ(entry->fields.count("cpl"), 1U);
        auto withoutCpl = entry->fields;
        withoutCpl.erase("cpl");
        EXPECT_FALSE(dgrad::parseAotInstance(catalog::FieldsMetadata{withoutCpl}, dgrad::GFX942));
        if(entry != &depthwise)
        {
            auto stated = entry->fields;
            stated["cpl"] = int64_t{2};
            EXPECT_FALSE(dgrad::parseAotInstance(catalog::FieldsMetadata{stated}, dgrad::GFX942));
        }
        // gfx1250's schedule and every knob past cpl are not gfx942's, even at 0.
        auto scheduled = entry->fields;
        scheduled["sched"] = std::string();
        EXPECT_FALSE(dgrad::parseAotInstance(catalog::FieldsMetadata{scheduled}, dgrad::GFX942));
        for(size_t i = 1; i < dgrad::GFX1250_DIRECT_KNOB_FIELDS.size(); ++i)
        {
            const std::string field(dgrad::GFX1250_DIRECT_KNOB_FIELDS[i].first);
            for(const int64_t value : {0, 1})
            {
                auto fields = entry->fields;
                fields[field] = value;
                EXPECT_FALSE(
                    dgrad::parseAotInstance(catalog::FieldsMetadata{fields}, dgrad::GFX942))
                    << field << "=" << value;
            }
        }
    }
    // A depthwise lane owns 1 or more channels; a negative count is refused.
    auto negative = depthwise.fields;
    negative["cpl"] = int64_t{-1};
    EXPECT_FALSE(dgrad::parseAotInstance(catalog::FieldsMetadata{negative}, dgrad::GFX942));
}

TEST(TestGfx942ConvDgradGeometry, UnknownOrIncompleteMetadataIsRefused)
{
    const auto& implicit = catalog::firstEntry(
        "fp16",
        [](const catalog::Entry& e) { return e.integer("kernel_family") == 0; },
        dgrad::GFX942);
    const auto& mfma = catalog::firstEntry(
        "fp16",
        [](const catalog::Entry& e) { return e.text("direct_variant") == "grouped_dgrad_mfma"; },
        dgrad::GFX942);
    const auto& helper = catalog::firstEntry(
        "fp16", [](const catalog::Entry& e) { return e.isHelper(); }, dgrad::GFX942);
    for(const auto* entry : {&implicit, &mfma, &helper})
    {
        ASSERT_TRUE(dgrad::parseAotInstance(catalog::FieldsMetadata{entry->fields}, dgrad::GFX942));
        for(const auto& [field, value] : std::vector<std::pair<std::string, catalog::FieldValue>>{
                {"direct_variant", std::string("dgrad_wmma")},
                {"direct_variant", std::string("bogus")},
                {"kernel_family", int64_t{3}},
                {"layout", std::string("NCHW")},
                {"wave_size", int64_t{32}},
                {"pipeline", std::string("wavelet")},
                {"halo_max_wo", int64_t{7}},
                {"block_w", int64_t{3}},
                {"dot2", int64_t{2}}})
        {
            auto fields = entry->fields;
            fields[field] = value;
            EXPECT_FALSE(dgrad::parseAotInstance(catalog::FieldsMetadata{fields}, dgrad::GFX942))
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
            EXPECT_FALSE(dgrad::parseAotInstance(catalog::FieldsMetadata{fields}, dgrad::GFX942))
                << entry->text("direct_variant") << " without " << field;
        }
    }
}

} // namespace
