// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvFwdTestCatalog.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx942ConvFwdGeometry.cpp
 * @brief hipkernel:Gfx942ConvFwdAot against the rocKE adapter over its whole shipped
 *        catalog: on every golden problem, the instances the engine's kernel matcher
 *        admits are the ones conv_fwd_aot_serves admits, the ones its candidate filter
 *        keeps are conv_fwd_aot_runtime_candidates, and the launch plan the geometry
 *        derives for each serving instance (grids, blocks, kernel arguments, batch split,
 *        workspace) folds to the adapter's digest (a few compared in full). The golden
 *        vectors are Gfx942ConvFwdTestVectors.hpp, which
 *        tools/make_conv_aot_test_vectors.py wrote from convolution_forward.py.
 *        Host-only.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
namespace conv = hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd;
namespace vectors = hip_kernel_provider::kernel_ingestor_engine::gfx942_fwd_test_vectors;
namespace catalog = hip_kernel_provider::kernel_ingestor_engine::conv_fwd_test_catalog;

using catalog::describe;

constexpr const char* GRAPH_MATCHER = "hipkernel.gfx942_conv_fwd_aot.graph_match";
constexpr const char* KERNEL_MATCHER = "hipkernel.gfx942_conv_fwd_aot.kernel_match";

/// How many mismatches a failing check names before it only counts them.
constexpr size_t REPORTED_MISMATCHES = 20;

const std::vector<catalog::Entry>& entries(const std::string& dtype)
{
    return catalog::gfx942Catalog(dtype);
}

/// The catalog of @p dtype as the engine loads it: each instance's kernel definition,
/// prepared by the kernel matcher's KernelPrepareFn as the state manager prepares it.
const std::vector<KernelDefinition>& kernels(const std::string& dtype)
{
    static const auto s_kernels = [] {
        hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols();
        const auto prepare = KernelPrepareRegistry::resolve(KERNEL_MATCHER);
        std::map<std::string, std::vector<KernelDefinition>> byDtype;
        for(const auto* name : {"fp16", "bf16"})
        {
            for(const auto& entry : catalog::gfx942Catalog(name))
            {
                auto kernel = catalog::kernelOf(entry);
                kernel.prepared = prepare(kernel);
                kernel.preparedBy = prepare;
                byDtype[name].push_back(std::move(kernel));
            }
        }
        return byDtype;
    }();
    return s_kernels.at(dtype);
}

/// What the engine decides for one served-matrix row on a wave64 gfx942: the graph match,
/// the kernel matcher on every instance of the row's dtype, then the candidate filter over
/// what it admitted, as KernelIngestorStateManager runs them. One flag per instance.
struct EngineDecision
{
    bool graphMatched = false;
    std::vector<bool> served;
    std::vector<bool> selected;
};

EngineDecision decide(const vectors::ServedRow& row)
{
    hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols();
    const auto graphMatch = GraphMatchRegistry::resolve(GRAPH_MATCHER);
    const auto matcher = KernelMatcherRegistry::resolve(KERNEL_MATCHER);
    const auto filter = CandidateFilterRegistry::resolve(KERNEL_MATCHER);
    const auto& dtypeKernels = kernels(row.dtype);

    EngineDecision decision;
    decision.served.assign(dtypeKernels.size(), false);
    decision.selected.assign(dtypeKernels.size(), false);
    const GraphFixture fixture(catalog::convGraph(row.problem, row.dtype),
                               gfx950Properties("gfx942"));
    const auto& context = fixture.context();
    const auto bound = graphMatch(context);
    decision.graphMatched = bound.has_value();
    if(!bound)
    {
        return decision;
    }
    std::vector<size_t> serving;
    std::vector<const KernelDefinition*> candidates;
    for(size_t i = 0; i < dtypeKernels.size(); ++i)
    {
        if(matcher(context, *bound, dtypeKernels[i]))
        {
            decision.served[i] = true;
            serving.push_back(i);
            candidates.push_back(&dtypeKernels[i]);
        }
    }
    const auto keep = filter(context, *bound, candidates);
    EXPECT_EQ(keep.size(), candidates.size());
    for(size_t j = 0; j < keep.size() && j < serving.size(); ++j)
    {
        decision.selected[serving[j]] = keep[j];
    }
    return decision;
}

/// decide() over every served-matrix row, once.
const std::vector<EngineDecision>& engineDecisions()
{
    static const auto s_decisions = [] {
        std::vector<EngineDecision> decisions;
        for(const auto& row : vectors::servedMatrix())
        {
            decisions.push_back(decide(row));
        }
        return decisions;
    }();
    return s_decisions;
}

/// Records mismatches, naming the first REPORTED_MISMATCHES.
class Mismatches
{
public:
    void add(const std::string& what)
    {
        if(_count++ < REPORTED_MISMATCHES)
        {
            ADD_FAILURE() << what;
        }
    }

    size_t count() const
    {
        return _count;
    }

private:
    size_t _count = 0;
};

/// Which catalog instances some served-matrix row serves. The vectors give every shipped
/// instance a problem it serves, so each one's admission and launch plan are compared, not
/// only its refusals; expectEveryInstanceServed() names any instance no row reached.
class ServedSomewhere
{
public:
    void mark(const std::string& dtype, size_t index)
    {
        auto& served = _served[dtype];
        served.resize(entries(dtype).size(), false);
        served.at(index) = true;
    }

    void expectEveryInstanceServed(const std::string& what) const
    {
        std::set<std::string> dtypes;
        for(const auto& instance : vectors::catalogInstances())
        {
            dtypes.insert(instance.dtype);
        }
        ASSERT_FALSE(dtypes.empty());
        Mismatches mismatches;
        for(const auto& dtype : dtypes)
        {
            const auto& dtypeEntries = entries(dtype);
            const auto found = _served.find(dtype);
            for(size_t i = 0; i < dtypeEntries.size(); ++i)
            {
                if(found == _served.end() || !found->second[i])
                {
                    mismatches.add(what + " serves " + dtype + " instance " + std::to_string(i)
                                   + " (" + dtypeEntries[i].kind() + ") on no row");
                }
            }
        }
        EXPECT_EQ(mismatches.count(), 0U);
    }

private:
    std::map<std::string, std::vector<bool>> _served;
};

std::string rowName(size_t index, const vectors::ServedRow& row)
{
    return "row " + std::to_string(index) + " " + describe(catalog::toProblem(row.problem)) + " "
           + row.dtype;
}

TEST(TestGfx942ConvFwdGeometry, EveryCatalogInstanceDecodes)
{
    // The adapter names the AOT problem block of each family as the geometry does.
    ASSERT_EQ(vectors::IMPLICIT_GEMM_KERNARGS.size(), conv::IMPLICIT_GEMM_KERNARG_NAMES.size());
    for(size_t i = 0; i < vectors::IMPLICIT_GEMM_KERNARGS.size(); ++i)
    {
        EXPECT_EQ(vectors::IMPLICIT_GEMM_KERNARGS[i], conv::IMPLICIT_GEMM_KERNARG_NAMES[i]);
    }
    ASSERT_EQ(vectors::DIRECT_KERNARGS.size(), conv::DIRECT_KERNARG_NAMES.size());
    for(size_t i = 0; i < vectors::DIRECT_KERNARGS.size(); ++i)
    {
        EXPECT_EQ(vectors::DIRECT_KERNARGS[i], conv::DIRECT_KERNARG_NAMES[i]);
    }
    for(const auto* dtype : {"fp16", "bf16"})
    {
        ASSERT_FALSE(entries(dtype).empty()) << dtype;
        std::set<std::string> instanceIds;
        for(const auto& entry : entries(dtype))
        {
            EXPECT_EQ(entry.dtype, dtype);
            EXPECT_EQ(entry.text("dtype"), dtype);
            EXPECT_EQ(entry.fields.size(), vectors::FIELDS.size());
            EXPECT_TRUE(entry.instance.has_value()) << entry.kind();
            EXPECT_EQ(entry.integer("wave_size"), 64) << entry.kind();
            EXPECT_TRUE(instanceIds.insert(entry.text("instance_id")).second) << entry.kind();
            // A recorded atom is one the geometry knows: gfx942's MFMA atoms.
            if(entry.text("atom") != "none")
            {
                EXPECT_TRUE(conv::nongroupedAtomTile(entry.text("atom"))) << entry.kind();
            }
        }
        EXPECT_EQ(kernels(dtype).size(), entries(dtype).size());
    }
}

TEST(TestGfx942ConvFwdGeometry, ServesExactlyWhatTheAdapterServesOverTheCatalog)
{
    // The engine's kernel matcher on a wave64 gfx942, over every catalog instance of the
    // problem's dtype, admits exactly what conv_fwd_aot_serves admits.
    Mismatches mismatches;
    size_t servedPairs = 0;
    size_t refusedPairs = 0;
    ServedSomewhere servedSomewhere;
    for(size_t r = 0; r < vectors::servedMatrix().size(); ++r)
    {
        const auto& row = vectors::servedMatrix()[r];
        const auto& dtypeEntries = entries(row.dtype);
        ASSERT_EQ(std::string_view(row.served).size(), (dtypeEntries.size() + 3) / 4);
        const auto& decision = engineDecisions()[r];
        for(size_t i = 0; i < dtypeEntries.size(); ++i)
        {
            const bool expected = catalog::bitAt(row.served, i);
            (expected ? servedPairs : refusedPairs) += 1;
            if(expected && decision.served[i])
            {
                servedSomewhere.mark(row.dtype, i);
            }
            if(decision.served[i] != expected)
            {
                mismatches.add(rowName(r, row) + ": the kernel matcher "
                               + (decision.served[i] ? "admits " : "refuses ")
                               + dtypeEntries[i].kind() + ", the adapter "
                               + (expected ? "serves it" : "does not")
                               + (decision.graphMatched ? "" : " (the graph match refused)"));
            }
        }
    }
    EXPECT_EQ(mismatches.count(), 0U);
    // Both verdicts occur, and every instance is admitted by both somewhere.
    EXPECT_GT(servedPairs, 0U);
    EXPECT_GT(refusedPairs, 0U);
    servedSomewhere.expectEveryInstanceServed("the kernel matcher and the adapter");
}

TEST(TestGfx942ConvFwdGeometry, SelectsExactlyWhatTheAdapterSelectsOverTheCatalog)
{
    // Of what the kernel matcher admits, the candidate filter keeps exactly
    // conv_fwd_aot_runtime_candidates (the widest vector widths, else every serving
    // instance; direct and group-merged instances always).
    Mismatches mismatches;
    size_t narrowed = 0;
    for(size_t r = 0; r < vectors::servedMatrix().size(); ++r)
    {
        const auto& row = vectors::servedMatrix()[r];
        const auto& dtypeEntries = entries(row.dtype);
        ASSERT_EQ(std::string_view(row.selected).size(), (dtypeEntries.size() + 3) / 4);
        const auto& decision = engineDecisions()[r];
        bool rowNarrowed = false;
        for(size_t i = 0; i < dtypeEntries.size(); ++i)
        {
            const bool expected = catalog::bitAt(row.selected, i);
            // The adapter selects among what it serves only.
            EXPECT_TRUE(!expected || catalog::bitAt(row.served, i)) << rowName(r, row);
            rowNarrowed = rowNarrowed || (catalog::bitAt(row.served, i) && !expected);
            if(decision.served[i] != catalog::bitAt(row.served, i))
            {
                continue; // ServesExactlyWhatTheAdapterServesOverTheCatalog names it.
            }
            if(decision.selected[i] != expected)
            {
                mismatches.add(rowName(r, row) + ": the candidate filter "
                               + (decision.selected[i] ? "keeps " : "drops ")
                               + dtypeEntries[i].kind() + ", the adapter "
                               + (expected ? "selects it" : "does not"));
            }
        }
        narrowed += rowNarrowed ? 1 : 0;
    }
    EXPECT_EQ(mismatches.count(), 0U);
    // The policy narrows some problems' serving sets and not others.
    EXPECT_GT(narrowed, 0U);
    EXPECT_LT(narrowed, vectors::servedMatrix().size());
}

TEST(TestGfx942ConvFwdGeometry, PlansMatchTheAdapterOverTheCatalog)
{
    // Every catalog instance of the problem's dtype through the geometry: the same
    // instances launch, and their plans fold to the adapter's digest. On a digest
    // mismatch, FullPlansMatchTheAdapter or make_conv_aot_test_vectors.py's plan_values
    // name the values.
    Mismatches mismatches;
    std::map<std::string, size_t> servedByKind;
    std::set<std::string> catalogKinds;
    ServedSomewhere launched;
    for(size_t r = 0; r < vectors::servedMatrix().size(); ++r)
    {
        const auto& row = vectors::servedMatrix()[r];
        const auto p = catalog::toProblem(row.problem);
        const auto& dtypeEntries = entries(row.dtype);
        uint64_t folded = catalog::FNV_OFFSET;
        for(size_t i = 0; i < dtypeEntries.size(); ++i)
        {
            const auto& entry = dtypeEntries[i];
            const auto kind = entry.kind().substr(0, entry.kind().rfind(' '));
            catalogKinds.insert(kind);
            const bool expected = catalog::bitAt(row.served, i);
            const auto plan
                = entry.instance ? catalog::geometryPlan(p, *entry.instance) : std::nullopt;
            if(plan.has_value() != expected)
            {
                mismatches.add(rowName(r, row) + ": the geometry "
                               + (plan ? "launches " : "refuses ") + entry.kind() + ", the adapter "
                               + (expected ? "serves it" : "does not"));
            }
            if(!plan)
            {
                continue;
            }
            catalog::fnvFold(folded, catalog::planDigest(*plan));
            ++servedByKind[kind];
            if(expected)
            {
                launched.mark(row.dtype, i);
            }
        }
        if(folded != row.digest)
        {
            mismatches.add(rowName(r, row) + ": the serving plans do not fold to the adapter's");
        }
    }
    EXPECT_EQ(mismatches.count(), 0U);
    // Every kind of the catalog (direct variant, atom, merge degree) is reached, and every
    // instance's plan is compared on some row.
    ASSERT_FALSE(catalogKinds.empty());
    for(const auto& kind : catalogKinds)
    {
        EXPECT_GT(servedByKind[kind], 0U) << kind;
    }
    launched.expectEveryInstanceServed("the geometry and the adapter");
}

TEST(TestGfx942ConvFwdGeometry, FullPlansMatchTheAdapter)
{
    ASSERT_FALSE(vectors::fullPlans().empty());
    bool batchSplit = false;
    for(const auto& full : vectors::fullPlans())
    {
        const auto& row = vectors::servedMatrix()[full.row];
        const auto p = catalog::toProblem(row.problem);
        const auto& entry = entries(row.dtype)[full.instance];
        SCOPED_TRACE(entry.kind() + " on " + describe(p));
        ASSERT_TRUE(entry.instance);
        const auto plan = catalog::geometryPlan(p, *entry.instance);
        ASSERT_TRUE(plan);
        EXPECT_EQ(*plan, full.values);
        batchSplit = batchSplit || full.values.front() > 0;
    }
    // A plan of several launches is among them.
    EXPECT_TRUE(batchSplit);
}

TEST(TestGfx942ConvFwdGeometry, NongroupedAtomsOfTheGfx942SweepTileAndFragment)
{
    // gfx942 has the 32x32x8 and 16x16x16 MFMA atoms only; the 16x16x32 and 32x32x16 ones
    // are gfx950's.
    EXPECT_EQ(conv::nongroupedAtomTile("16x16x16"), 16);
    EXPECT_EQ(conv::nongroupedAtomFragment("16x16x16"), 4);
    EXPECT_EQ(conv::nongroupedAtomTile("32x32x8"), 32);
    EXPECT_EQ(conv::nongroupedAtomFragment("32x32x8"), 4);
    // Every atom the catalog ships is one of them, and both are shipped.
    std::set<std::string> atoms;
    for(const auto* dtype : {"fp16", "bf16"})
    {
        for(const auto& entry : entries(dtype))
        {
            if(entry.text("atom") != "none")
            {
                atoms.insert(entry.text("atom"));
            }
        }
    }
    EXPECT_EQ(atoms, (std::set<std::string>{"16x16x16", "32x32x8"}));
}

/// One (instance, problem) case of the gfx942 catalog's tiled, col, grouped and
/// non-grouped direct instances and what gfx942_conv_fwd_aot_serves,
/// Gfx942ConvFwdAotSpec.grid() and .block() in convolution_forward.py give for it: whether
/// the instance serves the problem, and then its grid and block. The instances are the
/// catalog's (non-grouped with both gfx942 atoms). test_make_conv_aot_config.py recomputes
/// every row from the adapter.
struct DirectLaunchCase
{
    conv::DirectInstance instance;
    conv::Problem problem;
    bool served;
    unsigned int gridX;
    unsigned int gridY;
    unsigned int gridZ;
    unsigned int blockX;
};

using conv::DirectVariant;

// Instance: {filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w},
// variant, block_w, block_waves, block_h, block_q, block_groups, cpg, kpg, tile_h, tile_w,
// tile_k_out, ck, waves_m, waves_n, atom tile. Problem: Problem's field order. Keep one row
// per line.
// clang-format off
const std::vector<DirectLaunchCase>& directLaunchCasesGfx942()
{
    static const std::vector<DirectLaunchCase> s_cases{
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 8, 8, 33, 17, 3, 3, 1, 1, 1, 1, 1, 1, 8}, true, 5, 1, 34, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 48, 48, 13, 192, 3, 3, 1, 1, 1, 1, 1, 1, 48}, true, 48, 1, 14, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 63, 63, 33, 192, 3, 3, 1, 1, 1, 1, 1, 1, 63}, true, 48, 1, 51, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 16, 16, 3, 56, 3, 3, 1, 1, 1, 1, 1, 1, 16}, true, 14, 1, 2, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 64, 128, 17, 33, 3, 3, 1, 1, 1, 1, 1, 1, 64}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 65, 130, 8, 32, 3, 3, 1, 1, 1, 1, 1, 1, 65}, false, 0, 0, 0, 0},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 48, 48, 12, 96, 3, 3, 2, 2, 1, 1, 1, 1, 48}, true, 12, 1, 2, 64},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 16, 16, 5, 12, 3, 3, 2, 2, 1, 1, 1, 1, 16}, true, 2, 1, 1, 64},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 65, 65, 8, 64, 3, 3, 2, 2, 1, 1, 1, 1, 65}, true, 8, 2, 1, 64},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 1, 1, 2, 8, 3, 3, 2, 2, 1, 1, 1, 1, 1}, true, 1, 1, 2, 64},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 300, 600, 12, 12, 3, 3, 2, 2, 1, 1, 1, 1, 300}, false, 0, 0, 0, 0},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 64, 64, 7, 17, 5, 5, 2, 2, 1, 1, 1, 1, 64}, false, 0, 0, 0, 0},
        {{{7, 7, 1, 1, 3, 3, 1, 1}, DirectVariant::TILED, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 2, 2, 5, 28, 7, 7, 1, 1, 3, 3, 1, 1, 2}, true, 7, 1, 3, 64},
        {{{7, 7, 1, 1, 3, 3, 1, 1}, DirectVariant::TILED, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 96, 96, 33, 17, 7, 7, 1, 1, 3, 3, 1, 1, 96}, true, 5, 2, 34, 64},
        {{{7, 7, 1, 1, 3, 3, 1, 1}, DirectVariant::TILED, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 128, 128, 12, 33, 7, 7, 1, 1, 3, 3, 1, 1, 128}, true, 9, 2, 18, 64},
        {{{7, 7, 1, 1, 3, 3, 1, 1}, DirectVariant::TILED, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 5, 5, 8, 12, 7, 7, 1, 1, 3, 3, 1, 1, 5}, true, 3, 1, 8, 64},
        {{{7, 7, 1, 1, 3, 3, 1, 1}, DirectVariant::TILED, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 2, 2, 5, 48, 9, 9, 1, 1, 3, 3, 1, 1, 2}, false, 0, 0, 0, 0},
        {{{7, 7, 1, 1, 3, 3, 1, 1}, DirectVariant::TILED, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 32, 64, 2, 12, 7, 7, 1, 1, 3, 3, 1, 1, 32}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::COL, 1, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 300, 300, 8, 56, 3, 3, 1, 1, 1, 1, 1, 1, 300}, true, 56, 3, 2, 128},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::COL, 1, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 65, 65, 13, 30, 3, 3, 1, 1, 1, 1, 1, 1, 65}, true, 30, 1, 2, 128},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::COL, 1, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 63, 63, 28, 16, 3, 3, 1, 1, 1, 1, 1, 1, 63}, true, 16, 1, 4, 128},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::COL, 1, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 16, 16, 13, 96, 3, 3, 1, 1, 1, 1, 1, 1, 16}, true, 96, 1, 6, 128},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::COL, 1, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 1, 1, 31, 48, 3, 3, 1, 1, 1, 1, 2, 2, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::COL, 1, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 5, 10, 28, 5, 3, 3, 1, 1, 1, 1, 1, 1, 5}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 8, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {2, 192, 192, 17, 7, 3, 3, 1, 1, 1, 1, 1, 1, 48}, true, 1, 6, 4, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 8, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {1, 256, 256, 31, 16, 3, 3, 1, 1, 1, 1, 1, 1, 64}, true, 1, 8, 2, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 8, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {2, 192, 192, 12, 16, 3, 3, 1, 1, 1, 1, 1, 1, 48}, true, 1, 6, 2, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 8, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {1, 32, 32, 3, 7, 3, 3, 1, 1, 1, 1, 1, 1, 8}, true, 1, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 8, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {2, 64, 64, 33, 64, 3, 3, 1, 1, 1, 1, 1, 1, 2}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 8, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {1, 16, 16, 17, 64, 3, 3, 1, 1, 2, 2, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {3, 128, 128, 17, 28, 3, 3, 1, 1, 1, 1, 1, 1, 4}, true, 2, 4, 6, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {1, 32, 32, 28, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 4, 1, 2, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {3, 128, 128, 12, 48, 3, 3, 1, 1, 1, 1, 1, 1, 4}, true, 3, 4, 3, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {1, 512, 512, 56, 16, 3, 3, 1, 1, 1, 1, 1, 1, 16}, true, 1, 16, 4, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {2, 128, 128, 13, 12, 3, 3, 1, 1, 2, 2, 1, 1, 4}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {3, 8, 8, 8, 56, 3, 3, 1, 2, 1, 1, 1, 1, 4}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 2, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {3, 64, 64, 7, 96, 3, 3, 1, 1, 1, 1, 1, 1, 2}, true, 3, 1, 3, 128},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 2, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {1, 2048, 2048, 3, 5, 3, 3, 1, 1, 1, 1, 1, 1, 64}, true, 1, 32, 1, 128},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 2, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {2, 128, 128, 5, 192, 3, 3, 1, 1, 1, 1, 1, 1, 4}, true, 6, 2, 2, 128},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 2, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {3, 256, 256, 13, 192, 3, 3, 1, 1, 1, 1, 1, 1, 8}, true, 6, 4, 3, 128},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 2, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {2, 64, 64, 56, 64, 3, 3, 1, 1, 1, 1, 2, 2, 32}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 2, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {3, 64, 64, 8, 16, 5, 5, 1, 1, 1, 1, 1, 1, 8}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 4, 16}, {1, 32, 16, 31, 192, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 24, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 4, 16}, {3, 96, 16, 31, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 12, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 4, 16}, {2, 128, 96, 7, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 24, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 4, 16}, {3, 96, 64, 31, 12, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 6, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 4, 16}, {1, 256, 32, 28, 28, 5, 5, 1, 1, 1, 1, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 4, 16}, {2, 192, 320, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 2}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 48, 128, 16, 2, 4, 16}, {1, 64, 64, 31, 33, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 4, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 48, 128, 16, 2, 4, 16}, {1, 32, 96, 3, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 2, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 48, 128, 16, 2, 4, 16}, {3, 64, 16, 31, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 24, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 48, 128, 16, 2, 4, 16}, {1, 48, 96, 12, 48, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 2, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 48, 128, 16, 2, 4, 16}, {2, 96, 16, 28, 100, 3, 3, 1, 1, 1, 1, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 48, 128, 16, 2, 4, 16}, {1, 64, 160, 2, 17, 3, 3, 1, 1, 1, 1, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 64, 16, 2, 4, 32}, {1, 128, 128, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 2, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 64, 16, 2, 4, 32}, {3, 48, 32, 17, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 6, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 64, 16, 2, 4, 32}, {1, 16, 64, 3, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 1, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 64, 16, 2, 4, 32}, {3, 96, 64, 12, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 3, 1, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 64, 16, 2, 4, 32}, {3, 192, 128, 56, 12, 5, 5, 1, 1, 1, 1, 1, 1, 2}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 64, 16, 2, 4, 32}, {2, 64, 96, 33, 32, 3, 3, 1, 1, 1, 1, 2, 2, 2}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {3, 48, 128, 28, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 48, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {2, 96, 128, 12, 100, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 64, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {3, 64, 128, 2, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 12, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {3, 256, 96, 17, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 27, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {2, 16, 32, 17, 12, 3, 3, 1, 2, 1, 1, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {3, 96, 320, 56, 192, 5, 5, 1, 1, 1, 1, 1, 1, 2}, false, 0, 0, 0, 0},
    };
    return s_cases;
}
// clang-format on

TEST(TestGfx942ConvFwdGeometry, AotDirectVariantsLaunchWhereAndHowTheAdapterDoes)
{
    size_t served = 0;
    std::set<int64_t> atomTiles;
    for(size_t i = 0; i < directLaunchCasesGfx942().size(); ++i)
    {
        SCOPED_TRACE(i);
        const auto& entry = directLaunchCasesGfx942()[i];
        const auto geometry = conv::deriveGeometry(entry.problem);
        ASSERT_TRUE(geometry);
        const auto launch = conv::directInstanceLaunch(entry.problem, *geometry, entry.instance);
        ASSERT_EQ(launch.has_value(), entry.served);
        if(launch)
        {
            ++served;
            EXPECT_EQ(std::tie(launch->gridX, launch->gridY, launch->gridZ, launch->blockX),
                      std::tie(entry.gridX, entry.gridY, entry.gridZ, entry.blockX));
        }
        if(entry.instance.variant == DirectVariant::NONGROUPED)
        {
            atomTiles.insert(entry.instance.atomTile);
        }
    }
    // Both verdicts, and the non-grouped rows of both gfx942 atoms.
    EXPECT_GT(served, size_t{0});
    EXPECT_LT(served, directLaunchCasesGfx942().size());
    EXPECT_EQ(atomTiles, (std::set<int64_t>{16, 32}));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
