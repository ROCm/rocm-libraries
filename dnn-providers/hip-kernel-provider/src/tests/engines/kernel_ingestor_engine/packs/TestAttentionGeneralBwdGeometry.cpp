// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionGeneralBwdTestSupport.hpp"

/**
 * @file TestAttentionGeneralBwdGeometry.cpp
 * @brief hipkernel:AttentionGeneralBwd's plan at its edges: catalog selection, grids,
 *        runtime plan values, the shared workspace and the f32 kernargs.
 *
 * The replay corpus pins whole plans against rocKE. These cases name the rules those plans
 * follow, at the inputs where an off-by-one or a wrong rounding would show: a single query
 * row, a KV length that is not a whole number of tiles, the head-packing threshold, the
 * mirror pairing's CU guard and the workspace alignment.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace general_bwd_test;
using general_bwd::Launch;
using general_bwd::Plan;
using general_bwd::PlanInput;
using hipdnn_plugin_sdk::ingestor::BoundTokens;

/// A dense BHSD request of every operand, the default class (no CU count, so no tuned key).
PlanInput denseInput(int64_t b, int64_t hQ, int64_t hKv, int64_t sQ, int64_t sKv, int64_t d)
{
    PlanInput in;
    in.b = b;
    in.hQ = hQ;
    in.hK = hKv;
    in.hV = hKv;
    in.sQ = sQ;
    in.sKv = sKv;
    in.headSize = d;
    in.dtype = generated::Dtype::FP16;
    for(std::size_t i = 0; i < general_bwd::DATA_OPERANDS; ++i)
    {
        const auto operand = general_bwd::DATA[i];
        const int64_t h = general_bwd::detail::headsOf(in, operand);
        const int64_t s = general_bwd::detail::seqOf(in, operand);
        in.strides[i] = {h * s * d, s * d, d, 1};
    }
    in.lseDims = {b, hQ, sQ, 1};
    in.lseStrides = {hQ * sQ, sQ, 1, 1};
    return in;
}

/// The input @p representative stands for, as the catalog generator built it.
PlanInput representativeInput(const generated::Representative& representative)
{
    PlanInput in = denseInput(representative.b,
                              representative.hQ,
                              representative.hK,
                              representative.sQ,
                              representative.sKv,
                              representative.headSize);
    in.hV = representative.hV;
    in.dtype = representative.dtype;
    in.thd = representative.seqMode == generated::SeqMode::THD;
    in.left = representative.left;
    in.right = representative.right;
    in.bottomRight = representative.bottomRight;
    in.tensorAlignment = representative.alignment;
    if(representative.numCus != 0)
    {
        in.numCus = representative.numCus;
    }
    const int64_t d = representative.headSize;
    for(std::size_t i = 0; i < general_bwd::DATA_OPERANDS; ++i)
    {
        const auto operand = general_bwd::DATA[i];
        const int64_t h = general_bwd::detail::headsOf(in, operand);
        const int64_t s = general_bwd::detail::seqOf(in, operand);
        in.strides[i] = in.thd ? std::array<int64_t, 4>{0, d, h * d, 1}
                               : std::array<int64_t, 4>{h * s * d, s * d, d, 1};
    }
    in.qMult = representative.hQ * d;
    in.kvMult = representative.hK * d;
    if(in.thd)
    {
        in.lseStrides = {representative.sQ * representative.hQ, 1, representative.hQ, 1};
    }
    return in;
}

Plan planOf(std::variant<Plan, general_bwd::Decline> result)
{
    if(const auto* decline = std::get_if<general_bwd::Decline>(&result))
    {
        throw std::runtime_error("declined [" + std::string(general_bwd::reasonName(decline->code))
                                 + ": " + decline->detail + "]");
    }
    return std::get<Plan>(std::move(result));
}

const general_bwd::LaunchPlan& launchOf(const Plan& plan, Launch launch)
{
    for(const auto& candidate : plan.launches)
    {
        if(candidate.launch == launch)
        {
            return candidate;
        }
    }
    throw std::runtime_error("the plan has no '" + std::string(general_bwd::launchName(launch))
                             + "' launch");
}

const general_bwd::Kernarg& kernargOf(const general_bwd::LaunchPlan& launch, std::string_view name)
{
    for(const auto& kernarg : launch.kernargs)
    {
        if(kernarg.name == name)
        {
            return kernarg;
        }
    }
    throw std::runtime_error("no kernarg " + std::string(name));
}

int64_t ceilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

} // namespace

TEST(TestAttentionGeneralBwdGeometry, EveryRepresentativeResolvesToItsOwnCatalogRow)
{
    std::size_t resolved = 0;
    for(const auto& representative : generated::REPRESENTATIVES)
    {
        if(!representative.reproduces)
        {
            continue;
        }
        const auto& row = generated::CATALOG.at(representative.row);
        const auto result = general_bwd::resolvePlan(representativeInput(representative),
                                                     general_bwd::archTraitsOf(row.arch));
        const auto* plan = std::get_if<Plan>(&result);
        ASSERT_NE(plan, nullptr) << "row " << representative.row << " (" << row.mainKernel
                                 << ") declined: " << std::get<general_bwd::Decline>(result).detail;
        EXPECT_EQ(plan->row, &row)
            << "row " << representative.row << " resolved to " << plan->row->mainKernel;
        EXPECT_EQ(general_bwd::proveWorkspace(plan->workspace), "") << "row " << representative.row;
        ++resolved;
    }
    EXPECT_EQ(resolved, generated::CATALOG.size()) << "a catalog row no representative reaches";
}

TEST(TestAttentionGeneralBwdGeometry, GridsCoverASingleQueryRowAndAPartialKvTile)
{
    for(const auto& traits : {general_bwd::GFX942_TRAITS, general_bwd::GFX950_TRAITS})
    {
        const auto in = denseInput(3, 4, 4, 1, 300, 64);
        const auto& plan = planOf(general_bwd::resolvePlan(in, traits));
        const auto& row = *plan.row;
        ASSERT_NE(300 % row.mainBlockN, 0) << "the case needs a partial last KV tile";

        const auto& prep = launchOf(plan, Launch::PREP);
        EXPECT_EQ(prep.grid[0], static_cast<uint32_t>(ceilDiv(300, row.prepRowsPerCta)));
        EXPECT_EQ(prep.grid[1], 4u);
        EXPECT_EQ(prep.grid[2], 3u);
        EXPECT_EQ(prep.block, static_cast<uint32_t>(row.prepThreads));

        const auto& main = launchOf(plan, Launch::MAIN);
        const int64_t tiles = ceilDiv(300, row.mainBlockN);
        EXPECT_EQ(main.grid[0], static_cast<uint32_t>(tiles));
        EXPECT_EQ(main.grid[1], 4u);
        EXPECT_EQ(main.grid[2], 3u);
        EXPECT_EQ(main.block, static_cast<uint32_t>(row.mainThreads));
        EXPECT_EQ(kernargOf(main, "n_kv_tiles").intValue(), tiles);
        EXPECT_EQ(kernargOf(main, "S_q_max").intValue(), 1);
        EXPECT_EQ(kernargOf(main, "S_kv_max").intValue(), 300);

        // One query row still needs one convert CTA.
        const auto& dq = launchOf(plan, Launch::DQ);
        EXPECT_EQ(dq.grid[0], 1u);
        EXPECT_EQ(dq.grid[1], 4u);
        EXPECT_EQ(dq.grid[2], 3u);
        EXPECT_EQ(plan.launches.size(), 3u) << "direct dK/dV launches no converts";
    }
}

TEST(TestAttentionGeneralBwdGeometry, AtomicDkDvAddsTheConvertsAndTheirPrepRows)
{
    auto in = denseInput(2, 8, 4, 128, 128, 128);
    in.hV = 2;
    const auto& plan = planOf(general_bwd::resolvePlan(in, general_bwd::GFX942_TRAITS));
    EXPECT_EQ(plan.row->dkv, generated::DkvMode::ATOMIC);
    ASSERT_EQ(plan.launches.size(), 5u);
    EXPECT_EQ(launchOf(plan, Launch::PREP).grid[1], 8u + 4u + 2u);
    EXPECT_EQ(launchOf(plan, Launch::CONVERT_DK).grid[1], 4u);
    EXPECT_EQ(launchOf(plan, Launch::CONVERT_DV).grid[1], 2u);
    EXPECT_EQ(kernargOf(launchOf(plan, Launch::CONVERT_DK), "SRC").role, "ws.dk_acc");
    EXPECT_EQ(kernargOf(launchOf(plan, Launch::CONVERT_DV), "SRC").role, "ws.dv_acc");
    EXPECT_EQ(plan.workspace.regions.count("dk_acc"), 1u);
    EXPECT_EQ(plan.workspace.regions.count("dv_acc"), 1u);
}

TEST(TestAttentionGeneralBwdGeometry, PacksQueryHeadsOnlyWhileTheGroupFitsTheQueryTile)
{
    // h_q / h_kv = 4: one query row of four heads packs into a q tile.
    const auto& packed = planOf(
        general_bwd::resolvePlan(denseInput(2, 8, 2, 1, 256, 128), general_bwd::GFX942_TRAITS));
    EXPECT_TRUE(packed.row->headPack);
    EXPECT_EQ(packed.runtime.packHeads, 4);
    EXPECT_EQ(kernargOf(launchOf(packed, Launch::MAIN), "pack_heads").intValue(), 4);

    // G * s_q one past block_m does not.
    const int64_t blockM = packed.row->mainBlockM;
    const auto& unpacked = planOf(general_bwd::resolvePlan(
        denseInput(2, 8, 2, blockM / 4 + 1, 256, 128), general_bwd::GFX942_TRAITS));
    EXPECT_FALSE(unpacked.row->headPack);
    EXPECT_EQ(unpacked.runtime.packHeads, 0);

    // MHA has no group to pack.
    const auto& mha = planOf(
        general_bwd::resolvePlan(denseInput(2, 8, 8, 1, 256, 128), general_bwd::GFX942_TRAITS));
    EXPECT_FALSE(mha.row->headPack);
}

TEST(TestAttentionGeneralBwdGeometry, MirrorPairingHalvesTheGridAndNeverPairsAOneWaveGrid)
{
    std::size_t mirrored = 0;
    for(const auto& representative : generated::REPRESENTATIVES)
    {
        const auto& row = generated::CATALOG.at(representative.row);
        if(!representative.reproduces || !row.lbMirror)
        {
            continue;
        }
        auto in = representativeInput(representative);
        const auto& plan
            = planOf(general_bwd::resolvePlan(in, general_bwd::archTraitsOf(row.arch)));
        EXPECT_EQ(plan.runtime.pair, 1) << row.mainKernel;
        const auto& main = launchOf(plan, Launch::MAIN);
        EXPECT_EQ(main.grid[0], static_cast<uint32_t>(ceilDiv(plan.runtime.nKvTiles, 2)))
            << row.mainKernel;
        EXPECT_EQ(kernargOf(main, "pair").intValue(), 1);
        EXPECT_EQ(kernargOf(main, "lb_order").intValue(),
                  static_cast<int64_t>(generated::LbOrder::MIRROR));

        // Enough CUs that the unpaired grid fits one wave: never paired.
        in.numCus = int64_t{1} << 30;
        const auto wide = general_bwd::resolvePlan(in, general_bwd::archTraitsOf(row.arch));
        if(const auto* widePlan = std::get_if<Plan>(&wide))
        {
            EXPECT_EQ(widePlan->runtime.pair, 0) << row.mainKernel;
        }
        else
        {
            EXPECT_EQ(std::get<general_bwd::Decline>(wide).code,
                      generated::Reason::INSTANCE_UNAVAILABLE);
        }
        ++mirrored;
    }
    EXPECT_GT(mirrored, 0u) << "no catalog row pairs, so the guard is untested";
}

TEST(TestAttentionGeneralBwdGeometry, WorkspaceRegionsAre256ByteAlignedAndPresentOnlyWhenUsed)
{
    const auto& mha = planOf(
        general_bwd::resolvePlan(denseInput(2, 8, 8, 100, 130, 64), general_bwd::GFX950_TRAITS));
    const auto& ws = mha.workspace;
    EXPECT_EQ(ws.bytes % generated::WORKSPACE_ALIGN, 0);
    EXPECT_EQ(ws.scratchOffset, ws.bytes);
    EXPECT_EQ(ws.regions.count("lse2"), 1u);
    EXPECT_EQ(ws.regions.count("dsum"), 1u);
    EXPECT_EQ(ws.regions.count("dq_acc"), 1u);
    EXPECT_EQ(ws.regions.count("dk_acc"), 0u);
    EXPECT_EQ(ws.regions.count("worklist"), 0u);
    for(const auto& [name, region] : ws.regions)
    {
        EXPECT_EQ(region.offset % generated::WORKSPACE_ALIGN, 0) << name;
        EXPECT_LE(region.offset + region.bytes, ws.bytes) << name;
    }
    // fp32 rows of every (batch, query row, head), dq_acc also over the head dim.
    EXPECT_EQ(ws.regions.at("lse2").bytes, 4 * 2 * 100 * 8);
    EXPECT_EQ(ws.regions.at("dq_acc").bytes, 4 * 2 * 100 * 8 * 64);
    EXPECT_EQ(general_bwd::proveWorkspace(ws), "");
}

TEST(TestAttentionGeneralBwdGeometry, ProveWorkspaceRefusesEveryLayoutThatDoesNotFit)
{
    using general_bwd::Region;
    using general_bwd::WorkspaceLayout;
    EXPECT_EQ(general_bwd::proveWorkspace(WorkspaceLayout{
                  512, 512, {{"lse2", Region{0, 100}}, {"dsum", Region{256, 256}}}}),
              "");
    EXPECT_NE(general_bwd::proveWorkspace(WorkspaceLayout{
                  512, 512, {{"lse2", Region{0, 300}}, {"dsum", Region{256, 256}}}}),
              "")
        << "overlapping regions";
    EXPECT_NE(general_bwd::proveWorkspace(WorkspaceLayout{
                  768, 768, {{"lse2", Region{0, 100}}, {"dsum", Region{300, 256}}}}),
              "")
        << "a region off the 256-byte grid";
    EXPECT_NE(general_bwd::proveWorkspace(WorkspaceLayout{
                  512, 512, {{"lse2", Region{0, 100}}, {"dsum", Region{256, 512}}}}),
              "")
        << "a region past the end";
    EXPECT_NE(general_bwd::proveWorkspace(WorkspaceLayout{512, 0, {{"lse2", Region{0, 100}}}}), "")
        << "scratch inside the shared region";
    EXPECT_NE(general_bwd::proveWorkspace(WorkspaceLayout{500, 500, {{"lse2", Region{0, 100}}}}),
              "")
        << "a size off the 256-byte grid";
}

TEST(TestAttentionGeneralBwdGeometry, ScaleKernargsAreTheF32OfTheDoubleProduct)
{
    // Default scale 1/sqrt(128): scale_log2 = scale * log2(e), and fold_ds puts the scale in
    // ds_mult. Bits as Python's struct.pack('<f').
    const auto& defaults = planOf(
        general_bwd::resolvePlan(denseInput(2, 8, 8, 128, 128, 128), general_bwd::GFX942_TRAITS));
    const auto& main = launchOf(defaults, Launch::MAIN);
    EXPECT_EQ(kernargOf(main, "scale_log2").bits, 0x3e0293eeu);
    EXPECT_EQ(kernargOf(main, "ds_mult").bits, 0x3db504f3u);
    EXPECT_EQ(kernargOf(main, "dq_mult").bits, 0x3f800000u);

    auto in = denseInput(2, 8, 8, 128, 128, 64);
    in.scale = 0.5;
    const auto& explicitScale = planOf(general_bwd::resolvePlan(in, general_bwd::GFX942_TRAITS));
    const auto& scaled = launchOf(explicitScale, Launch::MAIN);
    EXPECT_EQ(kernargOf(scaled, "scale_log2").bits, 0x3f38aa3bu);
    EXPECT_EQ(kernargOf(scaled, "ds_mult").bits, 0x3f000000u);
}

TEST(TestAttentionGeneralBwdGeometry, TuningKeyStringsNameTheirTableEntries)
{
    for(const auto& entry : generated::TUNED_KEYS)
    {
        EXPECT_EQ(entry.key.substr(0, entry.arch.size()), entry.arch) << entry.key;
        EXPECT_GE(entry.id, 1) << entry.key;
        EXPECT_EQ(entry.xcdChunk, 0) << entry.key << ": no shipped key remaps XCDs";
    }
    for(const auto& row : generated::CATALOG)
    {
        EXPECT_NO_THROW(static_cast<void>(tuningKeyText(row))) << row.mainKernel;
    }
}

// ---------------------------------------------------------------------------
// The handlers address the workspace by the layout they recompute
// ---------------------------------------------------------------------------

namespace
{

/// What prepare() throws for @p bound on the dense MHA graph, or "" when it throws nothing.
std::string prepareFailure(std::string_view symbol, const BoundTokens& bound)
{
    registerNativeIngestorSymbols();
    const auto* handler
        = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(std::string(symbol));
    EXPECT_NE(handler, nullptr);
    if(handler == nullptr)
    {
        return {};
    }
    const GraphFixture fixture(bwdGraphJson({}), "gfx942", fullComputeUnits("gfx942"));
    const auto& plan = planOf(general_bwd::evaluate(fixture.context()));
    const auto kernel
        = symbol == MAIN_DISPATCH_SYMBOL ? mainKernelOf(*plan.row) : prepKernelOf(*plan.row);
    try
    {
        handler->prepare(fixture.context(), bound, kernel);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        return error.what();
    }
    catch(const std::exception& error)
    {
        return std::string("unexpected exception: ") + error.what();
    }
    return {};
}

BoundTokens denseMhaTokens()
{
    const GraphFixture fixture(bwdGraphJson({}), "gfx942", fullComputeUnits("gfx942"));
    return general_bwd::tokensFor(planOf(general_bwd::evaluate(fixture.context())));
}

} // namespace

TEST(TestAttentionGeneralBwdGeometry, PrepareRefusesWorkspaceTokensThatDisagreeWithThePlan)
{
    for(const auto symbol : {PREP_DISPATCH_SYMBOL, MAIN_DISPATCH_SYMBOL})
    {
        auto wrongSize = denseMhaTokens();
        wrongSize["ws.bytes"] = std::get<int64_t>(wrongSize["ws.bytes"]) + 256;
        EXPECT_NE(prepareFailure(symbol, wrongSize).find("ws.bytes"), std::string::npos) << symbol;

        auto missing = denseMhaTokens();
        missing.erase("ws.lse2.offset");
        EXPECT_NE(prepareFailure(symbol, missing).find("ws.lse2.offset"), std::string::npos)
            << symbol;

        auto extra = denseMhaTokens();
        extra["ws.dk_acc.offset"] = int64_t{0};
        EXPECT_NE(prepareFailure(symbol, extra).find("ws.dk_acc.offset"), std::string::npos)
            << symbol;

        auto otherKernel = denseMhaTokens();
        otherKernel["attention_general_bwd.main.head_pack"] = int64_t{1};
        EXPECT_NE(prepareFailure(symbol, otherKernel).find("head_pack"), std::string::npos)
            << symbol;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
