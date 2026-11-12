// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionGeneralBwdTestSupport.hpp"

/**
 * @file TestAttentionGeneralBwdMatch.cpp
 * @brief hipkernel:AttentionGeneralBwd's graph match, kernel match and pack criteria on
 *        hand-built graphs.
 *
 * One graph per reason code the predicate can reach from a hipDNN graph, each a single edit
 * of a graph the engine accepts, so the code the edit provokes is the one rocKE's
 * `attn_bwd_support` returns for it. Every decline is logged once at INFO under the engine's
 * name; a graph that is not one SdpaBackward node is declined silently. The corpus replay
 * covers the check order at scale; these cases name each code.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace general_bwd_test;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

constexpr std::string_view GFX942 = "gfx942";
constexpr std::string_view GFX950 = "gfx950";

std::optional<BoundTokens> graphMatch(const json& graph, std::string_view arch = GFX942)
{
    registerNativeIngestorSymbols();
    const auto matcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));
    const GraphFixture fixture(graph, arch, fullComputeUnits(arch));
    return matcher(fixture.context());
}

struct LoggedDecline
{
    std::string code;
    std::string detail;
    std::size_t lines = 0;
};

/// What the engine logged at INFO while declining @p graph.
LoggedDecline loggedDecline(const json& graph, std::string_view arch = GFX942)
{
    const std::string marker
        = std::string(ATTENTION_GENERAL_BWD_ENGINE_NAME) + " declined the graph [";
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_FALSE(graphMatch(graph, arch).has_value());
    LoggedDecline decline;
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        if(log.severity != HIPDNN_SEV_INFO || at == std::string::npos)
        {
            continue;
        }
        ++decline.lines;
        const auto start = at + marker.size();
        const auto colon = log.message.find(": ", start);
        const auto end = log.message.rfind(']');
        if(colon != std::string::npos && end != std::string::npos && colon < end)
        {
            decline.code = log.message.substr(start, colon - start);
            decline.detail = log.message.substr(colon + 2, end - colon - 2);
        }
    }
    return decline;
}

json graphWith(const GraphSpec& spec, const std::function<void(json&)>& edit = {})
{
    auto graph = bwdGraphJson(spec);
    if(edit)
    {
        edit(graph);
    }
    return graph;
}

json& parameters(json& graph)
{
    return graph["nodes"][0]["parameters"];
}

json& inputs(json& graph)
{
    return graph["nodes"][0]["inputs"];
}

void addTensor(json& graph, json tensor)
{
    graph["tensors"].push_back(std::move(tensor));
}

struct DeclineCase
{
    const char* name;
    std::function<json()> graph;
    const char* code;
};

std::vector<DeclineCase> declineCases()
{
    GraphSpec thd;
    thd.thd = true;
    return {
        {"DTYPE",
         [] {
             GraphSpec spec;
             spec.dtype = "float";
             return graphWith(spec);
         },
         "DTYPE"},
        {"OUT_DTYPE",
         [] { return graphWith({}, [](json& g) { tensorOf(g, DQ_UID)["data_type"] = "uint8"; }); },
         "OUT_DTYPE"},
        {"FP8",
         [] {
             return graphWith({},
                              [](json& g) { tensorOf(g, K_UID)["data_type"] = "fp8_e5m2_fnuz"; });
         },
         "FP8"},
        {"LSE_FORMAT_HalfStats",
         [] {
             return graphWith({}, [](json& g) { tensorOf(g, STATS_UID)["data_type"] = "half"; });
         },
         "LSE_FORMAT"},
        {"LSE_FORMAT_DenseStatsUnderThd",
         [thd] {
             return graphWith(thd, [](json& g) {
                 auto& stats = tensorOf(g, STATS_UID);
                 stats.erase("ragged_offset_tensor_uid");
                 stats.erase("ragged_offset_multiplier");
                 stats["strides"] = {8 * 128, 128, 1, 1};
             });
         },
         "LSE_FORMAT"},
        {"HEAD_DIM",
         [] {
             GraphSpec spec;
             spec.dQk = 96;
             spec.dV = 96;
             return graphWith(spec);
         },
         "HEAD_DIM"},
        {"HEAD_DIM_256_BWD",
         [] {
             GraphSpec spec;
             spec.dQk = 256;
             spec.dV = 256;
             return graphWith(spec);
         },
         "HEAD_DIM_256_BWD"},
        {"DQK_NE_DV",
         [] {
             GraphSpec spec;
             spec.dV = 64;
             return graphWith(spec);
         },
         "DQK_NE_DV"},
        {"GQA_RATIO",
         [] {
             GraphSpec spec;
             spec.hK = 3;
             spec.hV = 3;
             return graphWith(spec);
         },
         "GQA_RATIO"},
        {"DROPOUT",
         [] {
             return graphWith({}, [](json& g) { parameters(g)["dropout_probability"] = 0.125; });
         },
         "DROPOUT"},
        {"BIAS_IN_BACKWARD",
         [] {
             return graphWith({}, [](json& g) {
                 addTensor(g, tensorJson(11, "mask", "half", {2, 8, 128, 128}, bshd(8, 128, 128)));
                 inputs(g)["attn_mask_tensor_uid"] = 11;
             });
         },
         "BIAS_IN_BACKWARD"},
        {"DBIAS",
         [] {
             return graphWith({}, [](json& g) {
                 addTensor(g, tensorJson(11, "dbias", "half", {2, 8, 128, 128}, bshd(8, 128, 128)));
                 g["nodes"][0]["outputs"]["dbias_tensor_uid"] = 11;
             });
         },
         "DBIAS"},
        {"ALIBI",
         [] { return graphWith({}, [](json& g) { parameters(g)["alibi_mask"] = true; }); },
         "ALIBI"},
        {"SCALE_DEVICE_TENSOR",
         [] {
             return graphWith({}, [](json& g) {
                 auto scale = tensorJson(11, "scale", "float", {1, 1, 1, 1}, {1, 1, 1, 1});
                 scale["is_runtime_pass_by_value"] = true;
                 addTensor(g, std::move(scale));
                 inputs(g)["scale_tensor_uid"] = 11;
             });
         },
         "SCALE_DEVICE_TENSOR"},
        {"EMPTY",
         [] {
             GraphSpec spec;
             spec.sKv = 0;
             return graphWith(spec);
         },
         "EMPTY"},
        {"STRIDE_D",
         [] { return graphWith({}, [](json& g) { tensorOf(g, Q_UID)["strides"][3] = 2; }); },
         "STRIDE_D"},
        {"STRIDE_NEGATIVE",
         [] {
             return graphWith({}, [](json& g) {
                 auto& strides = tensorOf(g, Q_UID)["strides"];
                 strides[0] = -strides[0].get<int64_t>();
             });
         },
         "STRIDE_NEGATIVE"},
        {"OUTPUT_OVERLAP",
         [] { return graphWith({}, [](json& g) { tensorOf(g, DQ_UID)["strides"][1] = 64; }); },
         "OUTPUT_OVERLAP"},
        {"STRIDE_ALIGN",
         [] {
             GraphSpec spec;
             spec.alignment = 1;
             return graphWith(spec);
         },
         "STRIDE_ALIGN"},
        {"BAND_BOUND",
         [] {
             GraphSpec spec;
             spec.rightBound = int64_t{1} << 31;
             return graphWith(spec);
         },
         "BAND_BOUND"},
        {"INDEX_RANGE",
         [] {
             GraphSpec spec;
             spec.b = 1;
             spec.hQ = spec.hK = spec.hV = 1;
             spec.sQ = spec.sKv = int64_t{1} << 24;
             return graphWith(spec);
         },
         "INDEX_RANGE"},
        {"GRID_LIMIT",
         [] {
             GraphSpec spec;
             spec.b = 65536;
             spec.hQ = spec.hK = spec.hV = 1;
             spec.sQ = spec.sKv = 1;
             spec.dQk = spec.dV = 32;
             return graphWith(spec);
         },
         "GRID_LIMIT"},
        {"RAGGED_FORMAT_StatsOnTheKvTable",
         [thd] {
             return graphWith(thd, [](json& g) {
                 tensorOf(g, STATS_UID)["ragged_offset_tensor_uid"] = OFFSETS_KV_UID;
             });
         },
         "RAGGED_FORMAT"},
        {"RAGGED_FORMAT_Multiplier",
         [thd] {
             return graphWith(thd,
                              [](json& g) { tensorOf(g, K_UID)["ragged_offset_multiplier"] = 7; });
         },
         "RAGGED_FORMAT"},
        {"SEQ_LEN_FORMAT",
         [] {
             return graphWith({}, [](json& g) {
                 addTensor(g, tensorJson(11, "seq_len_q", "int32", {2, 1, 1, 1}, {1, 1, 1, 1}));
                 inputs(g)["seq_len_q_tensor_uid"] = 11;
             });
         },
         "SEQ_LEN_FORMAT"},
        {"NOT_YET_DECLARED_Padding",
         [] {
             return graphWith({}, [](json& g) {
                 addTensor(g, tensorJson(11, "seq_len_q", "int32", {2, 1, 1, 1}, {1, 1, 1, 1}));
                 addTensor(g, tensorJson(12, "seq_len_kv", "int32", {2, 1, 1, 1}, {1, 1, 1, 1}));
                 inputs(g)["seq_len_q_tensor_uid"] = 11;
                 inputs(g)["seq_len_kv_tensor_uid"] = 12;
                 parameters(g)["padding_mask"] = true;
             });
         },
         "NOT_YET_DECLARED"},
        {"UNSUPPORTED_FEATURE",
         [] {
             return graphWith({}, [](json& g) { g["nodes"][0]["compute_data_type"] = "double"; });
         },
         "UNSUPPORTED_FEATURE"},
        {"LOWERING_BothCausalFlags",
         [] {
             GraphSpec spec;
             spec.causal = true;
             spec.causalBottomRight = true;
             return graphWith(spec);
         },
         "LOWERING"},
        {"LOWERING_VirtualOutput",
         [] { return graphWith({}, [](json& g) { tensorOf(g, DQ_UID)["virtual"] = true; }); },
         "LOWERING"},
        {"LOWERING_OneByteScaleConstant",
         [] {
             return graphWith({}, [](json& g) {
                 auto scale = tensorJson(11, "scale", "fp8_e4m3", {1, 1, 1, 1}, {1, 1, 1, 1});
                 scale["value"] = 0x38;
                 scale["value_type"] = "Float8Value";
                 addTensor(g, std::move(scale));
                 inputs(g)["scale_tensor_uid"] = 11;
             });
         },
         "LOWERING"},
        {"LOWERING_UnconsumedRaggedTensor",
         [] {
             return graphWith({}, [](json& g) {
                 auto extra = tensorJson(11, "extra", "half", {2, 1, 1, 1}, {1, 1, 1, 1});
                 extra["ragged_offset_tensor_uid"] = OFFSETS_Q_UID;
                 addTensor(g, std::move(extra));
                 addTensor(
                     g,
                     tensorJson(OFFSETS_Q_UID, "offsets_q", "int32", {3, 1, 1, 1}, {1, 1, 1, 1}));
             });
         },
         "LOWERING"},
    };
}

std::ostream& operator<<(std::ostream& out, const DeclineCase& declineCase)
{
    return out << declineCase.name;
}

class TestAttentionGeneralBwdDeclines : public ::testing::TestWithParam<DeclineCase>
{
};

} // namespace

TEST(TestAttentionGeneralBwdMatch, AcceptsTheGraphsTheFamilyServesAndLogsNoDecline)
{
    std::vector<std::pair<const char*, json>> accepted;
    accepted.emplace_back("dense MHA", graphWith({}));
    {
        GraphSpec spec;
        spec.thd = true;
        spec.hK = spec.hV = 2;
        accepted.emplace_back("THD GQA with per-side multipliers", graphWith(spec));
    }
    {
        GraphSpec spec;
        spec.causalBottomRight = true;
        accepted.emplace_back("bottom-right causal", graphWith(spec));
    }
    {
        GraphSpec spec;
        spec.leftBound = 64;
        spec.rightBound = 0;
        accepted.emplace_back("sliding window", graphWith(spec));
    }
    {
        GraphSpec spec;
        spec.hK = 4;
        spec.hV = 2;
        accepted.emplace_back("h_k != h_v", graphWith(spec));
    }
    for(const auto arch : {GFX942, GFX950})
    {
        for(const auto& [what, graph] : accepted)
        {
            const auto recorder
                = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
            const auto bound = graphMatch(graph, arch);
            EXPECT_TRUE(bound.has_value()) << arch << ": " << what;
            EXPECT_FALSE(recorder.hasLogContaining(std::string(ATTENTION_GENERAL_BWD_ENGINE_NAME)
                                                   + " declined the graph"))
                << arch << ": " << what;
            if(bound.has_value())
            {
                EXPECT_TRUE(tryGetBoundInt(*bound, "ws.bytes").has_value());
                EXPECT_EQ(tryGetBoundInt(*bound, "ws.bytes"),
                          tryGetBoundInt(*bound, "ws.scratch_offset"));
            }
        }
    }
}

TEST(TestAttentionGeneralBwdMatch, PublishesAtomicDkDvRegionsOnlyWhenTheHeadsDiffer)
{
    const GraphSpec mha;
    const auto direct = graphMatch(graphWith(mha));
    ASSERT_TRUE(direct.has_value());
    EXPECT_FALSE(tryGetBoundInt(*direct, "ws.dk_acc.offset").has_value());
    EXPECT_EQ(tryGetBoundInt(*direct, "attention_general_bwd.main.dkv_mode"), 0);

    GraphSpec split = mha;
    split.hK = 4;
    split.hV = 2;
    const auto atomic = graphMatch(graphWith(split));
    ASSERT_TRUE(atomic.has_value());
    EXPECT_TRUE(tryGetBoundInt(*atomic, "ws.dk_acc.offset").has_value());
    EXPECT_TRUE(tryGetBoundInt(*atomic, "ws.dv_acc.offset").has_value());
    EXPECT_EQ(tryGetBoundInt(*atomic, "attention_general_bwd.main.dkv_mode"), 1);
}

TEST(TestAttentionGeneralBwdMatch, ReadsADeclaredAlignmentAsItsLargestPowerOfTwoDivisor)
{
    const auto stageVec = [](int64_t declared) {
        GraphSpec spec;
        spec.alignment = declared;
        const auto bound = graphMatch(graphWith(spec));
        EXPECT_TRUE(bound.has_value()) << declared;
        return bound ? tryGetBoundInt(*bound, "attention_general_bwd.main.stage_vec")
                     : std::nullopt;
    };
    EXPECT_EQ(stageVec(16), 8);
    EXPECT_EQ(stageVec(48), 8);
    // A pointer that is a multiple of 24 is only 8-byte aligned.
    EXPECT_EQ(stageVec(24), 1);
    EXPECT_EQ(stageVec(6), 1);

    GraphSpec odd;
    odd.alignment = 3;
    EXPECT_EQ(loggedDecline(graphWith(odd)).code, "STRIDE_ALIGN");
}

TEST_P(TestAttentionGeneralBwdDeclines, DeclinesWithTheCodeRockeReturnsAndLogsItOnce)
{
    const auto& param = GetParam();
    for(const auto arch : {GFX942, GFX950})
    {
        const auto decline = loggedDecline(param.graph(), arch);
        EXPECT_EQ(decline.code, param.code) << arch << ": " << decline.detail;
        EXPECT_FALSE(decline.detail.empty()) << arch;
        EXPECT_EQ(decline.lines, 1u) << arch;
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionGeneralBwdDeclines,
                         ::testing::ValuesIn(declineCases()),
                         [](const ::testing::TestParamInfo<DeclineCase>& info) {
                             return std::string(info.param.name);
                         });

TEST(TestAttentionGeneralBwdMatch, DeclinesAnUnshippedArchitectureFirst)
{
    const auto decline = loggedDecline(graphWith({}), "gfx1201");
    EXPECT_EQ(decline.code, "ARCH_UNSUPPORTED");
    EXPECT_NE(decline.detail.find("gfx942"), std::string::npos) << decline.detail;

    // Ahead of every graph check: a graph rocKE would decline on gfx942 declines on the arch.
    GraphSpec spec;
    spec.dQk = spec.dV = 96;
    EXPECT_EQ(loggedDecline(graphWith(spec), "gfx1201").code, "ARCH_UNSUPPORTED");
}

TEST(TestAttentionGeneralBwdMatch, IsSilentOnAGraphThatIsNotOneSdpaBackwardNode)
{
    auto twoNodes = graphWith({});
    twoNodes["nodes"].push_back(twoNodes["nodes"][0]);
    twoNodes["nodes"][1]["name"] = "SdpaBwd_1";

    auto noNodes = graphWith({});
    noNodes["nodes"] = json::array();

    for(const auto& graph : {twoNodes, noNodes})
    {
        const auto recorder
            = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        EXPECT_FALSE(graphMatch(graph).has_value());
        EXPECT_FALSE(recorder.hasLogContaining(std::string(ATTENTION_GENERAL_BWD_ENGINE_NAME)));
    }
}

TEST(TestAttentionGeneralBwdMatch, EachPackCriterionAcceptsOnlyItsOwnArchitecture)
{
    registerNativeIngestorSymbols();
    const std::vector<std::pair<std::string, std::string_view>> packs = {
        {"hipkernel.attention_general_bwd.prep_gfx942_match", GFX942},
        {"hipkernel.attention_general_bwd.main_gfx942_match", GFX942},
        {"hipkernel.attention_general_bwd.prep_gfx950_match", GFX950},
        {"hipkernel.attention_general_bwd.main_gfx950_match", GFX950},
        {"hipkernel.attention_general_bwd.main_splitdq_gfx950_match", GFX950},
    };
    const auto graph = graphWith({});
    for(const auto& [symbol, packArch] : packs)
    {
        const auto criterion = hipdnn_plugin_sdk::ingestor::GraphCriterionRegistry::resolve(symbol);
        for(const auto device : {GFX942, GFX950, std::string_view("gfx1201")})
        {
            const GraphFixture fixture(graph, device, fullComputeUnits(device));
            EXPECT_EQ(criterion(fixture.context(), BoundTokens{}), device == packArch)
                << symbol << " on " << device;
        }
    }
}

TEST(TestAttentionGeneralBwdMatch, KernelMatchSelectsTheRowsThePlanNames)
{
    registerNativeIngestorSymbols();
    const auto kernelMatcher = hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(KERNEL_MATCHER_SYMBOL));
    for(const auto arch : {GFX942, GFX950})
    {
        const GraphFixture fixture(graphWith({}), arch, fullComputeUnits(arch));
        const auto context = fixture.context();
        const auto verdict = general_bwd::evaluate(context);
        ASSERT_TRUE(std::holds_alternative<general_bwd::Plan>(verdict)) << arch;
        const auto& plan = std::get<general_bwd::Plan>(verdict);
        const auto bound = general_bwd::tokensFor(plan);

        const auto main = mainKernelOf(*plan.row);
        const auto prep = prepKernelOf(*plan.row);
        EXPECT_TRUE(kernelMatcher(context, bound, main)) << arch;
        EXPECT_TRUE(kernelMatcher(context, bound, prep)) << arch;

        // One field off, one kernel off.
        auto otherHeadPack = main;
        otherHeadPack.metadata["head_pack"] = !plan.row->headPack;
        EXPECT_FALSE(kernelMatcher(context, bound, otherHeadPack)) << arch;

        auto otherDqMode = main;
        otherDqMode.metadata["dq_mode"]
            = std::string(plan.row->dq == generated::DqMode::SPLIT ? "atomic" : "split");
        EXPECT_FALSE(kernelMatcher(context, bound, otherDqMode)) << arch;

        auto otherPrepRows = prep;
        otherPrepRows.metadata["rows_per_cta"] = plan.row->prepRowsPerCta + 1;
        EXPECT_FALSE(kernelMatcher(context, bound, otherPrepRows)) << arch;

        // A tuning key of this arch the plan did not pick, then one no arch knows.
        for(const auto& entry : generated::TUNED_KEYS)
        {
            if(entry.arch == arch && entry.id != plan.row->tuningKeyId)
            {
                auto otherKey = main;
                otherKey.metadata["tuning_key"] = std::string(entry.key);
                EXPECT_FALSE(kernelMatcher(context, bound, otherKey)) << arch;
                break;
            }
        }
        auto unknownKey = main;
        unknownKey.metadata["tuning_key"] = std::string("not|a|tuning|key");
        EXPECT_FALSE(kernelMatcher(context, bound, unknownKey)) << arch;

        auto unknownDtype = main;
        unknownDtype.metadata["dtype"] = std::string("fp16");
        EXPECT_FALSE(kernelMatcher(context, bound, unknownDtype))
            << arch << ": the KMD spells dtype FP16/BF16";

        auto missingField = main;
        missingField.metadata.erase("main_static_scale");
        EXPECT_FALSE(kernelMatcher(context, bound, missingField)) << arch;
    }
}

TEST(TestAttentionGeneralBwdMatch, KernelMatchSelectsOneKernelPerSlotForEveryCatalogRow)
{
    registerNativeIngestorSymbols();
    const auto kernelMatcher = hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(KERNEL_MATCHER_SYMBOL));
    for(const auto arch : {GFX942, GFX950})
    {
        const auto kernels = archKernels(arch);
        const GraphFixture fixture(graphWith({}), arch, fullComputeUnits(arch));
        const auto context = fixture.context();
        std::size_t checked = 0;
        for(const auto& row : generated::CATALOG)
        {
            if(row.arch != arch)
            {
                continue;
            }
            general_bwd::Plan plan;
            plan.row = &row;
            const auto bound = general_bwd::tokensFor(plan);
            std::size_t mains = 0;
            std::size_t preps = 0;
            for(const auto& kernel : kernels.main)
            {
                mains += kernelMatcher(context, bound, kernel) ? 1 : 0;
            }
            for(const auto& kernel : kernels.prep)
            {
                preps += kernelMatcher(context, bound, kernel) ? 1 : 0;
            }
            EXPECT_EQ(mains, 1u) << row.mainKernel << " / key " << row.tuningKeyId;
            EXPECT_EQ(preps, 1u) << row.prepKernel;
            ++checked;
        }
        EXPECT_GT(checked, 0u) << arch;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
