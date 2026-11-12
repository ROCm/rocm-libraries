// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(ENABLE_MEGAKERNEL_ENGINE)

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/custom_op_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestMegakernelMatchers.cpp
 * @brief Declines of hipkernel:Megakernel: each payload or binary-key mismatch declines,
 *        and the INFO line names the field that differs. Matcher-only: no device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;

constexpr std::string_view MODEL = "qwen38-27b-mxfp4-gfx942";
constexpr std::string_view SRC = "71ccfa7742a025b67d7d8e597072bf4f";
constexpr int64_t CUS = 304;

nlohmann::json goodPayload()
{
    return {{"schema", 1},
            {"model", MODEL},
            {"src", SRC},
            {"arch", "gfx942"},
            {"cus", CUS},
            {"kv", "ROCM_ATTN"},
            {"block", 784},
            {"M", 1},
            {"tp", 1},
            {"rt", 0},
            {"lm_head", false},
            {"nlayers", 64}};
}

/// The KMD record of the object goodPayload() names.
KernelDefinition goodKernel()
{
    KernelDefinition kernel;
    kernel.name = "megakernel.qwen38.M1";
    kernel.metadata = {{"model", std::string(MODEL)},
                       {"src", std::string(SRC)},
                       {"cus", CUS},
                       {"kv", std::string("ROCM_ATTN")},
                       {"block", int64_t{784}},
                       {"M", int64_t{1}},
                       {"tp", int64_t{1}},
                       {"rt", int64_t{0}},
                       {"lm_head", false},
                       {"block_threads", int64_t{512}},
                       {"lds_bytes", int64_t{0}}};
    return kernel;
}

struct Graph
{
    flatbuffers::FlatBufferBuilder builder;
    DeviceProperties properties;

    explicit Graph(const std::string& data,
                   const std::string& opId = "megakernel." + std::string(MODEL),
                   int multiProcessorCount = static_cast<int>(CUS))
    {
        const std::vector<int64_t> inputs{1, 2};
        const std::vector<uint8_t> bytes(data.begin(), data.end());
        const auto attributes = data_objects::CreateCustomOpAttributesDirect(
            builder, opId.c_str(), &inputs, nullptr, &bytes);
        const std::vector<flatbuffers::Offset<data_objects::Node>> nodes{
            data_objects::CreateNodeDirect(builder,
                                           "megakernel",
                                           data_objects::DataType::FLOAT,
                                           data_objects::NodeAttributes::CustomOpAttributes,
                                           attributes.Union())};
        const auto name = builder.CreateString("megakernel_test");
        const auto nodesVector = builder.CreateVector(nodes);
        data_objects::GraphBuilder graphBuilder(builder);
        graphBuilder.add_name(name);
        graphBuilder.add_nodes(nodesVector);
        builder.Finish(graphBuilder.Finish());

        properties.gcnArchName = "gfx942:sramecc+:xnack-";
        properties.multiProcessorCount = multiProcessorCount;
    }
};

struct Outcome
{
    bool matched = false;
    std::string cause; ///< The bracketed field of the decline line, empty when none logged.
};

std::string declineCause(const hipdnn_test_sdk::utilities::LogRecorderBase& recorder)
{
    const std::string marker = std::string(MEGAKERNEL_ENGINE_NAME) + " declined ";
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        const auto open = at == std::string::npos ? at : log.message.find(" [", at);
        const auto close = open == std::string::npos ? open : log.message.find(']', open);
        if(close != std::string::npos)
        {
            return log.message.substr(open + 2, close - open - 2);
        }
    }
    return {};
}

/// graph_match, then (when it binds) kernel_match against @p kernel, as a catalog build.
Outcome match(const Graph& graph, const KernelDefinition& kernel = goodKernel())
{
    registerNativeIngestorSymbols();
    const auto graphMatcher
        = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve("megakernel.match_custom_op");
    const auto kernelMatcher
        = hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve("megakernel.kernel_matches");
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        graph.builder.GetBufferPointer(), graph.builder.GetSize());
    const MatchContext context{wrapper, 0, graph.properties};
    Outcome outcome;
    const auto bound = graphMatcher(context);
    outcome.matched = bound.has_value() && kernelMatcher(context, *bound, kernel);
    outcome.cause = declineCause(recorder);
    return outcome;
}

Outcome matchPayload(const nlohmann::json& payload)
{
    return match(Graph(payload.dump()));
}

TEST(TestMegakernelMatchers, AcceptsThePayloadOfTheObjectsBinaryKey)
{
    const auto outcome = matchPayload(goodPayload());
    EXPECT_TRUE(outcome.matched);
    EXPECT_EQ(outcome.cause, "");
}

TEST(TestMegakernelMatchers, DeclinesACuCountOtherThanTheDevices)
{
    auto payload = goodPayload();
    payload["cus"] = 228;
    const auto outcome = matchPayload(payload);
    EXPECT_FALSE(outcome.matched);
    EXPECT_EQ(outcome.cause, "cus");
}

TEST(TestMegakernelMatchers, DeclinesADeviceWhoseCuCountIsNotTheObjects)
{
    // The payload and the kernel agree on 304; the device is a partitioned 38-CU part.
    const auto outcome = match(Graph(goodPayload().dump(), "megakernel." + std::string(MODEL), 38));
    EXPECT_FALSE(outcome.matched);
    EXPECT_EQ(outcome.cause, "cus");
}

TEST(TestMegakernelMatchers, DeclinesASrcNoObjectCarries)
{
    auto payload = goodPayload();
    payload["src"] = "0000000000000000deadbeefdeadbeef";
    const auto outcome = matchPayload(payload);
    EXPECT_FALSE(outcome.matched);
    EXPECT_EQ(outcome.cause, "src");
}

TEST(TestMegakernelMatchers, DeclinesAWidthOtherThanTheObjects)
{
    auto payload = goodPayload();
    payload["M"] = 8;
    const auto outcome = matchPayload(payload);
    EXPECT_FALSE(outcome.matched);
    EXPECT_EQ(outcome.cause, "M");
}

TEST(TestMegakernelMatchers, DeclinesAnotherSchemaVersion)
{
    auto payload = goodPayload();
    payload["schema"] = 2;
    const auto outcome = matchPayload(payload);
    EXPECT_FALSE(outcome.matched);
    EXPECT_EQ(outcome.cause, "schema");

    payload.erase("schema");
    EXPECT_EQ(matchPayload(payload).cause, "schema");
}

TEST(TestMegakernelMatchers, DeclinesMalformedJson)
{
    auto text = goodPayload().dump();
    text.pop_back(); // drop the closing brace
    const auto outcome = match(Graph(text));
    EXPECT_FALSE(outcome.matched);
    EXPECT_EQ(outcome.cause, "payload");

    EXPECT_EQ(match(Graph("[1, 2]")).cause, "payload");
}

TEST(TestMegakernelMatchers, DeclinesAKeyOfTheWrongType)
{
    auto payload = goodPayload();
    payload["lm_head"] = 0; // an int is not the bool the key is
    const auto outcome = matchPayload(payload);
    EXPECT_FALSE(outcome.matched);
    EXPECT_EQ(outcome.cause, "lm_head");
}

TEST(TestMegakernelMatchers, DeclinesAnOpIdThatNamesAnotherModel)
{
    const auto outcome = match(Graph(goodPayload().dump(), "megakernel.gemma4-26b-a4b-fp8-gfx942"));
    EXPECT_FALSE(outcome.matched);
    EXPECT_EQ(outcome.cause, "model");
}

TEST(TestMegakernelMatchers, IgnoresCustomOpsOfOtherPluginsSilently)
{
    const auto outcome = match(Graph(goodPayload().dump(), "example.rope"));
    EXPECT_FALSE(outcome.matched);
    EXPECT_EQ(outcome.cause, "");
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif
