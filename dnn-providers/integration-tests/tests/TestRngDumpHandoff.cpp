// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <gmock/gmock.h>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "HarnessTestSupport.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/RngDumpHandoff.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using namespace hipdnn_integration_tests::bundle;

namespace
{

flatbuffers::DetachedBuffer buildSdpaGraph(std::optional<float> dropoutProbability,
                                           std::optional<int64_t> rngDumpUid)
{
    flatbuffers::FlatBufferBuilder b;
    const std::vector<int64_t> dims = {1, 1, 2, 2};
    const std::vector<int64_t> strides = {4, 4, 2, 1};
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    for(int64_t uid = 1; uid <= 4; ++uid)
    {
        tensors.push_back(
            CreateTensorAttributesDirect(b, uid, "t", DataType::FLOAT, &strides, &dims));
    }
    if(rngDumpUid.has_value())
    {
        tensors.push_back(CreateTensorAttributesDirect(
            b, *rngDumpUid, "rng_dump", DataType::FLOAT, &strides, &dims));
    }
    SdpaAttributesBuilder attrs(b);
    attrs.add_q_tensor_uid(1);
    attrs.add_k_tensor_uid(2);
    attrs.add_v_tensor_uid(3);
    attrs.add_o_tensor_uid(4);
    if(dropoutProbability.has_value())
    {
        attrs.add_dropout_probability(*dropoutProbability);
    }
    if(rngDumpUid.has_value())
    {
        attrs.add_rng_dump_tensor_uid(*rngDumpUid);
    }
    const auto sdpa = attrs.Finish();
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "sdpa_fwd", DataType::FLOAT, NodeAttributes::SdpaAttributes, sdpa.Union()));
    b.Finish(CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes));
    return b.Release();
}

// The Philox graph the harness runs: q=1, k=2, v=3 inputs; o=4 and rng_dump=9 outputs, each
// holding four elements.
constexpr int64_t K_O_UID = 4;
constexpr int64_t K_DUMP_UID = 9;
constexpr size_t K_ELEMS = 4;

std::shared_ptr<IntegrationTestBundle> philoxBundle()
{
    auto bundle = std::make_shared<IntegrationTestBundle>();
    bundle->graphBuffer = buildSdpaGraph(0.5F, K_DUMP_UID);
    bundle->outputTensorUids = {K_O_UID, K_DUMP_UID};
    return bundle;
}

// The engine dumps `mask` and writes it to o as well, so a reference that computes o from the
// mask it was handed matches the engine exactly when it was handed the engine's mask.
void engineDumps(testing_support::HarnessMocks& mocks, const std::vector<float>& mask)
{
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([mask](GraphSession&,
                              const std::optional<LoadedEngine>&,
                              hipdnn_integration_tests::VariantPack& variantPack) {
            std::copy(mask.begin(), mask.end(), static_cast<float*>(variantPack.at(K_DUMP_UID)));
            std::copy(mask.begin(), mask.end(), static_cast<float*>(variantPack.at(K_O_UID)));
            return EngineOpResult::succeeded();
        });
}

// The CPU reference records the rng_dump it was handed and writes it to o.
void referenceEchoesDump(testing_support::HarnessMocks& mocks, std::vector<float>& seen)
{
    ON_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(
            [&seen](void*, size_t, const hipdnn_integration_tests::VariantPack& variantPack) {
                const auto* dump = static_cast<const float*>(variantPack.at(K_DUMP_UID));
                seen.assign(dump, dump + K_ELEMS);
                std::copy(dump, dump + K_ELEMS, static_cast<float*>(variantPack.at(K_O_UID)));
            });
}

void runCpuVerification(testing_support::HarnessMocks& mocks,
                        ::testing::TestPartResultArray* results)
{
    testing_support::ensureTestConfigInitialized();
    IntegrationBundleVerificationHarness harness(mocks.dependencies(
        testing_support::hostPolicy(hipdnn_integration_tests::VerificationMode::CPU)));
    harness.setBundle(philoxBundle(), "rng-dump-test-bundle");
    testing_support::driveHarness(harness, results);
}

Tensor<float> maskTensor(const std::vector<float>& values)
{
    Tensor<float> tensor({static_cast<int64_t>(values.size())}, {1});
    tensor.fillWithData(values.data(), values.size() * sizeof(float));
    return tensor;
}

} // namespace

TEST(TestRngDumpHandoff, FindsDumpOnlyWithPhiloxDropout)
{
    const auto withDump = buildSdpaGraph(0.25F, 9);
    const auto found = findSdpaRngDump(GraphWrapper(withDump.data(), withDump.size()));
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->uid, 9);
    EXPECT_FLOAT_EQ(found->dropoutProbability, 0.25F);

    const auto noDropout = buildSdpaGraph(std::nullopt, 9);
    EXPECT_FALSE(findSdpaRngDump(GraphWrapper(noDropout.data(), noDropout.size())).has_value());
    const auto noDump = buildSdpaGraph(0.25F, std::nullopt);
    EXPECT_FALSE(findSdpaRngDump(GraphWrapper(noDump.data(), noDump.size())).has_value());
}

TEST(TestRngDumpHandoff, ComparedUidsDropOnlyTheDump)
{
    const std::vector<int64_t> outputs = {4, 9, 10};
    EXPECT_EQ(comparedOutputUids(outputs, SdpaRngDump{9, 0.5F}), (std::vector<int64_t>{4, 10}));
    EXPECT_EQ(comparedOutputUids(outputs, std::nullopt), outputs);
}

TEST(TestRngDumpHandoff, AcceptsBernoulliMaskAtExpectedRate)
{
    std::mt19937 gen(7);
    std::bernoulli_distribution keep(0.7);
    std::vector<float> values(4096);
    for(auto& value : values)
    {
        value = keep(gen) ? 1.0F : 0.0F;
    }
    auto mask = maskTensor(values);
    EXPECT_EQ(checkRngDump(mask, 0.3F), "");
}

TEST(TestRngDumpHandoff, RejectsNonBinaryValues)
{
    auto mask = maskTensor({1.0F, 0.0F, 1.0F / 0.7F, 0.0F});
    EXPECT_NE(checkRngDump(mask, 0.3F).find("not 0 or 1"), std::string::npos);
}

TEST(TestRngDumpHandoff, RejectsAllOnesAndAllZeros)
{
    auto ones = maskTensor(std::vector<float>(4096, 1.0F));
    EXPECT_NE(checkRngDump(ones, 0.3F).find("keep rate"), std::string::npos);
    auto zeros = maskTensor(std::vector<float>(4096, 0.0F));
    EXPECT_NE(checkRngDump(zeros, 0.3F).find("keep rate"), std::string::npos);
}

TEST(TestRngDumpHandoff, RejectsNonFloatDump)
{
    Tensor<double> dump({4}, {1});
    dump.fillTensorWithValue(1.0F);
    EXPECT_EQ(checkRngDump(dump, 0.3F), "rng_dump must be FLOAT");
}

TEST(TestRngDumpHandoff, CopiesEngineMaskIntoReferenceBuffer)
{
    auto engine = maskTensor({1.0F, 0.0F, 0.0F, 1.0F});
    auto reference = maskTensor({-1.0F, -1.0F, -1.0F, -1.0F});
    copyRngDump(engine, reference);
    const auto* copied = static_cast<const float*>(reference.rawHostData());
    EXPECT_EQ((std::vector<float>(copied, copied + 4)),
              (std::vector<float>{1.0F, 0.0F, 0.0F, 1.0F}));

    auto wrongSize = maskTensor({1.0F, 0.0F});
    EXPECT_THROW(copyRngDump(engine, wrongSize), std::runtime_error);
}

TEST(TestRngDumpHandoff, ReferenceRunsOnTheEngineDump)
{
    testing_support::HarnessMocks mocks;
    const std::vector<float> mask = {1.0F, 0.0F, 0.0F, 1.0F};
    engineDumps(mocks, mask);
    std::vector<float> seen;
    referenceEchoesDump(mocks, seen);

    ::testing::TestPartResultArray results;
    runCpuVerification(mocks, &results);

    EXPECT_EQ(seen, mask);
    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST(TestRngDumpHandoff, InvalidEngineDumpFailsTheBundle)
{
    // o matches the reference, and rng_dump is never compared: only the dump check can fail it.
    testing_support::HarnessMocks mocks;
    engineDumps(mocks, {1.0F, 0.0F, 0.5F, 1.0F});
    std::vector<float> seen;
    referenceEchoesDump(mocks, seen);

    ::testing::TestPartResultArray results;
    runCpuVerification(mocks, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_THAT(testing_support::allMessages(results), ::testing::HasSubstr("not 0 or 1"));
}
