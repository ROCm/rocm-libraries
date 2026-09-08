// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Drives the harness under ScopedFakeTestPartResultReporter to verify
// executor outcomes (decline→SKIP, match→PASS, mismatch→FAIL).

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/VariantPackBuilder.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;

namespace
{

class TestGoldenHarnessFixture : public ::testing::Test
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;

    void SetUp() override
    {
        testing_support::ensureTestConfigInitialized();
        _scopedDir.emplace(scratch::makeDir("golden_harness_test_"));
        _tempDir = _scopedDir->path();
    }

    /// Writes and loads a golden-bearing bundle under the fixture's temp dir.
    std::shared_ptr<IntegrationTestBundle> loadRunnableBundle(const std::string& name) const
    {
        return fixtures::loadBundle(_tempDir, name, /*includeGoldenOutput=*/true);
    }

    /// Builds the real harness on top of `mocks`, drives it through one bundle, and
    /// captures every gtest disposition it issues.
    static void runCapturing(testing_support::HarnessMocks& mocks,
                             std::shared_ptr<IntegrationTestBundle> bundle,
                             ::testing::TestPartResultArray* results)
    {
        IntegrationBundleVerificationHarness harness(
            mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
        harness.setBundle(std::move(bundle), "unit-test-bundle");

        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, results);
        harness.SetUp();
        harness.TestBody();
    }
};

// uids: x=1, y=2, scale=3, bias=4, epsilon=5, prev_mean=8, prev_variance=9, momentum=10
std::shared_ptr<IntegrationTestBundle> makeRuntimePbvFillBundle()
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        {3, 1},
        {2, 3},
        /*withMeanVariance=*/false,
        /*overrideShapeEnabled=*/false,
        /*runtimeEpsilon=*/true,
        /*withRunningStatsAndMomentum=*/true,
        /*runtimeMomentum=*/true);
    auto bundle = std::make_shared<IntegrationTestBundle>();
    bundle->graphBuffer = builder.Release();
    bundle->outputTensorUids = {2};
    return bundle;
}

std::shared_ptr<IntegrationTestBundle> makeRuntimePassByValueBundle()
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> scalarDims = {1};
    const std::vector<int64_t> scalarStrides = {1};
    const std::vector<int64_t> outputDims = {1};
    const std::vector<int64_t> outputStrides = {1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   1,
                                                   "epsilon",
                                                   DataType::FLOAT,
                                                   &scalarStrides,
                                                   &scalarDims,
                                                   false,
                                                   TensorValue::NONE,
                                                   0,
                                                   true));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, 3, "scale", DataType::FLOAT, &scalarStrides, &scalarDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, 2, "output", DataType::FLOAT, &outputStrides, &outputDims));

    const std::vector<flatbuffers::Offset<Node>> nodes;
    const auto graph = CreateGraphDirect(builder,
                                         "runtime_pbv",
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         &tensors,
                                         &nodes);
    builder.Finish(graph);

    auto bundle = std::make_shared<IntegrationTestBundle>();
    bundle->graphBuffer = builder.Release();
    bundle->outputTensorUids = {2};
    return bundle;
}

TEST(TestBundleVerificationHarness, DeviceVariantPackUsesHostPointerForRuntimePassByValue)
{
    SKIP_IF_NO_DEVICES();
    auto bundle = makeRuntimePassByValueBundle();
    const auto wrapper = bundle->graphWrapper();
    const auto& tensorAttributes = wrapper.getTensorMap();

    TensorMap inputs;
    inputs.emplace(1, hipdnn_test_sdk::detail::createTensorFromAttribute(*tensorAttributes.at(1)));
    inputs.at(1)->fillTensorWithValue(0.01f);
    auto* expectedHostPointer = inputs.at(1)->rawHostData();

    inputs.emplace(3, hipdnn_test_sdk::detail::createTensorFromAttribute(*tensorAttributes.at(3)));

    static constexpr int64_t K_UNKNOWN_UID = 99;
    inputs.emplace(K_UNKNOWN_UID,
                   hipdnn_test_sdk::detail::createTensorFromAttribute(*tensorAttributes.at(3)));

    OutputTensors outputs;
    outputs.emplace(2, hipdnn_test_sdk::detail::createTensorFromAttribute(*tensorAttributes.at(2)));
    auto variantPack = detail::buildVariantPack(
        inputs, outputs, tensorAttributes, bundle->outputTensorUids, /*useDevice=*/true);

    ASSERT_EQ(variantPack.at(1), expectedHostPointer);
    EXPECT_FLOAT_EQ(*static_cast<const float*>(variantPack.at(1)), 0.01f);
    EXPECT_EQ(variantPack.at(3), inputs.at(3)->rawDeviceData());
    EXPECT_EQ(variantPack.at(K_UNKNOWN_UID), inputs.at(K_UNKNOWN_UID)->rawDeviceData());
    EXPECT_EQ(variantPack.at(2), outputs.at(2)->rawDeviceData());
}
} // namespace

TEST_F(TestGoldenHarnessFixture, GraphOnlyRuntimePbvValuesAreFilledEndToEnd)
{
    const auto runHarness = [](float& epsilon, float& momentum) {
        testing_support::HarnessMocks mocks;
        ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault([&epsilon, &momentum](GraphSession&,
                                                 const std::optional<LoadedEngine>&,
                                                 VariantPack& variantPack) {
                epsilon = *static_cast<const float*>(variantPack.at(5));
                momentum = *static_cast<const float*>(variantPack.at(10));
                EXPECT_GE(momentum, 0.0f);
                EXPECT_LE(momentum, 1.0f);
                return EngineOpResult::declinedBy("value-capture stub completed");
            });

        IntegrationBundleVerificationHarness harness(
            mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));

        auto bundle = makeRuntimePbvFillBundle();
        harness.setBundle(std::move(bundle), "runtime-pbv-fill");
        harness.inputFillRecipes().setGlobalSeed(42);

        ::testing::TestPartResultArray results;
        {
            const ::testing::ScopedFakeTestPartResultReporter reporter(
                ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
            harness.SetUp();
            harness.TestBody();
        }
        EXPECT_TRUE(testing_support::anySkipped(results));
        EXPECT_FALSE(testing_support::anyFailed(results));
    };

    float firstEpsilon = 0.0f;
    float firstMomentum = 0.0f;
    runHarness(firstEpsilon, firstMomentum);
    EXPECT_FLOAT_EQ(firstEpsilon, 1e-5f);

    float secondEpsilon = 0.0f;
    float secondMomentum = 0.0f;
    runHarness(secondEpsilon, secondMomentum);
    EXPECT_FLOAT_EQ(secondEpsilon, 1e-5f);
    EXPECT_FLOAT_EQ(secondMomentum, firstMomentum);
}

// Was ExecutorThrowsYieldsSkip: IGraphEngineRunner::execute() now answers "not
// mine" with EngineOpResult::declinedBy(...) instead of throwing
// EngineNotApplicableError. The outcome the test defends is unchanged (SKIP).
TEST_F(TestGoldenHarnessFixture, ExecutorDeclineYieldsSkip)
{
    testing_support::HarnessMocks mocks;
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(
            ::testing::Return(EngineOpResult::declinedBy("engine does not support this graph")));

    ::testing::TestPartResultArray results;
    runCapturing(mocks, loadRunnableBundle("throws"), &results);

    EXPECT_TRUE(testing_support::anySkipped(results));
    EXPECT_FALSE(testing_support::anyFailed(results));
}

TEST_F(TestGoldenHarnessFixture, MatchingOutputYieldsPass)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);

    ::testing::TestPartResultArray results;
    runCapturing(mocks, loadRunnableBundle("match"), &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST_F(TestGoldenHarnessFixture, MismatchingOutputYieldsFail)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE + 100.0f);

    ::testing::TestPartResultArray results;
    runCapturing(mocks, loadRunnableBundle("mismatch"), &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

// NOLINTEND(readability-identifier-naming)
