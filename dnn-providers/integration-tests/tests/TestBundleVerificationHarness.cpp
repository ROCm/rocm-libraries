// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Drives the harness under ScopedFakeTestPartResultReporter to verify
// executor outcomes (decline→SKIP, match→PASS, mismatch→FAIL).

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PackedElementTraits.hpp>
#include <hipdnn_data_sdk/utilities/PackedSubByteTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "RaggedGraphTestUtils.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/VariantPackBuilder.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using hipdnn_test_sdk::utilities::claimScratchDirectory;

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
        _scopedDir.emplace(claimScratchDirectory("golden_harness"));
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

// uids: x_a=1, scale_a=2, x_b=4, scale_b=5 (leaf); y_a=3, y_b=6 virtual; c=7 output.
// A is 32x128 with column-major strides {1, 32}.
std::shared_ptr<IntegrationTestBundle>
    makeMxMatmulBundle(hipdnn_flatbuffers_sdk::data_objects::DataType xType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 4}, {4, 32}, xType);
    auto bundle = std::make_shared<IntegrationTestBundle>();
    bundle->graphBuffer = builder.Release();
    bundle->outputTensorUids = {7};
    return bundle;
}

// Runs a graph-only bundle through the harness on the host. The outputs stay
// sentinels, so the verdict is not what the caller is testing.
void fillOnHost(IntegrationBundleVerificationHarness& harness,
                std::shared_ptr<IntegrationTestBundle> bundle)
{
    harness.setBundle(std::move(bundle), "unit-test-bundle");
    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);
}

// Reads the code in bit slot `slot` of a packed buffer, LSB-first.
uint8_t packedCodeAt(const uint8_t* packed, size_t slot, size_t bits)
{
    const size_t bitOffset = slot * bits;
    const size_t byteIndex = bitOffset / 8;
    const auto bitIndex = static_cast<unsigned>(bitOffset % 8);

    auto window = static_cast<unsigned>(packed[byteIndex]);
    if(bitIndex > 8 - bits)
    {
        window |= static_cast<unsigned>(packed[byteIndex + 1]) << 8;
    }
    return static_cast<uint8_t>((window >> bitIndex) & ((1u << bits) - 1));
}

// The packed set must hold the same uids as the bundle's tensors: the sub-byte
// operands packed and code-for-code equal at every coordinate, the scales
// byte-for-byte equal.
template <typename T>
void expectPackedInputsTwinBundleTensors(hipdnn_flatbuffers_sdk::data_objects::DataType xType)
{
    using Traits = hipdnn_data_sdk::utilities::PackedElementTraits<T>;
    using PackedTensor
        = hipdnn_data_sdk::utilities::PackedSubByteTensor<T, Traits::BITS_PER_ELEMENT>;

    testing_support::HarnessMocks mocks;
    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::CPU)));
    auto bundle = makeMxMatmulBundle(xType);
    fillOnHost(harness, bundle);

    ASSERT_TRUE(bundle->tensors.has_value());
    const auto& unpacked = *bundle->tensors;
    const auto& packed = harness.packedInputs();
    ASSERT_EQ(packed.size(), unpacked.size());

    for(const int64_t uid : {1, 4})
    {
        ASSERT_EQ(packed.count(uid), 1u) << "uid " << uid;
        ASSERT_NE(dynamic_cast<const PackedTensor*>(packed.at(uid).get()), nullptr)
            << "uid " << uid;
        const auto& expected
            = dynamic_cast<const hipdnn_data_sdk::utilities::Tensor<T>&>(*unpacked.at(uid));
        const auto* packedHost = static_cast<const uint8_t*>(packed.at(uid)->rawHostData());
        const auto& dims = expected.dims();
        const auto& strides = expected.strides();
        for(int64_t i0 = 0; i0 < dims[0]; ++i0)
        {
            for(int64_t i1 = 0; i1 < dims[1]; ++i1)
            {
                const auto slot = static_cast<size_t>((i0 * strides[0]) + (i1 * strides[1]));
                ASSERT_EQ(
                    packedCodeAt(packedHost, slot, Traits::BITS_PER_ELEMENT),
                    static_cast<uint8_t>(expected.getHostValue(i0, i1).data & Traits::CODE_MASK))
                    << "uid " << uid << " at (" << i0 << "," << i1 << ")";
            }
        }
    }

    for(const int64_t uid : {2, 5})
    {
        ASSERT_EQ(packed.count(uid), 1u) << "uid " << uid;
        const size_t bytes = unpacked.at(uid)->elementSpace() * unpacked.at(uid)->elementSize();
        EXPECT_EQ(
            std::memcmp(unpacked.at(uid)->rawHostData(), packed.at(uid)->rawHostData(), bytes), 0)
            << "uid " << uid;
    }
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

TEST_F(TestGoldenHarnessFixture, PackedInputsTwinBundleTensorsForFp4)
{
    expectPackedInputsTwinBundleTensors<hipdnn_data_sdk::types::fp4_e2m1>(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1);
}

TEST_F(TestGoldenHarnessFixture, PackedInputsTwinBundleTensorsForFp6)
{
    expectPackedInputsTwinBundleTensors<hipdnn_data_sdk::types::fp6_e2m3>(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E2M3);
}

TEST_F(TestGoldenHarnessFixture, NonSubByteBundleHasNoPackedInputs)
{
    testing_support::HarnessMocks mocks;
    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::CPU)));
    auto bundle = makeMxMatmulBundle(hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3);
    fillOnHost(harness, bundle);

    ASSERT_TRUE(bundle->tensors.has_value());
    EXPECT_TRUE(harness.packedInputs().empty());
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

// A declined graph must not pay for its inputs. Sweep bundles carry no tensor data,
// so the harness generates it -- and on the largest full-tier cases that fill alone
// is seconds per test, which is what put whole provider suites past their CI wall
// when every one of those cases was going to be declined anyway.
TEST_F(TestGoldenHarnessFixture, DeclinedGraphSkipsWithoutFillingInputs)
{
    testing_support::HarnessMocks mocks;
    ON_CALL(mocks.engineRunner, openGraph(::testing::_, ::testing::_))
        .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
            return testing_support::declinedSession();
        });

    auto bundle = makeRuntimePbvFillBundle();
    ASSERT_FALSE(bundle->tensors.has_value());

    ::testing::TestPartResultArray results;
    runCapturing(mocks, bundle, &results);

    EXPECT_TRUE(testing_support::anySkipped(results));
    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("Engine could not execute bundle"),
              std::string::npos);
    EXPECT_FALSE(bundle->tensors.has_value());
}

TEST_F(TestGoldenHarnessFixture, RaggedBundleWithoutBlobsIsUnverifiable)
{
    testing_support::HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);

    auto bundle = makeRuntimePbvFillBundle();
    bundle->graphBuffer = test_utils::markFirstTensorRagged(bundle->graphBuffer.data());
    ASSERT_FALSE(bundle->tensors.has_value());

    ::testing::TestPartResultArray results;
    runCapturing(mocks, bundle, &results);

    EXPECT_TRUE(testing_support::anySkipped(results));
    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("ragged inputs require golden blobs"),
              std::string::npos);
    EXPECT_FALSE(bundle->tensors.has_value());
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
