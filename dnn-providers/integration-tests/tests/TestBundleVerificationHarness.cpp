// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Drives the harness under ScopedFakeTestPartResultReporter to verify
// executor outcomes (decline→SKIP, match→PASS, mismatch→FAIL).

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PackedElementTraits.hpp>
#include <hipdnn_data_sdk/utilities/PackedSubByteTensor.hpp>
#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/RaggedLengthUids.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "RaggedGraphTestUtils.hpp"
#include "harness/Benchmark.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/LiveRows.hpp"
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

    /// Writes and loads the ragged bundle with `goldenOutput` as y. Its offsets
    /// {0, 20, 60} give batch 0 one sequence row and batch 1 two.
    std::shared_ptr<IntegrationTestBundle>
        loadRunnableRaggedBundle(const std::string& name,
                                 const std::vector<float>& goldenOutput) const
    {
        const auto dir = _tempDir / name;
        fixtures::writeRaggedBundleFiles(dir, name, {0, 20, 60}, goldenOutput);
        return fixtures::loadWrittenBundle(dir, name);
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

// uids: x=1 (input), z=2 (output). The GDN node has no input fills registered, so the
// harness cannot fill the bundle's inputs.
std::shared_ptr<IntegrationTestBundle> makeUnfillableBundle()
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {2, 3};
    const std::vector<int64_t> strides = {3, 1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 1, "x", DataType::FLOAT, &strides, &dims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 2, "z", DataType::FLOAT, &strides, &dims));

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "gdn",
                                     DataType::FLOAT,
                                     NodeAttributes::GdnAttributes,
                                     flatbuffers::Offset<void>()));

    const auto graph = CreateGraphDirect(
        builder, "unfillable", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    builder.Finish(graph);

    auto bundle = std::make_shared<IntegrationTestBundle>();
    bundle->graphBuffer = builder.Release();
    bundle->outputTensorUids = {2};
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

    ASSERT_FALSE(harness.inputs().empty());
    const auto& unpacked = harness.inputs();
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

// One output per element width the device fill handles (1, 2 and 4 bytes) and ones it
// leaves to the host (8 bytes: double and INT64), plus types whose sentinel is their
// largest value rather than a NaN. INT8, UINT8 and INT64 are built by the tensor
// factory but have no entry in the data type to native type mapping, which is what a
// dispatch keyed on the attribute's data type would trip over.
std::shared_ptr<IntegrationTestBundle> makeMixedTypeOutputBundle()
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {7};
    const std::vector<int64_t> strides = {1};
    const std::vector<DataType> types = {DataType::FLOAT,
                                         DataType::HALF,
                                         DataType::BFLOAT16,
                                         DataType::INT32,
                                         DataType::FP8_E4M3,
                                         DataType::DOUBLE,
                                         DataType::BOOLEAN,
                                         DataType::INT8,
                                         DataType::UINT8,
                                         DataType::INT64};

    auto bundle = std::make_shared<IntegrationTestBundle>();
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    for(size_t i = 0; i < types.size(); ++i)
    {
        const auto uid = static_cast<int64_t>(i) + 1;
        tensors.push_back(
            CreateTensorAttributesDirect(builder, uid, "output", types[i], &strides, &dims));
        bundle->outputTensorUids.push_back(uid);
    }

    const std::vector<flatbuffers::Offset<Node>> nodes;
    builder.Finish(CreateGraphDirect(builder,
                                     "mixed_types",
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     &tensors,
                                     &nodes));
    bundle->graphBuffer = builder.Release();
    return bundle;
}

// Writing the sentinel on the device instead of filling the host buffer and uploading
// it must leave every output holding the same bytes, or a tensor an engine never wrote
// would stop looking untouched.
TEST(TestBundleVerificationHarness, DeviceSentinelFillMatchesTheHostFill)
{
    SKIP_IF_NO_DEVICES();
    auto bundle = makeMixedTypeOutputBundle();
    const auto wrapper = bundle->graphWrapper();
    const auto& attributes = wrapper.getTensorMap();

    auto hostOutputs = detail::allocateSentinelOutputs(
        attributes, bundle->outputTensorUids, TensorMap{}, {}, /*onDevice=*/false);
    auto deviceOutputs = detail::allocateSentinelOutputs(
        attributes, bundle->outputTensorUids, TensorMap{}, {}, /*onDevice=*/true);

    for(const int64_t uid : bundle->outputTensorUids)
    {
        auto& expected = *hostOutputs.at(uid);
        auto& actual = *deviceOutputs.at(uid);
        ASSERT_EQ(expected.elementSpace(), actual.elementSpace()) << "uid " << uid;

        // The first non-const host access migrates the device-written bytes back.
        EXPECT_EQ(std::memcmp(expected.rawHostData(),
                              actual.rawHostData(),
                              expected.elementSpace() * expected.elementSize()),
                  0)
            << "uid " << uid;
    }
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

    EXPECT_FALSE(harness.inputs().empty());
    EXPECT_TRUE(harness.packedInputs().empty());
}

// A registered bundle outlives its tests, so what a run generates has to belong to the
// run: nothing is left on the bundle for the next one, and every run regenerates the
// same values from the same seed.
TEST_F(TestGoldenHarnessFixture, GeneratedInputsBelongToTheRunAndRepeatIdentically)
{
    auto bundle = makeMxMatmulBundle(hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3);

    testing_support::HarnessMocks firstMocks;
    IntegrationBundleVerificationHarness first(
        firstMocks.dependencies(testing_support::hostPolicy(VerificationMode::CPU)));
    fillOnHost(first, bundle);

    testing_support::HarnessMocks secondMocks;
    IntegrationBundleVerificationHarness second(
        secondMocks.dependencies(testing_support::hostPolicy(VerificationMode::CPU)));
    fillOnHost(second, bundle);

    EXPECT_FALSE(bundle->blobs.has_value());
    ASSERT_FALSE(first.inputs().empty());
    ASSERT_EQ(first.inputs().size(), second.inputs().size());
    for(const auto& [uid, tensor] : first.inputs())
    {
        auto& other = *second.inputs().at(uid);
        EXPECT_NE(tensor.get(), &other) << "uid " << uid;
        EXPECT_EQ(std::memcmp(tensor->rawHostData(),
                              other.rawHostData(),
                              tensor->elementSpace() * tensor->elementSize()),
                  0)
            << "uid " << uid;
    }
}

// Golden tensors are read by the run that needs them, so a bundle can be run again: the
// second run reads its own copy instead of finding the first one's.
TEST_F(TestGoldenHarnessFixture, GoldenBundleCanBeRunAgainBecauseEachRunReadsItsTensors)
{
    auto bundle = loadRunnableBundle("golden_run_twice");

    for(int run = 0; run < 2; ++run)
    {
        testing_support::HarnessMocks mocks;
        testing_support::engineWrites(
            mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);

        ::testing::TestPartResultArray results;
        runCapturing(mocks, bundle, &results);

        EXPECT_FALSE(testing_support::anyFailed(results)) << "run " << run;
        EXPECT_FALSE(testing_support::anySkipped(results)) << "run " << run;
    }
}

// A blob that cannot be read fails the test that needs it, with the reason, instead of
// the bundle having been dropped quietly when it was registered.
TEST_F(TestGoldenHarnessFixture, UnreadableGoldenBlobFailsTheRunWithTheReason)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);

    auto bundle = loadRunnableBundle("golden_bad_blob");
    ASSERT_TRUE(bundle->blobs.has_value());
    const auto blob = bundle->blobs->pathForUid(bundle->blobs->inputUids.front());
    std::ofstream(blob, std::ios::binary | std::ios::trunc) << "too short";

    ::testing::TestPartResultArray results;
    runCapturing(mocks, bundle, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("tensor data failed to load"),
              std::string::npos);
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
    ASSERT_FALSE(bundle->blobs.has_value());

    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_TRUE(testing_support::anySkipped(results));
    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("Engine could not execute bundle"),
              std::string::npos);
    EXPECT_TRUE(harness.inputs().empty());
}

TEST_F(TestGoldenHarnessFixture, RaggedBundleWithoutBlobsIsUnverifiable)
{
    testing_support::HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);

    auto bundle = makeRuntimePbvFillBundle();
    bundle->graphBuffer = test_utils::markFirstTensorRagged(bundle->graphBuffer.data());
    ASSERT_FALSE(bundle->blobs.has_value());

    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_TRUE(testing_support::anySkipped(results));
    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find(
                  "ragged inputs need exact offset values or golden blobs"),
              std::string::npos)
        << testing_support::allMessages(results);
    EXPECT_TRUE(harness.inputs().empty());
}

TEST_F(TestGoldenHarnessFixture, RaggedDeviceSentinelFillMatchesTheHostFill)
{
    SKIP_IF_NO_DEVICES();
    const auto bundle = loadRunnableRaggedBundle(
        "ragged_sentinel", std::vector<float>(fixtures::K_RAGGED_OUTPUT_ELEMS, 0.0f));
    const auto loadedTensors = bundle->loadTensors();
    const auto wrapper = bundle->graphWrapper();
    const auto& attributes = wrapper.getTensorMap();

    const auto lengthUids = hipdnn_test_sdk::utilities::raggedLengthUids(wrapper.getGraph());

    auto hostOutputs = detail::allocateSentinelOutputs(
        attributes, bundle->outputTensorUids, loadedTensors, lengthUids, /*onDevice=*/false);
    auto deviceOutputs = detail::allocateSentinelOutputs(
        attributes, bundle->outputTensorUids, loadedTensors, lengthUids, /*onDevice=*/true);

    for(const int64_t uid : bundle->outputTensorUids)
    {
        auto& expected = *hostOutputs.at(uid);
        auto& actual = *deviceOutputs.at(uid);
        ASSERT_NE(dynamic_cast<hipdnn_data_sdk::utilities::RaggedTensorBase<float>*>(&actual),
                  nullptr)
            << "uid " << uid;
        ASSERT_EQ(expected.elementSpace(), actual.elementSpace()) << "uid " << uid;

        EXPECT_EQ(std::memcmp(expected.rawHostData(),
                              actual.rawHostData(),
                              expected.elementSpace() * expected.elementSize()),
                  0)
            << "uid " << uid;
    }
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

TEST_F(TestGoldenHarnessFixture, RaggedMatchingOutputYieldsPass)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeRaggedOutput, fixtures::K_OUTPUT_VALUE);

    const std::vector<float> golden(fixtures::K_RAGGED_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);

    ::testing::TestPartResultArray results;
    runCapturing(mocks, loadRunnableRaggedBundle("ragged_match", golden), &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST_F(TestGoldenHarnessFixture, RaggedNaNGoldenRowIsNotCompared)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeRaggedOutput, fixtures::K_OUTPUT_VALUE);

    std::vector<float> golden(fixtures::K_RAGGED_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
    const auto batch1FirstRow = golden.begin() + 20;
    std::fill(batch1FirstRow, batch1FirstRow + 20, std::numeric_limits<float>::quiet_NaN());

    ::testing::TestPartResultArray results;
    runCapturing(mocks, loadRunnableRaggedBundle("ragged_nan_row", golden), &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST_F(TestGoldenHarnessFixture, RaggedInBlockMismatchYieldsFail)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeRaggedOutput, fixtures::K_OUTPUT_VALUE);

    std::vector<float> golden(fixtures::K_RAGGED_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
    golden[45] = fixtures::K_OUTPUT_VALUE + 100.0f;

    ::testing::TestPartResultArray results;
    runCapturing(mocks, loadRunnableRaggedBundle("ragged_mismatch", golden), &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

// An SDPA bundle whose Stats sits [H,T] with a gap between its batches (L7d) gives its
// outputs the elementSpace of the dense view only when the sentinel outputs are built with the
// node's seq_len lengths; the engine is then handed buffers that size and the golden
// comparison reads them.
TEST_F(TestGoldenHarnessFixture, RaggedSdpaSentinelOutputsTakeTheirLengthsFromSeqLen)
{
    const auto layout = fixtures::raggedSdpaGappedStatsCase();
    const auto dir = _tempDir / layout.name;
    fixtures::writeRaggedSdpaCase(dir, layout);
    const auto bundle = fixtures::loadWrittenBundle(dir, layout.name);
    const auto loadedTensors = bundle->loadTensors();
    const auto wrapper = bundle->graphWrapper();

    const auto outputs = detail::allocateSentinelOutputs(
        wrapper.getTensorMap(),
        bundle->outputTensorUids,
        loadedTensors,
        hipdnn_test_sdk::utilities::raggedLengthUids(wrapper.getGraph()),
        /*onDevice=*/false);

    for(const int64_t uid : {fixtures::K_SDPA_O_UID, fixtures::K_SDPA_STATS_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        const auto& output = outputs.at(uid);
        ASSERT_TRUE(output->raggedIterationInfo().has_value());
        EXPECT_EQ(output->raggedIterationInfo()->lengths, layout.lengths);
        EXPECT_EQ(output->elementSpace(),
                  uid == fixtures::K_SDPA_STATS_UID ? layout.statsSpace : layout.qkvoSpace);
    }
}

TEST_F(TestGoldenHarnessFixture, RaggedSdpaMatchingOutputYieldsPass)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeRaggedSdpaGappedOutputs, 0.0f);

    const auto layout = fixtures::raggedSdpaGappedStatsCase();
    const auto dir = _tempDir / layout.name;
    fixtures::writeRaggedSdpaCase(dir, layout);

    ::testing::TestPartResultArray results;
    runCapturing(mocks, fixtures::loadWrittenBundle(dir, layout.name), &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
}

// A VALUES recipe that cannot fill its tensor is an error in the bundle. The test fails
// before the engine runs and does not skip.
TEST_F(TestGoldenHarnessFixture, RejectsInvalidValuesFill)
{
    testing_support::HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);

    // An accepted claim for the engine under test makes the run publish a verdict
    // whose detail names the failure origin.
    const std::string engineName = "ENGINE_UNDER_TEST";
    SupportResult accepted;
    accepted.verdict = SupportVerdict::CLAIM_ACCEPTED;
    accepted.bundlePath = "invalid-values-fill";
    accepted.engineName = engineName;
    accepted.arch = "gfx942";
    accepted.platform = "linux";
    ON_CALL(mocks.claimObserver,
            observe(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(SupportObservation{SidecarState::CHECKED, {accepted}}));
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(mocks.reporter, verdicts);

    const auto sidecar = _tempDir / "invalid-values-fill.support.json";
    std::ofstream(sidecar) << R"({"version": 1, "claims": {}})";
    SupportClaimLocator locator;
    locator.sidecarPath = sidecar;
    locator.diagnosticPath = "invalid-values-fill";

    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)),
        LoadedEngine{11, engineName});
    harness.setBundle(makeRuntimePbvFillBundle(), "invalid-values-fill", locator);
    // x has 6 elements; an exact list of 7 cannot fill it.
    harness.inputFillRecipes().set(
        1, FillRecipe::values(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7}, /*repeat=*/false));

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
    const auto messages = testing_support::allMessages(results);
    EXPECT_NE(messages.find("input fill failed: uid 1: "), std::string::npos) << messages;
    EXPECT_NE(messages.find("invalid-values-fill"), std::string::npos) << messages;

    ASSERT_EQ(verdicts.size(), 1u);
    EXPECT_EQ(verdicts.front().reachedDepth, VerificationDepth::NOT_REACHED);
    const std::string origin
        = std::string("failed outside the engine (") + toString(FailureOrigin::HARNESS) + ")";
    EXPECT_NE(verdicts.front().detail.find(origin), std::string::npos) << verdicts.front().detail;
}

TEST_F(TestGoldenHarnessFixture, UnsupportedFillStaysUnverifiable)
{
    testing_support::HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);

    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
    harness.setBundle(makeUnfillableBundle(), "unit-test-bundle");

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_TRUE(testing_support::anySkipped(results));
    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("no input fill registered"),
              std::string::npos)
        << testing_support::allMessages(results);
}

// A sub-byte graph is filled twice, unpacked and packed. A VALUES fill of one of its
// other tensors gives the same contents in both sets.
TEST_F(TestGoldenHarnessFixture, ValuesFillOfASubByteGraphMatchesInBothInputSets)
{
    testing_support::HarnessMocks mocks;
    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::CPU)));
    harness.setBundle(makeMxMatmulBundle(hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1),
                      "unit-test-bundle");
    harness.inputFillRecipes().set(2, FillRecipe::values(std::vector<int64_t>{1, 2}));

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    ASSERT_EQ(harness.inputs().count(2), 1u);
    ASSERT_EQ(harness.packedInputs().count(2), 1u);
    auto& unpacked = *harness.inputs().at(2);
    auto& packed = *harness.packedInputs().at(2);

    std::vector<float> scales;
    for(auto* element : unpacked)
    {
        scales.push_back(
            static_cast<float>(*static_cast<hipdnn_data_sdk::types::fp8_e8m0*>(element)));
    }
    ASSERT_GE(scales.size(), 2u);
    EXPECT_EQ(scales[0], 1.0f);
    EXPECT_EQ(scales[1], 2.0f);
    EXPECT_EQ(std::memcmp(unpacked.rawHostData(),
                          packed.rawHostData(),
                          unpacked.elementSpace() * unpacked.elementSize()),
              0);
}

// ---- aliases ---------------------------------------------------------------

namespace
{

constexpr int64_t K_ALIAS_OWNER_UID = 1;
constexpr int64_t K_ALIAS_VIEW_UID = 2;
constexpr int64_t K_ALIAS_OUTPUT_UID = 3;
// The view starts at the second row of the owner.
constexpr int64_t K_ALIAS_BYTE_OFFSET = 3 * static_cast<int64_t>(sizeof(float));

const std::vector<int64_t> K_ALIAS_OWNER_DIMS = {2, 3};
const std::vector<int64_t> K_ALIAS_VIEW_DIMS = {1, 3};
const std::vector<int64_t> K_ALIAS_STRIDES = {3, 1};

// uids: x=1 [2, 3] (the owner), w=2 [1, 3] (the view), y=3 [2, 3] (the output).
std::shared_ptr<IntegrationTestBundle> makeAliasedConvBundle(std::vector<TensorAlias> aliases)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(
        builder, K_ALIAS_OWNER_UID, "x", DataType::FLOAT, &K_ALIAS_STRIDES, &K_ALIAS_OWNER_DIMS));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, K_ALIAS_VIEW_UID, "w", DataType::FLOAT, &K_ALIAS_STRIDES, &K_ALIAS_VIEW_DIMS));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, K_ALIAS_OUTPUT_UID, "y", DataType::FLOAT, &K_ALIAS_STRIDES, &K_ALIAS_OWNER_DIMS));

    const auto conv = CreateConvolutionFwdAttributesDirect(
        builder, K_ALIAS_OWNER_UID, K_ALIAS_VIEW_UID, K_ALIAS_OUTPUT_UID);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        builder, "conv", DataType::FLOAT, NodeAttributes::ConvolutionFwdAttributes, conv.Union()));
    builder.Finish(CreateGraphDirect(
        builder, "aliased", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes));

    auto bundle = std::make_shared<IntegrationTestBundle>();
    bundle->graphBuffer = builder.Release();
    bundle->outputTensorUids = {K_ALIAS_OUTPUT_UID};
    bundle->metadata.formatVersion = 2;
    bundle->metadata.aliases = std::move(aliases);
    return bundle;
}

std::vector<TensorAlias> viewOfOwnersSecondRow()
{
    return {{K_ALIAS_VIEW_UID, K_ALIAS_OWNER_UID, K_ALIAS_BYTE_OFFSET}};
}

// The owner's tensor alone, filled with 0, 1, 2, ... so each element is distinct.
TensorMap makeAliasOwnerInputs(const IntegrationTestBundle& bundle)
{
    const auto wrapper = bundle.graphWrapper();
    TensorMap inputs;
    inputs.emplace(K_ALIAS_OWNER_UID,
                   hipdnn_test_sdk::detail::createTensorFromAttribute(
                       *wrapper.getTensorMap().at(K_ALIAS_OWNER_UID)));
    auto* data = static_cast<float*>(inputs.at(K_ALIAS_OWNER_UID)->rawHostData());
    for(std::size_t i = 0; i < inputs.at(K_ALIAS_OWNER_UID)->elementSpace(); ++i)
    {
        data[i] = static_cast<float>(i);
    }
    inputs.at(K_ALIAS_OWNER_UID)->markHostModified();
    return inputs;
}

void* advanced(void* pointer, int64_t bytes)
{
    return static_cast<std::byte*>(pointer) + bytes;
}

// The view's elements, read through `viewPointer` with the view's own layout, are the
// elements of the owner's second row.
void expectViewReadsTheOwnersSecondRow(void* viewPointer,
                                       const hipdnn_data_sdk::utilities::ITensor& owner)
{
    const hipdnn_data_sdk::utilities::ShallowTensor<float> view(
        viewPointer, K_ALIAS_VIEW_DIMS, K_ALIAS_STRIDES);
    const auto& ownerTensor
        = dynamic_cast<const hipdnn_data_sdk::utilities::TensorBase<float>&>(owner);
    for(int64_t column = 0; column < K_ALIAS_VIEW_DIMS[1]; ++column)
    {
        EXPECT_EQ(view.getHostValue(int64_t{0}, column),
                  ownerTensor.getHostValue(int64_t{1}, column))
            << "column " << column;
    }
}

} // namespace

TEST(TestBundleVerificationHarness, AliasVariantPackPointsTheViewIntoTheOwnerOnTheHost)
{
    const auto bundle = makeAliasedConvBundle(viewOfOwnersSecondRow());
    const auto wrapper = bundle->graphWrapper();
    auto inputs = makeAliasOwnerInputs(*bundle);
    OutputTensors outputs;

    const auto variantPack = detail::buildVariantPack(inputs,
                                                      outputs,
                                                      wrapper.getTensorMap(),
                                                      bundle->outputTensorUids,
                                                      /*useDevice=*/false,
                                                      bundle->metadata.aliases);

    ASSERT_EQ(variantPack.count(K_ALIAS_VIEW_UID), 1u);
    EXPECT_EQ(variantPack.at(K_ALIAS_VIEW_UID),
              advanced(inputs.at(K_ALIAS_OWNER_UID)->rawHostData(), K_ALIAS_BYTE_OFFSET));
    expectViewReadsTheOwnersSecondRow(variantPack.at(K_ALIAS_VIEW_UID),
                                      *inputs.at(K_ALIAS_OWNER_UID));
}

TEST(TestBundleVerificationHarness, AliasVariantPackPointsTheViewIntoTheOwnerOnTheDevice)
{
    SKIP_IF_NO_DEVICES();
    const auto bundle = makeAliasedConvBundle(viewOfOwnersSecondRow());
    const auto wrapper = bundle->graphWrapper();
    auto inputs = makeAliasOwnerInputs(*bundle);
    OutputTensors outputs;

    const auto variantPack = detail::buildVariantPack(inputs,
                                                      outputs,
                                                      wrapper.getTensorMap(),
                                                      bundle->outputTensorUids,
                                                      /*useDevice=*/true,
                                                      bundle->metadata.aliases);

    ASSERT_EQ(variantPack.count(K_ALIAS_VIEW_UID), 1u);
    EXPECT_EQ(variantPack.at(K_ALIAS_VIEW_UID),
              advanced(inputs.at(K_ALIAS_OWNER_UID)->rawDeviceData(), K_ALIAS_BYTE_OFFSET));
}

// uid 4 is not a tensor of the graph; the builder needs only the owner's entry.
TEST(TestBundleVerificationHarness, AliasVariantPackGivesEachViewOfOneOwnerItsOffset)
{
    const auto bundle = makeAliasedConvBundle({});
    const auto wrapper = bundle->graphWrapper();
    auto inputs = makeAliasOwnerInputs(*bundle);
    OutputTensors outputs;

    for(const auto& aliases :
        {std::vector<TensorAlias>{{K_ALIAS_VIEW_UID, K_ALIAS_OWNER_UID, K_ALIAS_BYTE_OFFSET},
                                  {4, K_ALIAS_OWNER_UID, 0}},
         std::vector<TensorAlias>{{4, K_ALIAS_OWNER_UID, 0},
                                  {K_ALIAS_VIEW_UID, K_ALIAS_OWNER_UID, K_ALIAS_BYTE_OFFSET}}})
    {
        SCOPED_TRACE("first view " + std::to_string(aliases.front().uid));
        const auto variantPack = detail::buildVariantPack(inputs,
                                                          outputs,
                                                          wrapper.getTensorMap(),
                                                          bundle->outputTensorUids,
                                                          /*useDevice=*/false,
                                                          aliases);
        auto* ownerData = inputs.at(K_ALIAS_OWNER_UID)->rawHostData();
        EXPECT_EQ(variantPack.at(4), ownerData);
        EXPECT_EQ(variantPack.at(K_ALIAS_VIEW_UID), advanced(ownerData, K_ALIAS_BYTE_OFFSET));
        expectViewReadsTheOwnersSecondRow(variantPack.at(K_ALIAS_VIEW_UID),
                                          *inputs.at(K_ALIAS_OWNER_UID));
    }
}

TEST(TestBundleVerificationHarness, AliasVariantPackThrowsForAMissingOwnerOrAViewWithABuffer)
{
    const auto bundle = makeAliasedConvBundle({});
    const auto wrapper = bundle->graphWrapper();
    auto inputs = makeAliasOwnerInputs(*bundle);
    OutputTensors outputs;

    const std::vector<TensorAlias> missingOwner = {{K_ALIAS_VIEW_UID, 99, 0}};
    EXPECT_THROW(detail::buildVariantPack(inputs,
                                          outputs,
                                          wrapper.getTensorMap(),
                                          bundle->outputTensorUids,
                                          /*useDevice=*/false,
                                          missingOwner),
                 std::invalid_argument);

    inputs.emplace(K_ALIAS_VIEW_UID,
                   hipdnn_test_sdk::detail::createTensorFromAttribute(
                       *wrapper.getTensorMap().at(K_ALIAS_VIEW_UID)));
    EXPECT_THROW(detail::buildVariantPack(inputs,
                                          outputs,
                                          wrapper.getTensorMap(),
                                          bundle->outputTensorUids,
                                          /*useDevice=*/false,
                                          viewOfOwnersSecondRow()),
                 std::invalid_argument);
}

// The engine and the reference get the same view pointer, into the one owner that the
// harness filled; the view has no tensor of its own.
TEST_F(TestGoldenHarnessFixture, AliasViewPointsIntoTheOwnerForTheEngineAndTheReference)
{
    testing_support::HarnessMocks mocks;
    VariantPack enginePack;
    VariantPack referencePack;
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([&enginePack](GraphSession&,
                                     const std::optional<LoadedEngine>&,
                                     VariantPack& variantPack) {
            enginePack = variantPack;
            return EngineOpResult::succeeded();
        });
    ON_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([&referencePack](void*, std::size_t, const VariantPack& variantPack) {
            referencePack = variantPack;
        });

    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::CPU)));
    harness.setBundle(makeAliasedConvBundle(viewOfOwnersSecondRow()), "unit-test-bundle");

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_EQ(harness.inputs().count(K_ALIAS_VIEW_UID), 0u);
    ASSERT_EQ(harness.inputs().count(K_ALIAS_OWNER_UID), 1u);
    ASSERT_EQ(enginePack.count(K_ALIAS_VIEW_UID), 1u);
    ASSERT_EQ(referencePack.count(K_ALIAS_VIEW_UID), 1u);

    auto& owner = *harness.inputs().at(K_ALIAS_OWNER_UID);
    EXPECT_EQ(enginePack.at(K_ALIAS_OWNER_UID), owner.rawHostData());
    EXPECT_EQ(enginePack.at(K_ALIAS_VIEW_UID), advanced(owner.rawHostData(), K_ALIAS_BYTE_OFFSET));
    EXPECT_EQ(referencePack.at(K_ALIAS_VIEW_UID), enginePack.at(K_ALIAS_VIEW_UID));
    expectViewReadsTheOwnersSecondRow(enginePack.at(K_ALIAS_VIEW_UID), owner);
}

namespace
{

// uids of makeMxMatmulBundle(): scale_b (5) as a view of scale_a (2). Both are FP8_E8M0
// with 128 elements; x_a and x_b make it a sub-byte graph.
const std::vector<TensorAlias> K_SCALE_B_VIEW_OF_SCALE_A = {{5, 2, 0}};

} // namespace

TEST_F(TestGoldenHarnessFixture, AliasInASubByteGraphHasNoTensorInEitherInputSet)
{
    testing_support::HarnessMocks mocks;
    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::CPU)));
    auto bundle = makeMxMatmulBundle(hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1);
    bundle->metadata.formatVersion = 2;
    bundle->metadata.aliases = K_SCALE_B_VIEW_OF_SCALE_A;
    fillOnHost(harness, bundle);

    EXPECT_EQ(harness.inputs().count(5), 0u);
    EXPECT_EQ(harness.packedInputs().count(5), 0u);
    EXPECT_EQ(harness.inputs().count(2), 1u);
    EXPECT_EQ(harness.packedInputs().count(2), 1u);
}

TEST_F(TestGoldenHarnessFixture, AliasInASubByteGraphPointsTheDeviceViewIntoThePackedOwner)
{
    SKIP_IF_NO_DEVICES();
    testing_support::HarnessMocks mocks;
    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::CPU)));
    auto bundle = makeMxMatmulBundle(hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1);
    bundle->metadata.formatVersion = 2;
    bundle->metadata.aliases = K_SCALE_B_VIEW_OF_SCALE_A;
    fillOnHost(harness, bundle);
    ASSERT_EQ(harness.packedInputs().count(2), 1u);

    TensorMap packed = harness.packedInputs();
    OutputTensors outputs;
    const auto wrapper = bundle->graphWrapper();
    const std::vector<TensorAlias> offsetView = {{5, 2, 32}};
    const auto variantPack = detail::buildVariantPack(packed,
                                                      outputs,
                                                      wrapper.getTensorMap(),
                                                      bundle->outputTensorUids,
                                                      /*useDevice=*/true,
                                                      offsetView);
    ASSERT_EQ(variantPack.count(5), 1u);
    EXPECT_EQ(variantPack.at(5), advanced(packed.at(2)->rawDeviceData(), 32));
}

namespace
{

// Runs a bundle built in the test, without the load-time alias checks, and expects a
// harness failure that names the bundle.
void expectAliasFillFailsTheHarness(std::vector<TensorAlias> aliases, const std::string& reason)
{
    testing_support::HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);
    std::vector<std::string> unverifiable;
    testing_support::captureUnverifiable(mocks.reporter, unverifiable);

    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
    harness.setBundle(makeAliasedConvBundle(std::move(aliases)), "aliased-bundle");

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
    EXPECT_TRUE(unverifiable.empty());
    const auto messages = testing_support::allMessages(results);
    EXPECT_NE(messages.find("input fill failed"), std::string::npos) << messages;
    EXPECT_NE(messages.find(reason), std::string::npos) << messages;
    EXPECT_NE(messages.find("aliased-bundle"), std::string::npos) << messages;
}

} // namespace

// fillInputs() throws when an owner has no input tensor: here the owner is the output.
TEST_F(TestGoldenHarnessFixture, AliasFillThatThrowsFailsTheHarness)
{
    expectAliasFillFailsTheHarness({{K_ALIAS_VIEW_UID, K_ALIAS_OUTPUT_UID, 0}},
                                   "the owner has no input tensor to fill");
}

TEST_F(TestGoldenHarnessFixture, AliasWithAMissingOwnerFailsTheHarness)
{
    expectAliasFillFailsTheHarness({{K_ALIAS_VIEW_UID, 99, 0}},
                                   "alias view 2 of owner 99: the owner has no input tensor");
}

// The fill accepts a view that is an output, since it fills only owners; the variant
// pack then refuses it, and the run fails in the harness before the engine runs.
TEST_F(TestGoldenHarnessFixture, AliasVariantPackErrorFailsTheHarness)
{
    testing_support::HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);
    std::vector<std::string> unverifiable;
    testing_support::captureUnverifiable(mocks.reporter, unverifiable);

    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
    harness.setBundle(makeAliasedConvBundle({{K_ALIAS_OUTPUT_UID, K_ALIAS_OWNER_UID, 0}}),
                      "aliased-bundle");

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
    EXPECT_TRUE(unverifiable.empty());
    const auto messages = testing_support::allMessages(results);
    EXPECT_NE(messages.find("buildVariantPack: alias view 3 of owner 1"), std::string::npos)
        << messages;
    EXPECT_NE(messages.find("aliased-bundle"), std::string::npos) << messages;
}

TEST_F(TestGoldenHarnessFixture, AliasWithGoldenBlobsFailsTheHarness)
{
    testing_support::HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);

    auto bundle = loadRunnableBundle("aliased_golden");
    ASSERT_TRUE(bundle->blobs.has_value());
    bundle->metadata.aliases = {{2, 1, 0}};

    ::testing::TestPartResultArray results;
    runCapturing(mocks, bundle, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
    EXPECT_NE(testing_support::allMessages(results).find("aliases cannot be used with tensor data"),
              std::string::npos)
        << testing_support::allMessages(results);
}

// ---- start_offsets ---------------------------------------------------------

namespace
{

constexpr int64_t K_SHIFTED_INPUT_OFFSET = 2;
constexpr int64_t K_SHIFTED_OUTPUT_OFFSET = 3;
constexpr std::size_t K_ALIAS_OUTPUT_ELEMS = 6;
constexpr std::size_t K_ALIAS_OWNER_BYTES = 6 * sizeof(float);
constexpr std::size_t K_ALIAS_VIEW_BYTES = 3 * sizeof(float);

// Bytes from the last address aligned to K_START_OFFSET_BASE_ALIGNMENT to `pointer`.
int64_t bytesPastAlignment(const void* pointer)
{
    return static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(pointer)
                                % static_cast<std::uintptr_t>(K_START_OFFSET_BASE_ALIGNMENT));
}

// Writes `count` copies of `value` from `pointer` on. The pointer need not be aligned
// for a float.
void writeFloats(void* pointer, std::size_t count, float value)
{
    for(std::size_t i = 0; i < count; ++i)
    {
        std::memcpy(static_cast<std::byte*>(pointer) + (i * sizeof(float)), &value, sizeof(value));
    }
}

std::vector<std::byte> copyBytes(const void* pointer, std::size_t bytes)
{
    std::vector<std::byte> copy(bytes);
    std::memcpy(copy.data(), pointer, bytes);
    return copy;
}

std::size_t byteCount(const hipdnn_data_sdk::utilities::ITensor& tensor)
{
    return tensor.elementSpace() * tensor.elementSize();
}

float floatAt(hipdnn_data_sdk::utilities::ITensor& tensor, std::size_t index)
{
    float value = 0.0f;
    std::memcpy(&value,
                static_cast<const std::byte*>(tensor.rawHostData()) + (index * sizeof(float)),
                sizeof(value));
    return value;
}

// The aliased conv bundle with the owner and the output moved.
std::shared_ptr<IntegrationTestBundle> makeShiftedConvBundle(std::vector<TensorAlias> aliases = {})
{
    auto bundle = makeAliasedConvBundle(std::move(aliases));
    bundle->metadata.startOffsets = {{K_ALIAS_OWNER_UID, K_SHIFTED_INPUT_OFFSET},
                                     {K_ALIAS_OUTPUT_UID, K_SHIFTED_OUTPUT_OFFSET}};
    return bundle;
}

// The golden batchnorm bundle with x (uid 0) and the output moved.
void shiftGoldenBundle(IntegrationTestBundle& bundle)
{
    bundle.metadata.formatVersion = 2;
    bundle.metadata.startOffsets
        = {{0, K_SHIFTED_INPUT_OFFSET}, {fixtures::K_OUTPUT_UID, K_SHIFTED_OUTPUT_OFFSET}};
}

// The mocked engine writes `value` into every element of the golden bundle's output and
// records the offsets past alignment of x (uid 0) and of the output.
void engineWritesThroughAnyPointer(testing_support::HarnessMocks& mocks,
                                   float value,
                                   std::vector<int64_t>& engineOffsets)
{
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([value, &engineOffsets](GraphSession&,
                                               const std::optional<LoadedEngine>&,
                                               VariantPack& variantPack) {
            engineOffsets.push_back(bytesPastAlignment(variantPack.at(0)));
            engineOffsets.push_back(bytesPastAlignment(variantPack.at(fixtures::K_OUTPUT_UID)));
            writeFloats(variantPack.at(fixtures::K_OUTPUT_UID), fixtures::K_OUTPUT_ELEMS, value);
            return EngineOpResult::succeeded();
        });
}

} // namespace

// The engine reads a moved copy of each input in start_offsets and writes a moved output
// that reaches the comparison; the reference gets the tensors themselves.
TEST_F(TestGoldenHarnessFixture, StartOffsetEngineGetsMovedCopiesAndTheReferenceGetsTheTensors)
{
    testing_support::HarnessMocks mocks;
    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::CPU)));
    harness.setBundle(makeShiftedConvBundle(), "unit-test-bundle");

    const auto ownerData
        = [&harness] { return harness.inputs().at(K_ALIAS_OWNER_UID)->rawHostData(); };
    std::vector<int64_t> engineOffsets;
    bool engineOwnerIsTheTensor = true;
    bool engineOwnerBytesMatch = false;
    bool referenceOwnerIsTheTensor = false;
    int64_t referenceOutputOffset = -1;
    constexpr float K_VALUE = 2.5f;
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([&](GraphSession&, const std::optional<LoadedEngine>&, VariantPack& pack) {
            engineOffsets.push_back(bytesPastAlignment(pack.at(K_ALIAS_OWNER_UID)));
            engineOffsets.push_back(bytesPastAlignment(pack.at(K_ALIAS_OUTPUT_UID)));
            engineOwnerIsTheTensor = pack.at(K_ALIAS_OWNER_UID) == ownerData();
            engineOwnerBytesMatch = copyBytes(pack.at(K_ALIAS_OWNER_UID), K_ALIAS_OWNER_BYTES)
                                    == copyBytes(ownerData(), K_ALIAS_OWNER_BYTES);
            writeFloats(pack.at(K_ALIAS_OUTPUT_UID), K_ALIAS_OUTPUT_ELEMS, K_VALUE);
            return EngineOpResult::succeeded();
        });
    ON_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([&](void*, std::size_t, const VariantPack& pack) {
            referenceOwnerIsTheTensor = pack.at(K_ALIAS_OWNER_UID) == ownerData();
            referenceOutputOffset = bytesPastAlignment(pack.at(K_ALIAS_OUTPUT_UID));
            writeFloats(pack.at(K_ALIAS_OUTPUT_UID), K_ALIAS_OUTPUT_ELEMS, K_VALUE);
        });

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_EQ(engineOffsets,
              (std::vector<int64_t>{K_SHIFTED_INPUT_OFFSET, K_SHIFTED_OUTPUT_OFFSET}));
    EXPECT_FALSE(engineOwnerIsTheTensor);
    EXPECT_TRUE(engineOwnerBytesMatch);
    EXPECT_TRUE(referenceOwnerIsTheTensor);
    // A host float buffer is at least 4-byte aligned, so only a moved copy has offset 3.
    EXPECT_NE(referenceOutputOffset, K_SHIFTED_OUTPUT_OFFSET);
    EXPECT_NE(referenceOutputOffset, -1);
}

// A view of a moved owner points into the owner's moved copy for the engine, and into
// the owner itself for the reference.
TEST_F(TestGoldenHarnessFixture, StartOffsetViewOfAMovedOwnerMovesWithItForTheEngineOnly)
{
    testing_support::HarnessMocks mocks;
    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::CPU)));
    harness.setBundle(makeShiftedConvBundle(viewOfOwnersSecondRow()), "unit-test-bundle");

    const auto ownersSecondRow = [&harness] {
        return advanced(harness.inputs().at(K_ALIAS_OWNER_UID)->rawHostData(), K_ALIAS_BYTE_OFFSET);
    };
    int64_t engineOwnerOffset = -1;
    bool engineViewFollowsTheOwner = false;
    bool engineViewBytesMatch = false;
    bool referenceViewIsInTheTensor = false;
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([&](GraphSession&, const std::optional<LoadedEngine>&, VariantPack& pack) {
            engineOwnerOffset = bytesPastAlignment(pack.at(K_ALIAS_OWNER_UID));
            engineViewFollowsTheOwner
                = pack.at(K_ALIAS_VIEW_UID)
                  == advanced(pack.at(K_ALIAS_OWNER_UID), K_ALIAS_BYTE_OFFSET);
            engineViewBytesMatch = copyBytes(pack.at(K_ALIAS_VIEW_UID), K_ALIAS_VIEW_BYTES)
                                   == copyBytes(ownersSecondRow(), K_ALIAS_VIEW_BYTES);
            return EngineOpResult::succeeded();
        });
    ON_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([&](void*, std::size_t, const VariantPack& pack) {
            referenceViewIsInTheTensor = pack.at(K_ALIAS_VIEW_UID) == ownersSecondRow();
        });

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_EQ(engineOwnerOffset, K_SHIFTED_INPUT_OFFSET);
    EXPECT_TRUE(engineViewFollowsTheOwner);
    EXPECT_TRUE(engineViewBytesMatch);
    EXPECT_TRUE(referenceViewIsInTheTensor);
}

TEST_F(TestGoldenHarnessFixture, StartOffsetOutputThatMatchesTheGoldenDataPasses)
{
    testing_support::HarnessMocks mocks;
    std::vector<int64_t> engineOffsets;
    engineWritesThroughAnyPointer(mocks, fixtures::K_OUTPUT_VALUE, engineOffsets);
    auto bundle = loadRunnableBundle("shifted_match");
    shiftGoldenBundle(*bundle);

    ::testing::TestPartResultArray results;
    runCapturing(mocks, bundle, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
    EXPECT_EQ(engineOffsets,
              (std::vector<int64_t>{K_SHIFTED_INPUT_OFFSET, K_SHIFTED_OUTPUT_OFFSET}));
}

TEST_F(TestGoldenHarnessFixture, StartOffsetOutputThatDiffersFromTheGoldenDataFails)
{
    testing_support::HarnessMocks mocks;
    std::vector<int64_t> engineOffsets;
    engineWritesThroughAnyPointer(mocks, fixtures::K_OUTPUT_VALUE + 100.0f, engineOffsets);
    auto bundle = loadRunnableBundle("shifted_mismatch");
    shiftGoldenBundle(*bundle);

    ::testing::TestPartResultArray results;
    runCapturing(mocks, bundle, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST_F(TestGoldenHarnessFixture, StartOffsetBenchmarkGetsTheEnginePointers)
{
    testing_support::HarnessMocks mocks;
    std::vector<int64_t> engineOffsets;
    engineWritesThroughAnyPointer(mocks, fixtures::K_OUTPUT_VALUE, engineOffsets);
    std::vector<int64_t> benchmarkOffsets;
    ON_CALL(mocks.engineRunner, benchmark(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(
            [&benchmarkOffsets](GraphSession&, VariantPack& pack, const BenchmarkSettings&) {
                benchmarkOffsets.push_back(bytesPastAlignment(pack.at(0)));
                benchmarkOffsets.push_back(bytesPastAlignment(pack.at(fixtures::K_OUTPUT_UID)));
                hipdnn_frontend::AutotuneResult result;
                result.engineName = "TEST_ENGINE";
                result.succeeded = true;
                return result;
            });

    auto policy = testing_support::hostPolicy(VerificationMode::AUTO);
    policy.benchmark = BenchmarkSettings{};
    IntegrationBundleVerificationHarness harness(mocks.dependencies(policy));
    auto bundle = loadRunnableBundle("shifted_benchmark");
    shiftGoldenBundle(*bundle);
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_EQ(benchmarkOffsets,
              (std::vector<int64_t>{K_SHIFTED_INPUT_OFFSET, K_SHIFTED_OUTPUT_OFFSET}));
}

// benchmarkIfVerified() used to return right after calling the engine runner's
// benchmark(), without the waitForEngine()/copyOutputsBack() sequence runEngine() does.
// This exercises that exact path on real gfx942 device memory with a forced-to-fail
// benchmark (every candidate engine declining -- the case where the missing wait
// matters, since a real autotune pass only waits internally when some candidate
// succeeds), with start offsets in play so the benchmark's own staged device copies are
// real, not mocked.
//
// NOTE ON COVERAGE: an earlier version of this test tried to prove the missing wait via
// a StallGate-held stream, expecting the unfixed code to observably return before the
// stream drained. That signal turned out to be masked: StartOffsetStaging's device
// copies are freed by hipFree() when `staging` goes out of scope at the end of
// benchmarkIfVerified() either way, and hipFree() performs its own implicit device-wide
// synchronization before freeing (see hipdnn_data_sdk::utilities::StallGate's own
// destructor comment) -- so by the time benchmarkIfVerified() returns, the device is
// synchronized whether or not the explicit wait ran. The fix's real, currently
// unobservable-from-outside effect is copyOutputsBack() propagating the benchmark's
// write back into the real output tensor (benchmarkIfVerified() builds a fresh staged
// copy of it); nothing in the current public harness API exposes that tensor's
// post-benchmark value to a test. What this test still verifies directly: the exact
// code path runs on real device memory with a forced failure and leaves the device
// clean -- no crash, no leaked HIP error state.
TEST_F(TestGoldenHarnessFixture, BenchmarkFailureDoesNotCrashOrLeaveTheDeviceErrored)
{
    SKIP_IF_NO_DEVICES();

    testing_support::HarnessMocks mocks;
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([](GraphSession&, const std::optional<LoadedEngine>&, VariantPack& pack) {
            const std::vector<float> output(fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
            const auto status = hipMemcpy(pack.at(fixtures::K_OUTPUT_UID),
                                          output.data(),
                                          output.size() * sizeof(float),
                                          hipMemcpyHostToDevice);
            return status == hipSuccess ? EngineOpResult::succeeded()
                                        : EngineOpResult::failed(hipGetErrorString(status));
        });
    ON_CALL(mocks.engineRunner, benchmark(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([](GraphSession&, VariantPack& pack, const BenchmarkSettings&) {
            // A real candidate still does device work before declining; write through
            // the (possibly moved) output pointer so there is something for
            // waitForEngine()/copyOutputsBack() to actually act on.
            const std::vector<float> output(fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE + 1);
            EXPECT_EQ(hipMemcpy(pack.at(fixtures::K_OUTPUT_UID),
                                output.data(),
                                output.size() * sizeof(float),
                                hipMemcpyHostToDevice),
                      hipSuccess);
            hipdnn_frontend::AutotuneResult failed;
            failed.errorMessage = "forced failure: every candidate engine declined";
            failed.succeeded = false;
            return failed;
        });

    auto policy = testing_support::hostPolicy(VerificationMode::AUTO);
    policy.placement = TensorPlacement::DEVICE;
    policy.benchmark = BenchmarkSettings{};
    IntegrationBundleVerificationHarness harness(mocks.dependencies(policy));
    auto bundle = loadRunnableBundle("shifted_benchmark_forced_fail");
    shiftGoldenBundle(*bundle);
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    // reportBenchmark() fails the test on an unsuccessful AutotuneResult; that is
    // expected and unrelated to what this test pins.
    EXPECT_TRUE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results)) << testing_support::allMessages(results);
    EXPECT_NE(testing_support::allMessages(results).find("benchmark failed: forced failure"),
              std::string::npos)
        << testing_support::allMessages(results);

    // The device must come out of this clean: no crash reaching here at all, and no
    // HIP error left pending from freeing the staged start-offset buffers while the
    // mocked candidate's device write might still have been in flight.
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(hipGetLastError(), hipSuccess);
}

// A staging fault happens before the engine runs, so it fails the run in the harness and
// names the bundle. A view has no buffer of its own to move.
TEST_F(TestGoldenHarnessFixture, StartOffsetStagingErrorFailsTheHarnessNotTheEngine)
{
    testing_support::HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);
    std::vector<std::string> unverifiable;
    testing_support::captureUnverifiable(mocks.reporter, unverifiable);

    // An accepted claim for the engine under test makes the run publish a verdict whose
    // detail names the failure origin.
    const std::string engineName = "ENGINE_UNDER_TEST";
    SupportResult accepted;
    accepted.verdict = SupportVerdict::CLAIM_ACCEPTED;
    accepted.bundlePath = "shifted-bundle";
    accepted.engineName = engineName;
    accepted.arch = "gfx942";
    accepted.platform = "linux";
    ON_CALL(mocks.claimObserver,
            observe(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(SupportObservation{SidecarState::CHECKED, {accepted}}));
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(mocks.reporter, verdicts);

    const auto sidecar = _tempDir / "shifted-bundle.support.json";
    std::ofstream(sidecar) << R"({"version": 1, "claims": {}})";
    SupportClaimLocator locator;
    locator.sidecarPath = sidecar;
    locator.diagnosticPath = "shifted-bundle";

    auto bundle = makeAliasedConvBundle(viewOfOwnersSecondRow());
    bundle->metadata.startOffsets = {{K_ALIAS_VIEW_UID, K_SHIFTED_INPUT_OFFSET}};
    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::CPU)),
        LoadedEngine{11, engineName});
    harness.setBundle(bundle, "shifted-bundle", locator);

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
    EXPECT_TRUE(unverifiable.empty());
    const auto messages = testing_support::allMessages(results);
    EXPECT_NE(messages.find("start offset of tensor 2: the tensor has no buffer of its own"),
              std::string::npos)
        << messages;
    EXPECT_NE(messages.find("shifted-bundle"), std::string::npos) << messages;

    ASSERT_EQ(verdicts.size(), 1u);
    EXPECT_EQ(verdicts.front().reachedDepth, VerificationDepth::NOT_REACHED);
    const std::string origin
        = std::string("failed outside the engine (") + toString(FailureOrigin::HARNESS) + ")";
    EXPECT_NE(verdicts.front().detail.find(origin), std::string::npos) << verdicts.front().detail;
}

// The output's moved copy starts as the sentinel, so an element the engine does not write
// keeps the sentinel after the copy back.
TEST(TestBundleVerificationHarness, StartOffsetOutputKeepsTheSentinelWhereTheEngineDoesNotWrite)
{
    const auto bundle = makeAliasedConvBundle({});
    const auto wrapper = bundle->graphWrapper();
    auto inputs = makeAliasOwnerInputs(*bundle);
    auto outputs = detail::allocateSentinelOutputs(
        wrapper.getTensorMap(), bundle->outputTensorUids, TensorMap{}, {}, /*onDevice=*/false);
    auto variantPack = detail::buildVariantPack(inputs,
                                                outputs,
                                                wrapper.getTensorMap(),
                                                bundle->outputTensorUids,
                                                /*useDevice=*/false);

    detail::StartOffsetStaging staging(variantPack,
                                       inputs,
                                       outputs,
                                       {{K_ALIAS_OUTPUT_UID, K_SHIFTED_OUTPUT_OFFSET}},
                                       {},
                                       /*useDevice=*/false);
    ASSERT_EQ(bytesPastAlignment(variantPack.at(K_ALIAS_OUTPUT_UID)), K_SHIFTED_OUTPUT_OFFSET);
    writeFloats(variantPack.at(K_ALIAS_OUTPUT_UID), 1, 7.0f);

    auto& output = *outputs.at(K_ALIAS_OUTPUT_UID);
    EXPECT_TRUE(std::isnan(floatAt(output, 0)));
    EXPECT_EQ(staging.waitForEngine(), hipSuccess);
    staging.copyOutputsBack();
    detail::markOutputsModified(outputs, /*device=*/false);

    EXPECT_EQ(floatAt(output, 0), 7.0f);
    for(std::size_t i = 1; i < K_ALIAS_OUTPUT_ELEMS; ++i)
    {
        EXPECT_TRUE(std::isnan(floatAt(output, i))) << "element " << i;
    }
}

// A second entry would read the first moved copy, so its copy back would miss the tensor.
TEST(TestBundleVerificationHarness, StartOffsetStagingThrowsForAUidListedTwice)
{
    const auto bundle = makeAliasedConvBundle({});
    const auto wrapper = bundle->graphWrapper();
    auto inputs = makeAliasOwnerInputs(*bundle);
    auto outputs = detail::allocateSentinelOutputs(
        wrapper.getTensorMap(), bundle->outputTensorUids, TensorMap{}, {}, /*onDevice=*/false);
    auto variantPack = detail::buildVariantPack(inputs,
                                                outputs,
                                                wrapper.getTensorMap(),
                                                bundle->outputTensorUids,
                                                /*useDevice=*/false);

    const auto stageTwice = [&] {
        const detail::StartOffsetStaging staging(
            variantPack,
            inputs,
            outputs,
            {{K_ALIAS_OUTPUT_UID, K_SHIFTED_OUTPUT_OFFSET}, {K_ALIAS_OUTPUT_UID, 1}},
            {},
            /*useDevice=*/false);
    };
    EXPECT_THROW(stageTwice(), std::invalid_argument);
}

TEST(TestBundleVerificationHarness, StartOffsetStagingThrowsForATensorWithoutABuffer)
{
    const auto bundle = makeAliasedConvBundle(viewOfOwnersSecondRow());
    const auto wrapper = bundle->graphWrapper();
    auto inputs = makeAliasOwnerInputs(*bundle);
    OutputTensors outputs;
    auto variantPack = detail::buildVariantPack(inputs,
                                                outputs,
                                                wrapper.getTensorMap(),
                                                bundle->outputTensorUids,
                                                /*useDevice=*/false,
                                                bundle->metadata.aliases);

    const auto stage = [&](int64_t uid) {
        const detail::StartOffsetStaging staging(variantPack,
                                                 inputs,
                                                 outputs,
                                                 {{uid, K_SHIFTED_INPUT_OFFSET}},
                                                 bundle->metadata.aliases,
                                                 /*useDevice=*/false);
    };
    EXPECT_THROW(stage(K_ALIAS_VIEW_UID), std::invalid_argument);
    EXPECT_THROW(stage(99), std::invalid_argument);
}

// The device copies start the offset past a 256-byte aligned address and hold the
// tensor's bytes; the output's copy goes back into the output, around the sentinel.
TEST(TestBundleVerificationHarness, StartOffsetStagingPlacesTheDeviceCopyPastAnAlignedAddress)
{
    SKIP_IF_NO_DEVICES();
    const auto bundle = makeAliasedConvBundle(viewOfOwnersSecondRow());
    const auto wrapper = bundle->graphWrapper();
    auto inputs = makeAliasOwnerInputs(*bundle);
    auto outputs = detail::allocateSentinelOutputs(
        wrapper.getTensorMap(), bundle->outputTensorUids, TensorMap{}, {}, /*onDevice=*/true);
    auto variantPack = detail::buildVariantPack(inputs,
                                                outputs,
                                                wrapper.getTensorMap(),
                                                bundle->outputTensorUids,
                                                /*useDevice=*/true,
                                                bundle->metadata.aliases);
    auto& owner = *inputs.at(K_ALIAS_OWNER_UID);
    void* const ownerDevice = owner.rawDeviceData();

    detail::StartOffsetStaging staging(variantPack,
                                       inputs,
                                       outputs,
                                       {{K_ALIAS_OWNER_UID, K_SHIFTED_INPUT_OFFSET},
                                        {K_ALIAS_OUTPUT_UID, K_SHIFTED_OUTPUT_OFFSET}},
                                       bundle->metadata.aliases,
                                       /*useDevice=*/true);

    void* const ownerCopy = variantPack.at(K_ALIAS_OWNER_UID);
    void* const outputCopy = variantPack.at(K_ALIAS_OUTPUT_UID);
    EXPECT_NE(ownerCopy, ownerDevice);
    EXPECT_EQ(bytesPastAlignment(ownerCopy), K_SHIFTED_INPUT_OFFSET);
    EXPECT_EQ(bytesPastAlignment(outputCopy), K_SHIFTED_OUTPUT_OFFSET);
    EXPECT_EQ(variantPack.at(K_ALIAS_VIEW_UID), advanced(ownerCopy, K_ALIAS_BYTE_OFFSET));

    std::vector<std::byte> ownerCopyBytes(byteCount(owner));
    ASSERT_EQ(
        hipMemcpy(ownerCopyBytes.data(), ownerCopy, ownerCopyBytes.size(), hipMemcpyDeviceToHost),
        hipSuccess);
    EXPECT_EQ(ownerCopyBytes, copyBytes(owner.rawHostData(), byteCount(owner)));

    const float written = 7.0f;
    ASSERT_EQ(hipMemcpy(outputCopy, &written, sizeof(written), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(staging.waitForEngine(), hipSuccess);
    staging.copyOutputsBack();
    detail::markOutputsModified(outputs, /*device=*/true);

    auto& output = *outputs.at(K_ALIAS_OUTPUT_UID);
    EXPECT_EQ(floatAt(output, 0), written);
    for(std::size_t i = 1; i < K_ALIAS_OUTPUT_ELEMS; ++i)
    {
        EXPECT_TRUE(std::isnan(floatAt(output, i))) << "element " << i;
    }
}

// The whole engine run under a device placement: the engine gets the moved device
// copies, and its output reaches the golden comparison.
TEST_F(TestGoldenHarnessFixture, StartOffsetEngineRunOnTheDeviceReachesTheComparison)
{
    SKIP_IF_NO_DEVICES();
    testing_support::HarnessMocks mocks;
    std::vector<int64_t> engineOffsets;
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(
            [&engineOffsets](GraphSession&, const std::optional<LoadedEngine>&, VariantPack& pack) {
                engineOffsets.push_back(bytesPastAlignment(pack.at(0)));
                engineOffsets.push_back(bytesPastAlignment(pack.at(fixtures::K_OUTPUT_UID)));
                const std::vector<float> output(fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
                const auto status = hipMemcpy(pack.at(fixtures::K_OUTPUT_UID),
                                              output.data(),
                                              output.size() * sizeof(float),
                                              hipMemcpyHostToDevice);
                return status == hipSuccess ? EngineOpResult::succeeded()
                                            : EngineOpResult::failed(hipGetErrorString(status));
            });

    auto policy = testing_support::hostPolicy(VerificationMode::AUTO);
    policy.placement = TensorPlacement::DEVICE;
    IntegrationBundleVerificationHarness harness(mocks.dependencies(policy));
    auto bundle = loadRunnableBundle("shifted_device");
    shiftGoldenBundle(*bundle);
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results)) << testing_support::allMessages(results);
    EXPECT_EQ(engineOffsets,
              (std::vector<int64_t>{K_SHIFTED_INPUT_OFFSET, K_SHIFTED_OUTPUT_OFFSET}));
}

// ---- live_rows -------------------------------------------------------------

namespace
{

constexpr float K_NAN = std::numeric_limits<float>::quiet_NaN();
constexpr float K_NEGATIVE_INF = -std::numeric_limits<float>::infinity();

// Writes y as the live-row bundle with lengths {1, 0} expects: K_OUTPUT_VALUE in the
// live rows (batch 0, s = 0) and `pastValue` everywhere else.
std::vector<float> liveRowOutput(float pastValue)
{
    std::vector<float> values(fixtures::K_OUTPUT_ELEMS, pastValue);
    for(std::size_t h = 0; h < 3; ++h)
    {
        for(std::size_t d = 0; d < 5; ++d)
        {
            values[h * 20 + d] = fixtures::K_OUTPUT_VALUE;
        }
    }
    return values;
}

void writeHostOutput(void* pointer, const std::vector<float>& values)
{
    std::memcpy(pointer, values.data(), values.size() * sizeof(float));
}

} // namespace

TEST_F(TestGoldenHarnessFixture, LiveRowGoldenNaNPastTheLengthsIsNotCompared)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);

    ::testing::TestPartResultArray results;
    runCapturing(
        mocks, fixtures::writeLiveRowBundle(_tempDir, "live_rows", {1, 0}, K_NAN), &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
}

// Golden y holds -inf past the lengths and the engine writes a finite value there: only
// the lengths keep those positions out of the comparison.
TEST_F(TestGoldenHarnessFixture, LiveRowGoldenInfPastTheLengthsIsNotCompared)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);

    ::testing::TestPartResultArray results;
    runCapturing(mocks,
                 fixtures::writeLiveRowBundle(_tempDir, "live_rows_inf", {1, 0}, K_NEGATIVE_INF),
                 &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST_F(TestGoldenHarnessFixture, LiveRowGoldenInfWithoutLiveRowsFails)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);

    ::testing::TestPartResultArray results;
    runCapturing(mocks,
                 fixtures::writeLiveRowBundle(
                     _tempDir, "no_live_rows", {1, 0}, K_NEGATIVE_INF, /*withLiveRows=*/false),
                 &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST_F(TestGoldenHarnessFixture, LiveRowMismatchInALiveRowFails)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE + 100.0f);

    ::testing::TestPartResultArray results;
    runCapturing(
        mocks, fixtures::writeLiveRowBundle(_tempDir, "live_mismatch", {1, 0}, K_NAN), &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

// A length past the output's extent is an error in the bundle's data: the harness fails
// the test with the HARNESS origin and does not skip the rows or the comparison.
TEST_F(TestGoldenHarnessFixture, LiveRowLengthPastTheExtentFailsTheHarness)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);

    const std::string engineName = "ENGINE_UNDER_TEST";
    SupportResult accepted;
    accepted.verdict = SupportVerdict::CLAIM_ACCEPTED;
    accepted.bundlePath = "live-rows-bundle";
    accepted.engineName = engineName;
    accepted.arch = "gfx942";
    accepted.platform = "linux";
    ON_CALL(mocks.claimObserver,
            observe(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(SupportObservation{SidecarState::CHECKED, {accepted}}));
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(mocks.reporter, verdicts);

    const auto sidecar = _tempDir / "live-rows-bundle.support.json";
    std::ofstream(sidecar) << R"({"version": 1, "claims": {}})";
    SupportClaimLocator locator;
    locator.sidecarPath = sidecar;
    locator.diagnosticPath = "live-rows-bundle";

    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)),
        LoadedEngine{11, engineName});
    harness.setBundle(fixtures::writeLiveRowBundle(_tempDir, "live_too_long", {5, 0}, K_NAN),
                      "live-rows-bundle",
                      locator);

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    ASSERT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
    const auto messages = testing_support::allMessages(results);
    EXPECT_NE(messages.find("live_rows cannot be applied"), std::string::npos) << messages;
    EXPECT_NE(messages.find("length 5 of batch 0 is not in the range 0 to 4"), std::string::npos)
        << messages;

    ASSERT_EQ(verdicts.size(), 1u);
    const std::string origin
        = std::string("failed outside the engine (") + toString(FailureOrigin::HARNESS) + ")";
    EXPECT_NE(verdicts.front().detail.find(origin), std::string::npos) << verdicts.front().detail;
}

// The CPU reference's output is the expected side: it holds -inf past the lengths, and
// the INT64 lengths tensor keeps those positions out of the comparison.
TEST_F(TestGoldenHarnessFixture, LiveRowReferenceOutputIsComparedOnlyInTheLiveRows)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
    ON_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([](void*, std::size_t, const VariantPack& variantPack) {
            writeHostOutput(variantPack.at(fixtures::K_OUTPUT_UID), liveRowOutput(K_NEGATIVE_INF));
        });

    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::CPU)));
    harness.setBundle(fixtures::writeLiveRowBundle(_tempDir,
                                                   "live_rows_reference",
                                                   {1, 0},
                                                   K_NAN,
                                                   /*withLiveRows=*/true,
                                                   /*int64Lengths=*/true),
                      "unit-test-bundle");

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
}

// A GPU reference leaves its output on the device. Under the default validator a
// live-row output is copied to the host and compared there; under --validator gpu it is
// reported as not comparable.
TEST_F(TestGoldenHarnessFixture, LiveRowDeviceReferenceOutputIsComparedOnTheHost)
{
    SKIP_IF_NO_DEVICES();
    for(const auto validator : {ValidatorDevice::AUTO, ValidatorDevice::GPU})
    {
        SCOPED_TRACE(validator == ValidatorDevice::AUTO ? "auto" : "gpu");
        testing_support::HarnessMocks mocks;
        const auto copyToDevice = [](void* device, const std::vector<float>& values) {
            return hipMemcpy(
                device, values.data(), values.size() * sizeof(float), hipMemcpyHostToDevice);
        };
        ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault([&copyToDevice](GraphSession&,
                                           const std::optional<LoadedEngine>&,
                                           VariantPack& pack) {
                const auto status = copyToDevice(pack.at(fixtures::K_OUTPUT_UID),
                                                 liveRowOutput(fixtures::K_OUTPUT_VALUE));
                return status == hipSuccess ? EngineOpResult::succeeded()
                                            : EngineOpResult::failed(hipGetErrorString(status));
            });
        ON_CALL(mocks.gpuReference, requiresDeviceMemory()).WillByDefault(::testing::Return(true));
        ON_CALL(mocks.gpuReference, execute(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault([&copyToDevice](void*, std::size_t, const VariantPack& pack) {
                ASSERT_EQ(copyToDevice(pack.at(fixtures::K_OUTPUT_UID), liveRowOutput(K_NAN)),
                          hipSuccess);
            });

        auto policy = testing_support::hostPolicy(VerificationMode::GPU);
        policy.placement = TensorPlacement::DEVICE;
        policy.validator = validator;
        IntegrationBundleVerificationHarness harness(mocks.dependencies(policy));
        harness.setBundle(fixtures::writeLiveRowBundle(_tempDir,
                                                       validator == ValidatorDevice::AUTO
                                                           ? "live_rows_auto"
                                                           : "live_rows_gpu",
                                                       {1, 0},
                                                       K_NAN),
                          "unit-test-bundle");

        ::testing::TestPartResultArray results;
        testing_support::driveHarness(harness, &results);

        const auto messages = testing_support::allMessages(results);
        if(validator == ValidatorDevice::AUTO)
        {
            EXPECT_FALSE(testing_support::anyFailed(results)) << messages;
        }
        else
        {
            EXPECT_TRUE(testing_support::anyFailed(results));
            EXPECT_NE(messages.find("Output with live rows NOT COMPARABLE ON DEVICE"),
                      std::string::npos)
                << messages;
        }
        EXPECT_FALSE(testing_support::anySkipped(results)) << messages;
    }
}

// ---- ragged recipe bundles ---------------------------------------------------

namespace
{

constexpr float K_RAGGED_VALUE = 2.0f;

// Writes `value` into O (qkvoSpace elements) and Stats (statsSpace elements, or every
// element of a dense Stats) through `pack`, which holds host pointers.
void writeRaggedSdpaOutputs(const VariantPack& pack,
                            const fixtures::RaggedSdpaCase& c,
                            size_t statsElements,
                            float value)
{
    auto* o = static_cast<float*>(pack.at(fixtures::K_SDPA_O_UID));
    std::fill(o, o + c.qkvoSpace, value);
    auto* stats = static_cast<float*>(pack.at(fixtures::K_SDPA_STATS_UID));
    std::fill(stats, stats + statsElements, value);
}

// The engine and the CPU reference write `engineValue` and `referenceValue` into every
// output element. The GPU reference declines, so the CPU reference is the oracle.
void engineAndCpuReferenceWrite(testing_support::HarnessMocks& mocks,
                                const fixtures::RaggedSdpaCase& c,
                                float engineValue,
                                float referenceValue)
{
    ON_CALL(mocks.gpuReference, isApplicable(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(false));
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(
            [c, engineValue](GraphSession&, const std::optional<LoadedEngine>&, VariantPack& pack) {
                writeRaggedSdpaOutputs(pack, c, c.statsSpace, engineValue);
                return EngineOpResult::succeeded();
            });
    ON_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([c, referenceValue](void*, size_t, const VariantPack& pack) {
            writeRaggedSdpaOutputs(pack, c, c.statsSpace, referenceValue);
        });
}

// The offset layouts a recipe bundle must fill and run: contiguous, gapped (seq_len),
// leading gap and out of order (seq_len).
std::vector<fixtures::RaggedSdpaCase> raggedRecipeCases()
{
    return {fixtures::raggedSdpaHeadTokenStatsCase(),
            fixtures::raggedSdpaGappedStatsCase(),
            fixtures::raggedSdpaShiftedOriginCase(),
            fixtures::raggedSdpaNonMonotonicCase()};
}

} // namespace

// A recipe bundle whose offset and seq_len tensors have exact values is filled from
// them, run, and checked against the CPU reference, for int32 and int64 index tensors.
TEST_F(TestGoldenHarnessFixture, RaggedRecipeBundleFillsRunsAndPasses)
{
    for(const std::string indexType : {"int32", "int64"})
    {
        for(const auto& c : raggedRecipeCases())
        {
            SCOPED_TRACE(c.name + " " + indexType);
            testing_support::HarnessMocks mocks;
            engineAndCpuReferenceWrite(mocks, c, K_RAGGED_VALUE, K_RAGGED_VALUE);
            EXPECT_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_))
                .Times(1);

            fixtures::RaggedRecipeOptions options;
            options.indexType = indexType;
            const auto bundle = fixtures::writeRaggedSdpaRecipeBundle(
                _tempDir, "recipe_" + c.name + "_" + indexType, c, options);
            ASSERT_FALSE(bundle->blobs.has_value());

            IntegrationBundleVerificationHarness harness(
                mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
            harness.setBundle(bundle, "unit-test-bundle");
            ::testing::TestPartResultArray results;
            testing_support::driveHarness(harness, &results);

            const auto messages = testing_support::allMessages(results);
            EXPECT_FALSE(testing_support::anyFailed(results)) << messages;
            EXPECT_FALSE(testing_support::anySkipped(results)) << messages;

            const auto& inputs = harness.inputs();
            ASSERT_EQ(inputs.count(fixtures::K_SDPA_QKVO_OFFSET_UID), 1u);
            const auto offsets = detail::readLengths(*inputs.at(fixtures::K_SDPA_QKVO_OFFSET_UID));
            EXPECT_EQ(offsets, std::vector<int64_t>(c.qkvo.offsets.begin(), c.qkvo.offsets.end()));
        }
    }
}

// The comparison of a filled ragged recipe bundle is real: a single drifted live
// position fails. Perturbing only one element (not the whole buffer, as
// engineAndCpuReferenceWrite's uniform fill would) is what RaggedInBlockMismatchYieldsFail
// and RaggedRecipeLiveRowsLikeResolvesOnTheFilledQ also do -- it is the only way to prove
// the comparison is checking individual live positions rather than, say, nothing at all.
TEST_F(TestGoldenHarnessFixture, RaggedRecipeBundleInBlockMismatchFails)
{
    const auto c = fixtures::raggedSdpaGappedStatsCase();
    testing_support::HarnessMocks mocks;
    ON_CALL(mocks.gpuReference, isApplicable(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(false));
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([c](GraphSession&, const std::optional<LoadedEngine>&, VariantPack& pack) {
            writeRaggedSdpaOutputs(pack, c, c.statsSpace, K_RAGGED_VALUE);
            static_cast<float*>(pack.at(fixtures::K_SDPA_STATS_UID))[0] = K_RAGGED_VALUE + 1.0f;
            return EngineOpResult::succeeded();
        });
    ON_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([c](void*, size_t, const VariantPack& pack) {
            writeRaggedSdpaOutputs(pack, c, c.statsSpace, K_RAGGED_VALUE);
        });

    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
    harness.setBundle(fixtures::writeRaggedSdpaRecipeBundle(_tempDir, "recipe_mismatch", c),
                      "unit-test-bundle");
    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

// An offset or seq_len tensor without a values fill spec leaves the bundle unverifiable,
// and the engine does not run.
TEST_F(TestGoldenHarnessFixture, RaggedRecipeBundleWithoutOffsetValuesIsUnverifiable)
{
    const auto c = fixtures::raggedSdpaGappedStatsCase();
    for(const int64_t missing : {fixtures::K_SDPA_QKVO_OFFSET_UID, fixtures::K_SDPA_SEQ_LEN_KV_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(missing));
        testing_support::HarnessMocks mocks;
        EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);

        fixtures::RaggedRecipeOptions options;
        options.withoutValues = {missing};
        IntegrationBundleVerificationHarness harness(
            mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
        harness.setBundle(fixtures::writeRaggedSdpaRecipeBundle(
                              _tempDir, "recipe_no_values_" + std::to_string(missing), c, options),
                          "unit-test-bundle");
        ::testing::TestPartResultArray results;
        testing_support::driveHarness(harness, &results);

        const auto messages = testing_support::allMessages(results);
        EXPECT_TRUE(testing_support::anySkipped(results)) << messages;
        EXPECT_FALSE(testing_support::anyFailed(results)) << messages;
        EXPECT_NE(messages.find("ragged inputs need exact offset values or golden blobs; "
                                "tensor(s) "
                                + std::to_string(missing) + " have no values fill spec"),
                  std::string::npos)
            << messages;
    }
}

// seq_len_kv needs exact values even when K and V are dense: it bounds the keys that the
// engine and the references read.
TEST_F(TestGoldenHarnessFixture, RaggedRecipeBundleNeedsValuesForEverySeqLenOfARaggedNode)
{
    const auto c = fixtures::raggedSdpaGappedStatsCase();
    testing_support::HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);

    fixtures::RaggedRecipeOptions options;
    options.raggedKv = false;
    options.withoutValues = {fixtures::K_SDPA_SEQ_LEN_KV_UID};
    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
    harness.setBundle(
        fixtures::writeRaggedSdpaRecipeBundle(_tempDir, "recipe_dense_kv", c, options),
        "unit-test-bundle");
    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    const auto messages = testing_support::allMessages(results);
    EXPECT_TRUE(testing_support::anySkipped(results)) << messages;
    EXPECT_FALSE(testing_support::anyFailed(results)) << messages;
    EXPECT_NE(messages.find("tensor(s) " + std::to_string(fixtures::K_SDPA_SEQ_LEN_KV_UID)
                            + " have no values fill spec"),
              std::string::npos)
        << messages;
}

// Offsets that send a ragged input past its dense buffer fail the test before the engine
// or a reference runs. Batch 1 starts at row 4 and holds 3 rows: element 27 is the last
// one it reaches, past the 24 elements of Q, K and V.
TEST_F(TestGoldenHarnessFixture, RaggedRecipeOffsetsPastTheBufferFailBeforeTheEngine)
{
    auto c = fixtures::raggedSdpaGappedStatsCase();
    c.qkvo.offsets = {0, 16, 20};
    testing_support::HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(mocks.gpuReference, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);

    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
    harness.setBundle(fixtures::writeRaggedSdpaRecipeBundle(_tempDir, "recipe_past_buffer", c),
                      "unit-test-bundle");
    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    const auto messages = testing_support::allMessages(results);
    EXPECT_TRUE(testing_support::anyFailed(results)) << messages;
    EXPECT_FALSE(testing_support::anySkipped(results)) << messages;
    EXPECT_NE(messages.find("does not fit its buffer: its offsets reach element 27, past the 24 "
                            "elements of its buffer"),
              std::string::npos)
        << messages;
}

// A malformed ragged layout -- not past-the-buffer, but one raggedLayoutFootprint() cannot
// read at all -- also fails the test before the engine or a reference runs. This exercises
// checkRaggedLayouts()'s other failure branch (catch(const std::exception&)), which the
// above test does not reach.
TEST_F(TestGoldenHarnessFixture, RaggedRecipeInvalidOffsetMultiplierFailsBeforeTheEngine)
{
    auto c = fixtures::raggedSdpaGappedStatsCase();
    c.qkvo.offsetMultiplier = 0;
    testing_support::HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(mocks.gpuReference, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);

    IntegrationBundleVerificationHarness harness(
        mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
    harness.setBundle(fixtures::writeRaggedSdpaRecipeBundle(_tempDir, "recipe_bad_multiplier", c),
                      "unit-test-bundle");
    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    const auto messages = testing_support::allMessages(results);
    EXPECT_TRUE(testing_support::anyFailed(results)) << messages;
    EXPECT_FALSE(testing_support::anySkipped(results)) << messages;
    EXPECT_NE(messages.find("has no valid ragged layout: ragged_offset_multiplier 0 is less "
                            "than 1"),
              std::string::npos)
        << messages;
}

// A dense Stats beside the filled ragged Q takes its live rows from Q through a `like`
// entry. Positions past each live length differ and are not compared.
TEST_F(TestGoldenHarnessFixture, RaggedRecipeLiveRowsLikeResolvesOnTheFilledQ)
{
    auto c = fixtures::raggedSdpaGappedStatsCase();
    c.stats.dims = {2, 2, 3, 1};
    c.stats.strides = {6, 3, 1, 1};
    constexpr size_t K_DENSE_STATS = 12;
    // Q lengths are {2, 3} on a sequence extent of 3: only (b=0, s=2) is past a length.
    const auto statsWith = [](float live, float past) {
        std::vector<float> stats(K_DENSE_STATS, live);
        for(size_t h = 0; h < 2; ++h)
        {
            stats[h * 3 + 2] = past;
        }
        return stats;
    };

    // A live position that differs still fails.
    for(const bool live : {true, false})
    {
        const float engineLive = live ? K_RAGGED_VALUE : K_RAGGED_VALUE + 1.0f;
        SCOPED_TRACE(live ? "live rows agree" : "a live row differs");
        testing_support::HarnessMocks mocks;
        ON_CALL(mocks.gpuReference, isApplicable(::testing::_, ::testing::_))
            .WillByDefault(::testing::Return(false));
        ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault([c, statsWith, engineLive](GraphSession&,
                                                      const std::optional<LoadedEngine>&,
                                                      VariantPack& pack) {
                writeRaggedSdpaOutputs(pack, c, 0, K_RAGGED_VALUE);
                const auto stats = statsWith(engineLive, 99.0f);
                std::copy(stats.begin(),
                          stats.end(),
                          static_cast<float*>(pack.at(fixtures::K_SDPA_STATS_UID)));
                return EngineOpResult::succeeded();
            });
        ON_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault([c, statsWith](void*, size_t, const VariantPack& pack) {
                writeRaggedSdpaOutputs(pack, c, 0, K_RAGGED_VALUE);
                const auto stats = statsWith(K_RAGGED_VALUE, -5.0f);
                std::copy(stats.begin(),
                          stats.end(),
                          static_cast<float*>(pack.at(fixtures::K_SDPA_STATS_UID)));
            });

        fixtures::RaggedRecipeOptions options;
        options.raggedStats = false;
        options.extraMetadata = nlohmann::json::parse(
            R"({"live_rows": [{"uid": )" + std::to_string(fixtures::K_SDPA_STATS_UID)
            + R"(, "like": )" + std::to_string(fixtures::K_SDPA_Q_UID) + "}]}");
        IntegrationBundleVerificationHarness harness(
            mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
        harness.setBundle(fixtures::writeRaggedSdpaRecipeBundle(
                              _tempDir, live ? "recipe_like" : "recipe_like_mismatch", c, options),
                          "unit-test-bundle");
        ::testing::TestPartResultArray results;
        testing_support::driveHarness(harness, &results);

        const auto messages = testing_support::allMessages(results);
        EXPECT_EQ(testing_support::anyFailed(results), !live) << messages;
        EXPECT_FALSE(testing_support::anySkipped(results)) << messages;
    }
}

// A GPU reference leaves the ragged outputs of a recipe bundle on the device. Under the
// default validator they are copied to the host and compared there; under
// --validator gpu they are reported as not comparable.
TEST_F(TestGoldenHarnessFixture, RaggedRecipeDeviceReferenceOutputIsComparedOnTheHost)
{
    SKIP_IF_NO_DEVICES();
    const auto c = fixtures::raggedSdpaGappedStatsCase();
    const auto copyOutputs = [c](const VariantPack& pack) {
        const std::vector<float> o(c.qkvoSpace, K_RAGGED_VALUE);
        const std::vector<float> stats(c.statsSpace, K_RAGGED_VALUE);
        const auto oStatus = hipMemcpy(pack.at(fixtures::K_SDPA_O_UID),
                                       o.data(),
                                       o.size() * sizeof(float),
                                       hipMemcpyHostToDevice);
        const auto statsStatus = hipMemcpy(pack.at(fixtures::K_SDPA_STATS_UID),
                                           stats.data(),
                                           stats.size() * sizeof(float),
                                           hipMemcpyHostToDevice);
        return oStatus != hipSuccess ? oStatus : statsStatus;
    };
    for(const auto validator : {ValidatorDevice::AUTO, ValidatorDevice::GPU})
    {
        const bool isAuto = validator == ValidatorDevice::AUTO;
        SCOPED_TRACE(isAuto ? "auto" : "gpu");
        testing_support::HarnessMocks mocks;
        ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault([&copyOutputs](GraphSession&,
                                          const std::optional<LoadedEngine>&,
                                          VariantPack& pack) {
                const auto status = copyOutputs(pack);
                return status == hipSuccess ? EngineOpResult::succeeded()
                                            : EngineOpResult::failed(hipGetErrorString(status));
            });
        ON_CALL(mocks.gpuReference, requiresDeviceMemory()).WillByDefault(::testing::Return(true));
        ON_CALL(mocks.gpuReference, execute(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault([&copyOutputs](void*, std::size_t, const VariantPack& pack) {
                ASSERT_EQ(copyOutputs(pack), hipSuccess);
            });

        auto policy = testing_support::hostPolicy(VerificationMode::GPU);
        policy.placement = TensorPlacement::DEVICE;
        policy.validator = validator;
        IntegrationBundleVerificationHarness harness(mocks.dependencies(policy));
        harness.setBundle(fixtures::writeRaggedSdpaRecipeBundle(
                              _tempDir, isAuto ? "recipe_device_auto" : "recipe_device_gpu", c),
                          "unit-test-bundle");
        ::testing::TestPartResultArray results;
        testing_support::driveHarness(harness, &results);

        const auto messages = testing_support::allMessages(results);
        if(isAuto)
        {
            EXPECT_FALSE(testing_support::anyFailed(results)) << messages;
        }
        else
        {
            EXPECT_TRUE(testing_support::anyFailed(results));
            EXPECT_NE(messages.find("Ragged output NOT COMPARABLE ON DEVICE"), std::string::npos)
                << messages;
        }
        EXPECT_FALSE(testing_support::anySkipped(results)) << messages;
    }
}

// NOLINTEND(readability-identifier-naming)
