// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Drives the harness under ScopedFakeTestPartResultReporter to verify
// executor outcomes (decline→SKIP, match→PASS, mismatch→FAIL).

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
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
        attributes, bundle->outputTensorUids, TensorMap{}, /*onDevice=*/false);
    auto deviceOutputs = detail::allocateSentinelOutputs(
        attributes, bundle->outputTensorUids, TensorMap{}, /*onDevice=*/true);

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

// ---- placement ---------------------------------------------------------------------------

constexpr std::array<int64_t, 9> K_ALIGNMENTS = {1, 2, 4, 8, 16, 32, 64, 128, 256};

// Bug caught: an exact placement that lands on a 2N boundary (so a kernel needing only N
// never sees a pointer that is merely N-aligned), or one that misses N altogether. Checked
// over bases that are not themselves aligned, not just the allocator's 256-byte ones.
TEST(TestBundleVerificationHarness, PlacedAddressIsAlignedToNAndNotTo2NForEveryAlignment)
{
    for(const int64_t n : K_ALIGNMENTS)
    {
        for(std::uintptr_t base = 0x10000; base < 0x10000 + 600; base += 7)
        {
            const auto address = detail::placedAddress(base, n, detail::PlacedAt::EXACT);
            const auto un = static_cast<std::uintptr_t>(n);
            EXPECT_EQ(address % un, 0u) << "N=" << n << " base=" << base;
            EXPECT_NE(address % (2 * un), 0u) << "N=" << n << " base=" << base;
            EXPECT_GE(address, base);
            EXPECT_LE(address - base, detail::placementSlack(n)) << "N=" << n;
        }
    }
}

// Bug caught: an under-aligned pointer that still satisfies the declared alignment, which
// the backend would then accept and the case would silently stop testing the rejection.
TEST(TestBundleVerificationHarness, UnderPlacedAddressMissesNForEveryAlignmentAboveOne)
{
    for(const int64_t n : K_ALIGNMENTS)
    {
        if(n < 2)
        {
            continue;
        }
        for(std::uintptr_t base = 0x10000; base < 0x10000 + 600; base += 7)
        {
            const auto address = detail::placedAddress(base, n, detail::PlacedAt::UNDER);
            const auto un = static_cast<std::uintptr_t>(n);
            EXPECT_NE(address % un, 0u) << "N=" << n << " base=" << base;
            EXPECT_EQ(address % (un / 2), 0u) << "N=" << n << " base=" << base;
            EXPECT_LE(address - base, detail::placementSlack(n)) << "N=" << n;
        }
    }
}

// Bug caught: an alignment of 1 has no smaller alignment to fall under, and a non-positive
// one is a malformed graph; either silently producing an address would hide the mistake.
TEST(TestBundleVerificationHarness, PlacedAddressRejectsAlignmentsThatCannotBePlaced)
{
    EXPECT_THROW(detail::placedAddress(0x10000, 1, detail::PlacedAt::UNDER), std::invalid_argument);
    EXPECT_THROW(detail::placedAddress(0x10000, 0, detail::PlacedAt::EXACT), std::invalid_argument);
    EXPECT_THROW(detail::placedAddress(0x10000, -16, detail::PlacedAt::EXACT),
                 std::invalid_argument);
    EXPECT_THROW(detail::placementSlack(0), std::invalid_argument);
}

// Bug caught: a scalar passed by value (a host pointer) or an input of the sub-byte packed set
// being moved into a device shadow, which would hand the engine a device pointer where it
// expects a host value, or place bytes the engine never reads.
TEST(TestBundleVerificationHarness, PlacementPlanLeavesScalarsAndPackedInputsInPlace)
{
    auto bundle = makeRuntimePassByValueBundle();
    const auto wrapper = bundle->graphWrapper();
    const auto& attributes = wrapper.getTensorMap();

    TensorMap inputs;
    inputs.emplace(1, hipdnn_test_sdk::detail::createTensorFromAttribute(*attributes.at(1)));
    inputs.emplace(3, hipdnn_test_sdk::detail::createTensorFromAttribute(*attributes.at(3)));
    OutputTensors outputs;
    outputs.emplace(2, hipdnn_test_sdk::detail::createTensorFromAttribute(*attributes.at(2)));
    const auto pack = detail::buildVariantPack(
        inputs, outputs, attributes, bundle->outputTensorUids, /*useDevice=*/false);

    const auto plan = detail::planPlacement(pack, inputs, outputs, attributes, false);
    std::vector<int64_t> placeable;
    for(const auto& candidate : plan.candidates)
    {
        placeable.push_back(candidate.uid);
        EXPECT_EQ(candidate.alignment, 16) << "the graph's declared alignment is what is placed";
    }
    EXPECT_EQ(placeable, (std::vector<int64_t>{2, 3}));
    ASSERT_EQ(plan.leftInPlace.size(), 1u);
    EXPECT_NE(plan.leftInPlace.front().find("uid 1"), std::string::npos);

    const auto packedPlan = detail::planPlacement(pack, inputs, outputs, attributes, true);
    placeable.clear();
    for(const auto& candidate : packedPlan.candidates)
    {
        placeable.push_back(candidate.uid);
    }
    EXPECT_EQ(placeable, (std::vector<int64_t>{2})) << "only outputs move when inputs are packed";
    EXPECT_EQ(packedPlan.leftInPlace.size(), 2u);
}

// Bug caught: a pack entry with no tensor behind it reaching hipMalloc with a null tensor.
TEST(TestBundleVerificationHarness, PlacementPlanRejectsAPackEntryWithoutATensor)
{
    auto bundle = makeRuntimePassByValueBundle();
    const auto wrapper = bundle->graphWrapper();
    TensorMap inputs;
    OutputTensors outputs;
    VariantPack pack;
    pack[42] = nullptr;
    EXPECT_THROW(detail::planPlacement(pack, inputs, outputs, wrapper.getTensorMap(), false),
                 std::invalid_argument);
}

// Bug caught: "under" naming a uid that is not a device buffer silently placing nothing, so the
// case would run unplaced and then fail for the wrong reason.
TEST(TestBundleVerificationHarness, UnderPlacementOfAnUnplaceableUidFailsBeforeAnyDeviceCall)
{
    VariantPack pack;
    const std::vector<detail::PlacementCandidate> none;
    EXPECT_THROW(detail::PlacedVariantPack(pack, none, CasePlacement::UNDER, 9),
                 detail::PlacementError);
    EXPECT_THROW(detail::PlacedVariantPack(pack, none, CasePlacement::UNDER, std::nullopt),
                 detail::PlacementError);
}

// Bug caught: the default placement touching the pack, which would move every ordinary case.
TEST(TestBundleVerificationHarness, DefaultPlacementLeavesThePackAlone)
{
    const std::vector<int64_t> dims = {4};
    hipdnn_data_sdk::utilities::Tensor<float> tensor(dims);
    VariantPack pack;
    pack[1] = tensor.rawHostData();
    const void* before = pack.at(1);
    const std::vector<detail::PlacementCandidate> candidates = {{1, &tensor, false, 16}};

    const detail::PlacedVariantPack placed(pack, candidates, CasePlacement::DEFAULT, std::nullopt);
    EXPECT_EQ(pack.at(1), before);
    EXPECT_TRUE(placed.placedUids().empty());
}

// Bug caught: a device pointer that is still 2N-aligned after "exact" placement, an input whose
// contents did not follow it into the shadow, or an output that did not start as the sentinel.
TEST(TestBundleVerificationHarness, ExactPlacementMovesDeviceBuffersAtEveryAlignmentIntact)
{
    SKIP_IF_NO_DEVICES();

    for(const int64_t n : K_ALIGNMENTS)
    {
        std::vector<float> values(33);
        for(std::size_t i = 0; i < values.size(); ++i)
        {
            values[i] = static_cast<float>(i) + 0.25f;
        }
        const std::vector<int64_t> dims = {33};
        hipdnn_data_sdk::utilities::Tensor<float> input(dims);
        input.fillWithData(values.data(), values.size() * sizeof(float));
        hipdnn_data_sdk::utilities::Tensor<float> output(dims);
        output.fillWithSentinelValue();

        VariantPack pack;
        pack[1] = input.rawDeviceData();
        pack[2] = output.rawDeviceData();
        void* const originalInput = pack.at(1);
        void* const originalOutput = pack.at(2);
        const std::vector<detail::PlacementCandidate> candidates
            = {{1, &input, false, n}, {2, &output, true, n}};

        const detail::PlacedVariantPack placed(
            pack, candidates, CasePlacement::EXACT, std::nullopt);

        EXPECT_EQ(placed.placedUids(), (std::vector<int64_t>{1, 2})) << "N=" << n;
        for(const int64_t uid : {1, 2})
        {
            const auto address = reinterpret_cast<std::uintptr_t>(pack.at(uid));
            EXPECT_EQ(address % static_cast<std::uintptr_t>(n), 0u) << "N=" << n << " uid " << uid;
            EXPECT_NE(address % static_cast<std::uintptr_t>(2 * n), 0u)
                << "N=" << n << " uid " << uid;
        }
        EXPECT_NE(pack.at(1), originalInput);
        EXPECT_NE(pack.at(2), originalOutput);

        std::vector<float> seen(33, 0.0f);
        ASSERT_EQ(hipMemcpy(seen.data(), pack.at(1), 33 * sizeof(float), hipMemcpyDeviceToHost),
                  hipSuccess);
        for(int i = 0; i < 33; ++i)
        {
            EXPECT_FLOAT_EQ(seen[static_cast<std::size_t>(i)], static_cast<float>(i) + 0.25f)
                << "N=" << n << " element " << i;
        }
        ASSERT_EQ(hipMemcpy(seen.data(), pack.at(2), 33 * sizeof(float), hipMemcpyDeviceToHost),
                  hipSuccess);
        for(const float value : seen)
        {
            EXPECT_TRUE(std::isnan(value)) << "N=" << n << ": the output shadow lost the sentinel";
        }
    }
}

// Bug caught: a copy-back that drops the engine's writes, or one that overwrites elements the
// engine never wrote with something other than the sentinel, which would let an unwritten
// output pass as written.
TEST(TestBundleVerificationHarness, PlacementCopyBackKeepsWrittenValuesAndTheSentinelElsewhere)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {16};
    hipdnn_data_sdk::utilities::Tensor<float> output(dims);
    output.fillWithSentinelValue();
    VariantPack pack;
    pack[2] = output.rawDeviceData();
    const std::vector<detail::PlacementCandidate> candidates = {{2, &output, true, 64}};

    {
        detail::PlacedVariantPack placed(pack, candidates, CasePlacement::EXACT, std::nullopt);
        // The "engine": writes the first four elements only.
        const std::vector<float> written = {1.0f, 2.0f, 3.0f, 4.0f};
        ASSERT_EQ(hipMemcpy(pack.at(2), written.data(), 4 * sizeof(float), hipMemcpyHostToDevice),
                  hipSuccess);
        placed.copyBackOutputs();
    }

    output.markDeviceModified();
    const auto* host = static_cast<const float*>(output.rawHostData());
    for(int i = 0; i < 4; ++i)
    {
        EXPECT_FLOAT_EQ(host[i], static_cast<float>(i) + 1.0f);
    }
    for(int i = 4; i < 16; ++i)
    {
        EXPECT_TRUE(std::isnan(host[i])) << "element " << i << " lost its sentinel on copy-back";
    }
}

// Bug caught: "under" moving every buffer (so the backend's complaint could name any tensor),
// or leaving the named one on a boundary that still satisfies its declared alignment.
TEST(TestBundleVerificationHarness, UnderPlacementMovesOnlyTheNamedUidBelowItsAlignment)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {8};
    hipdnn_data_sdk::utilities::Tensor<float> first(dims);
    hipdnn_data_sdk::utilities::Tensor<float> second(dims);
    VariantPack pack;
    pack[1] = first.rawDeviceData();
    pack[2] = second.rawDeviceData();
    void* const untouched = pack.at(1);
    const std::vector<detail::PlacementCandidate> candidates
        = {{1, &first, false, 16}, {2, &second, false, 16}};

    const detail::PlacedVariantPack placed(pack, candidates, CasePlacement::UNDER, 2);
    EXPECT_EQ(placed.placedUids(), (std::vector<int64_t>{2}));
    EXPECT_EQ(pack.at(1), untouched);
    const auto address = reinterpret_cast<std::uintptr_t>(pack.at(2));
    EXPECT_NE(address % 16u, 0u);
    EXPECT_EQ(address % 8u, 0u);
}

// ---- padded leading strides ---------------------------------------------------------------
// A tensor whose rows are padded has gap elements between the end of a row and the start of the
// next. The harness sizes, places and copies a tensor by its element space (the offset of the
// last element, plus one), so a gap is carried along as bytes that belong to no element.

/// The element space of dims and strides, computed independently of the tensor classes.
int64_t elementSpaceOf(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    int64_t last = 0;
    for(std::size_t axis = 0; axis < dims.size(); ++axis)
    {
        last += (dims[axis] - 1) * strides[axis];
    }
    return last + 1;
}

/// The layouts the padded probe spaces render (Matmul.Stride, SdpaFwd.AsmStride,
/// SdpaFwd.Gfx950DenseStride), padded by an odd, a sub-vector and a vector-multiple pad.
struct PaddedLayout
{
    const char* name;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

std::vector<PaddedLayout> paddedLayouts()
{
    return {
        // Matmul A [3, 16, 32] with a leading dimension of 32 + 3, 32 + 4 and 32 + 1
        {"matmul_pad3", {3, 16, 32}, {int64_t{16} * 35, 35, 1}},
        {"matmul_pad4", {3, 16, 32}, {int64_t{16} * 36, 36, 1}},
        {"matmul_pad1", {3, 16, 32}, {int64_t{16} * 33, 33, 1}},
        // SDPA Q [b 2, hq 8, sq 64, d 128], BHSD: row stride d + 5
        {"bhsd_pad5", {2, 8, 64, 128}, {int64_t{8} * 64 * 133, int64_t{64} * 133, 133, 1}},
        // the same, BSHD: row stride hq * d + 7, head stride d
        {"bshd_pad7", {2, 8, 64, 128}, {int64_t{64} * (8 * 128 + 7), 128, 8 * 128 + 7, 1}},
        // V [2, 2, 96, 128] BSHD with a row of 2 * 128 + 8
        {"bshd_pad8", {2, 2, 96, 128}, {int64_t{96} * (2 * 128 + 8), 128, 2 * 128 + 8, 1}},
    };
}

std::shared_ptr<IntegrationTestBundle>
    makePaddedOutputBundle(const PaddedLayout& layout,
                           hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    auto bundle = std::make_shared<IntegrationTestBundle>();
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 1, "output", type, &layout.strides, &layout.dims));
    bundle->outputTensorUids.push_back(1);
    const std::vector<flatbuffers::Offset<Node>> nodes;
    builder.Finish(CreateGraphDirect(
        builder, "padded", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes));
    bundle->graphBuffer = builder.Release();
    return bundle;
}

// Bug caught: an output buffer sized by the element count (too short for the last row once the
// rows are padded, so the engine writes past it), or by a packed guess of the strides, for the
// layouts the padded probe spaces render.
TEST(TestBundleVerificationHarness, AllocatedOutputsOfPaddedStridesCoverEveryPaddedOffset)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    for(const auto& layout : paddedLayouts())
    {
        SCOPED_TRACE(layout.name);
        auto bundle = makePaddedOutputBundle(layout, DataType::FLOAT);
        const auto wrapper = bundle->graphWrapper();
        auto outputs = detail::allocateSentinelOutputs(wrapper.getTensorMap(),
                                                       bundle->outputTensorUids,
                                                       TensorMap{},
                                                       /*onDevice=*/false);

        auto& tensor = *outputs.at(1);
        const auto expectedSpace
            = static_cast<std::size_t>(elementSpaceOf(layout.dims, layout.strides));
        EXPECT_EQ(tensor.elementSpace(), expectedSpace);
        EXPECT_GT(tensor.elementSpace(), tensor.elementCount()) << "the rows are padded";
        EXPECT_FALSE(tensor.isPacked());

        // The largest offset any element reaches is the last byte of the allocation, and the
        // element there holds the sentinel like every other element.
        std::vector<int64_t> last;
        last.reserve(layout.dims.size());
        for(const int64_t dim : layout.dims)
        {
            last.push_back(dim - 1);
        }
        EXPECT_EQ(tensor.getIndex(last), static_cast<int64_t>(expectedSpace) - 1);
        auto& typed = dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<float>&>(tensor);
        EXPECT_TRUE(std::isnan(typed.getHostValue(last)));
        EXPECT_TRUE(std::isnan(typed.getHostValue(std::vector<int64_t>(last.size(), 0))));
    }
}

// Bug caught: the device sentinel fill (a memset of the allocation) and the host fill (the
// logical elements) leaving different values in an element of a padded output, which would make
// an element the engine never wrote look written on one side.
TEST(TestBundleVerificationHarness, DeviceSentinelOfAPaddedOutputHoldsTheHostSentinelInEveryElement)
{
    SKIP_IF_NO_DEVICES();
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    for(const auto type : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        const PaddedLayout layout{"padded", {3, 5, 7}, {int64_t{5} * 11, 11, 1}};
        auto bundle = makePaddedOutputBundle(layout, type);
        const auto wrapper = bundle->graphWrapper();
        auto hostOutputs = detail::allocateSentinelOutputs(
            wrapper.getTensorMap(), bundle->outputTensorUids, TensorMap{}, /*onDevice=*/false);
        auto deviceOutputs = detail::allocateSentinelOutputs(
            wrapper.getTensorMap(), bundle->outputTensorUids, TensorMap{}, /*onDevice=*/true);

        auto& expected = *hostOutputs.at(1);
        auto& actual = *deviceOutputs.at(1);
        ASSERT_EQ(expected.elementSpace(), actual.elementSpace());
        ASSERT_EQ(actual.elementSpace(),
                  static_cast<std::size_t>(elementSpaceOf(layout.dims, layout.strides)));

        // Non-const host access migrates the device-written bytes back. Only the elements are
        // compared: what the gaps hold is not part of the contract.
        const auto* expectedBytes = static_cast<const unsigned char*>(expected.rawHostData());
        const auto* actualBytes = static_cast<const unsigned char*>(actual.rawHostData());
        const auto size = expected.elementSize();
        for(int64_t i = 0; i < layout.dims[0]; ++i)
        {
            for(int64_t j = 0; j < layout.dims[1]; ++j)
            {
                for(int64_t k = 0; k < layout.dims[2]; ++k)
                {
                    const auto offset = static_cast<std::size_t>(expected.getIndex(i, j, k)) * size;
                    EXPECT_EQ(std::memcmp(expectedBytes + offset, actualBytes + offset, size), 0)
                        << "element " << i << "," << j << "," << k;
                }
            }
        }
    }
}

// Bug caught: a shadow sized by the element count (the last row would be cut off), a copy that
// takes only the logical elements (a gap byte the engine's kernel expects to find would be gone),
// or a copy back that writes a gap byte the engine never wrote, for padded inputs and outputs at
// every alignment the harness places at.
TEST(TestBundleVerificationHarness, PlacementOfPaddedTensorsCarriesEveryByteAndTheLastElement)
{
    SKIP_IF_NO_DEVICES();

    // [3, 4, 8] floats with a leading stride of 11 (a pad of 3): 129 floats, 96 of them elements
    const std::vector<int64_t> dims = {3, 4, 8};
    const std::vector<int64_t> strides = {int64_t{4} * 11, 11, 1};
    const auto space = static_cast<std::size_t>(elementSpaceOf(dims, strides));
    ASSERT_EQ(space, 129u);

    // the offsets that hold an element
    std::vector<std::size_t> elementOffsets;
    std::vector<bool> isElement(space, false);
    for(int64_t i = 0; i < dims[0]; ++i)
    {
        for(int64_t j = 0; j < dims[1]; ++j)
        {
            for(int64_t k = 0; k < dims[2]; ++k)
            {
                const auto offset = static_cast<std::size_t>(i * strides[0] + j * strides[1] + k);
                elementOffsets.push_back(offset);
                isElement[offset] = true;
            }
        }
    }
    ASSERT_EQ(elementOffsets.size(), 96u);
    ASSERT_TRUE(isElement.back()) << "the last offset holds the last element of the last row";

    constexpr std::uint32_t inputGap = 0xA5A5A5A5U;
    constexpr std::uint32_t outputGap = 0x5A5A5A5AU;
    const auto bitsOf = [](float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    };

    for(const int64_t n : K_ALIGNMENTS)
    {
        SCOPED_TRACE("N=" + std::to_string(n));
        hipdnn_data_sdk::utilities::Tensor<float> input(dims, strides);
        hipdnn_data_sdk::utilities::Tensor<float> output(dims, strides);
        {
            std::vector<std::uint32_t> inputBits(space, inputGap);
            std::vector<std::uint32_t> outputBits(space, outputGap);
            for(const auto offset : elementOffsets)
            {
                inputBits[offset] = bitsOf(static_cast<float>(offset));
                outputBits[offset] = 0x7FC00000U; // the sentinel NaN
            }
            ASSERT_EQ(input.fillWithData(inputBits.data(), space * sizeof(float)),
                      space * sizeof(float));
            ASSERT_EQ(output.fillWithData(outputBits.data(), space * sizeof(float)),
                      space * sizeof(float));
        }

        VariantPack pack;
        pack[1] = input.rawDeviceData();
        pack[2] = output.rawDeviceData();
        const std::vector<detail::PlacementCandidate> candidates
            = {{1, &input, false, n}, {2, &output, true, n}};
        {
            detail::PlacedVariantPack placed(pack, candidates, CasePlacement::EXACT, std::nullopt);
            ASSERT_EQ(placed.placedUids(), (std::vector<int64_t>{1, 2}));

            // the input shadow holds every element and every gap byte of the original
            std::vector<std::uint32_t> seen(space, 0);
            ASSERT_EQ(
                hipMemcpy(seen.data(), pack.at(1), space * sizeof(float), hipMemcpyDeviceToHost),
                hipSuccess);
            for(std::size_t offset = 0; offset < space; ++offset)
            {
                const auto expected
                    = isElement[offset] ? bitsOf(static_cast<float>(offset)) : inputGap;
                ASSERT_EQ(seen[offset], expected) << "input offset " << offset;
            }

            // the "engine" writes every element of the output, and only those: the last row's
            // last element sits at the end of the shadow
            ASSERT_EQ(
                hipMemcpy(seen.data(), pack.at(2), space * sizeof(float), hipMemcpyDeviceToHost),
                hipSuccess);
            for(std::size_t offset = 0; offset < space; ++offset)
            {
                ASSERT_EQ(seen[offset], isElement[offset] ? 0x7FC00000U : outputGap)
                    << "output shadow offset " << offset;
            }
            for(const auto offset : elementOffsets)
            {
                seen[offset] = bitsOf(static_cast<float>(offset) + 0.5f);
            }
            ASSERT_EQ(
                hipMemcpy(pack.at(2), seen.data(), space * sizeof(float), hipMemcpyHostToDevice),
                hipSuccess);
            placed.copyBackOutputs();
        }

        output.markDeviceModified();
        const auto* host = static_cast<const std::uint32_t*>(output.rawHostData());
        for(std::size_t offset = 0; offset < space; ++offset)
        {
            const auto expected
                = isElement[offset] ? bitsOf(static_cast<float>(offset) + 0.5f) : outputGap;
            ASSERT_EQ(host[offset], expected) << "output offset " << offset;
        }
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
    EXPECT_NE(testing_support::allMessages(results).find("ragged inputs require golden blobs"),
              std::string::npos);
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

    auto hostOutputs = detail::allocateSentinelOutputs(
        attributes, bundle->outputTensorUids, loadedTensors, /*onDevice=*/false);
    auto deviceOutputs = detail::allocateSentinelOutputs(
        attributes, bundle->outputTensorUids, loadedTensors, /*onDevice=*/true);

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

namespace
{

HarnessPolicy devicePolicy()
{
    auto policy = testing_support::hostPolicy(VerificationMode::AUTO);
    policy.placement = TensorPlacement::DEVICE;
    return policy;
}

// Writes the golden value into the whole output the way a working engine would, through the
// pointer the engine was handed, and records the address of every buffer it saw.
void engineWritesThroughTheDevicePointer(testing_support::HarnessMocks& mocks,
                                         std::unordered_map<int64_t, std::uintptr_t>& seen,
                                         bool writeOutput)
{
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([&seen, writeOutput](GraphSession&,
                                            const std::optional<LoadedEngine>&,
                                            VariantPack& variantPack) {
            for(const auto& [uid, pointer] : variantPack)
            {
                seen[uid] = reinterpret_cast<std::uintptr_t>(pointer);
            }
            if(writeOutput)
            {
                const std::vector<float> value(fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
                EXPECT_EQ(hipMemcpy(variantPack.at(fixtures::K_OUTPUT_UID),
                                    value.data(),
                                    value.size() * sizeof(float),
                                    hipMemcpyHostToDevice),
                          hipSuccess);
            }
            return EngineOpResult::succeeded();
        });
}

} // namespace

// Bug caught: a placement that never reaches the engine run (every pointer still 2N-aligned),
// or one that breaks the run it was meant to leave intact (inputs lost, outputs not copied back).
TEST_F(TestGoldenHarnessFixture, ExactPlacementReachesTheEngineAndTheRunStillPasses)
{
    SKIP_IF_NO_DEVICES();
    testing_support::HarnessMocks mocks;
    std::unordered_map<int64_t, std::uintptr_t> seen;
    engineWritesThroughTheDevicePointer(mocks, seen, /*writeOutput=*/true);

    auto bundle = loadRunnableBundle("exact_placement");
    bundle->metadata.placement = CasePlacement::EXACT;

    IntegrationBundleVerificationHarness harness(mocks.dependencies(devicePolicy()));
    harness.setBundle(std::move(bundle), "unit-test-bundle");
    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    ASSERT_EQ(seen.size(), 6u) << "x, mean, inv_variance, scale, bias and y are all device buffers";
    for(const auto& [uid, address] : seen)
    {
        EXPECT_EQ(address % 16u, 0u) << "uid " << uid;
        EXPECT_NE(address % 32u, 0u) << "uid " << uid << " is still aligned to 2N";
    }
}

// Bug caught: copy-back dressing an unwritten output up as written, so a placed run could not
// fail an engine that never wrote its output (the very thing the sentinel exists to catch).
TEST_F(TestGoldenHarnessFixture, ExactPlacementKeepsAnUnwrittenOutputWrong)
{
    SKIP_IF_NO_DEVICES();
    testing_support::HarnessMocks mocks;
    std::unordered_map<int64_t, std::uintptr_t> seen;
    engineWritesThroughTheDevicePointer(mocks, seen, /*writeOutput=*/false);

    auto bundle = loadRunnableBundle("unwritten_placement");
    bundle->metadata.placement = CasePlacement::EXACT;

    IntegrationBundleVerificationHarness harness(mocks.dependencies(devicePolicy()));
    harness.setBundle(std::move(bundle), "unit-test-bundle");
    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_TRUE(testing_support::anyFailed(results))
        << "an output the engine never wrote must still fail after the copy-back";
}

// Bug caught: a reference run, or an unplaced case, picking up the placement: the same bundle
// without `placement` must hand the engine the allocator's own pointers.
TEST_F(TestGoldenHarnessFixture, DefaultPlacementHandsTheEngineTheAllocatorsPointers)
{
    SKIP_IF_NO_DEVICES();
    testing_support::HarnessMocks mocks;
    std::unordered_map<int64_t, std::uintptr_t> seen;
    engineWritesThroughTheDevicePointer(mocks, seen, /*writeOutput=*/true);

    IntegrationBundleVerificationHarness harness(mocks.dependencies(devicePolicy()));
    harness.setBundle(loadRunnableBundle("default_placement"), "unit-test-bundle");
    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    ASSERT_EQ(seen.size(), 6u);
    for(const auto& [uid, address] : seen)
    {
        EXPECT_EQ(address % 256u, 0u) << "uid " << uid << " moved without a placement";
    }
}

// ---- the non-finite mask ------------------------------------------------------------------

namespace
{

constexpr float K_QUIET_NAN = std::numeric_limits<float>::quiet_NaN();
constexpr float K_POSITIVE_INF = std::numeric_limits<float>::infinity();

// The golden y: 3.5 everywhere except a NaN, a +inf and a -inf, which are what a correct
// engine produces for the poisoned inputs of a `special` domain.
std::vector<float> goldenWithNonFinite()
{
    std::vector<float> golden(fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
    golden[3] = K_QUIET_NAN;
    golden[7] = K_POSITIVE_INF;
    golden[9] = -K_POSITIVE_INF;
    return golden;
}

// The fixture bundle, with `golden` as its y.
std::shared_ptr<IntegrationTestBundle> loadBundleWithGolden(const std::filesystem::path& parent,
                                                            const std::string& name,
                                                            const std::vector<float>& golden)
{
    const auto dir = parent / name;
    fixtures::writeBundleFiles(dir, name, /*includeGoldenOutput=*/true);
    {
        std::ofstream out((dir / name).string() + ".tensor" + std::to_string(fixtures::K_OUTPUT_UID)
                              + ".bin",
                          std::ios::binary);
        out.write(reinterpret_cast<const char*>(golden.data()),
                  static_cast<std::streamsize>(golden.size() * sizeof(float)));
    }
    return fixtures::loadWrittenBundle(dir, name);
}

// An engine that writes `values` into y, except at `skip`, which it never touches.
void engineWritesExcept(testing_support::HarnessMocks& mocks,
                        const std::vector<float>& values,
                        std::optional<std::size_t> skip)
{
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([values, skip](GraphSession&,
                                      const std::optional<LoadedEngine>&,
                                      VariantPack& variantPack) {
            auto* y = static_cast<float*>(variantPack.at(fixtures::K_OUTPUT_UID));
            for(std::size_t i = 0; i < values.size(); ++i)
            {
                if(!skip.has_value() || i != *skip)
                {
                    y[i] = values[i];
                }
            }
            return EngineOpResult::succeeded();
        });
}

} // namespace

// Bug caught: the mask never reaching the harness comparison (a correct NaN, +inf and -inf
// still fail), or being on for every case (the same pair passing without it).
TEST_F(TestGoldenHarnessFixture, MaskedNonFiniteOutputsMatchByPositionAndUnmaskedOnesFail)
{
    for(const bool masked : {true, false})
    {
        testing_support::HarnessMocks mocks;
        engineWritesExcept(mocks, goldenWithNonFinite(), std::nullopt);

        auto bundle = loadBundleWithGolden(
            _tempDir, masked ? "masked_nonfinite" : "unmasked_nonfinite", goldenWithNonFinite());
        bundle->metadata.validator
            = masked ? CaseValidator::MATCH_NONFINITE : CaseValidator::DEFAULT;

        ::testing::TestPartResultArray results;
        runCapturing(mocks, std::move(bundle), &results);

        EXPECT_EQ(testing_support::anyFailed(results), !masked)
            << (masked ? "masked" : "unmasked") << ": " << testing_support::allMessages(results);
    }
}

// Bug caught: an output element the engine never wrote passing because the golden value there is
// a NaN and the sentinel is a NaN too. The engine's output starts as a payload NaN, and an
// element still holding it fails even though NaN matches NaN everywhere else.
TEST_F(TestGoldenHarnessFixture, MaskedRunFailsAnOutputElementTheEngineNeverWrote)
{
    testing_support::HarnessMocks mocks;
    // Everything written except element 3, which is NaN in the golden data.
    engineWritesExcept(mocks, goldenWithNonFinite(), 3);

    auto bundle = loadBundleWithGolden(_tempDir, "unwritten_nan", goldenWithNonFinite());
    bundle->metadata.validator = CaseValidator::MATCH_NONFINITE;

    ::testing::TestPartResultArray results;
    runCapturing(mocks, std::move(bundle), &results);

    EXPECT_TRUE(testing_support::anyFailed(results))
        << "the unwritten element still holds the sentinel and must fail";
}

// Bug caught: the host-only validator running under --validator gpu (which would read device
// memory through host pointers) or silently overriding the flag; the conflict is reported, and
// before the engine has spent its time.
TEST_F(TestGoldenHarnessFixture, MaskedCaseUnderAnExplicitDeviceValidatorIsAConfigurationError)
{
    testing_support::HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);

    auto bundle = loadBundleWithGolden(_tempDir, "masked_gpu_validator", goldenWithNonFinite());
    bundle->metadata.validator = CaseValidator::MATCH_NONFINITE;

    auto policy = testing_support::hostPolicy(VerificationMode::AUTO);
    policy.validator = ValidatorDevice::GPU;
    IntegrationBundleVerificationHarness harness(mocks.dependencies(std::move(policy)));
    harness.setBundle(std::move(bundle), "unit-test-bundle");
    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("--validator gpu"), std::string::npos)
        << testing_support::allMessages(results);
}
// NOLINTEND(readability-identifier-naming)
