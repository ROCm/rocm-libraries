// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvFwdLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"

/**
 * @file TestPackLaunchValues.cpp
 * @brief The saved launch values of each pack: every member of the launch state maps to
 *        the value of its own name and type, and to no other value. Each pack reads its
 *        values back into the same launch state, and refuses values outside its contract.
 *
 * Each member gets a distinct value, so a swapped or dropped mapping shows. The expected
 * names are written out again here: they are the stored contract, and a test that shared
 * the constants would not see a rename.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_plugin_sdk::ingestor::MetadataType;
using hipdnn_plugin_sdk::ingestor::MetadataValues;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;
using hipdnn_plugin_sdk::ingestor::toString;

TEST(TestPackLaunchValues, PointwiseValuesMatchTheV1Contract)
{
    const PointwiseBinding binding{11, 12, 13};

    const MetadataValues values = pointwiseLaunchValues(binding, 256);

    const MetadataValues expected{{"input_a.uid", int64_t{11}},
                                  {"input_b.uid", int64_t{12}},
                                  {"output.uid", int64_t{13}},
                                  {"block_size", int64_t{256}}};
    EXPECT_EQ(values, expected);
    EXPECT_EQ(POINTWISE_DISPATCH_SYMBOL_V1, "hipkernel.pointwise.dispatch.v1");
}

TEST(TestPackLaunchValues, ConvFwdValuesMatchTheV1Contract)
{
    const ConvFwdBinding binding{21, 22, 23};
    const ConvFwdExtents extents{31, 32, 33, 34, 35, 36, 37};

    const MetadataValues values = convFwdLaunchValues(binding, extents, 64);

    const MetadataValues expected{{"x.uid", int64_t{21}},
                                  {"w.uid", int64_t{22}},
                                  {"y.uid", int64_t{23}},
                                  {"n", int64_t{31}},
                                  {"c", int64_t{32}},
                                  {"h", int64_t{33}},
                                  {"width", int64_t{34}},
                                  {"k", int64_t{35}},
                                  {"r", int64_t{36}},
                                  {"s", int64_t{37}},
                                  {"block_size", int64_t{64}}};
    EXPECT_EQ(values, expected);
    EXPECT_EQ(CONV_FWD_DISPATCH_SYMBOL_V1, "hipkernel.conv_fwd.dispatch.v1");
}

TEST(TestPackLaunchValues, Gfx950AttentionDenseValuesMatchTheV1Contract)
{
    AttentionDenseBinding binding;
    binding.q = 41;
    binding.k = 42;
    binding.v = 43;
    binding.o = 44;
    binding.causal = 1;
    binding.slidingWindow = 45;
    binding.scale = 0.1F;

    AttentionDenseProblem problem;
    problem.batch = 51;
    problem.seqLenQ = 52;
    problem.seqLenKv = 53;
    problem.numQueryHeads = 54;
    problem.numKvHeads = 55;
    problem.headSize = 56;
    problem.dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;

    const MetadataValues values = gfx950AttentionDenseLaunchValues(binding, problem, 128);

    const MetadataValues expected{{"q.uid", int64_t{41}},
                                  {"k.uid", int64_t{42}},
                                  {"v.uid", int64_t{43}},
                                  {"o.uid", int64_t{44}},
                                  {"causal", int64_t{1}},
                                  {"sliding_window", int64_t{45}},
                                  {"batch", int64_t{51}},
                                  {"seqlen_q", int64_t{52}},
                                  {"seqlen_kv", int64_t{53}},
                                  {"num_query_heads", int64_t{54}},
                                  {"num_kv_heads", int64_t{55}},
                                  {"head_size", int64_t{56}},
                                  {"block_m", int64_t{128}},
                                  {"scale", static_cast<double>(0.1F)},
                                  {"data_type", std::string("BFLOAT16")},
                                  {"stride_layout", std::string("bshd")}};
    EXPECT_EQ(values, expected);
    EXPECT_EQ(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1,
              "hipkernel.gfx950_attention_dense.dispatch.v1");

    // The stored double converts back to the float the kernel takes, bit for bit.
    const auto stored = values.find("scale");
    ASSERT_NE(stored, values.end());
    const auto* storedScale = std::get_if<double>(&stored->second);
    ASSERT_NE(storedScale, nullptr);
    const auto restored = static_cast<float>(*storedScale);
    uint32_t restoredBits = 0;
    uint32_t originalBits = 0;
    std::memcpy(&restoredBits, &restored, sizeof(restoredBits));
    std::memcpy(&originalBits, &binding.scale, sizeof(originalBits));
    EXPECT_EQ(restoredBits, originalBits);
}

// ---------------------------------------------------------------------------
// Reading saved values back
// ---------------------------------------------------------------------------

PointwiseLaunchInputs pointwiseInputs()
{
    return PointwiseLaunchInputs{PointwiseBinding{11, 12, 13}, 256};
}

ConvFwdLaunchInputs convFwdInputs()
{
    // Distinct extents with r <= h and s <= width.
    return ConvFwdLaunchInputs{
        ConvFwdBinding{21, 22, 23}, ConvFwdExtents{31, 32, 33, 34, 35, 3, 4}, 64};
}

Gfx950AttentionDenseLaunchInputs gfx950AttentionDenseInputs()
{
    Gfx950AttentionDenseLaunchInputs launch;
    launch.binding.q = 41;
    launch.binding.k = 42;
    launch.binding.v = 43;
    launch.binding.o = 44;
    launch.binding.causal = 1;
    launch.binding.slidingWindow = 45;
    launch.binding.scale = 0.1F;
    launch.problem.batch = 51;
    launch.problem.seqLenQ = 52;
    launch.problem.seqLenKv = 53;
    launch.problem.numQueryHeads = 54;
    launch.problem.numKvHeads = 55;
    launch.problem.headSize = 56;
    launch.problem.dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    launch.blockM = 128;
    return launch;
}

SavedLaunchInputs savedPointwise()
{
    const auto launch = pointwiseInputs();
    return {std::string(POINTWISE_DISPATCH_SYMBOL_V1),
            pointwiseLaunchValues(launch.binding, launch.blockSize)};
}

SavedLaunchInputs savedConvFwd()
{
    const auto launch = convFwdInputs();
    return {std::string(CONV_FWD_DISPATCH_SYMBOL_V1),
            convFwdLaunchValues(launch.binding, launch.extents, launch.blockSize)};
}

SavedLaunchInputs savedGfx950AttentionDense()
{
    const auto launch = gfx950AttentionDenseInputs();
    return {std::string(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1),
            gfx950AttentionDenseLaunchValues(launch.binding, launch.problem, launch.blockM)};
}

// Expects `read` to refuse `inputs` as a valid plan that cannot be restored, with `phrase`
// in the message.
template <typename TReader>
void expectReadRefusal(TReader read, const SavedLaunchInputs& inputs, const std::string& phrase)
{
    serialization::expectIngestorPlanRefusal(
        [&]() { static_cast<void>(read(inputs)); }, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, phrase);
}

// The refusals every reader shares: a missing value, an extra value, an integer stored as a
// string or as an integer list, and the unversioned dispatch name, which is registered but
// names no contract.
template <typename TReader>
void expectCommonReadRefusals(TReader read,
                              const SavedLaunchInputs& valid,
                              const std::string& unversionedAlias,
                              const std::string& integerName)
{
    SavedLaunchInputs missing = valid;
    missing.values.erase(integerName);
    expectReadRefusal(read, missing, "needs launch value '" + integerName + "'");

    SavedLaunchInputs extra = valid;
    extra.values["unexpected"] = int64_t{1};
    expectReadRefusal(read, extra, "launch value 'unexpected' is not part of the contract");

    SavedLaunchInputs wrongType = valid;
    wrongType.values[integerName] = std::string("1");
    expectReadRefusal(read, wrongType, "launch value '" + integerName + "'");
    expectReadRefusal(read, wrongType, "must be int in [");
    expectReadRefusal(read, wrongType, "and the plan holds string");

    SavedLaunchInputs wrongList = valid;
    wrongList.values[integerName] = std::vector<int64_t>{1};
    expectReadRefusal(read, wrongList, "and the plan holds int_list");

    SavedLaunchInputs unversioned = valid;
    unversioned.dispatchSymbol = unversionedAlias;
    expectReadRefusal(read, unversioned, "names no launch-value contract");
}

TEST(TestPackLaunchValues, PointwiseInputsRoundTripThroughTheirValues)
{
    const auto expected = pointwiseInputs();

    const auto read = readPointwiseLaunchInputs(savedPointwise());

    EXPECT_EQ(read.binding.inputA, expected.binding.inputA);
    EXPECT_EQ(read.binding.inputB, expected.binding.inputB);
    EXPECT_EQ(read.binding.output, expected.binding.output);
    EXPECT_EQ(read.blockSize, expected.blockSize);
}

TEST(TestPackLaunchValues, ConvFwdInputsRoundTripThroughTheirValues)
{
    const auto expected = convFwdInputs();

    const auto read = readConvFwdLaunchInputs(savedConvFwd());

    EXPECT_EQ(read.binding.x, expected.binding.x);
    EXPECT_EQ(read.binding.w, expected.binding.w);
    EXPECT_EQ(read.binding.y, expected.binding.y);
    EXPECT_EQ(read.extents.n, expected.extents.n);
    EXPECT_EQ(read.extents.c, expected.extents.c);
    EXPECT_EQ(read.extents.h, expected.extents.h);
    EXPECT_EQ(read.extents.width, expected.extents.width);
    EXPECT_EQ(read.extents.k, expected.extents.k);
    EXPECT_EQ(read.extents.r, expected.extents.r);
    EXPECT_EQ(read.extents.s, expected.extents.s);
    EXPECT_EQ(read.blockSize, expected.blockSize);
}

TEST(TestPackLaunchValues, Gfx950AttentionDenseInputsRoundTripThroughTheirValues)
{
    const auto expected = gfx950AttentionDenseInputs();

    const auto read = readGfx950AttentionDenseLaunchInputs(savedGfx950AttentionDense());

    EXPECT_EQ(read.binding.q, expected.binding.q);
    EXPECT_EQ(read.binding.k, expected.binding.k);
    EXPECT_EQ(read.binding.v, expected.binding.v);
    EXPECT_EQ(read.binding.o, expected.binding.o);
    EXPECT_EQ(read.binding.causal, expected.binding.causal);
    EXPECT_EQ(read.binding.slidingWindow, expected.binding.slidingWindow);
    uint32_t readScaleBits = 0;
    uint32_t expectedScaleBits = 0;
    std::memcpy(&readScaleBits, &read.binding.scale, sizeof(readScaleBits));
    std::memcpy(&expectedScaleBits, &expected.binding.scale, sizeof(expectedScaleBits));
    EXPECT_EQ(readScaleBits, expectedScaleBits);
    EXPECT_EQ(read.problem.batch, expected.problem.batch);
    EXPECT_EQ(read.problem.seqLenQ, expected.problem.seqLenQ);
    EXPECT_EQ(read.problem.seqLenKv, expected.problem.seqLenKv);
    EXPECT_EQ(read.problem.numQueryHeads, expected.problem.numQueryHeads);
    EXPECT_EQ(read.problem.numKvHeads, expected.problem.numKvHeads);
    EXPECT_EQ(read.problem.headSize, expected.problem.headSize);
    EXPECT_EQ(read.problem.dataType, expected.problem.dataType);
    EXPECT_EQ(read.blockM, expected.blockM);
}

TEST(TestPackLaunchValues, PointwiseRefusesMalformedValues)
{
    const auto valid = savedPointwise();
    expectCommonReadRefusals(
        &readPointwiseLaunchInputs, valid, "hipkernel.pointwise.dispatch", "block_size");

    SavedLaunchInputs zeroBlock = valid;
    zeroBlock.values["block_size"] = int64_t{0};
    expectReadRefusal(&readPointwiseLaunchInputs, zeroBlock, "launch value 'block_size'");
}

TEST(TestPackLaunchValues, ConvFwdRefusesMalformedValues)
{
    const auto valid = savedConvFwd();
    expectCommonReadRefusals(&readConvFwdLaunchInputs, valid, "hipkernel.conv_fwd.dispatch", "h");

    SavedLaunchInputs filterTooTall = valid;
    filterTooTall.values["r"] = int64_t{34};
    expectReadRefusal(&readConvFwdLaunchInputs, filterTooTall, "must not exceed 'h' and 'width'");
}

TEST(TestPackLaunchValues, Gfx950AttentionDenseRefusesMalformedValues)
{
    const auto valid = savedGfx950AttentionDense();
    expectCommonReadRefusals(&readGfx950AttentionDenseLaunchInputs,
                             valid,
                             "hipkernel.gfx950_attention_dense.dispatch",
                             "batch");

    SavedLaunchInputs wideBatch = valid;
    wideBatch.values["batch"] = int64_t{2147483648};
    expectReadRefusal(&readGfx950AttentionDenseLaunchInputs, wideBatch, "launch value 'batch'");

    // 0.1 as a double is not a float value, so no float scale was saved as it.
    SavedLaunchInputs doubleScale = valid;
    doubleScale.values["scale"] = 0.1;
    expectReadRefusal(&readGfx950AttentionDenseLaunchInputs, doubleScale, "launch value 'scale'");

    SavedLaunchInputs floatType = valid;
    floatType.values["data_type"] = std::string("FLOAT");
    expectReadRefusal(&readGfx950AttentionDenseLaunchInputs, floatType, "launch value 'data_type'");

    SavedLaunchInputs otherLayout = valid;
    otherLayout.values["stride_layout"] = std::string("bhsd");
    expectReadRefusal(
        &readGfx950AttentionDenseLaunchInputs, otherLayout, "launch value 'stride_layout'");

    SavedLaunchInputs unbuiltTile = valid;
    unbuiltTile.values["block_m"] = int64_t{64};
    expectReadRefusal(&readGfx950AttentionDenseLaunchInputs, unbuiltTile, "launch value 'block_m'");
}

TEST(TestPackLaunchValues, Gfx950AttentionDenseRefusesAScaleOfAnotherType)
{
    const auto valid = savedGfx950AttentionDense();
    const std::string mustBeFloat
        = "must be " + toString(MetadataType::FLOAT) + ", and the plan holds ";

    SavedLaunchInputs intScale = valid;
    intScale.values["scale"] = int64_t{1};
    expectReadRefusal(&readGfx950AttentionDenseLaunchInputs, intScale, "launch value 'scale'");
    expectReadRefusal(
        &readGfx950AttentionDenseLaunchInputs, intScale, mustBeFloat + toString(MetadataType::INT));

    SavedLaunchInputs stringScale = valid;
    stringScale.values["scale"] = std::string("0.1");
    expectReadRefusal(&readGfx950AttentionDenseLaunchInputs,
                      stringScale,
                      mustBeFloat + toString(MetadataType::STRING));
}

TEST(TestPackLaunchValues, Gfx950AttentionDenseRefusesAStringValueOfAnotherType)
{
    const auto valid = savedGfx950AttentionDense();
    const std::string mustBeString
        = "must be " + toString(MetadataType::STRING) + ", and the plan holds ";

    SavedLaunchInputs intDataType = valid;
    intDataType.values["data_type"] = int64_t{1};
    expectReadRefusal(
        &readGfx950AttentionDenseLaunchInputs, intDataType, "launch value 'data_type'");
    expectReadRefusal(&readGfx950AttentionDenseLaunchInputs,
                      intDataType,
                      mustBeString + toString(MetadataType::INT));

    SavedLaunchInputs floatLayout = valid;
    floatLayout.values["stride_layout"] = 1.0;
    expectReadRefusal(
        &readGfx950AttentionDenseLaunchInputs, floatLayout, "launch value 'stride_layout'");
    expectReadRefusal(&readGfx950AttentionDenseLaunchInputs,
                      floatLayout,
                      mustBeString + toString(MetadataType::FLOAT));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
