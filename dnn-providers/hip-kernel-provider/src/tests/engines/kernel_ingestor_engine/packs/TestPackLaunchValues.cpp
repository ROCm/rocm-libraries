// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvFwdLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseLaunchValues.hpp"
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
using hipdnn_plugin_sdk::ingestor::MetadataValue;
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
    EXPECT_EQ(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2,
              "hipkernel.gfx950_attention_dense.dispatch.v2");

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

// A shape the gfx950 match serves, with distinct values, and a saved window the reader admits
// (see attention_dense::readLaunchInputs).
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
    launch.problem.seqLenQ = 1536;
    launch.problem.seqLenKv = 1600;
    launch.problem.numQueryHeads = 8;
    launch.problem.numKvHeads = 4;
    launch.problem.headSize = 128;
    launch.problem.dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    launch.blockM = 256;
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

// Saved under v2, the contract new plans carry.
SavedLaunchInputs savedGfx950AttentionDense()
{
    const auto launch = gfx950AttentionDenseInputs();
    return {std::string(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2),
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

// The scales a dense reader restores: the [2^-64, 2^4] its graph match serves, bounds
// included, and nothing else. A float scale outside it is refused with the match's reason;
// NaN and the infinities are refused before that, as no float launch argument holds them.
template <typename TReader>
void expectDenseScaleRangeRestores(TReader read, const SavedLaunchInputs& valid)
{
    for(const float scale : {0x1p-64F, 0x1p4F})
    {
        SCOPED_TRACE(scale);
        SavedLaunchInputs bound = valid;
        bound.values["scale"] = static_cast<double>(scale);
        EXPECT_EQ(read(bound).binding.scale, scale);
    }

    // The float just past each bound first, then zero, a negative and far-off scales.
    for(const float scale :
        {0x1.fffffep-65F, 0x1.000002p4F, 0.0F, -1.0F, 0x1p-65F, 0x1p5F, 0x1p-100F, 1e30F})
    {
        SCOPED_TRACE(scale);
        SavedLaunchInputs outside = valid;
        outside.values["scale"] = static_cast<double>(scale);
        expectReadRefusal(read, outside, "launch value 'scale'");
        expectReadRefusal(read, outside, "is outside the supported [2^-64, 2^4]");
    }

    for(const double scale : {std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity()})
    {
        SCOPED_TRACE(scale);
        SavedLaunchInputs notFinite = valid;
        notFinite.values["scale"] = scale;
        expectReadRefusal(read, notFinite, "launch value 'scale'");
        expectReadRefusal(read, notFinite, "must be a finite double");
    }
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

// v1 and v2 differ only in the kernel arguments the launch passes, so one value set reads
// the same under either name, and a version the pack never defined is refused.
TEST(TestPackLaunchValues, Gfx950AttentionDenseReadsTheSameValuesUnderV1AndV2)
{
    auto v1 = savedGfx950AttentionDense();
    v1.dispatchSymbol = std::string(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1);
    const auto fromV1 = readGfx950AttentionDenseLaunchInputs(v1);
    const auto fromV2 = readGfx950AttentionDenseLaunchInputs(savedGfx950AttentionDense());

    EXPECT_EQ(fromV1.binding.q, fromV2.binding.q);
    EXPECT_EQ(fromV1.binding.o, fromV2.binding.o);
    EXPECT_EQ(fromV1.problem.batch, fromV2.problem.batch);
    EXPECT_EQ(fromV1.problem.seqLenKv, fromV2.problem.seqLenKv);
    EXPECT_EQ(fromV1.problem.numQueryHeads, fromV2.problem.numQueryHeads);
    EXPECT_EQ(fromV1.problem.numKvHeads, fromV2.problem.numKvHeads);
    EXPECT_EQ(fromV1.blockM, fromV2.blockM);

    auto v3 = savedGfx950AttentionDense();
    v3.dispatchSymbol = "hipkernel.gfx950_attention_dense.dispatch.v3";
    expectReadRefusal(&readGfx950AttentionDenseLaunchInputs, v3, "names no launch-value contract");
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

TEST(TestPackLaunchValues, Gfx950AttentionDenseRefusesAScaleOutsideTheSupportedRange)
{
    expectDenseScaleRangeRestores(&readGfx950AttentionDenseLaunchInputs,
                                  savedGfx950AttentionDense());
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

// ---------------------------------------------------------------------------
// gfx942 dense attention
//
// One contract, v1, with gfx950's value set plus stats.uid. Its reader holds the restore to
// what the gfx942 graph and kernel match bind: besides the scale range and the mask and shape
// rules both dense readers share (below), a block_m gfx942 builds and a stats uid that is -1
// (none) or non-negative.
// ---------------------------------------------------------------------------

// A shape the gfx942 match serves, with distinct values. seqlen_q 1536 is a multiple of
// every block_m from 32 to 512 as well as of 96.
Gfx942AttentionDenseLaunchInputs gfx942AttentionDenseInputs()
{
    Gfx942AttentionDenseLaunchInputs launch;
    launch.binding.q = 61;
    launch.binding.k = 62;
    launch.binding.v = 63;
    launch.binding.o = 64;
    launch.binding.causal = 1;
    launch.binding.slidingWindow = 160;
    launch.binding.scale = 0.1F;
    launch.problem.batch = 71;
    launch.problem.seqLenQ = 1536;
    launch.problem.seqLenKv = 1600;
    launch.problem.numQueryHeads = 8;
    launch.problem.numKvHeads = 4;
    launch.problem.headSize = 128;
    launch.problem.dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;
    launch.blockM = 96;
    return launch;
}

SavedLaunchInputs savedGfx942AttentionDense()
{
    const auto launch = gfx942AttentionDenseInputs();
    return {std::string(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1),
            gfx942AttentionDenseLaunchValues(launch.binding, launch.problem, launch.blockM)};
}

TEST(TestPackLaunchValues, Gfx942AttentionDenseValuesMatchTheV1Contract)
{
    const auto launch = gfx942AttentionDenseInputs();

    const MetadataValues values
        = gfx942AttentionDenseLaunchValues(launch.binding, launch.problem, launch.blockM);

    const MetadataValues expected{{"q.uid", int64_t{61}},
                                  {"k.uid", int64_t{62}},
                                  {"v.uid", int64_t{63}},
                                  {"o.uid", int64_t{64}},
                                  {"causal", int64_t{1}},
                                  {"sliding_window", int64_t{160}},
                                  {"batch", int64_t{71}},
                                  {"seqlen_q", int64_t{1536}},
                                  {"seqlen_kv", int64_t{1600}},
                                  {"num_query_heads", int64_t{8}},
                                  {"num_kv_heads", int64_t{4}},
                                  {"head_size", int64_t{128}},
                                  {"block_m", int64_t{96}},
                                  {"scale", static_cast<double>(0.1F)},
                                  {"data_type", std::string("HALF")},
                                  {"stride_layout", std::string("bshd")},
                                  {"stats.uid", int64_t{-1}},
                                  {"grid_order", std::string("qb_minor")},
                                  {"persist_decode", std::string()},
                                  {"num_persistent", int64_t{0}}};
    EXPECT_EQ(values, expected);
    EXPECT_EQ(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1,
              "hipkernel.gfx942_attention_dense.dispatch.v1");

    // A plan whose graph binds a stats output saves that output's uid.
    auto withStats = launch;
    withStats.binding.stats = 65;
    const MetadataValues statsValues
        = gfx942AttentionDenseLaunchValues(withStats.binding, withStats.problem, withStats.blockM);
    EXPECT_EQ(statsValues.at("stats.uid"), MetadataValue{int64_t{65}});
    EXPECT_EQ(statsValues.size(), expected.size());
}

TEST(TestPackLaunchValues, Gfx942AttentionDenseInputsRoundTripThroughTheirValues)
{
    const auto expected = gfx942AttentionDenseInputs();

    const auto read = readGfx942AttentionDenseLaunchInputs(savedGfx942AttentionDense());

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
    EXPECT_EQ(read.binding.stats, ATTENTION_DENSE_NO_STATS);

    // The stats uid of a plan that writes the LSE reads back as itself.
    auto withStats = expected;
    withStats.binding.stats = 65;
    const SavedLaunchInputs saved{
        std::string(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1),
        gfx942AttentionDenseLaunchValues(withStats.binding, withStats.problem, withStats.blockM)};
    EXPECT_EQ(readGfx942AttentionDenseLaunchInputs(saved).binding.stats, 65);
}

// The launch grid round-trips: a persistent plan reads back its decode and CTA count, a grid
// plan its block order. Off the persistent grid persist_decode is empty and num_persistent 0.
TEST(TestPackLaunchValues, Gfx942AttentionDenseGridRoundTripsThroughItsValues)
{
    using attention_dense::AttentionDenseGrid;
    using attention_dense::AttentionDenseGridOrder;
    using attention_dense::AttentionDensePersistDecode;
    const auto launch = gfx942AttentionDenseInputs();
    std::vector<AttentionDenseGrid> grids;
    for(const auto order : {AttentionDenseGridOrder::QB_MINOR,
                            AttentionDenseGridOrder::BT_HKV_MINOR,
                            AttentionDenseGridOrder::HQ_MINOR_SWZ})
    {
        AttentionDenseGrid grid;
        grid.order = order;
        grids.push_back(grid);
    }
    for(const auto decode : {AttentionDensePersistDecode::QB_MAJOR,
                             AttentionDensePersistDecode::HKV_MAJOR,
                             AttentionDensePersistDecode::BT_HKV_MINOR,
                             AttentionDensePersistDecode::HQ_MINOR_SWZ})
    {
        AttentionDenseGrid grid;
        grid.order = AttentionDenseGridOrder::PERSISTENT;
        grid.decode = decode;
        grid.numPersistent = 304;
        grids.push_back(grid);
    }
    for(const auto& grid : grids)
    {
        const auto values
            = gfx942AttentionDenseLaunchValues(launch.binding, launch.problem, launch.blockM, grid);
        EXPECT_EQ(values.at("grid_order"),
                  MetadataValue{std::string(attention_dense::gridOrderName(grid.order))});
        EXPECT_EQ(values.at("persist_decode"),
                  MetadataValue{grid.persistent()
                                    ? std::string(attention_dense::persistDecodeName(grid.decode))
                                    : std::string()});
        EXPECT_EQ(values.at("num_persistent"),
                  MetadataValue{grid.persistent() ? int64_t{304} : int64_t{0}});
        const auto read = readGfx942AttentionDenseLaunchInputs(
            {std::string(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1), values});
        EXPECT_EQ(read.grid.order, grid.order);
        EXPECT_EQ(read.grid.persistent(), grid.persistent());
        if(grid.persistent())
        {
            EXPECT_EQ(read.grid.decode, grid.decode);
            EXPECT_EQ(read.grid.numPersistent, 304);
        }
    }
}

// A saved grid the kernel match never admits is refused: an unknown order, a persistent
// decode without runtime kernargs, a persistent grid without CTAs, persistent values off
// the persistent grid, hq_minor_swz at Hq % 8 != 0, and a decode past its 2^30 bound.
TEST(TestPackLaunchValues, Gfx942AttentionDenseRefusesAGridTheKernelMatchNeverAdmits)
{
    const auto valid = savedGfx942AttentionDense();
    const auto with = [&valid](std::initializer_list<std::pair<const char*, MetadataValue>> moved) {
        auto saved = valid;
        for(const auto& [name, value] : moved)
        {
            saved.values[name] = value;
        }
        return saved;
    };
    const MetadataValue persistent{std::string("persistent")};
    expectReadRefusal(&readGfx942AttentionDenseLaunchInputs,
                      with({{"grid_order", std::string("auto")}}),
                      "launch value 'grid_order'");
    for(const char* decode : {"auto", "gqa_pair", "gqa_pair_2phase", ""})
    {
        expectReadRefusal(&readGfx942AttentionDenseLaunchInputs,
                          with({{"grid_order", persistent},
                                {"persist_decode", std::string(decode)},
                                {"num_persistent", int64_t{304}}}),
                          "launch value 'persist_decode'");
    }
    expectReadRefusal(&readGfx942AttentionDenseLaunchInputs,
                      with({{"grid_order", persistent},
                            {"persist_decode", std::string("qb_major")},
                            {"num_persistent", int64_t{0}}}),
                      "launch value 'num_persistent'");
    expectReadRefusal(&readGfx942AttentionDenseLaunchInputs,
                      with({{"persist_decode", std::string("qb_major")}}),
                      "launch value 'persist_decode'");
    expectReadRefusal(&readGfx942AttentionDenseLaunchInputs,
                      with({{"num_persistent", int64_t{304}}}),
                      "launch value 'num_persistent'");
    expectReadRefusal(
        &readGfx942AttentionDenseLaunchInputs,
        with({{"grid_order", std::string("hq_minor_swz")}, {"num_query_heads", int64_t{12}}}),
        "needs num_query_heads (12) divisible by chiplet_num_xcds");
    expectReadRefusal(&readGfx942AttentionDenseLaunchInputs,
                      with({{"grid_order", persistent},
                            {"persist_decode", std::string("bt_hkv_minor")},
                            {"num_persistent", int64_t{304}},
                            {"batch", int64_t{1}},
                            {"seqlen_q", int64_t{256}},
                            {"seqlen_kv", int64_t{256}},
                            {"num_query_heads", int64_t{32768}},
                            {"num_kv_heads", int64_t{1}},
                            {"block_m", int64_t{256}},
                            {"sliding_window", int64_t{0}}}),
                      "must be below 2^30");
}

// stats.uid is -1 for a plan without a stats output, or the output's uid. graph_match binds
// no negative uid, so anything below -1 is not a saved plan. Whether the restored kernel
// writes the LSE is checked against it by the restore's signature comparison
// (TestIngestorPlanRestoreChecks), which this reader does not see.
TEST(TestPackLaunchValues, Gfx942AttentionDenseReadsAStatsUidOfNoneOrAUid)
{
    const auto valid = savedGfx942AttentionDense();

    for(const int64_t uid : {int64_t{-1}, int64_t{0}, int64_t{65}, int64_t{1} << 40})
    {
        SCOPED_TRACE(uid);
        SavedLaunchInputs saved = valid;
        saved.values["stats.uid"] = uid;
        EXPECT_EQ(readGfx942AttentionDenseLaunchInputs(saved).binding.stats, uid);
    }

    for(const int64_t uid : {int64_t{-2}, std::numeric_limits<int64_t>::min()})
    {
        SCOPED_TRACE(uid);
        SavedLaunchInputs belowNone = valid;
        belowNone.values["stats.uid"] = uid;
        expectReadRefusal(
            &readGfx942AttentionDenseLaunchInputs, belowNone, "launch value 'stats.uid'");
        expectReadRefusal(&readGfx942AttentionDenseLaunchInputs, belowNone, "must be int in [-1, ");
    }

    expectCommonReadRefusals(&readGfx942AttentionDenseLaunchInputs,
                             valid,
                             "hipkernel.gfx942_attention_dense.dispatch",
                             "stats.uid");
}

// gfx950 has no LSE kernel: neither of its contracts saves or reads a stats uid.
TEST(TestPackLaunchValues, Gfx950AttentionDenseSavesAndReadsNoStatsUid)
{
    const auto launch = gfx950AttentionDenseInputs();
    EXPECT_EQ(gfx950AttentionDenseLaunchValues(launch.binding, launch.problem, launch.blockM)
                  .count("stats.uid"),
              0U);

    for(const char* contract : {"hipkernel.gfx950_attention_dense.dispatch.v1",
                                "hipkernel.gfx950_attention_dense.dispatch.v2"})
    {
        SCOPED_TRACE(contract);
        SavedLaunchInputs withStats = savedGfx950AttentionDense();
        withStats.dispatchSymbol = contract;
        withStats.values["stats.uid"] = int64_t{-1};
        expectReadRefusal(&readGfx950AttentionDenseLaunchInputs,
                          withStats,
                          "launch value 'stats.uid' is not part of the contract");
    }
}

// gfx942 has one contract. gfx950's version names, and its values under them, are not it.
TEST(TestPackLaunchValues, Gfx942AttentionDenseReadsOnlyItsV1Contract)
{
    for(const char* other : {"hipkernel.gfx942_attention_dense.dispatch.v2",
                             "hipkernel.gfx950_attention_dense.dispatch.v1",
                             "hipkernel.gfx950_attention_dense.dispatch.v2"})
    {
        SCOPED_TRACE(other);
        auto saved = savedGfx942AttentionDense();
        saved.dispatchSymbol = other;
        expectReadRefusal(
            &readGfx942AttentionDenseLaunchInputs, saved, "names no launch-value contract");
        expectReadRefusal(&readGfx942AttentionDenseLaunchInputs,
                          saved,
                          "the handler reads the contract "
                          "'hipkernel.gfx942_attention_dense.dispatch.v1'");
    }
}

TEST(TestPackLaunchValues, Gfx942AttentionDenseRefusesMalformedValues)
{
    const auto valid = savedGfx942AttentionDense();
    expectCommonReadRefusals(&readGfx942AttentionDenseLaunchInputs,
                             valid,
                             "hipkernel.gfx942_attention_dense.dispatch",
                             "batch");

    SavedLaunchInputs wideBatch = valid;
    wideBatch.values["batch"] = int64_t{2147483648};
    expectReadRefusal(&readGfx942AttentionDenseLaunchInputs, wideBatch, "launch value 'batch'");

    // 0.1 as a double is not a float value, so no float scale was saved as it.
    SavedLaunchInputs doubleScale = valid;
    doubleScale.values["scale"] = 0.1;
    expectReadRefusal(&readGfx942AttentionDenseLaunchInputs, doubleScale, "launch value 'scale'");

    SavedLaunchInputs floatType = valid;
    floatType.values["data_type"] = std::string("FLOAT");
    expectReadRefusal(&readGfx942AttentionDenseLaunchInputs, floatType, "launch value 'data_type'");

    SavedLaunchInputs otherLayout = valid;
    otherLayout.values["stride_layout"] = std::string("bhsd");
    expectReadRefusal(
        &readGfx942AttentionDenseLaunchInputs, otherLayout, "launch value 'stride_layout'");
}

// gfx942 serves the shared dense scale range, so a saved scale outside it is refused alike.
TEST(TestPackLaunchValues, Gfx942AttentionDenseRefusesAScaleOutsideTheSupportedRange)
{
    expectDenseScaleRangeRestores(&readGfx942AttentionDenseLaunchInputs,
                                  savedGfx942AttentionDense());
}

// block_m is a positive multiple of 32 within a 1024-thread CTA. block_n is not saved, so
// the rest of the tile rule is not re-run.
TEST(TestPackLaunchValues, Gfx942AttentionDenseRefusesABlockMGfx942DoesNotBuild)
{
    const auto valid = savedGfx942AttentionDense();

    for(const int64_t blockM : {int64_t{48}, int64_t{0}, int64_t{-64}, int64_t{544}, int64_t{1024}})
    {
        SCOPED_TRACE(blockM);
        SavedLaunchInputs unbuilt = valid;
        unbuilt.values["block_m"] = blockM;
        expectReadRefusal(&readGfx942AttentionDenseLaunchInputs, unbuilt, "launch value 'block_m'");
        expectReadRefusal(&readGfx942AttentionDenseLaunchInputs,
                          unbuilt,
                          "is not a tile the kernel is built with");
    }

    for(const int64_t blockM : {int64_t{32}, int64_t{64}, int64_t{128}, int64_t{256}, int64_t{512}})
    {
        SCOPED_TRACE(blockM);
        SavedLaunchInputs built = valid;
        built.values["block_m"] = blockM;
        EXPECT_EQ(readGfx942AttentionDenseLaunchInputs(built).blockM, blockM);
    }
}

// ---------------------------------------------------------------------------
// Both dense readers: the mask and shape rules
//
// Both arches' graph and kernel match apply one set of mask and shape rules, so each reader
// refuses the same saved values. Both plans hold a causal window, seqlen_q 1536 against
// seqlen_kv 1600, and 8 query heads over 4 kv heads of 128 two-byte elements; only block_m
// differs.
// ---------------------------------------------------------------------------

struct AttentionDenseReader
{
    std::string_view label;
    AttentionDenseLaunchInputs (*read)(const SavedLaunchInputs&);
    SavedLaunchInputs (*saved)();
};

const AttentionDenseReader GFX942_ATTENTION_DENSE_READER{
    "Gfx942", &readGfx942AttentionDenseLaunchInputs, &savedGfx942AttentionDense};
const AttentionDenseReader GFX950_ATTENTION_DENSE_READER{
    "Gfx950", &readGfx950AttentionDenseLaunchInputs, &savedGfx950AttentionDense};

class TestAttentionDenseSavedValues : public ::testing::TestWithParam<const AttentionDenseReader*>
{
};

// causal and sliding_window are baked into the code object, so the launch never reads them;
// a plan holding a causal flag or window outside attention_dense::readLaunchInputs' mask
// rules (gfx950's window exception included) is refused rather than launched.
TEST_P(TestAttentionDenseSavedValues, RefusesAMaskOutsideTheMaskRules)
{
    const auto read = GetParam()->read;
    const auto valid = GetParam()->saved();

    for(const int64_t causal : {int64_t{2}, int64_t{-1}})
    {
        SCOPED_TRACE(causal);
        SavedLaunchInputs badCausal = valid;
        badCausal.values["causal"] = causal;
        expectReadRefusal(read, badCausal, "launch value 'causal'");
    }

    SavedLaunchInputs negativeWindow = valid;
    negativeWindow.values["sliding_window"] = int64_t{-1};
    expectReadRefusal(read, negativeWindow, "launch value 'sliding_window'");

    SavedLaunchInputs windowWithoutCausal = valid;
    windowWithoutCausal.values["causal"] = int64_t{0};
    expectReadRefusal(read, windowWithoutCausal, "launch value 'sliding_window'");

    // A 512-key window over seqlen_q 1536 and seqlen_kv 1024: row 1535 sees (1023, 1535], no
    // key.
    SavedLaunchInputs keylessRows = valid;
    keylessRows.values["seqlen_q"] = int64_t{1536};
    keylessRows.values["seqlen_kv"] = int64_t{1024};
    keylessRows.values["sliding_window"] = int64_t{512};
    expectReadRefusal(read, keylessRows, "launch value 'sliding_window'");
    expectReadRefusal(read, keylessRows, "no key");

    // One key wider and the last row sees the last key.
    SavedLaunchInputs lastRowReaches = keylessRows;
    lastRowReaches.values["sliding_window"] = int64_t{513};
    EXPECT_EQ(read(lastRowReaches).binding.slidingWindow, 513);

    // No window, causal or not, always restores.
    for(const int64_t causal : {int64_t{0}, int64_t{1}})
    {
        SCOPED_TRACE(causal);
        SavedLaunchInputs noWindow = keylessRows;
        noWindow.values["causal"] = causal;
        noWindow.values["sliding_window"] = int64_t{0};
        EXPECT_EQ(read(noWindow).binding.causal, causal);
    }
}

// Each of the next five refuses a shape the graph and kernel match never bind. All but
// head_size would launch out of bounds; the launch does not read head_size, but the size
// limits are computed from it.

// The kernel's kv head is the query head over num_query_heads / num_kv_heads: a group of 0
// for 2 over 4, and a kv head past the last for 6 over 4.
TEST_P(TestAttentionDenseSavedValues, RefusesQueryHeadsNotAMultipleOfKvHeads)
{
    const auto read = GetParam()->read;
    const auto valid = GetParam()->saved();

    for(const int64_t numQueryHeads : {int64_t{2}, int64_t{6}})
    {
        SCOPED_TRACE(numQueryHeads);
        SavedLaunchInputs ungrouped = valid;
        ungrouped.values["num_query_heads"] = numQueryHeads;
        expectReadRefusal(read, ungrouped, "launch value 'num_query_heads'");
        expectReadRefusal(read, ungrouped, "must be a multiple of 'num_kv_heads' 4");
    }
}

TEST_P(TestAttentionDenseSavedValues, RefusesAHeadSizeTheKernelIsNotBuiltFor)
{
    const auto read = GetParam()->read;
    const auto valid = GetParam()->saved();

    for(const int64_t headSize : {int64_t{32}, int64_t{96}, int64_t{256}})
    {
        SCOPED_TRACE(headSize);
        SavedLaunchInputs unbuilt = valid;
        unbuilt.values["head_size"] = headSize;
        expectReadRefusal(read, unbuilt, "launch value 'head_size'");
        expectReadRefusal(read, unbuilt, "must be 64 or 128");
    }

    SavedLaunchInputs built = valid;
    built.values["head_size"] = int64_t{64};
    EXPECT_EQ(read(built).problem.headSize, 64);
}

// One batch of 4 kv heads of 128 two-byte elements is 1024 bytes per key, so 2^21 keys is
// 2^31 bytes.
TEST_P(TestAttentionDenseSavedValues, RefusesKvOf2To31BytesOrMore)
{
    const auto read = GetParam()->read;
    auto valid = GetParam()->saved();
    valid.values["batch"] = int64_t{1};

    SavedLaunchInputs tooLong = valid;
    tooLong.values["seqlen_kv"] = int64_t{2097152};
    expectReadRefusal(
        read, tooLong, "launch values 'batch', 'seqlen_kv', 'num_kv_heads' and 'head_size'");
    expectReadRefusal(read, tooLong, "2^31 bytes or more");

    SavedLaunchInputs fits = valid;
    fits.values["seqlen_kv"] = int64_t{2097151};
    EXPECT_EQ(read(fits).problem.seqLenKv, 2097151);
}

// One batch of 8 query heads of 128 elements is 1024 elements per query row, so 2^31
// elements is 2^21 rows. The first multiple of block_m at or past that is refused, and the
// one before it restores: 2097216 and 2097120 at gfx942's 96, 2097152 and 2096896 at
// gfx950's 256.
TEST_P(TestAttentionDenseSavedValues, RefusesQoOf2To31ElementsOrMore)
{
    constexpr int64_t ROWS_AT_LIMIT = 2097152; // 2^21
    const auto read = GetParam()->read;
    auto valid = GetParam()->saved();
    valid.values["batch"] = int64_t{1};
    valid.values["sliding_window"] = int64_t{0};
    const int64_t blockM = read(valid).blockM;
    const int64_t firstPast = (ROWS_AT_LIMIT + blockM - 1) / blockM * blockM;

    SavedLaunchInputs tooLong = valid;
    tooLong.values["seqlen_q"] = firstPast;
    expectReadRefusal(
        read, tooLong, "launch values 'batch', 'seqlen_q', 'num_query_heads' and 'head_size'");
    expectReadRefusal(read, tooLong, "2^31 elements or more");

    SavedLaunchInputs fits = valid;
    fits.values["seqlen_q"] = firstPast - blockM;
    EXPECT_EQ(read(fits).problem.seqLenQ, firstPast - blockM);
}

// The grid is ceil(seqlen_q / block_m), and the last query tile is read and written whole.
TEST_P(TestAttentionDenseSavedValues, RefusesASeqLenQNotAMultipleOfBlockM)
{
    const auto read = GetParam()->read;
    const auto valid = GetParam()->saved();
    const int64_t blockM = read(valid).blockM;

    for(const int64_t seqLenQ : {int64_t{32}, int64_t{300}, int64_t{1568}})
    {
        SCOPED_TRACE(seqLenQ);
        SavedLaunchInputs ragged = valid;
        ragged.values["seqlen_q"] = seqLenQ;
        expectReadRefusal(read, ragged, "launch value 'seqlen_q'");
        expectReadRefusal(
            read, ragged, "must be a multiple of 'block_m' " + std::to_string(blockM));
    }

    SavedLaunchInputs oneTile = valid;
    oneTile.values["seqlen_q"] = blockM;
    EXPECT_EQ(read(oneTile).problem.seqLenQ, blockM);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionDenseSavedValues,
                         ::testing::Values(&GFX942_ATTENTION_DENSE_READER,
                                           &GFX950_ATTENTION_DENSE_READER),
                         [](const ::testing::TestParamInfo<const AttentionDenseReader*>& info) {
                             return std::string(info.param->label);
                         });

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
