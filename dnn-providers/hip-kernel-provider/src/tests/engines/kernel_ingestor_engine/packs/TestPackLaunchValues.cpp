// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
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

#ifdef HIPDNN_ENGINE_FLYDSL
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdDeviceHarness.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdLaunchValues.hpp"
#endif

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
    // key. gfx950 serves no window, so its reader refuses the plan. gfx942 restores it when
    // the plan binds no stats output and refuses it with one.
    SavedLaunchInputs keylessRows = valid;
    keylessRows.values["seqlen_q"] = int64_t{1536};
    keylessRows.values["seqlen_kv"] = int64_t{1024};
    keylessRows.values["sliding_window"] = int64_t{512};
    if(GetParam()->label == "Gfx942")
    {
        EXPECT_EQ(read(keylessRows).binding.slidingWindow, 512);

        SavedLaunchInputs keylessRowsWithStats = keylessRows;
        keylessRowsWithStats.values["stats.uid"] = int64_t{7};
        expectReadRefusal(read, keylessRowsWithStats, "launch value 'sliding_window'");
        expectReadRefusal(read, keylessRowsWithStats, "not yet supported with a stats output");

        // One key wider, the plan with stats restores.
        keylessRowsWithStats.values["sliding_window"] = int64_t{513};
        EXPECT_EQ(read(keylessRowsWithStats).binding.slidingWindow, 513);
    }
    else
    {
        expectReadRefusal(read, keylessRows, "launch value 'sliding_window'");
        expectReadRefusal(read, keylessRows, "this arch serves no sliding window");
    }

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

#ifdef HIPDNN_ENGINE_FLYDSL

// The eight FlyDSL SDPA backward contracts. The sample problem, its common values and each
// contract's row values are spelled out in FlydslSdpaBwdDeviceHarness.hpp (launch_sample); a
// contract writes the common values and its own row fields, and reads them back into the
// sample.
namespace sample = hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device::launch_sample;

TEST(TestPackLaunchValues, SplitPrepValuesMatchTheV1Contract)
{
    const auto launch = sample::prepInputs();

    const MetadataValues values = flydslSdpaBwdPrepLaunchValues(launch);

    EXPECT_EQ(values, sample::withRow(sample::prepRowValues()));
    EXPECT_EQ(FLYDSL_SDPA_BWD_SPLIT_PREP_DISPATCH_SYMBOL_V1,
              "hipkernel.flydsl_sdpa_bwd_split.prep.dispatch.v1");

    const auto read = readFlydslSdpaBwdPrepLaunchInputs(
        {std::string(FLYDSL_SDPA_BWD_SPLIT_PREP_DISPATCH_SYMBOL_V1), values});
    sample::expectSampleProblem(read.problem);
    sample::expectSampleUidsAndScalars(read.uids, read.scalars);
    EXPECT_EQ(read.headDim, 128);
    EXPECT_EQ(read.grid, "qblocks_by_heads_by_seqs");
}

TEST(TestPackLaunchValues, SplitDkdvValuesMatchTheV1Contract)
{
    const auto launch = sample::dkdvInputs();

    const MetadataValues values = flydslSdpaBwdDkdvLaunchValues(launch);

    EXPECT_EQ(values, sample::withRow(sample::dkdvRowValues()));
    EXPECT_EQ(FLYDSL_SDPA_BWD_SPLIT_DKDV_DISPATCH_SYMBOL_V1,
              "hipkernel.flydsl_sdpa_bwd_split.dkdv.dispatch.v1");

    const auto read = readFlydslSdpaBwdDkdvLaunchInputs(
        {std::string(FLYDSL_SDPA_BWD_SPLIT_DKDV_DISPATCH_SYMBOL_V1), values});
    sample::expectSampleProblem(read.problem);
    sample::expectSampleUidsAndScalars(read.uids, read.scalars);
    EXPECT_EQ(read.headDim, 128);
    EXPECT_EQ(read.blockThreads, 256);
    EXPECT_EQ(read.blockKv, 32);
    EXPECT_EQ(read.grid, "kheads_by_kvblocks_by_seqs");
}

TEST(TestPackLaunchValues, SplitDqValuesMatchTheV1Contract)
{
    const auto launch = sample::dqInputs();

    const MetadataValues values = flydslSdpaBwdDqLaunchValues(launch);

    EXPECT_EQ(values, sample::withRow(sample::dqRowValues()));
    EXPECT_EQ(FLYDSL_SDPA_BWD_SPLIT_DQ_DISPATCH_SYMBOL_V1,
              "hipkernel.flydsl_sdpa_bwd_split.dq.dispatch.v1");

    const auto read = readFlydslSdpaBwdDqLaunchInputs(
        {std::string(FLYDSL_SDPA_BWD_SPLIT_DQ_DISPATCH_SYMBOL_V1), values});
    sample::expectSampleProblem(read.problem);
    sample::expectSampleUidsAndScalars(read.uids, read.scalars);
    EXPECT_EQ(read.headDim, 128);
    EXPECT_EQ(read.blockThreads, 256);
    EXPECT_EQ(read.blockM, 64);
    EXPECT_EQ(read.grid, "qheads_by_qblocks_by_seqs");
}

TEST(TestPackLaunchValues, SplitDqReduceValuesMatchTheV1Contract)
{
    const auto launch = sample::dqInputs();

    const MetadataValues values = flydslSdpaBwdDqLaunchValues(launch);

    EXPECT_EQ(values, sample::withRow(sample::dqRowValues()));
    EXPECT_EQ(FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1,
              "hipkernel.flydsl_sdpa_bwd_split.dq_reduce.dispatch.v1");

    const auto read = readFlydslSdpaBwdDqReduceLaunchInputs(
        {std::string(FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1), values});
    sample::expectSampleProblem(read.problem);
    sample::expectSampleUidsAndScalars(read.uids, read.scalars);
    EXPECT_EQ(read.headDim, 128);
    EXPECT_EQ(read.blockThreads, 256);
    EXPECT_EQ(read.blockM, 64);
    EXPECT_EQ(read.grid, "qheads_by_qblocks_by_seqs");
}

TEST(TestPackLaunchValues, MergedValuesMatchTheV1Contract)
{
    const auto launch = sample::mergedInputs();

    const MetadataValues values = flydslSdpaBwdMergedLaunchValues(launch);

    EXPECT_EQ(values, sample::withRow(sample::mergedRowValues()));
    EXPECT_EQ(FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1,
              "hipkernel.flydsl_sdpa_bwd_merged.dispatch.v1");

    const auto read = readFlydslSdpaBwdMergedLaunchInputs(
        {std::string(FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1), values});
    sample::expectSampleProblem(read.problem, sample::mergedProblem());
    sample::expectSampleUidsAndScalars(read.uids, read.scalars);
    EXPECT_EQ(read.headDim, 128);
    EXPECT_EQ(read.blockThreads, 256);
    EXPECT_EQ(read.blockKv, 32);
    EXPECT_EQ(read.blockM, 64);
}

TEST(TestPackLaunchValues, MergedReduceValuesMatchTheV1Contract)
{
    const auto launch = sample::mergedInputs();

    const MetadataValues values = flydslSdpaBwdMergedLaunchValues(launch);

    EXPECT_EQ(values, sample::withRow(sample::mergedRowValues()));
    EXPECT_EQ(FLYDSL_SDPA_BWD_MERGED_REDUCE_DISPATCH_SYMBOL_V1,
              "hipkernel.flydsl_sdpa_bwd_merged_reduce.dispatch.v1");

    const auto read = readFlydslSdpaBwdMergedReduceLaunchInputs(
        {std::string(FLYDSL_SDPA_BWD_MERGED_REDUCE_DISPATCH_SYMBOL_V1), values});
    sample::expectSampleProblem(read.problem, sample::mergedProblem());
    sample::expectSampleUidsAndScalars(read.uids, read.scalars);
    EXPECT_EQ(read.headDim, 128);
    EXPECT_EQ(read.blockThreads, 256);
    EXPECT_EQ(read.blockKv, 32);
    EXPECT_EQ(read.blockM, 64);
}

TEST(TestPackLaunchValues, SplitDkdvU2ValuesMatchTheV1Contract)
{
    const auto launch = sample::dkdvU2Inputs();

    const MetadataValues values = flydslSdpaBwdDkdvLaunchValues(launch);

    EXPECT_EQ(values, sample::u2WithRow(sample::dkdvRowValues()));
    EXPECT_EQ(FLYDSL_SDPA_BWD_SPLIT_DKDV_U2_DISPATCH_SYMBOL_V1,
              "hipkernel.flydsl_sdpa_bwd_split.dkdv_u2.dispatch.v1");

    const auto read = readFlydslSdpaBwdDkdvU2LaunchInputs(
        {std::string(FLYDSL_SDPA_BWD_SPLIT_DKDV_U2_DISPATCH_SYMBOL_V1), values});
    sample::expectSampleProblem(read.problem, sample::u2Problem());
    sample::expectSampleUids(read.uids);
    sample::expectSampleScaleOnly(read.scalars);
    EXPECT_EQ(read.headDim, 128);
    EXPECT_EQ(read.blockThreads, 256);
    EXPECT_EQ(read.blockKv, 32);
    EXPECT_EQ(read.grid, "kheads_by_kvblocks_by_seqs");
}

TEST(TestPackLaunchValues, SplitDqU2ValuesMatchTheV1Contract)
{
    const auto launch = sample::dqU2Inputs();

    const MetadataValues values = flydslSdpaBwdDqLaunchValues(launch);

    EXPECT_EQ(values, sample::u2WithRow(sample::dqRowValues()));
    EXPECT_EQ(FLYDSL_SDPA_BWD_SPLIT_DQ_U2_DISPATCH_SYMBOL_V1,
              "hipkernel.flydsl_sdpa_bwd_split.dq_u2.dispatch.v1");

    const auto read = readFlydslSdpaBwdDqU2LaunchInputs(
        {std::string(FLYDSL_SDPA_BWD_SPLIT_DQ_U2_DISPATCH_SYMBOL_V1), values});
    sample::expectSampleProblem(read.problem, sample::u2Problem());
    sample::expectSampleUids(read.uids);
    sample::expectSampleScaleOnly(read.scalars);
    EXPECT_EQ(read.headDim, 128);
    EXPECT_EQ(read.blockThreads, 256);
    EXPECT_EQ(read.blockM, 64);
    EXPECT_EQ(read.grid, "qheads_by_qblocks_by_seqs");
}

// A contract serves the value heads of its own handler: a plan saved under another handler's
// contract is refused, so a `dkdv` plan never restores into the `dkdv_u2` launches.
TEST(TestPackLaunchValues, EachContractRefusesTheValueHeadsOfAnotherHandler)
{
    const std::string standardRule = "must be equal to num_kv_heads (4) for this contract, and the "
                                     "plan holds 2";
    const std::string u2Rule = "must be different from num_kv_heads for this contract, and the "
                               "plan holds 4";

    const auto u2Dkdv = sample::u2WithRow(sample::dkdvRowValues());
    expectReadRefusal(&readFlydslSdpaBwdDkdvLaunchInputs,
                      {std::string(FLYDSL_SDPA_BWD_SPLIT_DKDV_DISPATCH_SYMBOL_V1), u2Dkdv},
                      standardRule);
    const auto standardDkdv = sample::withRow(sample::dkdvRowValues());
    expectReadRefusal(&readFlydslSdpaBwdDkdvU2LaunchInputs,
                      {std::string(FLYDSL_SDPA_BWD_SPLIT_DKDV_U2_DISPATCH_SYMBOL_V1), standardDkdv},
                      u2Rule);

    const auto u2Dq = sample::u2WithRow(sample::dqRowValues());
    expectReadRefusal(&readFlydslSdpaBwdDqLaunchInputs,
                      {std::string(FLYDSL_SDPA_BWD_SPLIT_DQ_DISPATCH_SYMBOL_V1), u2Dq},
                      standardRule);
    expectReadRefusal(&readFlydslSdpaBwdDqReduceLaunchInputs,
                      {std::string(FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1), u2Dq},
                      standardRule);
    const auto standardDq = sample::withRow(sample::dqRowValues());
    expectReadRefusal(&readFlydslSdpaBwdDqU2LaunchInputs,
                      {std::string(FLYDSL_SDPA_BWD_SPLIT_DQ_U2_DISPATCH_SYMBOL_V1), standardDq},
                      u2Rule);

    auto u2Merged = sample::withRow(sample::mergedRowValues());
    u2Merged["num_v_heads"] = int64_t{2};
    expectReadRefusal(&readFlydslSdpaBwdMergedLaunchInputs,
                      {std::string(FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1), u2Merged},
                      standardRule);
    expectReadRefusal(&readFlydslSdpaBwdMergedReduceLaunchInputs,
                      {std::string(FLYDSL_SDPA_BWD_MERGED_REDUCE_DISPATCH_SYMBOL_V1), u2Merged},
                      standardRule);

    // The prep launch reads the values of either.
    EXPECT_NO_THROW(static_cast<void>(readFlydslSdpaBwdPrepLaunchInputs(
        {std::string(FLYDSL_SDPA_BWD_SPLIT_PREP_DISPATCH_SYMBOL_V1),
         sample::u2WithRow(sample::prepRowValues())})));
}

// V heads that differ from K's are served only without 4-byte data, a bias and a ragged tensor
// (the matcher declines them).
TEST(TestPackLaunchValues, SplitDqU2RefusesWhatTheMatcherDeclinesBesideValueHeads)
{
    const std::string why = "must be equal to num_kv_heads with 4-byte data, a bias or a ragged "
                            "tensor, and the plan holds 2";
    const auto valid = sample::u2WithRow(sample::dqRowValues());

    const std::vector<std::pair<std::string, std::function<void(MetadataValues&)>>> rows{
        {"a bias",
         [](MetadataValues& v) {
             v["bias.uid"] = int64_t{71};
             v["bias_element_bytes"] = int64_t{2};
         }},
        {"4-byte data",
         [](MetadataValues& v) {
             v["element_bytes"] = int64_t{4};
             v["bias_element_bytes"] = int64_t{4};
         }},
        {"a ragged Q",
         [](MetadataValues& v) {
             for(const char* tensor : {"q", "stats"})
             {
                 v[std::string(tensor) + ".ragged_offset_uid"] = int64_t{301};
                 v[std::string(tensor) + ".ragged_offset_multiplier"] = int64_t{3};
                 v[std::string(tensor) + ".stride_batch"] = int64_t{0};
             }
         }},
        {"a ragged K and no ragged Q", [](MetadataValues& v) {
             v["k.ragged_offset_uid"] = int64_t{302};
             v["k.ragged_offset_multiplier"] = int64_t{5};
             v["k.stride_batch"] = int64_t{0};
         }}};
    for(const auto& [label, edit] : rows)
    {
        SCOPED_TRACE(label);
        MetadataValues edited = valid;
        edit(edited);
        expectReadRefusal(&readFlydslSdpaBwdDqU2LaunchInputs,
                          {std::string(FLYDSL_SDPA_BWD_SPLIT_DQ_U2_DISPATCH_SYMBOL_V1), edited},
                          why);
    }
}

// An f32 bias is a launch of the split engine only, and a pass-by-value 32-bit seed is not a
// device one: the split reader keeps both.
TEST(TestPackLaunchValues, ASplitPlanKeepsAnF32BiasAndAPassByValueInt32Seed)
{
    auto values = sample::withRow(sample::dqRowValues());
    values["seed_width"] = std::string("int32");
    values["seed.data_type"] = std::string("INT32");
    values["offset.data_type"] = std::string("INT32");
    const auto read = readFlydslSdpaBwdDqReduceLaunchInputs(
        {std::string(FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1), values});
    EXPECT_EQ(read.problem.biasElementBytes, 4U);
    EXPECT_EQ(read.problem.seedWidth, flydsl_sdpa_bwd::SeedWidth::INT32);
    EXPECT_EQ(read.problem.seedForm, flydsl_sdpa_bwd::SeedForm::PASS_BY_VALUE);
}

// The merged engine declines an f32 bias, and serves a masked or biased head dim 192 problem
// like any other head dim.
TEST(TestPackLaunchValues, MergedRefusesWhatTheMergedMatcherDeclinesAndServesHeadDim192)
{
    const auto valid = sample::withRow(sample::mergedRowValues());
    const auto d192 = [](MetadataValues& v) {
        v["head_dim_qk"] = int64_t{192};
        v["head_dim_v"] = int64_t{192};
    };

    struct MergedReader
    {
        std::string contract;
        std::function<void(const SavedLaunchInputs&)> read;
    };
    const std::array<MergedReader, 2> readers{
        {{std::string(FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1),
          [](const SavedLaunchInputs& in) {
              static_cast<void>(readFlydslSdpaBwdMergedLaunchInputs(in));
          }},
         {std::string(FLYDSL_SDPA_BWD_MERGED_REDUCE_DISPATCH_SYMBOL_V1),
          [](const SavedLaunchInputs& in) {
              static_cast<void>(readFlydslSdpaBwdMergedReduceLaunchInputs(in));
          }}}};
    for(const auto& reader : readers)
    {
        SCOPED_TRACE(reader.contract);
        MetadataValues f32Bias = valid;
        f32Bias["bias_element_bytes"] = int64_t{4};
        expectReadRefusal(
            reader.read, {reader.contract, f32Bias}, "launch value 'bias_element_bytes'");
        expectReadRefusal(reader.read,
                          {reader.contract, f32Bias},
                          "must be 2 (the merged engine serves a 16-bit bias only), and the "
                          "plan holds 4");

        // A head dim 192 problem is served with a mask or a bias, with both and with neither.
        for(const bool masked : {false, true})
        {
            for(const bool biased : {false, true})
            {
                SCOPED_TRACE(std::string(masked ? "masked" : "unmasked")
                             + (biased ? " biased" : " unbiased"));
                MetadataValues served = valid;
                d192(served);
                served["masked"] = masked;
                if(!biased)
                {
                    served["bias.uid"] = flydsl_sdpa_bwd::NO_UID;
                    served["dbias.uid"] = flydsl_sdpa_bwd::NO_UID;
                }
                EXPECT_NO_THROW(reader.read({reader.contract, served}));
            }
        }
    }
    const auto read = readFlydslSdpaBwdMergedLaunchInputs(
        {std::string(FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1), [&] {
             MetadataValues plain = valid;
             d192(plain);
             plain["bias.uid"] = flydsl_sdpa_bwd::NO_UID;
             plain["dbias.uid"] = flydsl_sdpa_bwd::NO_UID;
             plain["masked"] = false;
             return plain;
         }()});
    EXPECT_EQ(read.problem.headDimQk, 192);
    EXPECT_EQ(read.problem.headDimV, 192);
}

// Each of the nine tensors saves its ragged offset under its own name and reads it back into
// its own member: no two tensors' entries are exchanged.
struct BwdTensorCase
{
    const char* name;
    flydsl_sdpa_bwd::RaggedOffset flydsl_sdpa_bwd::RaggedOffsets::*ragged;
    flydsl_sdpa_bwd::Strides flydsl_sdpa_bwd::SdpaBwdProblem::*strides;
};

const std::vector<BwdTensorCase>& bwdTensorCases()
{
    using flydsl_sdpa_bwd::RaggedOffsets;
    using flydsl_sdpa_bwd::SdpaBwdProblem;
    static const std::vector<BwdTensorCase> s_cases{
        {"q", &RaggedOffsets::q, &SdpaBwdProblem::q},
        {"k", &RaggedOffsets::k, &SdpaBwdProblem::k},
        {"v", &RaggedOffsets::v, &SdpaBwdProblem::v},
        {"o", &RaggedOffsets::o, &SdpaBwdProblem::o},
        {"do", &RaggedOffsets::dO, &SdpaBwdProblem::dO},
        {"dq", &RaggedOffsets::dQ, &SdpaBwdProblem::dQ},
        {"dk", &RaggedOffsets::dK, &SdpaBwdProblem::dK},
        {"dv", &RaggedOffsets::dV, &SdpaBwdProblem::dV},
        {"stats", &RaggedOffsets::stats, &SdpaBwdProblem::stats}};
    return s_cases;
}

TEST(TestPackLaunchValues, EveryTensorSavesAndRestoresItsOwnRaggedOffset)
{
    // The stats are ragged exactly when Q is, so Q and the stats are ragged together; every
    // other tensor is ragged alone. Each ragged offset has an uid and a multiplier of its own.
    const auto& cases = bwdTensorCases();
    for(size_t index = 0; index < cases.size(); ++index)
    {
        const std::string tensor = cases[index].name;
        if(tensor == "stats")
        {
            continue;
        }
        SCOPED_TRACE(tensor);
        const int64_t uid = 500 + static_cast<int64_t>(index);
        const int64_t multiplier = 20 + static_cast<int64_t>(index);
        const bool withStats = tensor == "q";

        auto launch = sample::dqInputs();
        launch.problem.ragged = {};
        for(const auto& each : cases)
        {
            (launch.problem.*each.strides).batch = 700;
        }
        launch.problem.ragged.*cases[index].ragged = {uid, multiplier};
        (launch.problem.*cases[index].strides).batch = 0;
        if(withStats)
        {
            launch.problem.ragged.stats = {600, 30};
            launch.problem.stats.batch = 0;
        }
        const auto wantUid = [&](const std::string& name) -> int64_t {
            if(name == tensor)
            {
                return uid;
            }
            return withStats && name == "stats" ? 600 : -1;
        };
        const auto wantMultiplier = [&](const std::string& name) -> int64_t {
            if(name == tensor)
            {
                return multiplier;
            }
            return withStats && name == "stats" ? 30 : 0;
        };

        // The writer puts the offset of the member under the tensor's name.
        const MetadataValues values = flydslSdpaBwdDqLaunchValues(launch);
        for(const auto& other : cases)
        {
            const std::string name = other.name;
            EXPECT_EQ(std::get<int64_t>(values.at(name + ".ragged_offset_uid")), wantUid(name))
                << name;
            EXPECT_EQ(std::get<int64_t>(values.at(name + ".ragged_offset_multiplier")),
                      wantMultiplier(name))
                << name;
        }

        // The reader puts the offset under the tensor's name into its member.
        const auto read = readFlydslSdpaBwdDqReduceLaunchInputs(
            {std::string(FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1), values});
        for(const auto& other : cases)
        {
            const std::string name = other.name;
            const auto& got = read.problem.ragged.*other.ragged;
            EXPECT_EQ(got.uid, wantUid(name)) << name;
            EXPECT_EQ(got.multiplier, wantMultiplier(name)) << name;
        }
    }
}

// The graph's window bounds are int64 and the launch clamps them itself, so a plan that saved a
// bound above INT32_MAX loads and holds the same bound.
TEST(TestPackLaunchValues,
     WindowBoundsBeyondTheLaunchRangeRoundTripThroughEveryContractThatReadsThem)
{
    constexpr int64_t BEYOND_INT32 = int64_t{1} << 40;
    constexpr int64_t LARGEST = std::numeric_limits<int64_t>::max();

    for(const auto& [left, right] : {std::pair<int64_t, int64_t>{BEYOND_INT32, LARGEST},
                                     std::pair<int64_t, int64_t>{LARGEST, BEYOND_INT32}})
    {
        SCOPED_TRACE(left);

        auto dq = sample::dqInputs();
        dq.problem.windowLeft = left;
        dq.problem.windowRight = right;
        const auto readDq = readFlydslSdpaBwdDqLaunchInputs(
            {std::string(FLYDSL_SDPA_BWD_SPLIT_DQ_DISPATCH_SYMBOL_V1),
             flydslSdpaBwdDqLaunchValues(dq)});
        EXPECT_EQ(readDq.problem.windowLeft, left);
        EXPECT_EQ(readDq.problem.windowRight, right);

        auto merged = sample::mergedInputs();
        merged.problem.windowLeft = left;
        merged.problem.windowRight = right;
        const auto readMerged = readFlydslSdpaBwdMergedLaunchInputs(
            {std::string(FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1),
             flydslSdpaBwdMergedLaunchValues(merged)});
        EXPECT_EQ(readMerged.problem.windowLeft, left);
        EXPECT_EQ(readMerged.problem.windowRight, right);
    }
}

// One row edits one value of a sample so that exactly one check of the shared reader refuses
// it. `name` and `expectation` are the two parts of the refusal message.
struct BwdReadRefusalRow
{
    std::string label;
    std::function<void(MetadataValues&)> edit;
    std::string name;
    std::string expectation;
};

// The edit that makes the sample's seed a device seed: the pass-by-value bindings go, the two
// tensor uids stay.
void makeSeedADeviceSeed(MetadataValues& values)
{
    values["seed_form"] = std::string("device");
    for(const char* name : {"seed.data_type",
                            "seed.user_supplied",
                            "offset.data_type",
                            "offset.user_supplied",
                            "offset.baked"})
    {
        values.erase(name);
    }
}

// The edit that gives the sample no seed: dropout stays on, the seed form and width go.
void dropTheSeed(MetadataValues& values)
{
    values["seed_form"] = std::string("none");
    values["seed_width"] = std::string("none");
    for(const char* name : {"seed.uid",
                            "seed.data_type",
                            "seed.user_supplied",
                            "offset.uid",
                            "offset.data_type",
                            "offset.user_supplied",
                            "offset.baked"})
    {
        values.erase(name);
    }
}

// One reader of a SDPA backward contract, with a sample of the values it reads.
struct BwdReader
{
    std::string label;
    std::string contract;
    MetadataValues valid;
    std::function<void(const SavedLaunchInputs&)> read;
    /// The values the contract has beyond the shared ones.
    std::vector<std::string> rowValues;
    /// The `grid` spellings the contract accepts; empty for a contract without a grid.
    std::vector<std::string> grids;
    /// The relation the contract requires between the value heads and the key heads.
    enum class ValueHeads
    {
        ANY,
        EQUAL,
        DIFFERENT
    } valueHeads
        = ValueHeads::EQUAL;
};

template <typename TReader>
BwdReader bwdReader(std::string label,
                    std::string_view contract,
                    MetadataValues valid,
                    TReader read,
                    std::vector<std::string> rowValues,
                    std::vector<std::string> grids,
                    BwdReader::ValueHeads valueHeads = BwdReader::ValueHeads::EQUAL)
{
    return BwdReader{std::move(label),
                     std::string(contract),
                     std::move(valid),
                     [read](const SavedLaunchInputs& inputs) { static_cast<void>(read(inputs)); },
                     std::move(rowValues),
                     std::move(grids),
                     valueHeads};
}

// Every reader of the eight contracts, each with a valid sample of its own.
const std::vector<BwdReader>& bwdReaders()
{
    using Heads = BwdReader::ValueHeads;
    const std::vector<std::string> dqRow{"head_dim", "block_threads", "block_m", "grid"};
    const std::vector<std::string> dkdvRow{"head_dim", "block_threads", "block_kv", "grid"};
    const std::vector<std::string> mergedRow{"head_dim", "block_threads", "block_kv", "block_m"};
    const std::vector<std::string> dqGrids{"qheads_by_qblocks_by_seqs"};
    const std::vector<std::string> dkdvGrids{"kheads_by_kvblocks_by_seqs",
                                             "kvblocks_by_kheads_by_seqs"};
    static const std::vector<BwdReader> s_readers{
        bwdReader("prep",
                  FLYDSL_SDPA_BWD_SPLIT_PREP_DISPATCH_SYMBOL_V1,
                  sample::withRow(sample::prepRowValues()),
                  &readFlydslSdpaBwdPrepLaunchInputs,
                  {"head_dim", "grid"},
                  {"qblocks_by_heads_by_seqs"},
                  Heads::ANY),
        bwdReader("dkdv",
                  FLYDSL_SDPA_BWD_SPLIT_DKDV_DISPATCH_SYMBOL_V1,
                  sample::withRow(sample::dkdvRowValues()),
                  &readFlydslSdpaBwdDkdvLaunchInputs,
                  dkdvRow,
                  dkdvGrids),
        bwdReader("dkdv_u2",
                  FLYDSL_SDPA_BWD_SPLIT_DKDV_U2_DISPATCH_SYMBOL_V1,
                  sample::u2WithRow(sample::dkdvRowValues()),
                  &readFlydslSdpaBwdDkdvU2LaunchInputs,
                  dkdvRow,
                  dkdvGrids,
                  Heads::DIFFERENT),
        bwdReader("dq",
                  FLYDSL_SDPA_BWD_SPLIT_DQ_DISPATCH_SYMBOL_V1,
                  sample::withRow(sample::dqRowValues()),
                  &readFlydslSdpaBwdDqLaunchInputs,
                  dqRow,
                  dqGrids),
        bwdReader("dq_reduce",
                  FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1,
                  sample::withRow(sample::dqRowValues()),
                  &readFlydslSdpaBwdDqReduceLaunchInputs,
                  dqRow,
                  dqGrids),
        bwdReader("dq_u2",
                  FLYDSL_SDPA_BWD_SPLIT_DQ_U2_DISPATCH_SYMBOL_V1,
                  sample::u2WithRow(sample::dqRowValues()),
                  &readFlydslSdpaBwdDqU2LaunchInputs,
                  dqRow,
                  dqGrids,
                  Heads::DIFFERENT),
        bwdReader("merged",
                  FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1,
                  sample::withRow(sample::mergedRowValues()),
                  &readFlydslSdpaBwdMergedLaunchInputs,
                  mergedRow,
                  {}),
        bwdReader("merged_reduce",
                  FLYDSL_SDPA_BWD_MERGED_REDUCE_DISPATCH_SYMBOL_V1,
                  sample::withRow(sample::mergedRowValues()),
                  &readFlydslSdpaBwdMergedReduceLaunchInputs,
                  mergedRow,
                  {})};
    return s_readers;
}

const BwdReader& bwdReader(std::string_view label)
{
    const auto& readers = bwdReaders();
    return *std::find_if(readers.begin(), readers.end(), [&](const BwdReader& reader) {
        return reader.label == label;
    });
}

// Expects @p reader to refuse @p edit of its sample, naming @p name and saying @p expectation.
void expectBwdReaderRefuses(const BwdReader& reader,
                            const std::function<void(MetadataValues&)>& edit,
                            const std::string& name,
                            const std::string& expectation)
{
    MetadataValues edited = reader.valid;
    edit(edited);
    const SavedLaunchInputs inputs{reader.contract, edited};
    expectReadRefusal(reader.read, inputs, name);
    expectReadRefusal(reader.read, inputs, expectation);
}

// Expects @p reader to accept @p edit of its sample.
void expectBwdReaderAccepts(const BwdReader& reader,
                            const std::function<void(MetadataValues&)>& edit)
{
    MetadataValues edited = reader.valid;
    edit(edited);
    const SavedLaunchInputs inputs{reader.contract, edited};
    EXPECT_NO_THROW(reader.read(inputs));
}

// The shared checks, run against the four readers whose sample is the shared problem. The
// values of one reader's own row are refused by EachReaderRefusesEachRowValueOutsideItsContract.
TEST(TestPackLaunchValues, EachReaderRefusesEachValueOutsideTheSharedContract)
{
    const std::string seedRule
        = "must be set exactly when dropout_p is above 0, with a seed width, and the plan holds ";
    const std::string lengthRule = "must be both length uids set with padding_mask and neither "
                                   "without it, and the plan holds ";

    const std::vector<BwdReadRefusalRow> rows{
        {"an element width of 3",
         [](MetadataValues& v) { v["element_bytes"] = int64_t{3}; },
         "launch value 'element_bytes'",
         "must be 2 or 4, and the plan holds 3"},
        {"kv heads that do not divide the heads",
         [](MetadataValues& v) { v["num_kv_heads"] = int64_t{3}; },
         "launch value 'num_kv_heads'",
         "must be a divisor of num_heads (8), and the plan holds 3"},
        {"a padding mask without a length uid",
         [](MetadataValues& v) { v["seq_len_q.uid"] = flydsl_sdpa_bwd::NO_UID; },
         "launch value 'seq_len_q.uid'",
         lengthRule + "-1 and 62"},
        {"length uids without a padding mask",
         [](MetadataValues& v) { v["padding_mask"] = false; },
         "launch value 'seq_len_q.uid'",
         lengthRule + "61 and 62"},
        {"a dbias uid without a bias",
         [](MetadataValues& v) { v["bias.uid"] = flydsl_sdpa_bwd::NO_UID; },
         "launch value 'dbias.uid'",
         "must be unset when there is no bias, and the plan holds 72"},
        {"a dropout probability of 1",
         [](MetadataValues& v) { v["dropout_p"] = 1.0; },
         "launch value 'dropout_p'",
         "must be in [0, 1), and the plan holds 1.000000"},
        {"a seed form without dropout",
         [](MetadataValues& v) { v["dropout_p"] = 0.0; },
         "launch value 'seed_form'",
         seedRule + "pass_by_value with width int64"},
        {"a seed form without a seed width",
         [](MetadataValues& v) { v["seed_width"] = std::string("none"); },
         "launch value 'seed_form'",
         seedRule + "pass_by_value with width none"},
        {"dropout without a seed form",
         dropTheSeed,
         "launch value 'seed_form'",
         seedRule + "none with width none"},
        {"a seed that is not the seed width",
         [](MetadataValues& v) { v["seed.data_type"] = std::string("INT32"); },
         "launch value 'seed.data_type'",
         "must be INT64, and the plan holds INT32"},
        {"an offset that is not the seed width",
         [](MetadataValues& v) { v["offset.data_type"] = std::string("INT32"); },
         "launch value 'offset.data_type'",
         "must be INT64, and the plan holds INT32"},
        {"a 32-bit seed width over a 64-bit seed",
         [](MetadataValues& v) { v["seed_width"] = std::string("int32"); },
         "launch value 'seed.data_type'",
         "must be INT32, and the plan holds INT64"},
        {"a device seed without a seed uid",
         [](MetadataValues& v) {
             makeSeedADeviceSeed(v);
             v["seed.uid"] = flydsl_sdpa_bwd::NO_UID;
         },
         "launch value 'seed.uid'",
         "must be a tensor uid, and the plan holds unset"},
        {"a device seed without an offset uid",
         [](MetadataValues& v) {
             makeSeedADeviceSeed(v);
             v["offset.uid"] = flydsl_sdpa_bwd::NO_UID;
         },
         "launch value 'seed.uid'",
         "must be a tensor uid, and the plan holds unset"},
        {"an unknown scale source",
         [](MetadataValues& v) { v["scale_source"] = std::string("bogus"); },
         "launch value 'scale_source'",
         "must be one of 'value' 'tensor', and the plan holds 'bogus'"},
        {"an unknown seed form",
         [](MetadataValues& v) { v["seed_form"] = std::string("bogus"); },
         "launch value 'seed_form'",
         "must be one of 'none' 'device' 'pass_by_value', and the plan holds 'bogus'"},
        {"an unknown seed width",
         [](MetadataValues& v) { v["seed_width"] = std::string("bogus"); },
         "launch value 'seed_width'",
         "must be one of 'none' 'int32' 'int64', and the plan holds 'bogus'"},
        {"a ragged offset uid below the unset sentinel",
         [](MetadataValues& v) { v["q.ragged_offset_uid"] = int64_t{-2}; },
         "launch value 'q.ragged_offset_uid'",
         "must be int in [-1, 9223372036854775807], and the plan holds -2"},
        {"a negative ragged offset multiplier",
         [](MetadataValues& v) { v["q.ragged_offset_multiplier"] = int64_t{-1}; },
         "launch value 'q.ragged_offset_multiplier'",
         "must be int in [0, 9223372036854775807], and the plan holds -1"},
        {"a ragged offset uid with a zero multiplier",
         [](MetadataValues& v) { v["k.ragged_offset_multiplier"] = int64_t{0}; },
         "launch value 'k.ragged_offset_multiplier'",
         "must be at least 1 beside a ragged offset uid, and the plan holds 0"},
        {"a ragged offset multiplier without a uid",
         [](MetadataValues& v) { v["v.ragged_offset_multiplier"] = int64_t{2}; },
         "launch value 'v.ragged_offset_multiplier'",
         "must be 0 without a ragged offset uid, and the plan holds 2"},
        {"a ragged offset without a padding mask",
         [](MetadataValues& v) {
             v["padding_mask"] = false;
             v["seq_len_q.uid"] = flydsl_sdpa_bwd::NO_UID;
             v["seq_len_kv.uid"] = flydsl_sdpa_bwd::NO_UID;
         },
         "launch value 'q.ragged_offset_uid'",
         "must be unset without padding_mask, and the plan holds 301"},
        {"a ragged tensor with a batch stride",
         [](MetadataValues& v) { v["o.stride_batch"] = int64_t{5}; },
         "launch value 'o.stride_batch'",
         "must be 0 on a ragged tensor, and the plan holds 5"},
        {"ragged stats with a batch stride",
         [](MetadataValues& v) { v["stats.stride_batch"] = int64_t{4}; },
         "launch value 'stats.stride_batch'",
         "must be 0 on a ragged tensor, and the plan holds 4"},
        {"a negative stats stride",
         [](MetadataValues& v) { v["stats.stride_head"] = int64_t{-1}; },
         "launch value 'stats.stride_head'",
         "must be int in [0, 9223372036854775807], and the plan holds -1"},
        {"ragged stats beside a dense Q",
         [](MetadataValues& v) {
             v["q.ragged_offset_uid"] = flydsl_sdpa_bwd::NO_UID;
             v["q.ragged_offset_multiplier"] = int64_t{0};
         },
         "launch value 'stats.ragged_offset_uid'",
         "must be set exactly when q.ragged_offset_uid is, and the plan holds 309"},
        {"dense stats beside a ragged Q",
         [](MetadataValues& v) {
             v["stats.ragged_offset_uid"] = flydsl_sdpa_bwd::NO_UID;
             v["stats.ragged_offset_multiplier"] = int64_t{0};
         },
         "launch value 'stats.ragged_offset_uid'",
         "must be set exactly when q.ragged_offset_uid is, and the plan holds -1"},
        {"a left window bound below the unbounded sentinel",
         [](MetadataValues& v) { v["window_left"] = int64_t{-2}; },
         "launch value 'window_left'",
         "must be int in [-1, 9223372036854775807], and the plan holds -2"},
        {"a right window bound below the unbounded sentinel",
         [](MetadataValues& v) { v["window_right"] = int64_t{-2}; },
         "launch value 'window_right'",
         "must be int in [-1, 9223372036854775807], and the plan holds -2"},
        {"a left window bound of 0, which the matcher declines",
         [](MetadataValues& v) { v["window_left"] = int64_t{0}; },
         "launch value 'window_left'",
         "must be -1 or at least 1, and the plan holds 0"},
        {"a device seed of 32 bits, which the matcher declines",
         [](MetadataValues& v) {
             makeSeedADeviceSeed(v);
             v["seed_width"] = std::string("int32");
         },
         "launch value 'seed_width'",
         "must be int64 beside a device seed (the kernels read 8 bytes there), and the plan "
         "holds int32"},
        {"a bias width of 3",
         [](MetadataValues& v) { v["bias_element_bytes"] = int64_t{3}; },
         "launch value 'bias_element_bytes'",
         "must be 2 or 4, and the plan holds 3"},
        {"a bias width above 4",
         [](MetadataValues& v) { v["bias_element_bytes"] = int64_t{5}; },
         "launch value 'bias_element_bytes'",
         "must be int in [2, 4], and the plan holds 5"},
        {"a bias width of 4 without a bias",
         [](MetadataValues& v) {
             v["bias.uid"] = flydsl_sdpa_bwd::NO_UID;
             v["dbias.uid"] = flydsl_sdpa_bwd::NO_UID;
         },
         "launch value 'bias_element_bytes'",
         "must be equal to element_bytes (2) when there is no bias, and the plan holds 4"},
        {"a 16-bit bias beside 4-byte data",
         [](MetadataValues& v) {
             v["element_bytes"] = int64_t{4};
             v["bias_element_bytes"] = int64_t{2};
         },
         "launch value 'bias_element_bytes'",
         "must be 4 beside 4-byte data, and the plan holds 2"},
        {"no value heads",
         [](MetadataValues& v) { v["num_v_heads"] = int64_t{0}; },
         "launch value 'num_v_heads'",
         "must be int in [1, 2147483647], and the plan holds 0"},
        {"value heads that do not divide the heads",
         [](MetadataValues& v) { v["num_v_heads"] = int64_t{3}; },
         "launch value 'num_v_heads'",
         "must be a divisor of num_heads (8), and the plan holds 3"},
        {"an extent beyond INT32_MAX",
         [](MetadataValues& v) { v["batch"] = int64_t{2147483648}; },
         "launch value 'batch'",
         "must be int in [1, 2147483647], and the plan holds 2147483648"},
        {"a Q/K head dim of 0",
         [](MetadataValues& v) { v["head_dim_qk"] = int64_t{0}; },
         "launch value 'head_dim_qk'",
         "must be int in [1, 512], and the plan holds 0"},
        {"a Q/K head dim above 512",
         [](MetadataValues& v) { v["head_dim_qk"] = int64_t{513}; },
         "launch value 'head_dim_qk'",
         "must be int in [1, 512], and the plan holds 513"},
        {"a V head dim of 0",
         [](MetadataValues& v) { v["head_dim_v"] = int64_t{0}; },
         "launch value 'head_dim_v'",
         "must be int in [1, 512], and the plan holds 0"},
        {"a V head dim above 512",
         [](MetadataValues& v) { v["head_dim_v"] = int64_t{513}; },
         "launch value 'head_dim_v'",
         "must be int in [1, 512], and the plan holds 513"},
        {"a scale above 2^20",
         [](MetadataValues& v) { v["scale_value"] = 1048577.0; },
         "launch value 'scale_value'",
         "must be a double that a float holds exactly, in [-1048576.000000, 1048576.000000], and "
         "the plan holds 1048577.000000"},
        {"a scale no float holds",
         [](MetadataValues& v) { v["scale_value"] = 0.1; },
         "launch value 'scale_value'",
         "must be a double that a float holds exactly, in [-1048576.000000, 1048576.000000], and "
         "the plan holds 0.100000"},
        {"a diagonal offset above INT32_MAX",
         [](MetadataValues& v) { v["diagonal_offset"] = int64_t{2147483648}; },
         "launch value 'diagonal_offset'",
         "must be int in [-2147483647, 2147483647], and the plan holds 2147483648"},
        {"a diagonal offset below -INT32_MAX",
         [](MetadataValues& v) { v["diagonal_offset"] = int64_t{-2147483648}; },
         "launch value 'diagonal_offset'",
         "must be int in [-2147483647, 2147483647], and the plan holds -2147483648"},
        {"a dropout probability below 0",
         [](MetadataValues& v) { v["dropout_p"] = -0.5; },
         "launch value 'dropout_p'",
         "must be a double that a float holds exactly, in [0.000000, 1.000000], and the plan "
         "holds -0.500000"},
        {"a dropout probability no float holds",
         [](MetadataValues& v) { v["dropout_p"] = 0.1; },
         "launch value 'dropout_p'",
         "must be a double that a float holds exactly, in [0.000000, 1.000000], and the plan "
         "holds 0.100000"},
        {"a negative stride on the first strided tensor",
         [](MetadataValues& v) { v["q.stride_head"] = int64_t{-1}; },
         "launch value 'q.stride_head'",
         "must be int in [0, 9223372036854775807], and the plan holds -1"},
        {"a negative stride on the last strided tensor",
         [](MetadataValues& v) { v["dv.stride_seq"] = int64_t{-1}; },
         "launch value 'dv.stride_seq'",
         "must be int in [0, 9223372036854775807], and the plan holds -1"}};

    // The sample of every one of these readers is the shared problem, so every row applies to
    // each of them: the shared checks are one function, and a reader that stopped calling it
    // would show here.
    for(const char* label : {"prep", "dkdv", "dq", "dq_reduce"})
    {
        const auto& reader = bwdReader(label);
        for(const auto& row : rows)
        {
            SCOPED_TRACE(std::string(label) + ": " + row.label);
            expectBwdReaderRefuses(reader, row.edit, row.name, row.expectation);
        }
    }
}

// The values a contract has beyond the shared ones, refused one reader at a time: a head dim of
// 0 or above 512, a block of 0 threads or more than a block holds, a tile of 0 rows or beyond
// INT32_MAX, a grid spelling of another launch, a grid on a contract that has none, and the
// value heads of another handler.
TEST(TestPackLaunchValues, EachReaderRefusesEachRowValueOutsideItsContract)
{
    const auto has = [](const BwdReader& reader, const char* value) {
        return std::find(reader.rowValues.begin(), reader.rowValues.end(), value)
               != reader.rowValues.end();
    };
    const auto set = [](const char* name, int64_t value) {
        return [=](MetadataValues& v) { v[name] = value; };
    };
    const std::string threads = "must be int in [1, 1024], and the plan holds ";
    const std::string rows = "must be int in [1, 2147483647], and the plan holds ";

    for(const auto& reader : bwdReaders())
    {
        SCOPED_TRACE(reader.label);
        EXPECT_TRUE(has(reader, "head_dim"));
        expectBwdReaderRefuses(reader,
                               set("head_dim", 0),
                               "launch value 'head_dim'",
                               "must be int in [1, 512], and the plan holds 0");
        expectBwdReaderRefuses(reader,
                               set("head_dim", 513),
                               "launch value 'head_dim'",
                               "must be int in [1, 512], and the plan holds 513");
        if(has(reader, "block_threads"))
        {
            expectBwdReaderRefuses(
                reader, set("block_threads", 0), "launch value 'block_threads'", threads + "0");
            expectBwdReaderRefuses(reader,
                                   set("block_threads", 1025),
                                   "launch value 'block_threads'",
                                   threads + "1025");
        }
        for(const char* tile : {"block_kv", "block_m"})
        {
            if(has(reader, tile))
            {
                const std::string name = std::string("launch value '") + tile + "'";
                expectBwdReaderRefuses(reader, set(tile, 0), name, rows + "0");
                expectBwdReaderRefuses(reader, set(tile, 2147483648), name, rows + "2147483648");
            }
        }
        if(reader.grids.empty())
        {
            expectBwdReaderRefuses(
                reader,
                [](MetadataValues& v) { v["grid"] = std::string("qheads_by_qblocks_by_seqs"); },
                "launch value 'grid'",
                "is not part of the contract");
        }
        else
        {
            // The spelling of a launch this contract is not: the prep grid for the others, a dq
            // grid for prep.
            const std::string other
                = reader.label == "prep" ? "qheads_by_qblocks_by_seqs" : "qblocks_by_heads_by_seqs";
            std::string accepted;
            for(const auto& grid : reader.grids)
            {
                accepted += (accepted.empty() ? "'" : " '") + grid + "'";
            }
            std::string message = "must be one of ";
            message.append(accepted).append(", and the plan holds '").append(other).append("'");
            expectBwdReaderRefuses(
                reader,
                [&](MetadataValues& v) { v["grid"] = other; },
                "launch value 'grid'",
                message);
        }
    }

    // The value heads: a contract of `Hk == Hv` refuses other value heads, a `u2` contract
    // refuses equal ones, and prep serves either.
    for(const auto& reader : bwdReaders())
    {
        SCOPED_TRACE(reader.label);
        switch(reader.valueHeads)
        {
        case BwdReader::ValueHeads::EQUAL:
            expectBwdReaderRefuses(reader,
                                   set("num_v_heads", 2),
                                   "launch value 'num_v_heads'",
                                   "must be equal to num_kv_heads (4) for this contract, and the "
                                   "plan holds 2");
            break;
        case BwdReader::ValueHeads::DIFFERENT:
            expectBwdReaderRefuses(reader,
                                   set("num_v_heads", 4),
                                   "launch value 'num_v_heads'",
                                   "must be different from num_kv_heads for this contract, and "
                                   "the plan holds 4");
            break;
        case BwdReader::ValueHeads::ANY:
        default:
            EXPECT_NO_THROW(
                reader.read({reader.contract, sample::u2WithRow(sample::prepRowValues())}));
            EXPECT_NO_THROW(reader.read({reader.contract, reader.valid}));
            break;
        }
    }
}

// A value exactly at a limit of its range reads back, beside the rows that refuse the value one
// step beyond it.
TEST(TestPackLaunchValues, EachReaderAcceptsEachRowValueAtItsLimit)
{
    const auto set = [](const char* name, int64_t value) {
        return [=](MetadataValues& v) { v[name] = value; };
    };
    for(const auto& reader : bwdReaders())
    {
        SCOPED_TRACE(reader.label);
        for(const int64_t headDim : {int64_t{1}, int64_t{512}})
        {
            expectBwdReaderAccepts(reader, set("head_dim", headDim));
        }
        for(const char* name : {"block_threads", "block_kv", "block_m"})
        {
            if(std::find(reader.rowValues.begin(), reader.rowValues.end(), name)
               == reader.rowValues.end())
            {
                continue;
            }
            const int64_t largest = std::string(name) == "block_threads" ? 1024 : 2147483647;
            for(const int64_t value : {int64_t{1}, largest})
            {
                SCOPED_TRACE(std::string(name) + " " + std::to_string(value));
                expectBwdReaderAccepts(reader, set(name, value));
            }
        }
        for(const auto& grid : reader.grids)
        {
            SCOPED_TRACE(grid);
            expectBwdReaderAccepts(reader, [&](MetadataValues& v) { v["grid"] = grid; });
        }
    }
}

// A shared value exactly at a limit of its range reads back into its own member of the problem.
TEST(TestPackLaunchValues, TheSharedReaderAcceptsEachValueAtItsLimit)
{
    constexpr int64_t INT32_MAX_VALUE = 2147483647;
    constexpr int64_t INT64_MAX_VALUE = std::numeric_limits<int64_t>::max();
    const auto& reader = bwdReader("dq_reduce");

    struct Row
    {
        std::string label;
        std::function<void(MetadataValues&)> edit;
        std::function<void(const flydsl_sdpa_bwd::SdpaBwdProblem&)> check;
    };
    const std::vector<Row> rows{
        {"a batch of INT32_MAX",
         [&](MetadataValues& v) { v["batch"] = INT32_MAX_VALUE; },
         [&](const auto& p) { EXPECT_EQ(p.batch, INT32_MAX_VALUE); }},
        {"a Q length of INT32_MAX",
         [&](MetadataValues& v) { v["seqlen_q"] = INT32_MAX_VALUE; },
         [&](const auto& p) { EXPECT_EQ(p.seqLenQ, INT32_MAX_VALUE); }},
        {"a K length of INT32_MAX",
         [&](MetadataValues& v) { v["seqlen_kv"] = INT32_MAX_VALUE; },
         [&](const auto& p) { EXPECT_EQ(p.seqLenKv, INT32_MAX_VALUE); }},
        {"head dims of 512",
         [](MetadataValues& v) {
             v["head_dim_qk"] = int64_t{512};
             v["head_dim_v"] = int64_t{512};
         },
         [](const auto& p) {
             EXPECT_EQ(p.headDimQk, 512);
             EXPECT_EQ(p.headDimV, 512);
         }},
        {"head dims of 1",
         [](MetadataValues& v) {
             v["head_dim_qk"] = int64_t{1};
             v["head_dim_v"] = int64_t{1};
         },
         [](const auto& p) {
             EXPECT_EQ(p.headDimQk, 1);
             EXPECT_EQ(p.headDimV, 1);
         }},
        {"4-byte data and bias",
         [](MetadataValues& v) {
             v["element_bytes"] = int64_t{4};
             v["bias_element_bytes"] = int64_t{4};
             v["num_v_heads"] = int64_t{4};
         },
         [](const auto& p) {
             EXPECT_EQ(p.elementBytes, 4U);
             EXPECT_EQ(p.biasElementBytes, 4U);
         }},
        {"a scale of 2^20",
         [](MetadataValues& v) { v["scale_value"] = 1048576.0; },
         [](const auto& p) { EXPECT_EQ(p.scaleValue, 1048576.0F); }},
        {"a scale of -2^20",
         [](MetadataValues& v) { v["scale_value"] = -1048576.0; },
         [](const auto& p) { EXPECT_EQ(p.scaleValue, -1048576.0F); }},
        {"a diagonal offset of INT32_MAX",
         [&](MetadataValues& v) { v["diagonal_offset"] = INT32_MAX_VALUE; },
         [&](const auto& p) { EXPECT_EQ(p.diagonalOffset, INT32_MAX_VALUE); }},
        {"a diagonal offset of -INT32_MAX",
         [&](MetadataValues& v) { v["diagonal_offset"] = -INT32_MAX_VALUE; },
         [&](const auto& p) { EXPECT_EQ(p.diagonalOffset, -INT32_MAX_VALUE); }},
        {"the largest dropout probability below 1",
         [](MetadataValues& v) { v["dropout_p"] = 1.0 - 1.0 / 16777216.0; },
         [](const auto& p) { EXPECT_LT(p.dropoutP, 1.0F); }},
        {"strides of 0",
         [](MetadataValues& v) {
             v["q.stride_head"] = int64_t{0};
             v["dv.stride_seq"] = int64_t{0};
         },
         [](const auto& p) {
             EXPECT_EQ(p.q.head, 0);
             EXPECT_EQ(p.dV.seq, 0);
         }},
        {"strides of INT64_MAX",
         [&](MetadataValues& v) {
             v["q.stride_head"] = INT64_MAX_VALUE;
             v["dv.stride_seq"] = INT64_MAX_VALUE;
         },
         [&](const auto& p) {
             EXPECT_EQ(p.q.head, INT64_MAX_VALUE);
             EXPECT_EQ(p.dV.seq, INT64_MAX_VALUE);
         }}};
    for(const auto& row : rows)
    {
        SCOPED_TRACE(row.label);
        MetadataValues edited = reader.valid;
        row.edit(edited);
        const auto read = readFlydslSdpaBwdDqReduceLaunchInputs({reader.contract, edited});
        row.check(read.problem);
    }
}

// A launch that holds no scalar for a pass-by-value operand cannot be saved: the handler has a
// defect, so the refusal is an internal error and not a load refusal.
TEST(TestPackLaunchValues, SplitDqRefusesToWriteAPassByValueOperandTheLaunchHoldsNoScalarFor)
{
    const std::string scaleWhy = "cannot save the SDPA backward launch: the scale is a tensor and "
                                 "the launch holds no scalar for it";
    const std::string seedWhy = "cannot save the SDPA backward launch: the seed is pass-by-value "
                                "and the launch holds no scalar for it";

    auto noScale = sample::dqInputs();
    noScale.scalars.scale.reset();
    serialization::expectPluginInternalError(
        [&]() { static_cast<void>(flydslSdpaBwdDqLaunchValues(noScale)); }, scaleWhy);

    auto noSeed = sample::dqInputs();
    noSeed.scalars.seed.reset();
    serialization::expectPluginInternalError(
        [&]() { static_cast<void>(flydslSdpaBwdDqLaunchValues(noSeed)); }, seedWhy);

    auto noOffset = sample::dqInputs();
    noOffset.scalars.offset.reset();
    serialization::expectPluginInternalError(
        [&]() { static_cast<void>(flydslSdpaBwdDqLaunchValues(noOffset)); }, seedWhy);
}

#endif // HIPDNN_ENGINE_FLYDSL

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
