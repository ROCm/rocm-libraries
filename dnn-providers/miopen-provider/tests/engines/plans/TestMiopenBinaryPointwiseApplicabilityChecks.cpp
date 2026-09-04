// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "engines/plans/MiopenBinaryPointwiseApplicabilityChecks.hpp"

using namespace miopen_plugin;
using namespace miopen_plugin::binary_pointwise_applicability;

namespace
{

using DataType      = hipdnn_flatbuffers_sdk::data_objects::DataType;
using PointwiseMode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

struct BinaryTensorIds
{
    static constexpr int64_t IN_0  = 1;
    static constexpr int64_t IN_1  = 2;
    static constexpr int64_t OUT_0 = 3;
};

struct BinaryTensorConfig
{
    int64_t uid;
    std::string name;
    DataType dataType;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    bool isVirtual = false;
};

BinaryTensorConfig validIn0(DataType dt = DataType::FLOAT)
{
    return {BinaryTensorIds::IN_0, "in_0", dt, {1, 4, 4, 4}, {64, 16, 4, 1}};
}
BinaryTensorConfig validIn1(DataType dt = DataType::FLOAT)
{
    return {BinaryTensorIds::IN_1, "in_1", dt, {1, 4, 4, 4}, {64, 16, 4, 1}};
}
BinaryTensorConfig validOut0(DataType dt = DataType::FLOAT)
{
    return {BinaryTensorIds::OUT_0, "out_0", dt, {1, 4, 4, 4}, {64, 16, 4, 1}};
}

struct ModeTestCase
{
    std::string name;
    bool shouldPass;
    PointwiseMode mode;

    friend std::ostream& operator<<(std::ostream& os, const ModeTestCase& tc)
    {
        return os << tc.name;
    }
};

struct CheckTensorsTestCase
{
    std::string name;
    bool shouldPass;
    PointwiseMode mode;
    bool withIn1;
    std::vector<BinaryTensorConfig> tensors;

    friend std::ostream& operator<<(std::ostream& os, const CheckTensorsTestCase& tc)
    {
        return os << tc.name;
    }
};

struct IsBinaryPointwiseSupportedTestCase
{
    std::string name;
    bool shouldPass;
    PointwiseMode mode;
    DataType computeType;
    bool withIn1;
    std::vector<BinaryTensorConfig> tensors;
    bool addExtraNode;

    friend std::ostream& operator<<(std::ostream& os,
                                    const IsBinaryPointwiseSupportedTestCase& tc)
    {
        return os << tc.name;
    }
};

flatbuffers::FlatBufferBuilder buildBinaryPointwiseGraph(
    PointwiseMode mode,
    const std::vector<BinaryTensorConfig>& tensors,
    bool withIn1         = true,
    DataType computeType = DataType::FLOAT,
    bool addExtraNode    = false)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorOffsets;
    for(const auto& cfg : tensors)
    {
        tensorOffsets.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                cfg.uid,
                cfg.name.c_str(),
                cfg.dataType,
                &cfg.strides,
                &cfg.dims,
                cfg.isVirtual));
    }

    auto pwAttrs = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        mode,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        BinaryTensorIds::IN_0,
        withIn1 ? flatbuffers::Optional<int64_t>(BinaryTensorIds::IN_1) : flatbuffers::nullopt,
        flatbuffers::nullopt,
        BinaryTensorIds::OUT_0);

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "binary_pointwise",
        computeType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pwAttrs.Union()));

    if(addExtraNode)
    {
        auto extraAttrs = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
            builder,
            PointwiseMode::ADD,
            flatbuffers::nullopt,
            flatbuffers::nullopt,
            flatbuffers::nullopt,
            flatbuffers::nullopt,
            BinaryTensorIds::IN_0,
            flatbuffers::Optional<int64_t>(BinaryTensorIds::IN_1),
            flatbuffers::nullopt,
            BinaryTensorIds::OUT_0);
        nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
            builder,
            "extra_node",
            computeType,
            hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
            extraAttrs.Union()));
    }

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        DataType::FLOAT,
        DataType::HALF,
        DataType::BFLOAT16,
        &tensorOffsets,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

std::vector<ModeTestCase> getModeTestCases()
{
    return {
        {"AcceptsAdd",      true,  PointwiseMode::ADD},
        {"AcceptsSub",      true,  PointwiseMode::SUB},
        {"AcceptsMul",      true,  PointwiseMode::MUL},
        {"RejectsMin",      false, PointwiseMode::MIN_OP},
        {"RejectsMax",      false, PointwiseMode::MAX_OP},
        {"RejectsDiv",      false, PointwiseMode::DIV},
        {"RejectsReluFwd",  false, PointwiseMode::RELU_FWD},
        {"RejectsIdentity", false, PointwiseMode::IDENTITY},
    };
}

std::vector<CheckTensorsTestCase> getCheckTensorsTestCases()
{
    return {
        {"AcceptsValidFloat",
        true, PointwiseMode::ADD, true,
        {validIn0(DataType::FLOAT), validIn1(DataType::FLOAT), validOut0(DataType::FLOAT)}},

        {"AcceptsValidHalf",
        true, PointwiseMode::ADD, true,
        {validIn0(DataType::HALF), validIn1(DataType::HALF), validOut0(DataType::HALF)}},

        {"AcceptsRank1",
        true, PointwiseMode::ADD, true,
        {{BinaryTensorIds::IN_0,  "in_0",  DataType::FLOAT, {64}, {1}},
        {BinaryTensorIds::IN_1,  "in_1",  DataType::FLOAT, {64}, {1}},
        {BinaryTensorIds::OUT_0, "out_0", DataType::FLOAT, {64}, {1}}}},

        {"AcceptsRank2",
        true, PointwiseMode::ADD, true,
        {{BinaryTensorIds::IN_0,  "in_0",  DataType::FLOAT, {8, 8}, {8, 1}},
        {BinaryTensorIds::IN_1,  "in_1",  DataType::FLOAT, {8, 8}, {8, 1}},
        {BinaryTensorIds::OUT_0, "out_0", DataType::FLOAT, {8, 8}, {8, 1}}}},

        {"AcceptsRank3",
        true, PointwiseMode::ADD, true,
        {{BinaryTensorIds::IN_0,  "in_0",  DataType::FLOAT, {1, 4, 4}, {16, 4, 1}},
        {BinaryTensorIds::IN_1,  "in_1",  DataType::FLOAT, {1, 4, 4}, {16, 4, 1}},
        {BinaryTensorIds::OUT_0, "out_0", DataType::FLOAT, {1, 4, 4}, {16, 4, 1}}}},

        {"AcceptsRank4",
        true, PointwiseMode::ADD, true,
        {validIn0(), validIn1(), validOut0()}},

        {"RejectsMissingIn1",
        false, PointwiseMode::ADD, false,
        {validIn0(), validOut0()}},

        {"RejectsVirtualIn0",
        false, PointwiseMode::ADD, true,
        {{BinaryTensorIds::IN_0, "in_0", DataType::FLOAT, {1,4,4,4}, {64,16,4,1}, true},
        validIn1(),
        validOut0()}},

        {"RejectsVirtualIn1",
        false, PointwiseMode::ADD, true,
        {validIn0(),
        {BinaryTensorIds::IN_1, "in_1", DataType::FLOAT, {1,4,4,4}, {64,16,4,1}, true},
        validOut0()}},

        {"RejectsVirtualOut0",
        false, PointwiseMode::ADD, true,
        {validIn0(),
        validIn1(),
        {BinaryTensorIds::OUT_0, "out_0", DataType::FLOAT, {1,4,4,4}, {64,16,4,1}, true}}},

        {"RejectsUint8",
        false, PointwiseMode::ADD, true,
        {{BinaryTensorIds::IN_0,  "in_0",  DataType::UINT8, {1,4,4,4}, {64,16,4,1}},
        {BinaryTensorIds::IN_1,  "in_1",  DataType::UINT8, {1,4,4,4}, {64,16,4,1}},
        {BinaryTensorIds::OUT_0, "out_0", DataType::UINT8, {1,4,4,4}, {64,16,4,1}}}},

        {"RejectsMismatch_In0FloatIn1Half",
        false, PointwiseMode::ADD, true,
        {validIn0(DataType::FLOAT), validIn1(DataType::HALF), validOut0(DataType::FLOAT)}},

        {"RejectsMismatch_OutHalf",
        false, PointwiseMode::ADD, true,
        {validIn0(DataType::FLOAT), validIn1(DataType::FLOAT), validOut0(DataType::HALF)}},

        {"RejectsMismatch_In1DiffFromOut",
        false, PointwiseMode::ADD, true,
        {validIn0(DataType::HALF), validIn1(DataType::FLOAT), validOut0(DataType::HALF)}},

        {"RejectsRank5",
        false, PointwiseMode::ADD, true,
        {{BinaryTensorIds::IN_0,  "in_0",  DataType::FLOAT, {1,2,2,2,2}, {16,8,4,2,1}},
        {BinaryTensorIds::IN_1,  "in_1",  DataType::FLOAT, {1,2,2,2,2}, {16,8,4,2,1}},
        {BinaryTensorIds::OUT_0, "out_0", DataType::FLOAT, {1,2,2,2,2}, {16,8,4,2,1}}}},

        {"RejectsMismatchedRanks",
        false, PointwiseMode::ADD, true,
        {{BinaryTensorIds::IN_0,  "in_0",  DataType::FLOAT, {1,4,4,4}, {64,16,4,1}},
        {BinaryTensorIds::IN_1,  "in_1",  DataType::FLOAT, {4,4,4},   {16,4,1}},
        validOut0()}},

        {"RejectsMismatch_In0ElemCount",
        false, PointwiseMode::ADD, true,
        {{BinaryTensorIds::IN_0, "in_0", DataType::FLOAT, {1,8,8,8}, {512,64,8,1}},
        validIn1(),
        validOut0()}},

        {"RejectsMismatch_In1ElemCount",
        false, PointwiseMode::ADD, true,
        {validIn0(),
        {BinaryTensorIds::IN_1, "in_1", DataType::FLOAT, {1,8,4,4}, {128,16,4,1}},
        validOut0()}},

        {"RejectsMismatch_OutElemCount",
        false, PointwiseMode::ADD, true,
        {validIn0(),
        validIn1(),
        {BinaryTensorIds::OUT_0, "out_0", DataType::FLOAT, {1,2,2,2}, {8,4,2,1}}}},
    };
}

std::vector<IsBinaryPointwiseSupportedTestCase> getIsBinaryPointwiseSupportedTestCases()
{
    const auto validTensors
        = std::vector<BinaryTensorConfig>{validIn0(), validIn1(), validOut0()};

    return {
        {"AcceptsAdd", true,  PointwiseMode::ADD, DataType::FLOAT, true, validTensors, false},
        {"AcceptsSub", true,  PointwiseMode::SUB, DataType::FLOAT, true, validTensors, false},
        {"AcceptsMul", true,  PointwiseMode::MUL, DataType::FLOAT, true, validTensors, false},

        {"RejectsMultiNodeGraph",
        false, PointwiseMode::ADD, DataType::FLOAT, true, validTensors, true},

        {"RejectsHalfComputeType",
        false, PointwiseMode::ADD, DataType::HALF, true, validTensors, false},

        {"RejectsMinMode",
        false, PointwiseMode::MIN_OP, DataType::FLOAT, true, validTensors, false},

        {"RejectsReluFwdMode",
        false, PointwiseMode::RELU_FWD, DataType::FLOAT, true, validTensors, false},

        {"RejectsMissingIn1",
        false, PointwiseMode::ADD, DataType::FLOAT, false,
        {validIn0(), validOut0()}, false},

        {"RejectsVirtualIn0",
        false, PointwiseMode::ADD, DataType::FLOAT, true,
        {{BinaryTensorIds::IN_0, "in_0", DataType::FLOAT, {1,4,4,4}, {64,16,4,1}, true},
        validIn1(),
        validOut0()},
        false},
    };
}

} // namespace

class TestCheckModeSupported : public ::testing::TestWithParam<ModeTestCase>
{
};

TEST_P(TestCheckModeSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    flatbuffers::FlatBufferBuilder builder;
    auto offset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        tc.mode,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        BinaryTensorIds::IN_0,
        flatbuffers::Optional<int64_t>(BinaryTensorIds::IN_1),
        flatbuffers::nullopt,
        BinaryTensorIds::OUT_0);
    builder.Finish(offset);

    const auto* attrs =
        flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    if(tc.shouldPass)
        EXPECT_NO_THROW({ checkModeSupported(*attrs); });
    else
        EXPECT_THROW({ checkModeSupported(*attrs); }, hipdnn_plugin_sdk::HipdnnPluginException);
}

INSTANTIATE_TEST_SUITE_P(AllCases, TestCheckModeSupported, testing::ValuesIn(getModeTestCases()));

class TestCheckTensorsSupported : public ::testing::TestWithParam<CheckTensorsTestCase>
{
};

TEST_P(TestCheckTensorsSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    auto graphBuilder = buildBinaryPointwiseGraph(tc.mode, tc.tensors, tc.withIn1);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const auto& node  = graph.getNode(0);
    auto*       attrs = node.attributes_as_PointwiseAttributes();
    ASSERT_NE(attrs, nullptr);

    if(tc.shouldPass)
        EXPECT_NO_THROW({ checkTensorsSupported(*attrs, graph.getTensorMap()); });
    else
        EXPECT_THROW({ checkTensorsSupported(*attrs, graph.getTensorMap()); },
                    hipdnn_plugin_sdk::HipdnnPluginException);
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                        TestCheckTensorsSupported,
                        testing::ValuesIn(getCheckTensorsTestCases()));

class TestIsBinaryPointwiseSupported
    : public ::testing::TestWithParam<IsBinaryPointwiseSupportedTestCase>
{
};

TEST_P(TestIsBinaryPointwiseSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    auto graphBuilder = buildBinaryPointwiseGraph(
        tc.mode, tc.tensors, tc.withIn1, tc.computeType, tc.addExtraNode);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    EXPECT_EQ(isBinaryPointwiseSupported(graph), tc.shouldPass);
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                        TestIsBinaryPointwiseSupported,
                        testing::ValuesIn(getIsBinaryPointwiseSupportedTestCases()));

TEST(TestBinaryPointwiseApplicabilityChecks, RejectsNonPointwiseGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(isBinaryPointwiseSupported(graph));
}