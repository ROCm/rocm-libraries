// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/descriptors/hip/standalone_pointwise/kernels/StandalonePointwiseContract.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/utilities/PointwiseValidation.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/PointwiseOperationFunctors.hpp>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace
{
namespace data = hipdnn_flatbuffers_sdk::data_objects;
namespace contract = hkp_standalone_pointwise;
namespace reference = hipdnn_test_sdk::utilities::pointwise;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

constexpr PackSymbols SYMBOLS{"hipkernel:StandalonePointwise",
                              "hipkernel.standalone_pointwise.graph_match",
                              "",
                              "hipkernel.standalone_pointwise.kernel_match",
                              "hipkernel.standalone_pointwise.score",
                              "hipkernel.standalone_pointwise.dispatch.v1",
                              "",
                              "",
                              ""};

data::GraphT graph(data::PointwiseMode mode = data::PointwiseMode::ADD)
{
    data::GraphT result;
    for(int64_t uid = 1; uid <= 4; ++uid)
    {
        auto tensor = std::make_unique<data::TensorAttributesT>();
        tensor->uid = uid;
        tensor->data_type = uid == 3 ? data::DataType::BOOLEAN : data::DataType::FLOAT;
        tensor->dims = {2, 3};
        tensor->strides = {8, 2};
        result.tensors.push_back(std::move(tensor));
    }
    data::PointwiseAttributesT attributes;
    attributes.operation = mode;
    attributes.in_0_tensor_uid = 1;
    attributes.out_0_tensor_uid = 4;
    if(hipdnn_flatbuffers_sdk::utilities::isBinaryPointwiseMode(mode)
       || mode == data::PointwiseMode::BINARY_SELECT)
    {
        attributes.in_1_tensor_uid = 2;
    }
    if(mode == data::PointwiseMode::BINARY_SELECT)
    {
        attributes.in_2_tensor_uid = 3;
    }
    if(mode == data::PointwiseMode::CMP_GT)
    {
        result.tensors[3]->data_type = data::DataType::BOOLEAN;
    }
    auto node = std::make_unique<data::NodeT>();
    node->compute_data_type = data::DataType::FLOAT;
    node->attributes.Set(attributes);
    result.nodes.push_back(std::move(node));
    return result;
}

flatbuffers::FlatBufferBuilder pack(const data::GraphT& graph)
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(data::Graph::Pack(builder, &graph));
    return builder;
}

bool matches(const data::GraphT& graph)
{
    const GraphFixture fixture(pack(graph));
    return matchesGraph(SYMBOLS, fixture.context()).has_value();
}

TEST(TestStandalonePointwise, EveryReferencedUnaryAndBinaryModeMatches)
{
    for(const auto mode : {data::PointwiseMode::ABS,
                           data::PointwiseMode::NEG,
                           data::PointwiseMode::IDENTITY,
                           data::PointwiseMode::RELU_FWD,
                           data::PointwiseMode::SIGMOID_FWD,
                           data::PointwiseMode::TANH_FWD,
                           data::PointwiseMode::GELU_FWD,
                           data::PointwiseMode::GELU_APPROX_TANH_FWD,
                           data::PointwiseMode::SWISH_FWD,
                           data::PointwiseMode::ADD,
                           data::PointwiseMode::SUB,
                           data::PointwiseMode::MUL,
                           data::PointwiseMode::MIN_OP,
                           data::PointwiseMode::MAX_OP,
                           data::PointwiseMode::CMP_GT,
                           data::PointwiseMode::RELU_BWD,
                           data::PointwiseMode::SIGMOID_BWD,
                           data::PointwiseMode::TANH_BWD})
    {
        EXPECT_TRUE(matches(graph(mode))) << data::EnumNamePointwiseMode(mode);
    }
}

TEST(TestStandalonePointwise, RanksOneThroughFiveUseRightAlignedBroadcastAndPaddedStrides)
{
    for(size_t rank = 1; rank <= 5; ++rank)
    {
        auto input = graph();
        input.tensors[0]->dims = std::vector<int64_t>(rank, 2);
        input.tensors[0]->strides.resize(rank);
        input.tensors[3]->dims = input.tensors[0]->dims;
        int64_t stride = 3;
        for(size_t axis = rank; axis > 0; --axis)
        {
            input.tensors[0]->strides[axis - 1] = stride;
            stride *= 2;
        }
        input.tensors[3]->strides = input.tensors[0]->strides;
        input.tensors[1]->dims = {2};
        input.tensors[1]->strides = {7};
        EXPECT_TRUE(matches(input)) << rank;
        input.tensors[1]->dims = {3};
        EXPECT_FALSE(matches(input)) << rank;
    }
}

TEST(TestStandalonePointwise, DeclinesMalformedUnsupportedAndOverlappingOperands)
{
    using Mutation = void (*)(data::GraphT&);
    const std::vector<Mutation> mutations{
        [](data::GraphT& g) { g.tensors[0]->dims = {}; },
        [](data::GraphT& g) { g.tensors[0]->dims = {1, 1, 1, 1, 1, 1}; },
        [](data::GraphT& g) { g.tensors[0]->strides = {}; },
        [](data::GraphT& g) { g.tensors[0]->strides[0] = 0; },
        [](data::GraphT& g) { g.tensors[0]->strides[0] = -1; },
        [](data::GraphT& g) { g.tensors[0]->dims[0] = 0; },
        [](data::GraphT& g) { g.tensors[0]->dims[0] = std::numeric_limits<int64_t>::max(); },
        [](data::GraphT& g) { g.tensors[0]->virtual_ = true; },
        [](data::GraphT& g) { g.tensors[0]->ragged_offset_tensor_uid = 3; },
        [](data::GraphT& g) { g.tensors[3]->strides = {1, 1}; },
        [](data::GraphT& g) { g.tensors[0]->data_type = data::DataType::DOUBLE; },
        [](data::GraphT& g) { g.nodes[0]->compute_data_type = data::DataType::HALF; },
        [](data::GraphT& g) {
            g.nodes[0]->attributes.AsPointwiseAttributes()->in_1_tensor_uid = 99;
        },
        [](data::GraphT& g) {
            g.nodes[0]->attributes.AsPointwiseAttributes()->in_1_tensor_uid = flatbuffers::nullopt;
        },
        [](data::GraphT& g) {
            g.nodes[0]->attributes.AsPointwiseAttributes()->in_2_tensor_uid = 3;
        },
        [](data::GraphT& g) {
            g.nodes[0]->attributes.AsPointwiseAttributes()->out_0_tensor_uid = 1;
        },
        [](data::GraphT& g) { g.nodes.emplace_back(std::make_unique<data::NodeT>(*g.nodes[0])); }};
    for(size_t index = 0; index < mutations.size(); ++index)
    {
        auto input = graph();
        mutations[index](input);
        EXPECT_FALSE(matches(input)) << index;
    }
}

TEST(TestStandalonePointwise, DeclinesParametersAndValueOperandsPendingTheirSeparateContract)
{
    using Mutation = void (*)(data::GraphT&);
    const std::vector<Mutation> mutations{
        [](data::GraphT& g) { g.tensors[1]->is_runtime_pass_by_value = true; },
        [](data::GraphT& g) { g.tensors[1]->value.Set(data::Float32Value(2.0f)); },
        [](data::GraphT& g) { g.tensors[3]->is_runtime_pass_by_value = true; },
        [](data::GraphT& g) {
            g.nodes[0]->attributes.AsPointwiseAttributes()->relu_lower_clip = 0.0f;
        },
        [](data::GraphT& g) {
            g.nodes[0]->attributes.AsPointwiseAttributes()->relu_upper_clip = 6.0f;
        },
        [](data::GraphT& g) {
            g.nodes[0]->attributes.AsPointwiseAttributes()->relu_lower_clip_slope = 0.0f;
        },
        [](data::GraphT& g) { g.nodes[0]->attributes.AsPointwiseAttributes()->swish_beta = 1.0f; },
        [](data::GraphT& g) { g.nodes[0]->attributes.AsPointwiseAttributes()->elu_alpha = 1.0f; },
        [](data::GraphT& g) {
            g.nodes[0]->attributes.AsPointwiseAttributes()->softplus_beta = 1.0f;
        },
        [](data::GraphT& g) {
            g.nodes[0]->attributes.AsPointwiseAttributes()->axis_tensor_uid = 0;
        }};
    for(size_t index = 0; index < mutations.size(); ++index)
    {
        auto input = graph();
        input.tensors[1]->dims = {1};
        input.tensors[1]->strides = {1};
        mutations[index](input);
        EXPECT_FALSE(matches(input)) << index;
    }
    auto relu = graph(data::PointwiseMode::RELU_FWD);
    relu.nodes[0]->attributes.AsPointwiseAttributes()->relu_lower_clip_slope = 0.0f;
    EXPECT_FALSE(matches(relu));
    auto swish = graph(data::PointwiseMode::SWISH_FWD);
    swish.nodes[0]->attributes.AsPointwiseAttributes()->swish_beta = 1.0f;
    EXPECT_FALSE(matches(swish));
}

TEST(TestStandalonePointwise, DeclinesModesOutsideTheUnaryBinaryReferenceContract)
{
    for(const auto mode : {data::PointwiseMode::BINARY_SELECT,
                           data::PointwiseMode::GEN_INDEX,
                           data::PointwiseMode::ELU_FWD,
                           data::PointwiseMode::ELU_BWD,
                           data::PointwiseMode::SOFTPLUS_FWD,
                           data::PointwiseMode::SOFTPLUS_BWD,
                           data::PointwiseMode::SWISH_BWD,
                           data::PointwiseMode::LOGICAL_NOT,
                           data::PointwiseMode::DIV,
                           data::PointwiseMode::UNSET})
    {
        EXPECT_FALSE(matches(graph(mode))) << data::EnumNamePointwiseMode(mode);
    }
}

TEST(TestStandalonePointwise, MixedStorageAndComparisonDtypesAreExplicit)
{
    auto input = graph();
    input.tensors[0]->data_type = data::DataType::HALF;
    EXPECT_TRUE(matches(input));
    input.tensors[1]->data_type = data::DataType::BFLOAT16;
    EXPECT_FALSE(matches(input));
    input.tensors[1]->data_type = data::DataType::HALF;
    EXPECT_TRUE(matches(input));
    input.tensors[3]->data_type = data::DataType::BFLOAT16;
    EXPECT_FALSE(matches(input));
    input = graph(data::PointwiseMode::CMP_GT);
    EXPECT_TRUE(matches(input));
    input.tensors[3]->data_type = data::DataType::FLOAT;
    EXPECT_FALSE(matches(input));
    for(const auto type : {data::DataType::FLOAT, data::DataType::HALF, data::DataType::BFLOAT16})
    {
        input = graph(data::PointwiseMode::IDENTITY);
        input.tensors[0]->data_type = type;
        input.tensors[3]->data_type = type;
        EXPECT_TRUE(matches(input));
    }
}

TEST(TestStandalonePointwise, LaunchMetadataRetainsSavedPlanLayout)
{
    static_assert(std::is_standard_layout_v<contract::Metadata>);
    static_assert(std::is_trivially_copyable_v<contract::Metadata>);
    EXPECT_EQ(sizeof(contract::Metadata), 184);
    EXPECT_EQ(offsetof(contract::Metadata, dims), 8);
    EXPECT_EQ(offsetof(contract::Metadata, strides), 48);
    EXPECT_EQ(offsetof(contract::Metadata, types), 168);
    EXPECT_EQ(offsetof(contract::Metadata, operation), 180);
    EXPECT_EQ(contract::FLOAT, 0);
    EXPECT_EQ(contract::HALF, 1);
    EXPECT_EQ(contract::BFLOAT16, 2);
    EXPECT_EQ(contract::BOOLEAN, 3);
}

TEST(TestStandalonePointwise, DynamicOffsetsMatchRightAlignedBroadcast)
{
    contract::Metadata meta;
    const std::array<int64_t, 5> dims{1, 1, 2, 3, 4};
    std::copy(std::begin(dims), std::end(dims), std::begin(meta.dims));
    meta.strides[0][2] = 40;
    meta.strides[0][3] = 9;
    meta.strides[0][4] = 2;
    meta.strides[1][4] = 3;
    for(int64_t index = 0; index < 24; ++index)
    {
        EXPECT_EQ(contract::offset(meta, 0, index),
                  (index / 12) * 40 + ((index / 4) % 3) * 9 + (index % 4) * 2);
        EXPECT_EQ(contract::offset(meta, 1, index), (index % 4) * 3);
    }
}

TEST(TestStandalonePointwise, EverySupportedFormulaMatchesCPUReferenceFunctors)
{
    contract::Metadata meta;
    const auto check = [&meta](data::PointwiseMode mode, float a, float b, float expected) {
        meta.operation = static_cast<int32_t>(mode);
        EXPECT_NEAR(contract::evaluate(meta, a, b), expected, 2e-6f)
            << data::EnumNamePointwiseMode(mode) << " a=" << a << " b=" << b;
    };
    for(const float a : {-3.0f, -0.5f, 0.0f, 0.7f, 2.0f, 4.0f})
    {
        check(data::PointwiseMode::ABS, a, 0, reference::AbsoluteValue<float>{}(a));
        check(data::PointwiseMode::NEG, a, 0, reference::Negation<float>{}(a));
        check(data::PointwiseMode::IDENTITY, a, 0, reference::Identity<float>{}(a));
        check(data::PointwiseMode::RELU_FWD, a, 0, reference::ReluForward<float>{}(a));
        check(data::PointwiseMode::SIGMOID_FWD, a, 0, reference::SigmoidForward<float>{}(a));
        check(data::PointwiseMode::TANH_FWD, a, 0, reference::TanhForward<float>{}(a));
        check(data::PointwiseMode::GELU_FWD, a, 0, reference::GeluForward<float>{}(a));
        check(data::PointwiseMode::GELU_APPROX_TANH_FWD,
              a,
              0,
              static_cast<float>(reference::GeluApproxTanhForward<double>{}(a)));
        check(data::PointwiseMode::SWISH_FWD, a, 0, reference::SwishForward<float>{}(a));
        for(const float b : {-3.0f, -0.5f, 0.0f, 0.8f, 2.0f, 4.0f})
        {
            check(data::PointwiseMode::ADD, a, b, reference::Add<float>{}(a, b));
            check(data::PointwiseMode::SUB, a, b, reference::Subtract<float>{}(a, b));
            check(data::PointwiseMode::MUL, a, b, reference::Multiply<float>{}(a, b));
            check(data::PointwiseMode::MIN_OP, a, b, reference::Min<float>{}(a, b));
            check(data::PointwiseMode::MAX_OP, a, b, reference::Max<float>{}(a, b));
            check(data::PointwiseMode::CMP_GT,
                  a,
                  b,
                  reference::CompareGreater<float>{}(a, b) ? 1.0f : 0.0f);
            check(data::PointwiseMode::RELU_BWD, a, b, reference::ReluBackward<float>{}(a, b));
            check(
                data::PointwiseMode::SIGMOID_BWD, a, b, reference::SigmoidBackward<float>{}(a, b));
            check(data::PointwiseMode::TANH_BWD, a, b, reference::TanhBackward<float>{}(a, b));
        }
    }
}

TEST(TestStandalonePointwise, ReferenceBoundaryAndNaNSelectionSemantics)
{
    contract::Metadata meta;
    meta.operation = static_cast<int32_t>(data::PointwiseMode::RELU_BWD);
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    for(const float input : {-1.0f, -0.0f, 0.0f, 1.0f, nan})
    {
        EXPECT_EQ(contract::evaluate(meta, 2.0f, input),
                  reference::ReluBackward<float>{}(2.0f, input));
    }
    meta.operation = static_cast<int32_t>(data::PointwiseMode::MAX_OP);
    EXPECT_TRUE(std::isnan(contract::evaluate(meta, nan, 1.0f)));
    EXPECT_EQ(contract::evaluate(meta, 1.0f, nan), 1.0f);
    EXPECT_TRUE(std::signbit(contract::evaluate(meta, -0.0f, 0.0f)));
    meta.operation = static_cast<int32_t>(data::PointwiseMode::MIN_OP);
    EXPECT_TRUE(std::isnan(contract::evaluate(meta, nan, 1.0f)));
    EXPECT_EQ(contract::evaluate(meta, 1.0f, nan), 1.0f);
    EXPECT_TRUE(std::signbit(contract::evaluate(meta, -0.0f, 0.0f)));
}

TEST(TestStandalonePointwise, PlanRejectsMissingBindingsBeforeCompiling)
{
    const GraphFixture fixture(pack(graph()));
    auto kernel = makeKernel(256, "FLOAT");
    kernel.metadata = {{"contract_version", int64_t{1}}};
    const auto& handler = dispatchHandler(SYMBOLS);
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), {}, kernel), 0U);
    EXPECT_TRUE(handler.supportsSaving());
    EXPECT_THROW(handler.prepare(fixture.context(), {}, kernel),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    kernel.metadata["contract_version"] = int64_t{2};
    EXPECT_THROW(handler.prepare(fixture.context(), {{"standalone.elements", int64_t{6}}}, kernel),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestStandalonePointwise, PlanRestoreValidatesLaunchMetadataBeforeLoadingCode)
{
    using namespace hipdnn_plugin_sdk::ingestor;
    SavedLaunchInputs inputs{
        "hipkernel.standalone_pointwise.dispatch.v1",
        {{"elements", int64_t{6}}, {"arity", int64_t{2}}, {"operation", int64_t{2}}}};
    const std::array<int64_t, 5> dims{1, 1, 1, 2, 3};
    const std::array<int64_t, 5> strides{0, 0, 0, 8, 2};
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        inputs.values.emplace("dim." + std::to_string(axis), dims[axis]);
    }
    for(size_t operand = 0; operand < 3; ++operand)
    {
        const auto root = std::to_string(operand);
        inputs.values.emplace("uid." + root, static_cast<int64_t>(operand + 1));
        inputs.values.emplace("type." + root, int64_t{0});
        for(size_t axis = 0; axis < 5; ++axis)
        {
            inputs.values.emplace("stride." + root + "." + std::to_string(axis), strides[axis]);
        }
    }
    SavedKernelCode code;
    code.symbol = "StandalonePointwise";
    const KernelArgument pointer{"global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    code.recordedSignature
        = {pointer,
           pointer,
           pointer,
           {"by_value", static_cast<uint32_t>(sizeof(contract::Metadata)), 0, ""}};
    const auto& handler = dispatchHandler(SYMBOLS);
    const std::vector<std::pair<std::string, MetadataValue>> invalidValues{
        {"arity", int64_t{1}},
        {"operation", int64_t{22}},
        {"elements", int64_t{7}},
        {"uid.0", int64_t{3}},
        {"type.0", int64_t{3}},
        {"dim.4", int64_t{0}},
        {"stride.2.3", int64_t{2}},
        {"stride.0.4", int64_t{-1}},
        {"unexpected", int64_t{1}}};
    for(const auto& [name, value] : invalidValues)
    {
        auto mutated = inputs;
        mutated.values[name] = value;
        try
        {
            static_cast<void>(handler.restoreLaunch(mutated, code, 0));
            FAIL() << name;
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
        {
            EXPECT_NE(error.getMessage().find("launch"), std::string::npos)
                << name << ": " << error.getMessage();
        }
    }
}

} // namespace
#endif
