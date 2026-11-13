// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "PackedKernelSource.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanCapture.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanRestore.hpp"
#include "engines/kernel_ingestor_engine/descriptors/hip/standalone_pointwise/kernels/StandalonePointwiseContractV2.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/CpuReferencePointwise.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/PointwiseOperationFunctors.hpp>
#include <memory>
#include <type_traits>
#include <vector>

namespace
{
namespace data = hipdnn_flatbuffers_sdk::data_objects;
namespace contract = hkp_standalone_pointwise_v2;
namespace reference = hipdnn_test_sdk::utilities::pointwise;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

constexpr PackSymbols SYMBOLS{"hipkernel:StandalonePointwise",
                              "hipkernel.standalone_pointwise.graph_match.v2",
                              "",
                              "hipkernel.standalone_pointwise.kernel_match.v2",
                              "hipkernel.standalone_pointwise.score.v2",
                              "hipkernel.standalone_pointwise.dispatch.v2",
                              "",
                              "",
                              ""};

TEST(TestStandalonePointwiseV2, LaunchMetadataRetainsSavedPlanLayout)
{
    static_assert(std::is_standard_layout_v<contract::Metadata>);
    static_assert(std::is_trivially_copyable_v<contract::Metadata>);
    EXPECT_EQ(sizeof(contract::Metadata), 272);
    EXPECT_EQ(offsetof(contract::Metadata, dims), 8);
    EXPECT_EQ(offsetof(contract::Metadata, strides), 48);
    EXPECT_EQ(offsetof(contract::Metadata, types), 208);
    EXPECT_EQ(offsetof(contract::Metadata, scalarMask), 224);
    EXPECT_EQ(offsetof(contract::Metadata, operation), 228);
    EXPECT_EQ(offsetof(contract::Metadata, lower), 232);
    EXPECT_EQ(offsetof(contract::Metadata, upper), 236);
    EXPECT_EQ(offsetof(contract::Metadata, slope), 240);
    EXPECT_EQ(offsetof(contract::Metadata, beta), 244);
    EXPECT_EQ(offsetof(contract::Metadata, scalars), 248);
    EXPECT_EQ(offsetof(contract::Metadata, reluParameterized), 260);
    EXPECT_EQ(offsetof(contract::Metadata, alpha), 264);
    EXPECT_EQ(offsetof(contract::Metadata, axis), 268);
    EXPECT_EQ(contract::FLOAT, 0);
    EXPECT_EQ(contract::HALF, 1);
    EXPECT_EQ(contract::BFLOAT16, 2);
    EXPECT_EQ(contract::BOOLEAN, 3);
    EXPECT_EQ(contract::INT32, 4);
}

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
    if(mode == data::PointwiseMode::ADD || mode == data::PointwiseMode::RELU_BWD
       || mode == data::PointwiseMode::ELU_BWD || mode == data::PointwiseMode::SOFTPLUS_BWD
       || mode == data::PointwiseMode::SWISH_BWD || mode == data::PointwiseMode::CMP_GT
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

TEST(TestStandalonePointwiseV2, RanksOneThroughFiveUseRightAlignedBroadcastAndPaddedStrides)
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

TEST(TestStandalonePointwiseV2, DeclinesMalformedUnsupportedAndOverlappingOperands)
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
            g.nodes[0]->attributes.AsPointwiseAttributes()->in_2_tensor_uid = 3;
        },
        [](data::GraphT& g) {
            g.nodes[0]->attributes.AsPointwiseAttributes()->out_0_tensor_uid = 1;
        }};
    for(size_t index = 0; index < mutations.size(); ++index)
    {
        auto input = graph();
        mutations[index](input);
        EXPECT_FALSE(matches(input)) << index;
    }
}

TEST(TestStandalonePointwiseV2, SelectRequiresBooleanThirdOperandAndUniformValueStorage)
{
    auto input = graph(data::PointwiseMode::BINARY_SELECT);
    EXPECT_TRUE(matches(input));
    input.tensors[2]->dims = {3};
    input.tensors[2]->strides = {1};
    EXPECT_TRUE(matches(input));
    input.tensors[2]->data_type = data::DataType::FLOAT;
    EXPECT_FALSE(matches(input));
    input.tensors[2]->data_type = data::DataType::BOOLEAN;
    input.tensors[1]->data_type = data::DataType::HALF;
    EXPECT_FALSE(matches(input));
}

TEST(TestStandalonePointwiseV2, ScalarValueAndRuntimePBVAreSupportedOnlyForSingleElements)
{
    auto input = graph();
    input.tensors[1]->dims = {1};
    input.tensors[1]->strides = {1};
    input.tensors[1]->is_runtime_pass_by_value = true;
    EXPECT_TRUE(matches(input));
    input.tensors[1]->value.Set(data::Float32Value(2.0f));
    EXPECT_FALSE(matches(input));
    input.tensors[1]->is_runtime_pass_by_value = false;
    EXPECT_TRUE(matches(input));
    input.tensors[1]->value.Set(data::BoolValue(true));
    EXPECT_FALSE(matches(input));
    input.tensors[1]->value.Set(data::Float32Value(2.0f));
    input.tensors[1]->dims = {2};
    EXPECT_FALSE(matches(input));
}

TEST(TestStandalonePointwiseV2, ReluParametersAndSwishBetaAreModeSpecific)
{
    auto input = graph(data::PointwiseMode::RELU_FWD);
    auto* attributes = input.nodes[0]->attributes.AsPointwiseAttributes();
    attributes->relu_lower_clip = -1.0f;
    attributes->relu_upper_clip = 3.0f;
    attributes->relu_lower_clip_slope = 0.2f;
    EXPECT_TRUE(matches(input));
    attributes->relu_lower_clip = 4.0f;
    EXPECT_FALSE(matches(input));
    attributes->relu_lower_clip = -1.0f;
    attributes->swish_beta = 2.0f;
    EXPECT_FALSE(matches(input));
    input = graph(data::PointwiseMode::SWISH_FWD);
    input.nodes[0]->attributes.AsPointwiseAttributes()->swish_beta = -2.0f;
    EXPECT_TRUE(matches(input));
    input.nodes[0]->attributes.AsPointwiseAttributes()->swish_beta
        = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(matches(input));
}

TEST(TestStandalonePointwiseV2, DeclinesUnreferencedModesAndIgnoredParameters)
{
    for(const auto mode : {data::PointwiseMode::GEN_INDEX,
                           data::PointwiseMode::LOGICAL_NOT,
                           data::PointwiseMode::UNSET})
    {
        EXPECT_FALSE(matches(graph(mode)));
    }
    auto input = graph();
    input.nodes[0]->attributes.AsPointwiseAttributes()->elu_alpha = 1.0f;
    EXPECT_FALSE(matches(input));
    input = graph();
    input.nodes[0]->attributes.AsPointwiseAttributes()->softplus_beta = 1.0f;
    EXPECT_FALSE(matches(input));
    input = graph();
    input.nodes[0]->attributes.AsPointwiseAttributes()->axis_tensor_uid = 0;
    EXPECT_FALSE(matches(input));
}

TEST(TestStandalonePointwiseV2, MixedStorageAndComparisonDtypesAreExplicit)
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
}

TEST(TestStandalonePointwiseV2, EluSoftplusAndIndexParametersAreValidated)
{
    for(const auto mode : {data::PointwiseMode::ELU_FWD, data::PointwiseMode::ELU_BWD})
    {
        auto input = graph(mode);
        input.nodes[0]->attributes.AsPointwiseAttributes()->elu_alpha = 1.75f;
        EXPECT_TRUE(matches(input));
        input.nodes[0]->attributes.AsPointwiseAttributes()->elu_alpha
            = std::numeric_limits<float>::quiet_NaN();
        EXPECT_FALSE(matches(input));
    }
    for(const auto mode : {data::PointwiseMode::SOFTPLUS_FWD, data::PointwiseMode::SOFTPLUS_BWD})
    {
        auto input = graph(mode);
        input.nodes[0]->attributes.AsPointwiseAttributes()->softplus_beta = 1.5f;
        EXPECT_TRUE(matches(input));
        input.nodes[0]->attributes.AsPointwiseAttributes()->softplus_beta = 0.0f;
        EXPECT_FALSE(matches(input));
    }
    auto index = graph(data::PointwiseMode::GEN_INDEX);
    index.nodes[0]->attributes.AsPointwiseAttributes()->axis_tensor_uid = 1;
    EXPECT_TRUE(matches(index));
    index.tensors[3]->data_type = data::DataType::INT32;
    EXPECT_TRUE(matches(index));
    index.nodes[0]->attributes.AsPointwiseAttributes()->axis_tensor_uid = 2;
    EXPECT_FALSE(matches(index));
    index.nodes[0]->attributes.AsPointwiseAttributes()->axis_tensor_uid = -1;
    EXPECT_FALSE(matches(index));
}

TEST(TestStandalonePointwiseV2, IndexCPUReferenceUsesAxisRatherThanTensorUid)
{
    const hipdnn_data_sdk::utilities::Tensor<float> input({2, 3});
    hipdnn_data_sdk::utilities::Tensor<int32_t> output({2, 3}, {8, 2});
    hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl<int32_t, float>::pointwiseCompute(
        data::PointwiseMode::GEN_INDEX,
        output,
        input,
        0.0f,
        std::numeric_limits<float>::max(),
        0.0f,
        1.0f,
        1.0f,
        1.0f,
        int64_t{1});
    for(int64_t row = 0; row < 2; ++row)
    {
        for(int64_t column = 0; column < 3; ++column)
        {
            EXPECT_EQ(output.getHostValue(row, column), column);
        }
    }
}

TEST(TestStandalonePointwiseV2, DynamicOffsetsMatchRightAlignedBroadcast)
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

TEST(TestStandalonePointwiseV2, EverySupportedFormulaMatchesCPUReferenceFunctors)
{
    contract::Metadata meta;
    meta.lower = -0.5f;
    meta.upper = 2.0f;
    meta.slope = 0.25f;
    meta.beta = 1.5f;
    meta.reluParameterized = 1;
    meta.alpha = 1.75f;
    const auto check = [&meta](data::PointwiseMode mode, float a, float b, float expected) {
        meta.operation = static_cast<int32_t>(mode);
        EXPECT_NEAR(contract::evaluate(meta, a, b, false), expected, 2e-6f)
            << data::EnumNamePointwiseMode(mode) << " a=" << a << " b=" << b;
    };
    for(const float a : {-3.0f, -0.5f, 0.0f, 0.7f, 2.0f, 4.0f})
    {
        check(data::PointwiseMode::ABS, a, 0, reference::AbsoluteValue<float>{}(a));
        check(data::PointwiseMode::NEG, a, 0, reference::Negation<float>{}(a));
        check(data::PointwiseMode::IDENTITY, a, 0, reference::Identity<float>{}(a));
        check(data::PointwiseMode::RELU_FWD,
              a,
              0,
              reference::ReluForward<float>{meta.lower, meta.upper, meta.slope}(a));
        check(data::PointwiseMode::SIGMOID_FWD, a, 0, reference::SigmoidForward<float>{}(a));
        check(data::PointwiseMode::TANH_FWD, a, 0, reference::TanhForward<float>{}(a));
        check(data::PointwiseMode::GELU_FWD, a, 0, reference::GeluForward<float>{}(a));
        check(data::PointwiseMode::GELU_APPROX_TANH_FWD,
              a,
              0,
              static_cast<float>(reference::GeluApproxTanhForward<double>{}(a)));
        check(data::PointwiseMode::SWISH_FWD, a, 0, reference::SwishForward<float>{meta.beta}(a));
        check(data::PointwiseMode::ELU_FWD, a, 0, reference::EluForward<float>{meta.alpha}(a));
        check(data::PointwiseMode::SOFTPLUS_FWD,
              a,
              0,
              reference::SoftplusForward<float>{meta.beta}(a));
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
            check(data::PointwiseMode::RELU_BWD,
                  a,
                  b,
                  reference::ParameterizedReluBackward<float>{meta.lower, meta.upper, meta.slope}(
                      a, b));
            check(
                data::PointwiseMode::SIGMOID_BWD, a, b, reference::SigmoidBackward<float>{}(a, b));
            check(data::PointwiseMode::TANH_BWD, a, b, reference::TanhBackward<float>{}(a, b));
            check(data::PointwiseMode::ELU_BWD,
                  a,
                  b,
                  reference::EluBackward<float>{meta.alpha}(a, b));
            check(data::PointwiseMode::SOFTPLUS_BWD,
                  a,
                  b,
                  reference::SoftplusBackward<float>{meta.beta}(a, b));
            check(data::PointwiseMode::SWISH_BWD,
                  a,
                  b,
                  reference::SwishBackward<float>{meta.beta}(a, b));
            meta.operation = static_cast<int32_t>(data::PointwiseMode::BINARY_SELECT);
            for(const bool mask : {false, true})
            {
                EXPECT_EQ(contract::evaluate(meta, a, b, mask),
                          reference::BinarySelect<float>{}(a, b, mask));
            }
        }
    }
}

TEST(TestStandalonePointwiseV2, PlanRejectsMissingBindingsBeforeCompiling)
{
    const GraphFixture fixture(pack(graph()));
    auto kernel = makeKernel(256, "FLOAT");
    kernel.metadata = {{"contract_version", int64_t{2}}};
    const auto& handler = dispatchHandler(SYMBOLS);
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), {}, kernel), 0U);
    EXPECT_TRUE(handler.supportsSaving());
    EXPECT_THROW(handler.prepare(fixture.context(), {}, kernel),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestStandalonePointwiseV2, ReferenceBoundaryAndNaNSelectionSemantics)
{
    contract::Metadata meta;
    meta.operation = static_cast<int32_t>(data::PointwiseMode::RELU_BWD);
    meta.upper = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(contract::evaluate(meta, 2.0f, nan, false),
              reference::ReluBackward<float>{}(2.0f, nan));
    meta.reluParameterized = 1;
    meta.lower = -1.0f;
    meta.upper = 1.0f;
    meta.slope = 0.25f;
    for(const float input : {-1.0f, 0.0f, 1.0f, 2.0f, nan})
    {
        EXPECT_EQ(contract::evaluate(meta, 2.0f, input, false),
                  (reference::ParameterizedReluBackward<float>{-1.0f, 1.0f, 0.25f}(2.0f, input)));
    }
    meta.operation = static_cast<int32_t>(data::PointwiseMode::MAX_OP);
    EXPECT_TRUE(std::isnan(contract::evaluate(meta, nan, 1.0f, false)));
    EXPECT_EQ(contract::evaluate(meta, 1.0f, nan, false), 1.0f);
    meta.operation = static_cast<int32_t>(data::PointwiseMode::MIN_OP);
    EXPECT_TRUE(std::isnan(contract::evaluate(meta, nan, 1.0f, false)));
    EXPECT_EQ(contract::evaluate(meta, 1.0f, nan, false), 1.0f);
    meta.operation = static_cast<int32_t>(data::PointwiseMode::BINARY_SELECT);
    EXPECT_EQ(contract::evaluate(meta, nan, 1.0f, false), 1.0f);
    EXPECT_TRUE(std::isnan(contract::evaluate(meta, nan, 1.0f, true)));
    EXPECT_TRUE(std::signbit(contract::evaluate(meta, -0.0f, 1.0f, true)));
}

TEST(TestStandalonePointwiseV2, PlanRestoreValidatesLaunchMetadataBeforeLoadingCode)
{
    using namespace hipdnn_plugin_sdk::ingestor;
    SavedLaunchInputs inputs{"hipkernel.standalone_pointwise.dispatch.v2",
                             {{"elements", int64_t{6}},
                              {"arity", int64_t{2}},
                              {"operation", int64_t{2}},
                              {"scalar_mask", int64_t{0}},
                              {"relu_parameterized", int64_t{0}},
                              {"lower", 0.0},
                              {"upper", 3.0},
                              {"slope", 0.0},
                              {"beta", 1.0}}};
    inputs.values.emplace("alpha", 1.0);
    inputs.values.emplace("axis", int64_t{0});
    const std::array<int64_t, 5> dims{1, 1, 1, 2, 3};
    const std::array<int64_t, 5> strides{0, 0, 0, 8, 2};
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        inputs.values.emplace("dim." + std::to_string(axis), dims[axis]);
    }
    for(size_t operand = 0; operand < 4; ++operand)
    {
        const auto root = std::to_string(operand);
        inputs.values.emplace("uid." + root,
                              operand == 2 ? int64_t{0} : static_cast<int64_t>(operand + 1));
        inputs.values.emplace("type." + root, int64_t{0});
        if(operand < 3)
        {
            inputs.values.emplace("kind." + root, int64_t{0});
            inputs.values.emplace("scalar." + root, 0.0);
        }
        for(size_t axis = 0; axis < 5; ++axis)
        {
            inputs.values.emplace("stride." + root + "." + std::to_string(axis),
                                  operand == 2 ? int64_t{0} : strides[axis]);
        }
    }
    SavedKernelCode code;
    code.symbol = "StandalonePointwiseV2";
    const KernelArgument pointer{"global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    code.recordedSignature
        = {pointer,
           pointer,
           pointer,
           pointer,
           {"by_value", static_cast<uint32_t>(sizeof(contract::Metadata)), 0, ""}};
    const auto& handler = dispatchHandler(SYMBOLS);
    const std::vector<std::pair<std::string, MetadataValue>> invalidValues{
        {"arity", int64_t{1}},
        {"operation", int64_t{22}},
        {"elements", int64_t{7}},
        {"uid.0", int64_t{4}},
        {"kind.0", int64_t{2}},
        {"dim.4", int64_t{0}},
        {"stride.3.3", int64_t{2}},
        {"stride.2.4", int64_t{1}},
        {"stride.0.4", int64_t{-1}},
        {"lower", 4.0}};
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

TEST(TestStandalonePointwiseV2, BothPlanContractVersionsRemainIndependentlyRegistered)
{
    auto legacy = SYMBOLS;
    legacy.dispatch = "hipkernel.standalone_pointwise.dispatch.v1";
    const auto& original = dispatchHandler(legacy);
    const auto& current = dispatchHandler(SYMBOLS);
    EXPECT_NE(&original, &current);
    EXPECT_TRUE(original.supportsSaving());
    EXPECT_TRUE(current.supportsSaving());
}

TEST(TestStandalonePointwiseV2, IrrelevantParametersDeclineEvenAtTheirDefaultValues)
{
    using Mutation = void (*)(data::PointwiseAttributesT&);
    const std::vector<Mutation> mutations{
        [](data::PointwiseAttributesT& a) { a.relu_lower_clip = 0.0f; },
        [](data::PointwiseAttributesT& a) {
            a.relu_upper_clip = std::numeric_limits<float>::max();
        },
        [](data::PointwiseAttributesT& a) { a.relu_lower_clip_slope = 0.0f; },
        [](data::PointwiseAttributesT& a) { a.swish_beta = 1.0f; },
        [](data::PointwiseAttributesT& a) { a.elu_alpha = 1.0f; },
        [](data::PointwiseAttributesT& a) { a.softplus_beta = 1.0f; },
        [](data::PointwiseAttributesT& a) { a.axis_tensor_uid = 0; }};
    for(size_t index = 0; index < mutations.size(); ++index)
    {
        auto input = graph();
        mutations[index](*input.nodes[0]->attributes.AsPointwiseAttributes());
        EXPECT_FALSE(matches(input)) << index;
    }
}

TEST(TestGpuStandalonePointwiseV2Plan, BakedValuesRoundTripAndRuntimeValuesAreReadOnEachLaunch)
{
    SKIP_IF_NO_DEVICES();
    namespace provider = hip_kernel_provider::kernel_ingestor_engine;
    namespace fixtures = hip_kernel_provider::testing;
    using namespace hipdnn_plugin_sdk::ingestor;
    const auto root = fixtures::descriptorSetRoot(HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR);
    hipDeviceProp_t properties{};
    std::string arch;
    std::filesystem::path directory;
    ASSERT_NO_FATAL_FAILURE(
        fixtures::findPackedArchDirectoryUnder(root, properties, arch, directory));
    if(directory.empty())
    {
        GTEST_SKIP() << "no production descriptors packed for " << arch;
    }
    KernelDefinition kernel;
    ASSERT_NO_FATAL_FAILURE(
        fixtures::readPackedKernelDefinition(directory, "standalone_pointwise.kdp.json", kernel));
    const auto& handler = dispatchHandler(SYMBOLS);
    const Handle handle;

    for(const bool runtime : {false, true})
    {
        SCOPED_TRACE(runtime);
        auto g = graph();
        g.tensors[1]->dims = {1};
        g.tensors[1]->strides = {1};
        g.tensors[1]->is_runtime_pass_by_value = runtime;
        if(!runtime)
        {
            g.tensors[1]->value.Set(data::Float32Value(1.25f));
        }
        const GraphFixture fixture(pack(g), currentDeviceProperties());
        const auto bound = matchesGraph(SYMBOLS, fixture.context());
        ASSERT_TRUE(bound.has_value());
        const GenericPlan<Handle> plan({kernel, &handler}, fixture.context(), *bound);
        std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>> restored;
        if(runtime)
        {
            EXPECT_FALSE(handler.saveLaunchInputs(plan.prepared()).has_value());
            EXPECT_THROW(provider::captureIngestorPlan(plan, handle),
                         hipdnn_plugin_sdk::HipdnnPluginException);
        }
        else
        {
            const auto payload = provider::captureIngestorPlan(plan, handle);
            EXPECT_EQ(payload.dispatchSymbol, SYMBOLS.dispatch);
            const auto bytes = provider::serialization::encodeIngestorPlan(payload);
            restored = provider::restoreIngestorPlan(bytes.data(), bytes.size(), handle);
            ASSERT_NE(restored, nullptr);
        }
        hipdnn_data_sdk::utilities::Tensor<float> input({2, 3}, {8, 2});
        hipdnn_data_sdk::utilities::Tensor<float> output({2, 3}, {8, 2});
        input.fillWithValue(0.5f);
        float scalar = 1.25f;
        std::vector<hipdnnPluginDeviceBuffer_t> buffers{{1, input.rawDeviceData()},
                                                        {4, output.rawDeviceData()}};
        if(runtime)
        {
            buffers.push_back({2, &scalar});
        }
        const hipdnn_plugin_sdk::IPlan<Handle>* executing = &plan;
        if(restored)
        {
            executing = restored.get();
        }
        for(const auto value : {1.25f, -2.0f})
        {
            scalar = value;
            executing->execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()));
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            output.markDeviceModified();
            static_cast<void>(output.rawHostData());
            EXPECT_EQ(output.getHostValue(1, 2), 0.5f + (runtime ? scalar : 1.25f));
        }
    }
}

} // namespace
#endif
