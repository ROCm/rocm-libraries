// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_HIP_CONV_FWD_BIAS_ACT_DW2D_RDNA)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"

// ---- hipkernel:hip_conv_fwd_bias_act_dw2d_rdna on the GPU ----
//
// Each case pins the engine, builds a depthwise 2-D (or 1-D) conv-forward node whose virtual
// output feeds a pointwise ADD with a per-output-channel bias, whose virtual output feeds a
// forward activation (or, for the NoBias cases, feeds the activation directly), runs it on the
// device and checks it against the CPU reference graph executor. The shape cases are the
// depthwise 2-D engine's suite (every undilated and dilated specialization in every dtype, 1-D,
// and the generic variant: NHWC, a channel multiplier of 2, other filters, paddings, strides and
// dilations), each with one of the activations in turn; the sweep cases run every activation and
// parameter setting in every dtype on one graph, with and without the bias. 3-D depthwise and
// non-depthwise 2-D graphs, an activation the kernel does not apply, conv + bias without an
// activation and the conv node alone must be declined.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:hip_conv_fwd_bias_act_dw2d_rdna";

// The targets the bundle declares in configs/hip_conv_fwd_bias_act_dw2d_rdna.yaml;
// the engine is not offered on any other device.
constexpr std::array<std::string_view, 10> RDNA_TARGETS{"gfx1100",
                                                        "gfx1101",
                                                        "gfx1102",
                                                        "gfx1103",
                                                        "gfx1150",
                                                        "gfx1151",
                                                        "gfx1152",
                                                        "gfx1153",
                                                        "gfx1200",
                                                        "gfx1201"};

constexpr int64_t X_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t Y_UID = 3;
constexpr int64_t B_UID = 4;

/// A forward activation and the attributes the graph sets on it; unset ones keep the
/// reference's defaults.
struct ActivationSpec
{
    const char* name;
    PointwiseMode mode;
    std::optional<float> reluLowerClip;
    std::optional<float> reluUpperClip;
    std::optional<float> reluLowerClipSlope;
    std::optional<float> eluAlpha;
    std::optional<float> swishBeta;
    std::optional<float> softplusBeta;
};

const std::vector<ActivationSpec>& activations()
{
    // clang-format off
    static const std::vector<ActivationSpec> s_activations{
        // name           mode                                lower  upper  slope  alpha  swish  softplus
        {"Relu",          PointwiseMode::RELU_FWD,             {},    {},    {},    {},    {},    {}},
        {"ReluLower",     PointwiseMode::RELU_FWD,             0.1F,  {},    {},    {},    {},    {}},
        {"ReluClamp",     PointwiseMode::RELU_FWD,             0.1F,  0.5F,  {},    {},    {},    {}},
        {"LeakyRelu",     PointwiseMode::RELU_FWD,             {},    {},    0.1F,  {},    {},    {}},
        {"Relu6",         PointwiseMode::RELU_FWD,             {},    6.0F,  {},    {},    {},    {}},
        {"Sigmoid",       PointwiseMode::SIGMOID_FWD,          {},    {},    {},    {},    {},    {}},
        {"Tanh",          PointwiseMode::TANH_FWD,             {},    {},    {},    {},    {},    {}},
        {"Elu",           PointwiseMode::ELU_FWD,              {},    {},    {},    {},    {},    {}},
        {"EluAlpha0p5",   PointwiseMode::ELU_FWD,              {},    {},    {},    0.5F,  {},    {}},
        {"Gelu",          PointwiseMode::GELU_FWD,             {},    {},    {},    {},    {},    {}},
        {"GeluTanh",      PointwiseMode::GELU_APPROX_TANH_FWD, {},    {},    {},    {},    {},    {}},
        {"Swish",         PointwiseMode::SWISH_FWD,            {},    {},    {},    {},    {},    {}},
        {"SwishBeta1p5",  PointwiseMode::SWISH_FWD,            {},    {},    {},    {},    1.5F,  {}},
        {"Softplus",      PointwiseMode::SOFTPLUS_FWD,         {},    {},    {},    {},    {},    {}},
        {"SoftplusBeta2", PointwiseMode::SOFTPLUS_FWD,         {},    {},    {},    {},    {},    2.0F},
        {"Identity",      PointwiseMode::IDENTITY,             {},    {},    {},    {},    {},    {}},
    };
    // clang-format on
    return s_activations;
}

/// Spatial vectors have one entry per spatial dimension, outermost first.
struct BiasActCase
{
    std::string name;
    DataType dtype;
    bool channelsLast;
    int64_t n, c, k, groups;
    std::vector<int64_t> in, filter, prePad, postPad, stride, dilation;
    /// ADD(bias, conv) rather than ADD(conv, bias).
    bool biasFirst;
    ActivationSpec activation;
    /// conv -> ADD(bias) -> activation; false: conv -> activation.
    bool withBias = true;
};

std::ostream& operator<<(std::ostream& os, const BiasActCase& biasActCase)
{
    return os << biasActCase.name;
}

/// Strides of @p dims (NC-spatial order) packed channels-first or channels-last.
std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims, bool channelsLast)
{
    std::vector<int64_t> strides(dims.size(), 1);
    if(!channelsLast)
    {
        for(size_t i = dims.size() - 1; i-- > 0;)
        {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        return strides;
    }
    int64_t expected = dims[1];
    strides[1] = 1;
    for(size_t i = dims.size(); i-- > 2;)
    {
        strides[i] = expected;
        expected *= dims[i];
    }
    strides[0] = expected;
    return strides;
}

std::vector<int64_t> concat(std::vector<int64_t> head, const std::vector<int64_t>& tail)
{
    head.insert(head.end(), tail.begin(), tail.end());
    return head;
}

std::vector<int64_t> outputDims(const BiasActCase& biasActCase)
{
    std::vector<int64_t> yDims{biasActCase.n, biasActCase.k};
    for(size_t i = 0; i < biasActCase.in.size(); ++i)
    {
        const int64_t span = biasActCase.in[i] + biasActCase.prePad[i] + biasActCase.postPad[i]
                             - biasActCase.dilation[i] * (biasActCase.filter[i] - 1) - 1;
        yDims.push_back(span / biasActCase.stride[i] + 1);
    }
    return yDims;
}

PointwiseAttributes activationAttributes(const ActivationSpec& spec)
{
    PointwiseAttributes attributes;
    attributes.set_name("activation").set_mode(spec.mode).set_compute_data_type(DataType::FLOAT);
    if(spec.reluLowerClip)
    {
        attributes.set_relu_lower_clip(*spec.reluLowerClip);
    }
    if(spec.reluUpperClip)
    {
        attributes.set_relu_upper_clip(*spec.reluUpperClip);
    }
    if(spec.reluLowerClipSlope)
    {
        attributes.set_relu_lower_clip_slope(*spec.reluLowerClipSlope);
    }
    if(spec.eluAlpha)
    {
        attributes.set_elu_alpha(*spec.eluAlpha);
    }
    if(spec.swishBeta)
    {
        attributes.set_swish_beta(*spec.swishBeta);
    }
    if(spec.softplusBeta)
    {
        attributes.set_softplus_beta(*spec.softplusBeta);
    }
    return attributes;
}

/// @p yOut receives the output tensor for a caller-registered validator. Without
/// @p withActivation the graph is conv + bias (or the conv node alone for a case without a
/// bias).
std::shared_ptr<Graph> buildGraph(const BiasActCase& biasActCase,
                                  std::shared_ptr<TensorAttributes>* yOut = nullptr,
                                  bool withActivation = true)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(biasActCase.name)
        .set_io_data_type(biasActCase.dtype)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // hipDNN carries no group attribute: groups = x channels / w channels.
    const auto xDims = concat({biasActCase.n, biasActCase.c}, biasActCase.in);
    const auto wDims
        = concat({biasActCase.k, biasActCase.c / biasActCase.groups}, biasActCase.filter);
    std::vector<int64_t> bDims(biasActCase.in.size() + 2, 1);
    bDims[1] = biasActCase.k;

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(X_UID)
        .set_name("X")
        .set_dim(xDims)
        .set_stride(packedStrides(xDims, biasActCase.channelsLast))
        .set_data_type(biasActCase.dtype);

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(W_UID)
        .set_name("W")
        .set_dim(wDims)
        .set_stride(packedStrides(wDims, biasActCase.channelsLast))
        .set_data_type(biasActCase.dtype);

    auto b = std::make_shared<TensorAttributes>();
    b->set_uid(B_UID)
        .set_name("B")
        .set_dim(bDims)
        .set_stride(packedStrides(bDims, biasActCase.channelsLast))
        .set_data_type(biasActCase.dtype);

    ConvFpropAttributes convAttrs;
    convAttrs.set_name("conv_fwd")
        .set_pre_padding(biasActCase.prePad)
        .set_post_padding(biasActCase.postPad)
        .set_stride(biasActCase.stride)
        .set_dilation(biasActCase.dilation)
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);
    auto convOut = graph->conv_fprop(x, w, convAttrs);

    auto sum = convOut;
    if(biasActCase.withBias)
    {
        PointwiseAttributes addAttrs;
        addAttrs.set_name("bias")
            .set_mode(PointwiseMode::ADD)
            .set_compute_data_type(DataType::FLOAT);
        sum = biasActCase.biasFirst ? graph->pointwise(b, convOut, addAttrs)
                                    : graph->pointwise(convOut, b, addAttrs);
    }
    auto y = withActivation ? graph->pointwise(sum, activationAttributes(biasActCase.activation))
                            : sum;
    y->set_uid(Y_UID).set_name("Y").set_output(true).set_data_type(biasActCase.dtype);
    const auto yDims = outputDims(biasActCase);
    // infer_properties_node() derives channels-first strides; pin y to the case layout.
    y->set_dim(yDims).set_stride(packedStrides(yDims, biasActCase.channelsLast));
    if(yOut != nullptr)
    {
        *yOut = y;
    }
    return graph;
}

/// FLOAT: each output is a filter-taps sum plus the bias, accumulated in float on both sides
/// but in a different order, then an activation whose slope is at most 1.13 (GELU) and whose
/// fp32 evaluation differs by a few epsilons between the device and the host libraries;
/// 2 x (terms + 4) epsilons covers both.
float floatToleranceEpsilons(const BiasActCase& biasActCase)
{
    int64_t taps = 1;
    for(const auto extent : biasActCase.filter)
    {
        taps *= extent;
    }
    return 2.0F * static_cast<float>(taps + 1 + 4);
}

/// HALF / BFLOAT16: the 16-bit tolerances of the depthwise 2-D engine's suite. Both sides keep
/// the intermediates in fp32 and round once, so neither the bias nor the activation adds a
/// rounding of its own.
constexpr float HALF_ABSOLUTE_TOLERANCE = 0.001F;
float halfRelativeTolerance(DataType dtype)
{
    return dtype == DataType::HALF ? 0.002F : 0.016F;
}

const std::vector<BiasActCase>& cases()
{
    static const std::vector<BiasActCase> s_cases = [] {
        constexpr auto F32 = DataType::FLOAT;
        constexpr auto F16 = DataType::HALF;
        constexpr auto BF16 = DataType::BFLOAT16;
        // The depthwise 2-D engine's shapes, without their activation.
        // clang-format off
        const std::vector<BiasActCase> shapes{
            // name                 dtype NHWC  n   c    k    g    in          filter    prePad    postPad   stride    dilation   biasFirst
            // Specialized, undilated (NCHW, K == C, "same" pre-padding).
            {"F32K3S1",             F32, false, 2, 32,  32,  32, {14, 16},   {3, 3},   {1, 1},   {1, 1},   {1, 1},   {1, 1}, false, {}},
            {"F16K3S2",             F16, false, 1, 48,  48,  48, {15, 17},   {3, 3},   {1, 1},   {1, 1},   {2, 2},   {1, 1}, false, {}},
            {"F32K3S2PostPad0",     F32, false, 1, 32,  32,  32, {16, 16},   {3, 3},   {1, 1},   {0, 0},   {2, 2},   {1, 1}, true, {}},
            {"Bf16K5S1",            BF16, false, 1, 32, 32,  32, {12, 13},   {5, 5},   {2, 2},   {2, 2},   {1, 1},   {1, 1}, false, {}},
            {"F32K5S2",             F32, false, 1, 32,  32,  32, {17, 19},   {5, 5},   {2, 2},   {2, 2},   {2, 2},   {1, 1}, false, {}},
            {"F16K7S1",             F16, false, 1, 64,  64,  64, {14, 14},   {7, 7},   {3, 3},   {3, 3},   {1, 1},   {1, 1}, true, {}},
            {"F32K7S2",             F32, false, 1, 32,  32,  32, {15, 15},   {7, 7},   {3, 3},   {3, 3},   {2, 2},   {1, 1}, false, {}},
            {"F32K1x31",            F32, false, 1, 32,  32,  32, {1, 200},   {1, 31},  {0, 15},  {0, 15},  {1, 1},   {1, 1}, false, {}},
            {"Bf16Conv1dNclK31",    BF16, false, 2, 32, 32,  32, {150},      {31},     {15},     {15},     {1},      {1}, true, {}},
            // Specialized, dilated (stride 1, pre-padding = dilation x (k - 1) / 2).
            {"F32K3D2",             F32, false, 1, 32,  32,  32, {20, 21},   {3, 3},   {2, 2},   {2, 2},   {1, 1},   {2, 2}, false, {}},
            {"F16K3D4",             F16, false, 1, 32,  32,  32, {24, 25},   {3, 3},   {4, 4},   {4, 4},   {1, 1},   {4, 4}, false, {}},
            {"F16K3D6",             F16, false, 1, 32,  32,  32, {33, 33},   {3, 3},   {6, 6},   {6, 6},   {1, 1},   {6, 6}, true, {}},
            {"F32K3D12",            F32, false, 1, 32,  32,  32, {40, 37},   {3, 3},   {12, 12}, {12, 12}, {1, 1},   {12, 12}, false, {}},
            {"Bf16K3D18",           BF16, false, 1, 32, 32,  32, {41, 40},   {3, 3},   {18, 18}, {18, 18}, {1, 1},   {18, 18}, false, {}},
            {"F32K5D2",             F32, false, 1, 32,  32,  32, {19, 22},   {5, 5},   {4, 4},   {4, 4},   {1, 1},   {2, 2}, true, {}},
            // Generic.
            {"F32NhwcK3S1",         F32, true,  2, 32,  32,  32, {13, 15},   {3, 3},   {1, 1},   {1, 1},   {1, 1},   {1, 1}, false, {}},
            {"F16NhwcK7S2",         F16, true,  1, 48,  48,  48, {15, 15},   {7, 7},   {3, 3},   {3, 3},   {2, 2},   {1, 1}, false, {}},
            {"F16Mult2K3",          F16, false, 1, 16,  32,  16, {12, 12},   {3, 3},   {1, 1},   {1, 1},   {1, 1},   {1, 1}, true, {}},
            {"F32K3Pad0",           F32, false, 1, 32,  32,  32, {12, 12},   {3, 3},   {0, 0},   {0, 0},   {1, 1},   {1, 1}, false, {}},
            {"Bf16K9S1",            BF16, false, 1, 32, 32,  32, {16, 16},   {9, 9},   {4, 4},   {4, 4},   {1, 1},   {1, 1}, false, {}},
            {"F32K3D3",             F32, false, 1, 32,  32,  32, {20, 20},   {3, 3},   {3, 3},   {3, 3},   {1, 1},   {3, 3}, true, {}},
            {"F32K3x5Hw",           F32, false, 1, 32,  32,  32, {17, 23},   {3, 5},   {1, 2},   {1, 2},   {1, 2},   {2, 1}, false, {}},
            {"F32Conv1dNlcK31",     F32, true,  1, 32,  32,  32, {120},      {31},     {15},     {15},     {1},      {1}, false, {}},
        };
        // clang-format on
        const auto& acts = activations();
        std::vector<BiasActCase> all;
        // Each shape with the next activation in turn, so every activation meets several shapes.
        for(size_t i = 0; i < shapes.size(); ++i)
        {
            auto shaped = shapes[i];
            shaped.activation = acts[i % acts.size()];
            shaped.name += shaped.activation.name;
            all.push_back(shaped);
        }
        // The kernels the sweep below does not reach without a bias: 1-D, the dilation grid and
        // the generic one in both layouts.
        for(const size_t shape : {size_t{8}, size_t{11}, size_t{16}, size_t{17}})
        {
            // Bf16Conv1dNclK31, F16K3D6, F16NhwcK7S2, F16Mult2K3
            auto noBias = shapes[shape];
            noBias.withBias = false;
            noBias.activation = acts[(shape + 3) % acts.size()];
            noBias.name = "NoBias" + noBias.name + noBias.activation.name;
            all.push_back(noBias);
        }
        // Every activation in every dtype on one graph, with and without the bias.
        const auto& sweepShape = shapes[0]; // F32K3S1
        for(const bool withBias : {true, false})
        {
            for(const auto& [dtype, prefix] : {std::pair{F32, "SweepF32"},
                                               std::pair{F16, "SweepF16"},
                                               std::pair{BF16, "SweepBf16"}})
            {
                for(const auto& activation : acts)
                {
                    auto swept = sweepShape;
                    swept.dtype = dtype;
                    swept.activation = activation;
                    swept.withBias = withBias;
                    swept.name = std::string(withBias ? "" : "NoBias") + prefix + activation.name;
                    all.push_back(swept);
                }
            }
        }
        return all;
    }();
    return s_cases;
}

} // namespace

class IntegrationGpuHipConvFwdBiasActDw2dRdna
    : public test_utilities::IntegrationGraphVerificationHarness<float, BiasActCase>
{
protected:
    void SetUp() override
    {
        test_utilities::IntegrationGraphVerificationHarness<float, BiasActCase>::SetUp();
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        hipDeviceProp_t properties{};
        ASSERT_EQ(hipGetDeviceProperties(&properties, _deviceId), hipSuccess);
        const std::string_view arch = properties.gcnArchName;
        const bool rdna = std::any_of(
            RDNA_TARGETS.begin(), RDNA_TARGETS.end(), [arch](std::string_view target) {
                return hipdnn_plugin_sdk::archMatches(
                    arch, target, hipdnn_plugin_sdk::ArchMatchMode::PREFIX);
            });
        if(!rdna)
        {
            GTEST_SKIP() << ENGINE_NAME << " ships for the gfx11 and gfx12 RDNA targets only, not "
                         << arch;
        }
    }

    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir{
        "hip-conv-fwd-bias-act-dw2d-rdna",
        hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }
};

TEST_P(IntegrationGpuHipConvFwdBiasActDw2dRdna, MatchesTheCpuReference)
{
    const auto& biasActCase = GetParam();
    std::shared_ptr<TensorAttributes> y;
    auto graph = buildGraph(biasActCase, &y);
    graph->set_preferred_engine_id_ext(engineId());

    auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t servingEngineId = 0;
    ASSERT_EQ(graph->get_execution_plan_engine_id(servingEngineId).code, ErrorCode::OK);
    ASSERT_EQ(servingEngineId, engineId());

    test_utilities::GraphVerificationContext context(*graph);
    if(biasActCase.dtype == DataType::FLOAT)
    {
        registerValidatorsForOutputs(context, floatToleranceEpsilons(biasActCase));
    }
    else
    {
        registerValidator(
            context, y, HALF_ABSOLUTE_TOLERANCE, halfRelativeTolerance(biasActCase.dtype));
    }
    verifyBuiltGraph(context, 0);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuHipConvFwdBiasActDw2dRdna,
                         ::testing::ValuesIn(cases()),
                         [](const ::testing::TestParamInfo<BiasActCase>& info) {
                             return info.param.name;
                         });

namespace
{

int64_t elementCount(const std::vector<int64_t>& dims)
{
    int64_t count = 1;
    for(const auto extent : dims)
    {
        count *= extent;
    }
    return count;
}

/// @p count elements of @p dtype with small, exactly representable values.
std::vector<uint8_t> patternBytes(DataType dtype, int64_t count, int seed)
{
    const size_t elementSize = dtype == DataType::FLOAT ? 4 : 2;
    std::vector<uint8_t> bytes(static_cast<size_t>(count) * elementSize);
    for(int64_t i = 0; i < count; ++i)
    {
        const float value = static_cast<float>((i * 7 + seed) % 9 - 4) * 0.25F;
        uint8_t* slot = bytes.data() + static_cast<size_t>(i) * elementSize;
        if(dtype == DataType::FLOAT)
        {
            std::memcpy(slot, &value, 4);
        }
        else if(dtype == DataType::HALF)
        {
            const auto half = static_cast<_Float16>(value);
            std::memcpy(slot, &half, 2);
        }
        else
        {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, 4);
            const auto upper = static_cast<uint16_t>(bits >> 16);
            std::memcpy(slot, &upper, 2);
        }
    }
    return bytes;
}

/// Device copies of x, w and the bias and a y buffer for one case, freed on destruction.
class CaseBuffers
{
public:
    explicit CaseBuffers(const BiasActCase& biasActCase)
    {
        const auto xDims = concat({biasActCase.n, biasActCase.c}, biasActCase.in);
        const auto wDims
            = concat({biasActCase.k, biasActCase.c / biasActCase.groups}, biasActCase.filter);
        const size_t elementSize = biasActCase.dtype == DataType::FLOAT ? 4 : 2;
        const auto x = patternBytes(biasActCase.dtype, elementCount(xDims), 1);
        const auto w = patternBytes(biasActCase.dtype, elementCount(wDims), 5);
        const auto b = patternBytes(biasActCase.dtype, biasActCase.k, 3);
        _yBytes = static_cast<size_t>(elementCount(outputDims(biasActCase))) * elementSize;
        EXPECT_EQ(hipMalloc(&_x, x.size()), hipSuccess);
        EXPECT_EQ(hipMalloc(&_w, w.size()), hipSuccess);
        EXPECT_EQ(hipMalloc(&_b, b.size()), hipSuccess);
        EXPECT_EQ(hipMalloc(&_y, _yBytes), hipSuccess);
        EXPECT_EQ(hipMemcpy(_x, x.data(), x.size(), hipMemcpyHostToDevice), hipSuccess);
        EXPECT_EQ(hipMemcpy(_w, w.data(), w.size(), hipMemcpyHostToDevice), hipSuccess);
        EXPECT_EQ(hipMemcpy(_b, b.data(), b.size(), hipMemcpyHostToDevice), hipSuccess);
    }
    ~CaseBuffers()
    {
        (void)hipFree(_x);
        (void)hipFree(_w);
        (void)hipFree(_b);
        (void)hipFree(_y);
    }
    CaseBuffers(const CaseBuffers&) = delete;
    CaseBuffers& operator=(const CaseBuffers&) = delete;

    std::unordered_map<int64_t, void*> variantPack(bool withBias) const
    {
        if(!withBias)
        {
            return {{X_UID, _x}, {W_UID, _w}, {Y_UID, _y}};
        }
        return {{X_UID, _x}, {W_UID, _w}, {B_UID, _b}, {Y_UID, _y}};
    }

    /// Clears y, runs @p graph, and returns y's bytes.
    std::vector<uint8_t> run(Graph& graph, hipdnnHandle_t handle, hipStream_t stream, bool withBias)
    {
        EXPECT_EQ(hipMemset(_y, 0xff, _yBytes), hipSuccess);
        auto pack = variantPack(withBias);
        const auto result = graph.execute(handle, pack, nullptr);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        std::vector<uint8_t> y(_yBytes);
        EXPECT_EQ(hipMemcpy(y.data(), _y, _yBytes, hipMemcpyDeviceToHost), hipSuccess);
        return y;
    }

private:
    void* _x = nullptr;
    void* _w = nullptr;
    void* _b = nullptr;
    void* _y = nullptr;
    size_t _yBytes = 0;
};

/// The cases a saved plan must reproduce: the kernel shapes (specialized undilated, specialized
/// on the dilation grid in fp32 and 16-bit, generic NCHW and NHWC), with activations whose saved
/// parameters change the output (a clamp, a swish beta, a leaky slope), and a graph without a
/// bias.
bool savesThisCase(const BiasActCase& biasActCase)
{
    return biasActCase.name == "F32K3S2PostPad0ReluClamp" || biasActCase.name == "F16K3D4GeluTanh"
           || biasActCase.name == "F32K3D12SwishBeta1p5" || biasActCase.name == "F32K3Pad0ReluClamp"
           || biasActCase.name == "NoBiasF16NhwcK7S2LeakyRelu";
}

/// Whether the engine is among those offered for @p graph.
bool offered(Graph& graph, hipdnnHandle_t handle, int64_t engine)
{
    EXPECT_EQ(graph.build_operation_graph(handle).code, ErrorCode::OK);
    // No engine at all may serve it, which the frontend reports as an error; either way the
    // question is only whether this one is offered.
    std::vector<int64_t> engines;
    graph.get_ranked_engine_ids(engines);
    return std::find(engines.begin(), engines.end(), engine) != engines.end();
}

} // namespace

// The engine reports plan serialization, and a plan saved in either form, loaded into a fresh
// handle, launches the same kernel on the same geometry with the same activation: its output
// is bit-identical to the output of the plan it was saved from.
TEST_P(IntegrationGpuHipConvFwdBiasActDw2dRdna, SavedPlanReproducesTheOutput)
{
    const auto& biasActCase = GetParam();
    if(!savesThisCase(biasActCase))
    {
        GTEST_SKIP() << "saved plans are checked on five representative graphs";
    }
    CaseBuffers buffers(biasActCase);

    for(const auto form : {SaveForm::PLAN_ONLY, SaveForm::GRAPH_AND_PLAN})
    {
        SCOPED_TRACE(form == SaveForm::PLAN_ONLY ? "plan only" : "graph and plan");

        std::vector<uint8_t> bytes;
        std::vector<uint8_t> expected;
        {
            auto graph = buildGraph(biasActCase);
            graph->set_preferred_engine_id_ext(engineId());
            auto result = graph->build(_handle);
            ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

            std::vector<BehaviorNote> notes;
            result = graph->get_behavior_notes_for_engine(engineId(), notes);
            ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
            ASSERT_NE(std::find(notes.begin(),
                                notes.end(),
                                BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION),
                      notes.end());

            expected = buffers.run(*graph, _handle, _stream, biasActCase.withBias);
            auto [saved, error] = detail::saveInForm(*graph, form);
            ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;
            bytes = std::move(saved);
        }

        ASSERT_NO_FATAL_FAILURE(detail::replaceHandleWithAFreshOne(_handle, _stream));

        Graph loaded;
        const auto result = detail::loadInForm(loaded, _handle, bytes, form);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        int64_t servingEngineId = 0;
        ASSERT_EQ(loaded.get_execution_plan_engine_id(servingEngineId).code, ErrorCode::OK);
        EXPECT_EQ(servingEngineId, engineId());

        const auto actual = buffers.run(loaded, _handle, _stream, biasActCase.withBias);
        EXPECT_TRUE(actual == expected) << "the loaded plan's output differs";
    }
}

// 3-D depthwise and non-depthwise 2-D convolutions are left to other engines; so are an
// activation the kernel does not apply, conv + bias without an activation, and the conv node
// alone.
TEST_P(IntegrationGpuHipConvFwdBiasActDw2dRdna, DeclinesWhatItDoesNotServe)
{
    const auto& base = GetParam();
    if(base.name != "SweepF32Relu")
    {
        GTEST_SKIP() << "one representative graph is enough";
    }
    BiasActCase depthwise3d = base;
    depthwise3d.name = "F32Depthwise3dRelu";
    depthwise3d.in = {5, 12, 14};
    depthwise3d.filter = {3, 3, 3};
    depthwise3d.prePad = {1, 1, 1};
    depthwise3d.postPad = {1, 1, 1};
    depthwise3d.stride = {1, 1, 1};
    depthwise3d.dilation = {1, 1, 1};
    EXPECT_FALSE(offered(*buildGraph(depthwise3d), _handle, engineId())) << "3-D depthwise";

    BiasActCase dense = base;
    dense.name = "F32Dense2dRelu";
    dense.groups = 1;
    EXPECT_FALSE(offered(*buildGraph(dense), _handle, engineId())) << "non-depthwise 2-D";

    BiasActCase exp = base;
    exp.name = "F32Exp";
    exp.activation = ActivationSpec{"Exp", PointwiseMode::EXP, {}, {}, {}, {}, {}, {}};
    EXPECT_FALSE(offered(*buildGraph(exp), _handle, engineId())) << "EXP";

    EXPECT_FALSE(offered(*buildGraph(base, nullptr, /*withActivation=*/false), _handle, engineId()))
        << "conv + bias";

    BiasActCase convOnly = base;
    convOnly.name = "F32ConvOnly";
    convOnly.withBias = false;
    EXPECT_FALSE(
        offered(*buildGraph(convOnly, nullptr, /*withActivation=*/false), _handle, engineId()))
        << "conv alone";
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_HIP_CONV_FWD_BIAS_ACT_DW2D_RDNA
