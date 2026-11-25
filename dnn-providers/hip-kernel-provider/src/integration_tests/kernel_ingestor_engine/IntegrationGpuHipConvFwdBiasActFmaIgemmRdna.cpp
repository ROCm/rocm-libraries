// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA)

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

// ---- hipkernel:hip_conv_fwd_bias_act_fma_igemm_rdna on the GPU ----
//
// Each case pins the engine, builds a conv-forward node whose virtual output feeds a pointwise
// ADD with a per-output-channel bias, whose virtual output feeds a forward activation (or, for
// the NoBias cases, feeds the activation directly), runs it on the device and checks it against
// the CPU reference graph executor. The shape cases are the
// conv + bias engine's suite (1-D, 2-D and 3-D, both packed layouts, stride, padding,
// asymmetric padding, dilation, groups, every channel tile and every dtype), each with one of
// the activations in turn; the sweep cases run every activation and parameter setting in every
// dtype on one graph, with and without the bias. Graphs the conv + bias engine declines, an
// activation the kernel does not apply, conv + bias without an activation and the conv node
// alone must be declined.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:hip_conv_fwd_bias_act_fma_igemm_rdna";

// The targets the bundle declares in configs/hip_conv_fwd_bias_act_fma_igemm_rdna.yaml;
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
    /// The intermediates (convolution output, biased sum) in the data dtype instead of fp32.
    bool dataDtypeIntermediates = false;
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
/// @p withActivation the graph is conv + bias, the conv + bias engine's graph (or the conv node
/// alone for a case without a bias).
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
    if(biasActCase.dataDtypeIntermediates)
    {
        convOut->set_data_type(biasActCase.dtype);
    }

    auto sum = convOut;
    if(biasActCase.withBias)
    {
        PointwiseAttributes addAttrs;
        addAttrs.set_name("bias")
            .set_mode(PointwiseMode::ADD)
            .set_compute_data_type(DataType::FLOAT);
        sum = biasActCase.biasFirst ? graph->pointwise(b, convOut, addAttrs)
                                    : graph->pointwise(convOut, b, addAttrs);
        if(biasActCase.dataDtypeIntermediates)
        {
            sum->set_data_type(biasActCase.dtype);
        }
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

/// FLOAT: each output is a (C / groups) x filter-taps sum plus the bias, accumulated in float
/// on both sides but in a different order, then an activation whose slope is at most 1.13
/// (GELU) and whose fp32 evaluation differs by a few epsilons between the device and the host
/// libraries; 2 x (terms + 4) epsilons covers both.
float floatToleranceEpsilons(const BiasActCase& biasActCase)
{
    int64_t taps = 1;
    for(const auto extent : biasActCase.filter)
    {
        taps *= extent;
    }
    const int64_t terms = biasActCase.c / biasActCase.groups * taps + 1;
    return 2.0F * static_cast<float>(terms + 4);
}

/// HALF / BFLOAT16: the 16-bit tolerances of the unfused engine's suite when both sides keep
/// the intermediates in fp32 and round once. With data-dtype intermediates the reference also
/// rounds the convolution output and the biased sum to 16 bits and the kernel does not, which
/// moves the result by up to 2^-p of each (p is 11 for HALF and 8 for BFLOAT16; inputs in
/// [-1, 1], activation slope at most 1.13); the relative tolerance covers it except near zero,
/// so 2 x 2^-p is added to the absolute one.
float halfAbsoluteTolerance(const BiasActCase& biasActCase)
{
    if(!biasActCase.dataDtypeIntermediates)
    {
        return 0.001F;
    }
    return 0.001F + 2.0F * (biasActCase.dtype == DataType::HALF ? 1.0F / 2048 : 1.0F / 256);
}
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
        // The conv + bias engine's shapes, without their activation.
        // clang-format off
        const std::vector<BiasActCase> shapes{
            // name                 dtype NHWC  n   c    k  g   in            filter     prePad     postPad    stride     dilation   biasFirst
            {"F32Unit3x3",          F32, false, 1,  2,   3, 1, {8, 8},       {3, 3},    {0, 0},    {0, 0},    {1, 1},    {1, 1},    false, {}},
            {"F32Pad1",             F32, false, 2,  8,  16, 1, {14, 14},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false, {}},
            {"F32Pad1BiasFirst",    F32, false, 2,  8,  16, 1, {14, 14},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    true,  {}},
            {"F32Stride2Pad1",      F32, false, 2,  8,  16, 1, {15, 17},     {3, 3},    {1, 1},    {1, 1},    {2, 2},    {1, 1},    false, {}},
            {"F32Dilation2",        F32, false, 1,  4,   8, 1, {16, 16},     {3, 3},    {2, 2},    {2, 2},    {1, 1},    {2, 2},    false, {}},
            {"F32AsymPad1x5",       F32, false, 1,  4,   4, 1, {10, 20},     {1, 5},    {0, 2},    {0, 1},    {1, 2},    {1, 1},    false, {}},
            {"F32Stem7x7Stride2",   F32, false, 1,  3,   8, 1, {32, 32},     {7, 7},    {3, 3},    {3, 3},    {2, 2},    {1, 1},    false, {}},
            {"F32TileN64C40",       F32, false, 2, 40,  48, 1, {12, 12},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false, {}},
            {"F32TileN128K200",     F32, false, 1, 64, 200, 1, {10, 10},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false, {}},
            {"F32Groups4Dil2",      F32, false, 1, 32,  64, 4, {20, 20},     {3, 3},    {2, 2},    {2, 2},    {1, 1},    {2, 2},    false, {}},
            {"F32Groups2K96",       F32, false, 2, 48, 192, 2, {9, 11},      {3, 3},    {1, 1},    {1, 1},    {2, 1},    {1, 1},    true,  {}},
            {"F32Conv1dDil3",       F32, false, 2, 32,  32, 1, {1, 300},     {1, 7},    {0, 9},    {0, 9},    {1, 1},    {1, 3},    false, {}},
            {"F32Conv1dNclDil3",    F32, false, 2, 32,  32, 1, {300},        {7},       {9},       {9},       {1},       {3},       false, {}},
            {"F16Conv1dNlcStride2", F16, true,  1, 24,  48, 1, {301},        {5},       {2},       {1},       {2},       {1},       false, {}},
            {"F32NhwcPad1",         F32, true,  2, 24,  40, 1, {13, 11},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false, {}},
            {"F32NhwcGroups2S2",    F32, true,  1, 64, 128, 2, {16, 16},     {3, 3},    {1, 1},    {1, 1},    {2, 2},    {1, 1},    true,  {}},
            {"F32Conv3dPad1",       F32, false, 1, 16,  32, 1, {6, 10, 12},  {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, false, {}},
            {"F32Conv3dS2Dil2",     F32, false, 2,  8,  24, 1, {9, 11, 13},  {3, 3, 3}, {2, 1, 2}, {2, 1, 2}, {2, 1, 2}, {2, 1, 2}, false, {}},
            {"F32Conv3dTemporal",   F32, false, 1, 32,  64, 1, {7, 8, 8},    {3, 1, 1}, {1, 0, 0}, {1, 0, 0}, {1, 1, 1}, {1, 1, 1}, false, {}},
            {"F32Ndhwc3dGroups2",   F32, true,  1, 32,  64, 2, {5, 9, 9},    {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 2, 2}, {1, 1, 1}, false, {}},
            {"F16Pad1",             F16, false, 2,  8,  16, 1, {14, 14},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false, {}},
            {"F16Stride2Dil2",      F16, false, 1,  4,   8, 1, {19, 19},     {3, 3},    {2, 2},    {2, 2},    {2, 2},    {2, 2},    false, {}},
            {"F16Groups8K128",      F16, false, 1, 64, 128, 8, {16, 16},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false, {}},
            {"F16NhwcTileN128",     F16, true,  1, 48, 160, 1, {12, 12},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    true,  {}},
            {"F16Conv3dPad1",       F16, false, 1, 16,  32, 1, {6, 10, 12},  {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, false, {}},
            {"Bf16Pad1",            BF16, false, 2,  8,  16, 1, {14, 14},    {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false, {}},
            {"Bf16Pointwise1x1",    BF16, false, 4, 16,  32, 1, {7, 7},      {1, 1},    {0, 0},    {0, 0},    {1, 1},    {1, 1},    true,  {}},
            {"Bf16TileN128C72",     BF16, false, 1, 72, 160, 1, {14, 14},    {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false, {}},
            {"Bf16NdhwcConv3d",     BF16, true,  1, 24,  48, 1, {4, 8, 8},   {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, false, {}},
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
        // Every activation in every dtype on one graph, with and without the bias.
        const auto& sweepShape = shapes[1]; // F32Pad1
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
        // The intermediates in the data dtype, with and without the bias.
        for(const bool withBias : {true, false})
        {
            for(const auto& [dtype, prefix] : {std::pair{F16, "F16"}, std::pair{BF16, "Bf16"}})
            {
                for(const size_t act : {size_t{0}, size_t{9}}) // Relu, Gelu
                {
                    auto narrow = sweepShape;
                    narrow.dtype = dtype;
                    narrow.activation = acts[act];
                    narrow.withBias = withBias;
                    narrow.dataDtypeIntermediates = true;
                    narrow.name = std::string(withBias ? "" : "NoBias") + "DataDtypeIntermediates"
                                  + prefix + acts[act].name;
                    all.push_back(narrow);
                }
            }
        }
        return all;
    }();
    return s_cases;
}

} // namespace

class IntegrationGpuHipConvFwdBiasActFmaIgemmRdna
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
        "hip-conv-fwd-bias-act-fma-igemm-rdna",
        hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }
};

TEST_P(IntegrationGpuHipConvFwdBiasActFmaIgemmRdna, MatchesTheCpuReference)
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
        registerValidator(context,
                          y,
                          halfAbsoluteTolerance(biasActCase),
                          halfRelativeTolerance(biasActCase.dtype));
    }
    verifyBuiltGraph(context, 0);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuHipConvFwdBiasActFmaIgemmRdna,
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

/// The cases a saved plan must reproduce: the kernel shapes (2-D NCHW, 3-D, NHWC / NDHWC
/// 16-bit), with activations whose saved parameters change the output (a clamp, a leaky ReLU,
/// a swish beta), and a graph without a bias.
bool savesThisCase(const BiasActCase& biasActCase)
{
    return biasActCase.name == "F32Groups2K96GeluTanh"
           || biasActCase.name == "F32Conv3dTemporalReluClamp"
           || biasActCase.name == "F32Ndhwc3dGroups2LeakyRelu"
           || biasActCase.name == "F16NhwcTileN128Elu"
           || biasActCase.name == "Bf16NdhwcConv3dSwishBeta1p5"
           || biasActCase.name == "NoBiasSweepF16LeakyRelu";
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
TEST_P(IntegrationGpuHipConvFwdBiasActFmaIgemmRdna, SavedPlanReproducesTheOutput)
{
    const auto& biasActCase = GetParam();
    if(!savesThisCase(biasActCase))
    {
        GTEST_SKIP() << "saved plans are checked on six representative graphs";
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

// Depthwise (and any group with fewer than 16 output channels) is left to other engines, as
// for the conv + bias engine; so are an activation the kernel does not apply, conv + bias
// without an activation, and the conv node alone.
TEST_P(IntegrationGpuHipConvFwdBiasActFmaIgemmRdna, DeclinesWhatItDoesNotServe)
{
    const auto& base = GetParam();
    if(base.name != "SweepF32Relu")
    {
        GTEST_SKIP() << "one representative graph is enough";
    }
    BiasActCase depthwise = base;
    depthwise.name = "F32DepthwiseRelu";
    depthwise.groups = depthwise.c;
    depthwise.k = depthwise.c;
    EXPECT_FALSE(offered(*buildGraph(depthwise), _handle, engineId())) << "depthwise";

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

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA
