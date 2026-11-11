// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_HIP_CONV_FWD_RDNA)

#include <algorithm>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

// ---- hipkernel:hip_conv_fwd_rdna on the GPU ----
//
// Each case pins the engine, builds one conv-forward graph, runs it on the device and
// checks it against the CPU reference graph executor. The cases cover 1-D (as H = 1), 2-D
// and 3-D problems, both packed layouts, stride, padding, asymmetric padding, dilation,
// non-square filters, groups, each channel tile width (K per group up to 32, up to 64,
// above 64), channel counts that leave a partial 16-channel chunk, and every dtype.
// Depthwise graphs must be declined.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:hip_conv_fwd_rdna";

/// Spatial vectors have one entry per spatial dimension, outermost first.
struct ConvCase
{
    std::string name;
    DataType dtype;
    bool channelsLast;
    int64_t n, c, k, groups;
    std::vector<int64_t> in, filter, prePad, postPad, stride, dilation;
};

std::ostream& operator<<(std::ostream& os, const ConvCase& convCase)
{
    return os << convCase.name;
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

/// @p yOut receives the output tensor for a caller-registered validator.
std::shared_ptr<Graph> buildGraph(const ConvCase& convCase,
                                  std::shared_ptr<TensorAttributes>* yOut = nullptr)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(convCase.name)
        .set_io_data_type(convCase.dtype)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // hipDNN carries no group attribute: groups = x channels / w channels.
    const auto xDims = concat({convCase.n, convCase.c}, convCase.in);
    const auto wDims = concat({convCase.k, convCase.c / convCase.groups}, convCase.filter);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim(xDims)
        .set_stride(packedStrides(xDims, convCase.channelsLast))
        .set_data_type(convCase.dtype);

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(2)
        .set_name("W")
        .set_dim(wDims)
        .set_stride(packedStrides(wDims, convCase.channelsLast))
        .set_data_type(convCase.dtype);

    ConvFpropAttributes attrs;
    attrs.set_name("conv_fwd")
        .set_pre_padding(convCase.prePad)
        .set_post_padding(convCase.postPad)
        .set_stride(convCase.stride)
        .set_dilation(convCase.dilation)
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

    auto y = graph->conv_fprop(x, w, attrs);
    y->set_uid(3).set_name("Y").set_output(true).set_data_type(convCase.dtype);
    if(convCase.channelsLast)
    {
        // infer_properties_node() derives channels-first strides; pin y to the case layout.
        std::vector<int64_t> yDims{convCase.n, convCase.k};
        for(size_t i = 0; i < convCase.in.size(); ++i)
        {
            const int64_t span = convCase.in[i] + convCase.prePad[i] + convCase.postPad[i]
                                 - convCase.dilation[i] * (convCase.filter[i] - 1) - 1;
            yDims.push_back(span / convCase.stride[i] + 1);
        }
        y->set_dim(yDims).set_stride(packedStrides(yDims, true));
    }
    if(yOut != nullptr)
    {
        *yOut = y;
    }
    return graph;
}

/// FLOAT: each output is a (C / groups) x filter-taps sum accumulated in float on both sides
/// but in a different order, so the budget is that many float epsilons.
float floatToleranceEpsilons(const ConvCase& convCase)
{
    int64_t taps = 1;
    for(const auto extent : convCase.filter)
    {
        taps *= extent;
    }
    return static_cast<float>(convCase.c / convCase.groups * taps);
}

/// HALF / BFLOAT16: the fixture's epsilons are float's, which only an output bit-identical
/// to the reference meets. The two float sums differ in their last bits, which can move the
/// one rounding to 16 bits by an ulp, so these use the 16-bit tolerances of the dgrad suite.
constexpr float HALF_ABSOLUTE_TOLERANCE = 0.001F;
float halfRelativeTolerance(DataType dtype)
{
    return dtype == DataType::HALF ? 0.002F : 0.016F;
}

const std::vector<ConvCase>& cases()
{
    constexpr auto F32 = DataType::FLOAT;
    constexpr auto F16 = DataType::HALF;
    constexpr auto BF16 = DataType::BFLOAT16;
    // clang-format off
    static const std::vector<ConvCase> s_cases{
        // name                 dtype NHWC  n   c    k  g   in            filter     prePad     postPad    stride     dilation
        {"F32Unit3x3",          F32, false, 1,  2,   3, 1, {8, 8},       {3, 3},    {0, 0},    {0, 0},    {1, 1},    {1, 1}},
        {"F32Pad1",             F32, false, 2,  8,  16, 1, {14, 14},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1}},
        {"F32Stride2Pad1",      F32, false, 2,  8,  16, 1, {15, 17},     {3, 3},    {1, 1},    {1, 1},    {2, 2},    {1, 1}},
        {"F32Dilation2",        F32, false, 1,  4,   8, 1, {16, 16},     {3, 3},    {2, 2},    {2, 2},    {1, 1},    {2, 2}},
        {"F32AsymPad1x5",       F32, false, 1,  4,   4, 1, {10, 20},     {1, 5},    {0, 2},    {0, 1},    {1, 2},    {1, 1}},
        {"F32Stem7x7Stride2",   F32, false, 1,  3,   8, 1, {32, 32},     {7, 7},    {3, 3},    {3, 3},    {2, 2},    {1, 1}},
        {"F32TileN64C40",       F32, false, 2, 40,  48, 1, {12, 12},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1}},
        {"F32TileN128K200",     F32, false, 1, 64, 200, 1, {10, 10},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1}},
        {"F32Groups4Dil2",      F32, false, 1, 32,  64, 4, {20, 20},     {3, 3},    {2, 2},    {2, 2},    {1, 1},    {2, 2}},
        {"F32Groups2K96",       F32, false, 2, 48, 192, 2, {9, 11},      {3, 3},    {1, 1},    {1, 1},    {2, 1},    {1, 1}},
        {"F32Conv1dDil3",       F32, false, 2, 32,  32, 1, {1, 300},     {1, 7},    {0, 9},    {0, 9},    {1, 1},    {1, 3}},
        {"F32NhwcPad1",         F32, true,  2, 24,  40, 1, {13, 11},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1}},
        {"F32NhwcGroups2S2",    F32, true,  1, 64, 128, 2, {16, 16},     {3, 3},    {1, 1},    {1, 1},    {2, 2},    {1, 1}},
        {"F32Conv3dPad1",       F32, false, 1, 16,  32, 1, {6, 10, 12},  {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        {"F32Conv3dS2Dil2",     F32, false, 2,  8,  24, 1, {9, 11, 13},  {3, 3, 3}, {2, 1, 2}, {2, 1, 2}, {2, 1, 2}, {2, 1, 2}},
        {"F32Conv3dTemporal",   F32, false, 1, 32,  64, 1, {7, 8, 8},    {3, 1, 1}, {1, 0, 0}, {1, 0, 0}, {1, 1, 1}, {1, 1, 1}},
        {"F32Ndhwc3dGroups2",   F32, true,  1, 32,  64, 2, {5, 9, 9},    {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 2, 2}, {1, 1, 1}},
        {"F16Pad1",             F16, false, 2,  8,  16, 1, {14, 14},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1}},
        {"F16Stride2Dil2",      F16, false, 1,  4,   8, 1, {19, 19},     {3, 3},    {2, 2},    {2, 2},    {2, 2},    {2, 2}},
        {"F16Groups8K128",      F16, false, 1, 64, 128, 8, {16, 16},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1}},
        {"F16NhwcTileN128",     F16, true,  1, 48, 160, 1, {12, 12},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1}},
        {"F16Conv3dPad1",       F16, false, 1, 16,  32, 1, {6, 10, 12},  {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        {"Bf16Pad1",            BF16, false, 2,  8,  16, 1, {14, 14},    {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1}},
        {"Bf16Pointwise1x1",    BF16, false, 4, 16,  32, 1, {7, 7},      {1, 1},    {0, 0},    {0, 0},    {1, 1},    {1, 1}},
        {"Bf16TileN128C72",     BF16, false, 1, 72, 160, 1, {14, 14},    {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1}},
        {"Bf16NdhwcConv3d",     BF16, true,  1, 24,  48, 1, {4, 8, 8},   {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
    };
    // clang-format on
    return s_cases;
}

} // namespace

class IntegrationGpuHipConvFwdRdna
    : public test_utilities::IntegrationGraphVerificationHarness<float, ConvCase>
{
protected:
    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir{
        "hip-conv-fwd-rdna", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }
};

TEST_P(IntegrationGpuHipConvFwdRdna, MatchesTheCpuReference)
{
    const auto& convCase = GetParam();
    std::shared_ptr<TensorAttributes> y;
    auto graph = buildGraph(convCase, &y);
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
    if(convCase.dtype == DataType::FLOAT)
    {
        registerValidatorsForOutputs(context, floatToleranceEpsilons(convCase));
    }
    else
    {
        registerValidator(
            context, y, HALF_ABSOLUTE_TOLERANCE, halfRelativeTolerance(convCase.dtype));
    }
    verifyBuiltGraph(context, 0);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuHipConvFwdRdna,
                         ::testing::ValuesIn(cases()),
                         [](const ::testing::TestParamInfo<ConvCase>& info) {
                             return info.param.name;
                         });

// Depthwise (and any group with fewer than 16 output channels) is left to other kernel
// families.
TEST_P(IntegrationGpuHipConvFwdRdna, DeclinesNarrowGroups)
{
    const auto& base = GetParam();
    if(base.name != "F32Pad1")
    {
        GTEST_SKIP() << "one representative graph is enough";
    }
    ConvCase depthwise = base;
    depthwise.name = "F32Depthwise";
    depthwise.groups = depthwise.c;
    depthwise.k = depthwise.c;

    auto graph = buildGraph(depthwise);
    ASSERT_EQ(graph->build_operation_graph(_handle).code, ErrorCode::OK);
    // No engine at all may serve it here, which the frontend reports as an error; either way
    // this one must not be offered.
    std::vector<int64_t> engines;
    graph->get_ranked_engine_ids(engines);
    EXPECT_EQ(std::find(engines.begin(), engines.end(), engineId()), engines.end());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_HIP_CONV_FWD_RDNA
