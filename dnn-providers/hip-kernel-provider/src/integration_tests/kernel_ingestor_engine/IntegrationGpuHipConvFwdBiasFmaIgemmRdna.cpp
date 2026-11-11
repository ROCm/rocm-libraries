// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_HIP_CONV_FWD_BIAS_FMA_IGEMM_RDNA)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"

// ---- hipkernel:hip_conv_fwd_bias_fma_igemm_rdna on the GPU ----
//
// Each case pins the engine, builds a conv-forward node whose virtual output feeds a pointwise
// ADD with a per-output-channel bias, runs it on the device and checks it against the CPU
// reference graph executor. The cases cover the convolutions of the unfused engine's suite
// (1-D, 2-D and 3-D, both packed layouts, stride, padding, asymmetric padding, dilation,
// groups, every channel tile and every dtype) and both ADD operand orders. Graphs the unfused
// engine declines, and an ADD of a full tensor rather than a bias, must be declined.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:hip_conv_fwd_bias_fma_igemm_rdna";

constexpr int64_t X_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t Y_UID = 3;
constexpr int64_t B_UID = 4;

/// Spatial vectors have one entry per spatial dimension, outermost first.
struct BiasCase
{
    std::string name;
    DataType dtype;
    bool channelsLast;
    int64_t n, c, k, groups;
    std::vector<int64_t> in, filter, prePad, postPad, stride, dilation;
    /// ADD(bias, conv) rather than ADD(conv, bias).
    bool biasFirst;
};

std::ostream& operator<<(std::ostream& os, const BiasCase& biasCase)
{
    return os << biasCase.name;
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

std::vector<int64_t> outputDims(const BiasCase& biasCase)
{
    std::vector<int64_t> yDims{biasCase.n, biasCase.k};
    for(size_t i = 0; i < biasCase.in.size(); ++i)
    {
        const int64_t span = biasCase.in[i] + biasCase.prePad[i] + biasCase.postPad[i]
                             - biasCase.dilation[i] * (biasCase.filter[i] - 1) - 1;
        yDims.push_back(span / biasCase.stride[i] + 1);
    }
    return yDims;
}

/// (1, K, 1...) of the convolution's rank, or the full output shape when @p fullTensor.
std::vector<int64_t> biasDims(const BiasCase& biasCase, bool fullTensor)
{
    if(fullTensor)
    {
        return outputDims(biasCase);
    }
    std::vector<int64_t> dims(biasCase.in.size() + 2, 1);
    dims[1] = biasCase.k;
    return dims;
}

/// @p yOut receives the output tensor for a caller-registered validator. @p fullTensorAdd
/// adds a tensor of the output shape instead of a bias.
std::shared_ptr<Graph> buildGraph(const BiasCase& biasCase,
                                  std::shared_ptr<TensorAttributes>* yOut = nullptr,
                                  bool fullTensorAdd = false)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(biasCase.name)
        .set_io_data_type(biasCase.dtype)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // hipDNN carries no group attribute: groups = x channels / w channels.
    const auto xDims = concat({biasCase.n, biasCase.c}, biasCase.in);
    const auto wDims = concat({biasCase.k, biasCase.c / biasCase.groups}, biasCase.filter);
    const auto bDims = biasDims(biasCase, fullTensorAdd);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(X_UID)
        .set_name("X")
        .set_dim(xDims)
        .set_stride(packedStrides(xDims, biasCase.channelsLast))
        .set_data_type(biasCase.dtype);

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(W_UID)
        .set_name("W")
        .set_dim(wDims)
        .set_stride(packedStrides(wDims, biasCase.channelsLast))
        .set_data_type(biasCase.dtype);

    auto b = std::make_shared<TensorAttributes>();
    b->set_uid(B_UID)
        .set_name("B")
        .set_dim(bDims)
        .set_stride(packedStrides(bDims, biasCase.channelsLast))
        .set_data_type(biasCase.dtype);

    ConvFpropAttributes convAttrs;
    convAttrs.set_name("conv_fwd")
        .set_pre_padding(biasCase.prePad)
        .set_post_padding(biasCase.postPad)
        .set_stride(biasCase.stride)
        .set_dilation(biasCase.dilation)
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);
    auto convOut = graph->conv_fprop(x, w, convAttrs);
    // The CPU reference has no ADD of a FLOAT and a 16-bit operand into a 16-bit output, so
    // 16-bit cases carry the convolution output in the data dtype. The engine serves either
    // form and keeps the sum in fp32 regardless.
    if(biasCase.dtype != DataType::FLOAT)
    {
        convOut->set_data_type(biasCase.dtype);
    }

    PointwiseAttributes addAttrs;
    addAttrs.set_name("bias").set_mode(PointwiseMode::ADD).set_compute_data_type(DataType::FLOAT);
    auto y = biasCase.biasFirst ? graph->pointwise(b, convOut, addAttrs)
                                : graph->pointwise(convOut, b, addAttrs);
    y->set_uid(Y_UID).set_name("Y").set_output(true).set_data_type(biasCase.dtype);
    const auto yDims = outputDims(biasCase);
    // infer_properties_node() derives channels-first strides; pin y to the case layout.
    y->set_dim(yDims).set_stride(packedStrides(yDims, biasCase.channelsLast));
    if(yOut != nullptr)
    {
        *yOut = y;
    }
    return graph;
}

/// FLOAT: each output is a (C / groups) x filter-taps sum plus the bias, accumulated in float
/// on both sides but in a different order, so the budget is that many float epsilons.
float floatToleranceEpsilons(const BiasCase& biasCase)
{
    int64_t taps = 1;
    for(const auto extent : biasCase.filter)
    {
        taps *= extent;
    }
    const int64_t terms = biasCase.c / biasCase.groups * taps + 1;
    return static_cast<float>(terms);
}

/// HALF / BFLOAT16: the 16-bit tolerances of the unfused engine's suite, plus what the
/// reference's extra rounding costs. The reference rounds the convolution output to 16 bits
/// before the ADD (see buildGraph) and the kernel does not, which moves the result by up to
/// 2^-p of |conv| = |y - bias| <= |y| + 1 (inputs in [-1, 1]); p is 11 for HALF and 8 for
/// BFLOAT16. The |y| part is within the relative tolerance; the 2^-p part is added to the
/// absolute one, where a near-zero sum of a large convolution output and its bias needs it.
float halfAbsoluteTolerance(DataType dtype)
{
    return 0.001F + (dtype == DataType::HALF ? 1.0F / 2048 : 1.0F / 256);
}
float halfRelativeTolerance(DataType dtype)
{
    return dtype == DataType::HALF ? 0.002F : 0.016F;
}

const std::vector<BiasCase>& cases()
{
    constexpr auto F32 = DataType::FLOAT;
    constexpr auto F16 = DataType::HALF;
    constexpr auto BF16 = DataType::BFLOAT16;
    // clang-format off
    static const std::vector<BiasCase> s_cases{
        // name                 dtype NHWC  n   c    k  g   in            filter     prePad     postPad    stride     dilation   biasFirst
        {"F32Unit3x3",          F32, false, 1,  2,   3, 1, {8, 8},       {3, 3},    {0, 0},    {0, 0},    {1, 1},    {1, 1},    false},
        {"F32Pad1",             F32, false, 2,  8,  16, 1, {14, 14},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false},
        {"F32Pad1BiasFirst",    F32, false, 2,  8,  16, 1, {14, 14},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    true},
        {"F32Stride2Pad1",      F32, false, 2,  8,  16, 1, {15, 17},     {3, 3},    {1, 1},    {1, 1},    {2, 2},    {1, 1},    false},
        {"F32Dilation2",        F32, false, 1,  4,   8, 1, {16, 16},     {3, 3},    {2, 2},    {2, 2},    {1, 1},    {2, 2},    false},
        {"F32AsymPad1x5",       F32, false, 1,  4,   4, 1, {10, 20},     {1, 5},    {0, 2},    {0, 1},    {1, 2},    {1, 1},    false},
        {"F32Stem7x7Stride2",   F32, false, 1,  3,   8, 1, {32, 32},     {7, 7},    {3, 3},    {3, 3},    {2, 2},    {1, 1},    false},
        {"F32TileN64C40",       F32, false, 2, 40,  48, 1, {12, 12},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false},
        {"F32TileN128K200",     F32, false, 1, 64, 200, 1, {10, 10},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false},
        {"F32Groups4Dil2",      F32, false, 1, 32,  64, 4, {20, 20},     {3, 3},    {2, 2},    {2, 2},    {1, 1},    {2, 2},    false},
        {"F32Groups2K96",       F32, false, 2, 48, 192, 2, {9, 11},      {3, 3},    {1, 1},    {1, 1},    {2, 1},    {1, 1},    true},
        {"F32Conv1dDil3",       F32, false, 2, 32,  32, 1, {1, 300},     {1, 7},    {0, 9},    {0, 9},    {1, 1},    {1, 3},    false},
        {"F32Conv1dNclDil3",    F32, false, 2, 32,  32, 1, {300},        {7},       {9},       {9},       {1},       {3},       false},
        {"F16Conv1dNlcStride2", F16, true,  1, 24,  48, 1, {301},        {5},       {2},       {1},       {2},       {1},       false},
        {"F32NhwcPad1",         F32, true,  2, 24,  40, 1, {13, 11},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false},
        {"F32NhwcGroups2S2",    F32, true,  1, 64, 128, 2, {16, 16},     {3, 3},    {1, 1},    {1, 1},    {2, 2},    {1, 1},    true},
        {"F32Conv3dPad1",       F32, false, 1, 16,  32, 1, {6, 10, 12},  {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, false},
        {"F32Conv3dS2Dil2",     F32, false, 2,  8,  24, 1, {9, 11, 13},  {3, 3, 3}, {2, 1, 2}, {2, 1, 2}, {2, 1, 2}, {2, 1, 2}, false},
        {"F32Conv3dTemporal",   F32, false, 1, 32,  64, 1, {7, 8, 8},    {3, 1, 1}, {1, 0, 0}, {1, 0, 0}, {1, 1, 1}, {1, 1, 1}, false},
        {"F32Ndhwc3dGroups2",   F32, true,  1, 32,  64, 2, {5, 9, 9},    {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 2, 2}, {1, 1, 1}, false},
        {"F16Pad1",             F16, false, 2,  8,  16, 1, {14, 14},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false},
        {"F16Stride2Dil2",      F16, false, 1,  4,   8, 1, {19, 19},     {3, 3},    {2, 2},    {2, 2},    {2, 2},    {2, 2},    false},
        {"F16Groups8K128",      F16, false, 1, 64, 128, 8, {16, 16},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false},
        {"F16NhwcTileN128",     F16, true,  1, 48, 160, 1, {12, 12},     {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    true},
        {"F16Conv3dPad1",       F16, false, 1, 16,  32, 1, {6, 10, 12},  {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, false},
        {"Bf16Pad1",            BF16, false, 2,  8,  16, 1, {14, 14},    {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false},
        {"Bf16Pointwise1x1",    BF16, false, 4, 16,  32, 1, {7, 7},      {1, 1},    {0, 0},    {0, 0},    {1, 1},    {1, 1},    true},
        {"Bf16TileN128C72",     BF16, false, 1, 72, 160, 1, {14, 14},    {3, 3},    {1, 1},    {1, 1},    {1, 1},    {1, 1},    false},
        {"Bf16NdhwcConv3d",     BF16, true,  1, 24,  48, 1, {4, 8, 8},   {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, false},
    };
    // clang-format on
    return s_cases;
}

} // namespace

class IntegrationGpuHipConvFwdBiasFmaIgemmRdna
    : public test_utilities::IntegrationGraphVerificationHarness<float, BiasCase>
{
protected:
    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir{
        "hip-conv-fwd-bias-fma-igemm-rdna",
        hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }
};

TEST_P(IntegrationGpuHipConvFwdBiasFmaIgemmRdna, MatchesTheCpuReference)
{
    const auto& biasCase = GetParam();
    std::shared_ptr<TensorAttributes> y;
    auto graph = buildGraph(biasCase, &y);
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
    if(biasCase.dtype == DataType::FLOAT)
    {
        registerValidatorsForOutputs(context, floatToleranceEpsilons(biasCase));
    }
    else
    {
        registerValidator(context,
                          y,
                          halfAbsoluteTolerance(biasCase.dtype),
                          halfRelativeTolerance(biasCase.dtype));
    }
    verifyBuiltGraph(context, 0);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuHipConvFwdBiasFmaIgemmRdna,
                         ::testing::ValuesIn(cases()),
                         [](const ::testing::TestParamInfo<BiasCase>& info) {
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
    explicit CaseBuffers(const BiasCase& biasCase)
    {
        const auto xDims = concat({biasCase.n, biasCase.c}, biasCase.in);
        const auto wDims = concat({biasCase.k, biasCase.c / biasCase.groups}, biasCase.filter);
        const size_t elementSize = biasCase.dtype == DataType::FLOAT ? 4 : 2;
        const auto x = patternBytes(biasCase.dtype, elementCount(xDims), 1);
        const auto w = patternBytes(biasCase.dtype, elementCount(wDims), 5);
        const auto b = patternBytes(biasCase.dtype, biasCase.k, 3);
        _yBytes = static_cast<size_t>(elementCount(outputDims(biasCase))) * elementSize;
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

    std::unordered_map<int64_t, void*> variantPack() const
    {
        return {{X_UID, _x}, {W_UID, _w}, {B_UID, _b}, {Y_UID, _y}};
    }

    /// Clears y, runs @p graph, and returns y's bytes.
    std::vector<uint8_t> run(Graph& graph, hipdnnHandle_t handle, hipStream_t stream)
    {
        EXPECT_EQ(hipMemset(_y, 0xff, _yBytes), hipSuccess);
        auto pack = variantPack();
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

/// One case per kernel shape that a saved plan must reproduce: 2-D NCHW, 3-D, NHWC 16-bit.
bool savesThisCase(const BiasCase& biasCase)
{
    return biasCase.name == "F32Groups2K96" || biasCase.name == "F32Conv3dS2Dil2"
           || biasCase.name == "F16NhwcTileN128" || biasCase.name == "Bf16NdhwcConv3d";
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
// handle, launches the same kernel on the same geometry: its output is bit-identical to the
// output of the plan it was saved from.
TEST_P(IntegrationGpuHipConvFwdBiasFmaIgemmRdna, SavedPlanReproducesTheOutput)
{
    const auto& biasCase = GetParam();
    if(!savesThisCase(biasCase))
    {
        GTEST_SKIP() << "saved plans are checked on four representative graphs";
    }
    CaseBuffers buffers(biasCase);

    for(const auto form : {SaveForm::PLAN_ONLY, SaveForm::GRAPH_AND_PLAN})
    {
        SCOPED_TRACE(form == SaveForm::PLAN_ONLY ? "plan only" : "graph and plan");

        std::vector<uint8_t> bytes;
        std::vector<uint8_t> expected;
        {
            auto graph = buildGraph(biasCase);
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

            expected = buffers.run(*graph, _handle, _stream);
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

        const auto actual = buffers.run(loaded, _handle, _stream);
        EXPECT_TRUE(actual == expected) << "the loaded plan's output differs";
    }
}

// Depthwise (and any group with fewer than 16 output channels) is left to other engines, as
// for the unfused engine; an ADD of a full tensor is not a bias.
TEST_P(IntegrationGpuHipConvFwdBiasFmaIgemmRdna, DeclinesWhatItDoesNotServe)
{
    const auto& base = GetParam();
    if(base.name != "F32Pad1")
    {
        GTEST_SKIP() << "one representative graph is enough";
    }
    BiasCase depthwise = base;
    depthwise.name = "F32Depthwise";
    depthwise.groups = depthwise.c;
    depthwise.k = depthwise.c;
    EXPECT_FALSE(offered(*buildGraph(depthwise), _handle, engineId())) << "depthwise";

    EXPECT_FALSE(offered(*buildGraph(base, nullptr, /*fullTensorAdd=*/true), _handle, engineId()))
        << "full-tensor ADD";
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_HIP_CONV_FWD_BIAS_FMA_IGEMM_RDNA
