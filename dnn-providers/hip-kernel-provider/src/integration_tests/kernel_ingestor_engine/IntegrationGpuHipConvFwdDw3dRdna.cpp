// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_HIP_CONV_FWD_DW3D_RDNA)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"

// ---- hipkernel:hip_conv_fwd_dw3d_rdna on the GPU ----
//
// Each case pins the engine, builds one depthwise 3-D conv-forward graph (groups == C, K a
// multiple of C), runs it on the device and checks it against the CPU reference graph
// executor. The cases cover every specialized filter and H / W stride in both layouts and every
// dtype, the two-depth tile and its one-depth fallback (depth stride or dilation above 1), a
// channel multiplier of 2, and the generic variant: other filters, H / W strides and
// dilations, and N x K beyond the uniform-k grid. 2-D depthwise and non-depthwise 3-D graphs
// must be declined.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:hip_conv_fwd_dw3d_rdna";

// The targets the bundle declares in configs/hip_conv_fwd_dw3d_rdna.yaml;
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

/// FLOAT: each output is a filter-taps sum accumulated in float on both sides but in a
/// different order, so the budget is that many float epsilons.
float floatToleranceEpsilons(const ConvCase& convCase)
{
    int64_t taps = 1;
    for(const auto extent : convCase.filter)
    {
        taps *= extent;
    }
    return static_cast<float>(taps);
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
        // name                    dtype NDHWC n   c    k    g    in            filter     prePad     postPad    stride     dilation
        // Specialized, NCDHW (uniform-k): two output depths, then one (depth stride / dilation 2).
        {"F32K333S1Td2",           F32, false, 2, 16,  16,  16, {6, 12, 14},  {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        {"F32K333DepthStride2",    F32, false, 1, 8,    8,   8, {9, 10, 11},  {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {2, 1, 1}, {1, 1, 1}},
        {"F32K333DepthDil2",       F32, false, 1, 8,    8,   8, {9, 10, 11},  {3, 3, 3}, {2, 1, 1}, {2, 1, 1}, {1, 1, 1}, {2, 1, 1}},
        {"F32K555S2",              F32, false, 1, 12,  12,  12, {7, 17, 19},  {5, 5, 5}, {2, 2, 2}, {2, 2, 2}, {1, 2, 2}, {1, 1, 1}},
        {"F32K311Temporal",        F32, false, 2, 24,  24,  24, {8, 9, 10},   {3, 1, 1}, {1, 0, 0}, {1, 0, 0}, {1, 1, 1}, {1, 1, 1}},
        {"F32K133S2",              F32, false, 1, 16,  16,  16, {4, 15, 13},  {1, 3, 3}, {0, 1, 1}, {0, 1, 1}, {1, 2, 2}, {1, 1, 1}},
        {"F32K333PadHw",           F32, false, 1, 8,    8,   8, {6, 10, 12},  {3, 3, 3}, {1, 0, 2}, {1, 0, 2}, {1, 1, 1}, {1, 1, 1}},
        {"F16K333Mult2",           F16, false, 1, 8,   16,   8, {5, 9, 11},   {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        {"Bf16K555S1",             BF16, false, 1, 8,   8,   8, {6, 12, 21},  {5, 5, 5}, {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}},
        // Specialized, NDHWC.
        {"F32NdhwcK333S1",         F32, true,  2, 16,  16,  16, {5, 10, 12},  {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        {"F16NdhwcK555S2",         F16, true,  1, 24,  24,  24, {6, 15, 17},  {5, 5, 5}, {2, 2, 2}, {2, 2, 2}, {1, 2, 2}, {1, 1, 1}},
        {"Bf16NdhwcK133S1",        BF16, true, 1, 32,  64,  32, {3, 9, 9},    {1, 3, 3}, {0, 1, 1}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        {"F32NdhwcK555PadHw",      F32, true,  1, 16,  16,  16, {6, 12, 14},  {5, 5, 5}, {2, 1, 3}, {2, 3, 1}, {1, 2, 2}, {1, 1, 1}},
        // Generic: other filters, H / W strides and dilations.
        {"F32GenericK234",         F32, false, 1, 8,    8,   8, {7, 11, 13},  {2, 3, 4}, {0, 1, 2}, {1, 1, 1}, {1, 3, 2}, {1, 2, 1}},
        {"F16NdhwcGenericK777",    F16, true,  1, 16,  16,  16, {9, 9, 9},    {7, 7, 7}, {3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}},
        {"Bf16GenericK333HwDil2",  BF16, false, 1, 8,  16,   8, {6, 13, 13},  {3, 3, 3}, {1, 2, 2}, {1, 2, 2}, {1, 1, 1}, {1, 2, 2}},
        // Generic: N x K above the uniform-k grid's y extent.
        {"F32GenericWideNK",       F32, false, 2, 33000, 33000, 33000, {2, 2, 2}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
    };
    // clang-format on
    return s_cases;
}

} // namespace

class IntegrationGpuHipConvFwdDw3dRdna
    : public test_utilities::IntegrationGraphVerificationHarness<float, ConvCase>
{
protected:
    void SetUp() override
    {
        test_utilities::IntegrationGraphVerificationHarness<float, ConvCase>::SetUp();
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
        "hip-conv-fwd-dw3d-rdna", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }
};

TEST_P(IntegrationGpuHipConvFwdDw3dRdna, MatchesTheCpuReference)
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
                         IntegrationGpuHipConvFwdDw3dRdna,
                         ::testing::ValuesIn(cases()),
                         [](const ::testing::TestParamInfo<ConvCase>& info) {
                             return info.param.name;
                         });

// 2-D depthwise and non-depthwise 3-D convolutions are left to other engines.
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

/// Device copies of x and w and a y buffer for one case, freed on destruction.
class CaseBuffers
{
public:
    explicit CaseBuffers(const ConvCase& convCase)
    {
        const auto xDims = concat({convCase.n, convCase.c}, convCase.in);
        const auto wDims = concat({convCase.k, convCase.c / convCase.groups}, convCase.filter);
        int64_t yCount = convCase.n * convCase.k;
        for(size_t i = 0; i < convCase.in.size(); ++i)
        {
            yCount *= (convCase.in[i] + convCase.prePad[i] + convCase.postPad[i]
                       - convCase.dilation[i] * (convCase.filter[i] - 1) - 1)
                          / convCase.stride[i]
                      + 1;
        }
        const auto x = patternBytes(convCase.dtype, elementCount(xDims), 1);
        const auto w = patternBytes(convCase.dtype, elementCount(wDims), 5);
        _yBytes = static_cast<size_t>(yCount) * (convCase.dtype == DataType::FLOAT ? 4 : 2);
        EXPECT_EQ(hipMalloc(&_x, x.size()), hipSuccess);
        EXPECT_EQ(hipMalloc(&_w, w.size()), hipSuccess);
        EXPECT_EQ(hipMalloc(&_y, _yBytes), hipSuccess);
        EXPECT_EQ(hipMemcpy(_x, x.data(), x.size(), hipMemcpyHostToDevice), hipSuccess);
        EXPECT_EQ(hipMemcpy(_w, w.data(), w.size(), hipMemcpyHostToDevice), hipSuccess);
    }
    ~CaseBuffers()
    {
        (void)hipFree(_x);
        (void)hipFree(_w);
        (void)hipFree(_y);
    }
    CaseBuffers(const CaseBuffers&) = delete;
    CaseBuffers& operator=(const CaseBuffers&) = delete;

    std::unordered_map<int64_t, void*> variantPack() const
    {
        return {{1, _x}, {2, _w}, {3, _y}};
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
    void* _y = nullptr;
    size_t _yBytes = 0;
};

/// One case per kernel shape a saved plan must reproduce: specialized NCDHW two-depth,
/// specialized NDHWC 16-bit, generic.
bool savesThisCase(const ConvCase& convCase)
{
    return convCase.name == "F32K333S1Td2" || convCase.name == "F16NdhwcK555S2"
           || convCase.name == "F32GenericK234";
}

} // namespace

// The engine reports plan serialization, and a plan saved in either form, loaded into a fresh
// handle, launches the same kernel on the same geometry: its output is bit-identical to the
// output of the plan it was saved from.
TEST_P(IntegrationGpuHipConvFwdDw3dRdna, SavedPlanReproducesTheOutput)
{
    const auto& convCase = GetParam();
    if(!savesThisCase(convCase))
    {
        GTEST_SKIP() << "saved plans are checked on three representative graphs";
    }
    CaseBuffers buffers(convCase);

    for(const auto form : {SaveForm::PLAN_ONLY, SaveForm::GRAPH_AND_PLAN})
    {
        SCOPED_TRACE(form == SaveForm::PLAN_ONLY ? "plan only" : "graph and plan");

        std::vector<uint8_t> bytes;
        std::vector<uint8_t> expected;
        {
            auto graph = buildGraph(convCase);
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

TEST_P(IntegrationGpuHipConvFwdDw3dRdna, DeclinesWhatItDoesNotServe)
{
    const auto& base = GetParam();
    if(base.name != "F32K333S1Td2")
    {
        GTEST_SKIP() << "one representative graph is enough";
    }
    const auto offered = [this](const ConvCase& convCase) {
        auto graph = buildGraph(convCase);
        EXPECT_EQ(graph->build_operation_graph(_handle).code, ErrorCode::OK);
        // No engine at all may serve it, which the frontend reports as an error; either way
        // the question is only whether this one is offered.
        std::vector<int64_t> engines;
        graph->get_ranked_engine_ids(engines);
        return std::find(engines.begin(), engines.end(), engineId()) != engines.end();
    };

    ConvCase depthwise2d = base;
    depthwise2d.name = "F32Depthwise2d";
    depthwise2d.in = {12, 14};
    depthwise2d.filter = {3, 3};
    depthwise2d.prePad = {1, 1};
    depthwise2d.postPad = {1, 1};
    depthwise2d.stride = {1, 1};
    depthwise2d.dilation = {1, 1};
    EXPECT_FALSE(offered(depthwise2d)) << "2-D depthwise";

    ConvCase dense = base;
    dense.name = "F32Dense3d";
    dense.groups = 1;
    EXPECT_FALSE(offered(dense)) << "non-depthwise 3-D";
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_HIP_CONV_FWD_DW3D_RDNA
