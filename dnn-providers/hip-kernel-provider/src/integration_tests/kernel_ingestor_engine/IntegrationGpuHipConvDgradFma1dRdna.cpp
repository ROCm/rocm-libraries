// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_HIP_CONV_DGRAD_FMA_1D_RDNA)

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
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"

// ---- hipkernel:hip_conv_dgrad_fma_1d_rdna on the GPU ----
//
// Each case pins the engine, builds one 1-D conv_dgrad graph that is not depthwise, runs it on
// the device and checks dx against the CPU reference graph executor. The cases cover every
// specialized variant (k = 16 / 8 / 4 / 9 at stride 8 / 4 / 2 / 5 in every dtype, at every
// channel tile, the dx channel count picking the tile), dy channel counts that are not
// multiples of the 32-channel LDS stage, padding (one where it adds a block to the grid), dx
// positions no tap reaches, rank 4 with H == 1 and a pre-padding other than the post-padding;
// and the generic variant: NLC, groups, dilation (also on a built shape), unbuilt shapes, a
// stride wider than the filter, a pointwise filter. Depthwise and 2-D conv_dgrad and the
// conv_fprop of the same problem must be declined.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:hip_conv_dgrad_fma_1d_rdna";

// The targets the bundle declares in configs/hip_conv_dgrad_fma_1d_rdna.yaml;
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

constexpr int64_t DY_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t DX_UID = 3;

/// A conv_dgrad problem in the forward view: dx is (n, c, in), w is (k, c / groups, filter),
/// dy is (n, k, out). Spatial vectors have one entry per spatial dimension, outermost first.
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

std::vector<int64_t> dxDims(const ConvCase& convCase)
{
    return concat({convCase.n, convCase.c}, convCase.in);
}

std::vector<int64_t> wDims(const ConvCase& convCase)
{
    return concat({convCase.k, convCase.c / convCase.groups}, convCase.filter);
}

std::vector<int64_t> dyDims(const ConvCase& convCase)
{
    std::vector<int64_t> dims{convCase.n, convCase.k};
    for(size_t i = 0; i < convCase.in.size(); ++i)
    {
        const int64_t span = convCase.in[i] + convCase.prePad[i] + convCase.postPad[i]
                             - convCase.dilation[i] * (convCase.filter[i] - 1) - 1;
        dims.push_back(span / convCase.stride[i] + 1);
    }
    return dims;
}

std::shared_ptr<TensorAttributes> tensor(int64_t uid,
                                         const char* name,
                                         const std::vector<int64_t>& dims,
                                         const ConvCase& convCase)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_uid(uid)
        .set_name(name)
        .set_dim(dims)
        .set_stride(packedStrides(dims, convCase.channelsLast))
        .set_data_type(convCase.dtype);
    return t;
}

/// @p dxOut receives dx for a caller-registered validator.
std::shared_ptr<Graph> buildGraph(const ConvCase& convCase,
                                  std::shared_ptr<TensorAttributes>* dxOut = nullptr)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(convCase.name)
        .set_io_data_type(convCase.dtype)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // hipDNN carries no group attribute: groups = dx channels / w channels.
    auto dy = tensor(DY_UID, "DY", dyDims(convCase), convCase);
    auto w = tensor(W_UID, "W", wDims(convCase), convCase);

    ConvDgradAttributes attrs;
    attrs.set_name("conv_dgrad")
        .set_pre_padding(convCase.prePad)
        .set_post_padding(convCase.postPad)
        .set_stride(convCase.stride)
        .set_dilation(convCase.dilation)
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

    auto dx = graph->conv_dgrad(dy, w, attrs);
    const auto dims = dxDims(convCase);
    dx->set_uid(DX_UID)
        .set_name("DX")
        .set_dim(dims)
        .set_stride(packedStrides(dims, convCase.channelsLast))
        .set_output(true)
        .set_data_type(convCase.dtype);
    if(dxOut != nullptr)
    {
        *dxOut = dx;
    }
    return graph;
}

/// FLOAT: each dx element sums at most filter taps x dy channels per group products,
/// accumulated in float on both sides but in a different order, so the budget is that many
/// float epsilons.
float floatToleranceEpsilons(const ConvCase& convCase)
{
    int64_t products = convCase.k / convCase.groups;
    for(const auto extent : convCase.filter)
    {
        products *= extent;
    }
    return static_cast<float>(products);
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
        // name                 dtype NLC   n   c    k    g    in             filter      prePad     postPad    stride     dilation
        // Specialized: every built variant (shape x dtype x channel tile).
        {"F32K16S8Cw4",             F32, false, 2, 35, 24, 1, {304}, {16}, {0}, {0}, {8}, {1}},
        {"F32K16S8Cw2",             F32, false, 1, 12, 40, 1, {567}, {16}, {4}, {4}, {8}, {1}},
        {"F32K16S8Cw1",             F32, false, 1, 5, 33, 1, {1039}, {16}, {1}, {1}, {8}, {1}},
        {"F32K8S4Cw8",              F32, false, 2, 67, 48, 1, {260}, {8}, {0}, {0}, {4}, {1}},
        {"F32K8S4Cw4",              F32, false, 1, 24, 24, 1, {152}, {8}, {0}, {0}, {4}, {1}},
        {"F32K8S4Cw2",              F32, false, 1, 12, 40, 1, {283}, {8}, {2}, {2}, {4}, {1}},
        {"F32K8S4Cw1",              F32, false, 2, 5, 33, 1, {519}, {8}, {1}, {1}, {4}, {1}},
        {"F32K4S2Cw8",              F32, false, 1, 67, 48, 1, {130}, {4}, {0}, {0}, {2}, {1}},
        {"F32K4S2Cw4",              F32, false, 1, 24, 24, 1, {76}, {4}, {0}, {0}, {2}, {1}},
        {"F32K4S2Cw2",              F32, false, 2, 12, 40, 1, {141}, {4}, {1}, {1}, {2}, {1}},
        {"F32K4S2Cw1",              F32, false, 1, 5, 33, 1, {259}, {4}, {1}, {1}, {2}, {1}},
        {"F32K9S5Cw4",              F32, false, 1, 35, 48, 1, {324}, {9}, {0}, {0}, {5}, {1}},
        {"F32K9S5Cw2",              F32, false, 2, 12, 24, 1, {189}, {9}, {0}, {0}, {5}, {1}},
        {"F32K9S5Cw1",              F32, false, 1, 5, 40, 1, {354}, {9}, {2}, {2}, {5}, {1}},
        {"F16K16S8Cw4",             F16, false, 1, 35, 33, 1, {1039}, {16}, {1}, {1}, {8}, {1}},
        {"F16K16S8Cw2",             F16, false, 2, 12, 48, 1, {520}, {16}, {0}, {0}, {8}, {1}},
        {"F16K16S8Cw1",             F16, false, 1, 5, 24, 1, {304}, {16}, {0}, {0}, {8}, {1}},
        {"F16K8S4Cw8",              F16, false, 1, 67, 40, 1, {283}, {8}, {2}, {2}, {4}, {1}},
        {"F16K8S4Cw4",              F16, false, 2, 24, 33, 1, {519}, {8}, {1}, {1}, {4}, {1}},
        {"F16K8S4Cw2",              F16, false, 1, 12, 48, 1, {260}, {8}, {0}, {0}, {4}, {1}},
        {"F16K8S4Cw1",              F16, false, 1, 5, 24, 1, {152}, {8}, {0}, {0}, {4}, {1}},
        {"F16K4S2Cw8",              F16, false, 2, 67, 40, 1, {141}, {4}, {1}, {1}, {2}, {1}},
        {"F16K4S2Cw4",              F16, false, 1, 24, 33, 1, {259}, {4}, {1}, {1}, {2}, {1}},
        {"F16K4S2Cw2",              F16, false, 1, 12, 48, 1, {130}, {4}, {0}, {0}, {2}, {1}},
        {"F16K4S2Cw1",              F16, false, 2, 5, 24, 1, {76}, {4}, {0}, {0}, {2}, {1}},
        {"F16K9S5Cw2",              F16, false, 1, 19, 40, 1, {354}, {9}, {2}, {2}, {5}, {1}},
        {"F16K9S5Cw1",              F16, false, 1, 5, 33, 1, {648}, {9}, {1}, {1}, {5}, {1}},
        {"Bf16K16S8Cw4",            BF16, false, 2, 35, 48, 1, {520}, {16}, {0}, {0}, {8}, {1}},
        {"Bf16K16S8Cw2",            BF16, false, 1, 12, 24, 1, {304}, {16}, {0}, {0}, {8}, {1}},
        {"Bf16K16S8Cw1",            BF16, false, 1, 5, 40, 1, {567}, {16}, {4}, {4}, {8}, {1}},
        {"Bf16K8S4Cw8",             BF16, false, 2, 67, 33, 1, {519}, {8}, {1}, {1}, {4}, {1}},
        {"Bf16K8S4Cw4",             BF16, false, 1, 24, 48, 1, {260}, {8}, {0}, {0}, {4}, {1}},
        {"Bf16K8S4Cw2",             BF16, false, 1, 12, 24, 1, {152}, {8}, {0}, {0}, {4}, {1}},
        {"Bf16K8S4Cw1",             BF16, false, 2, 5, 40, 1, {283}, {8}, {2}, {2}, {4}, {1}},
        {"Bf16K4S2Cw8",             BF16, false, 1, 67, 33, 1, {259}, {4}, {1}, {1}, {2}, {1}},
        {"Bf16K4S2Cw4",             BF16, false, 1, 24, 48, 1, {130}, {4}, {0}, {0}, {2}, {1}},
        {"Bf16K4S2Cw2",             BF16, false, 2, 12, 24, 1, {76}, {4}, {0}, {0}, {2}, {1}},
        {"Bf16K4S2Cw1",             BF16, false, 1, 5, 40, 1, {141}, {4}, {1}, {1}, {2}, {1}},
        {"Bf16K9S5Cw2",             BF16, false, 1, 19, 33, 1, {648}, {9}, {1}, {1}, {5}, {1}},
        {"Bf16K9S5Cw1",             BF16, false, 2, 5, 48, 1, {324}, {9}, {0}, {0}, {5}, {1}},
        // Specialized, rank 4 with H == 1 and a pre-padding other than the post-padding.
        {"F32K16S8H1",              F32, false, 1, 64, 40, 1, {1, 523}, {1, 16}, {0, 4}, {0, 4}, {1, 8}, {1, 1}},
        {"Bf16K9S5H1",              BF16, false, 2, 40, 24, 1, {1, 304}, {1, 9}, {0, 2}, {0, 2}, {1, 5}, {1, 1}},
        {"F16K8S4PrePad3",          F16, false, 1, 48, 32, 1, {301}, {8}, {3}, {1}, {4}, {1}},
        {"F32K8S4Pad2Edge",         F32, false, 1, 64, 24, 1, {512}, {8}, {2}, {2}, {4}, {1}},
        // Generic.
        {"F32K8S4Dil2",             F32, false, 1, 64, 24, 1, {300}, {8}, {4}, {4}, {4}, {2}},
        {"F32NlcK8S4",              F32, true, 2, 48, 32, 1, {258}, {8}, {2}, {2}, {4}, {1}},
        {"F16Groups2K4S2",          F16, false, 1, 32, 48, 2, {200}, {4}, {1}, {1}, {2}, {1}},
        {"F32Groups4K3S1Dil2",      F32, false, 1, 32, 64, 4, {150}, {3}, {2}, {2}, {1}, {2}},
        {"Bf16K7S3",                BF16, false, 1, 40, 24, 1, {301}, {7}, {2}, {2}, {3}, {1}},
        {"F32K3S4",                 F32, false, 1, 24, 16, 1, {203}, {3}, {1}, {1}, {4}, {1}},
        {"F16K1S1",                 F16, false, 2, 64, 32, 1, {97}, {1}, {0}, {0}, {1}, {1}},
        {"Bf16NlcGroups3K5S2Dil3",  BF16, true, 1, 24, 36, 3, {180}, {5}, {4}, {4}, {2}, {3}},
        {"F32NlcH1K9S5",            F32, true, 1, 32, 16, 1, {1, 304}, {1, 9}, {0, 2}, {0, 2}, {1, 5}, {1, 1}},
    };
    // clang-format on
    return s_cases;
}

} // namespace

class IntegrationGpuHipConvDgradFma1dRdna
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
        "hip-conv-dgrad-fma-1d-rdna", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }
};

TEST_P(IntegrationGpuHipConvDgradFma1dRdna, MatchesTheCpuReference)
{
    const auto& convCase = GetParam();
    std::shared_ptr<TensorAttributes> dx;
    auto graph = buildGraph(convCase, &dx);
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
            context, dx, HALF_ABSOLUTE_TOLERANCE, halfRelativeTolerance(convCase.dtype));
    }
    verifyBuiltGraph(context, 0);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuHipConvDgradFma1dRdna,
                         ::testing::ValuesIn(cases()),
                         [](const ::testing::TestParamInfo<ConvCase>& info) {
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

/// Device copies of dy and w and a dx buffer for one case, freed on destruction.
class CaseBuffers
{
public:
    explicit CaseBuffers(const ConvCase& convCase)
    {
        const auto dy = patternBytes(convCase.dtype, elementCount(dyDims(convCase)), 1);
        const auto w = patternBytes(convCase.dtype, elementCount(wDims(convCase)), 5);
        _dxBytes = static_cast<size_t>(elementCount(dxDims(convCase)))
                   * (convCase.dtype == DataType::FLOAT ? 4 : 2);
        EXPECT_EQ(hipMalloc(&_dy, dy.size()), hipSuccess);
        EXPECT_EQ(hipMalloc(&_w, w.size()), hipSuccess);
        EXPECT_EQ(hipMalloc(&_dx, _dxBytes), hipSuccess);
        EXPECT_EQ(hipMemcpy(_dy, dy.data(), dy.size(), hipMemcpyHostToDevice), hipSuccess);
        EXPECT_EQ(hipMemcpy(_w, w.data(), w.size(), hipMemcpyHostToDevice), hipSuccess);
    }
    ~CaseBuffers()
    {
        (void)hipFree(_dy);
        (void)hipFree(_w);
        (void)hipFree(_dx);
    }
    CaseBuffers(const CaseBuffers&) = delete;
    CaseBuffers& operator=(const CaseBuffers&) = delete;

    std::unordered_map<int64_t, void*> variantPack() const
    {
        return {{DY_UID, _dy}, {W_UID, _w}, {DX_UID, _dx}};
    }

    /// Fills dx with a pattern, runs @p graph, and returns dx's bytes.
    std::vector<uint8_t> run(Graph& graph, hipdnnHandle_t handle, hipStream_t stream)
    {
        EXPECT_EQ(hipMemset(_dx, 0xff, _dxBytes), hipSuccess);
        auto pack = variantPack();
        const auto result = graph.execute(handle, pack, nullptr);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        std::vector<uint8_t> dx(_dxBytes);
        EXPECT_EQ(hipMemcpy(dx.data(), _dx, _dxBytes, hipMemcpyDeviceToHost), hipSuccess);
        return dx;
    }

private:
    void* _dy = nullptr;
    void* _w = nullptr;
    void* _dx = nullptr;
    size_t _dxBytes = 0;
};

/// One case per kernel shape a saved plan must reproduce: specialized fp32 (weights read as
/// floats), specialized bf16 with an odd filter width (16-bit weights unpacked from words),
/// generic.
bool savesThisCase(const ConvCase& convCase)
{
    return convCase.name == "F32K16S8Cw4" || convCase.name == "Bf16K9S5Cw2"
           || convCase.name == "F32NlcK8S4";
}

} // namespace

// The engine reports plan serialization, and a plan saved in either form, loaded into a fresh
// handle, launches the same kernel on the same geometry: its output is bit-identical to the
// output of the plan it was saved from.
TEST_P(IntegrationGpuHipConvDgradFma1dRdna, SavedPlanReproducesTheOutput)
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

// Depthwise and 2-D conv_dgrad and the conv_fprop of the same problem are left to other engines.
TEST_P(IntegrationGpuHipConvDgradFma1dRdna, DeclinesWhatItDoesNotServe)
{
    const auto& base = GetParam();
    if(base.name != "F32K8S4Cw8")
    {
        GTEST_SKIP() << "one representative graph is enough";
    }
    const auto offered = [this](Graph& graph) {
        EXPECT_EQ(graph.build_operation_graph(_handle).code, ErrorCode::OK);
        // No engine at all may serve it, which the frontend reports as an error; either way
        // the question is only whether this one is offered.
        std::vector<int64_t> engines;
        graph.get_ranked_engine_ids(engines);
        return std::find(engines.begin(), engines.end(), engineId()) != engines.end();
    };

    ConvCase depthwise = base;
    depthwise.name = "F32Depthwise";
    depthwise.k = base.c;
    depthwise.groups = base.c;
    EXPECT_FALSE(offered(*buildGraph(depthwise))) << "depthwise conv_dgrad";

    ConvCase twoD = base;
    twoD.name = "F32Dgrad2d";
    twoD.in = {14, 16};
    twoD.filter = {3, 3};
    twoD.prePad = {1, 1};
    twoD.postPad = {1, 1};
    twoD.stride = {2, 2};
    twoD.dilation = {1, 1};
    EXPECT_FALSE(offered(*buildGraph(twoD))) << "2-D conv_dgrad";

    // The same problem as a forward convolution.
    auto forward = std::make_shared<Graph>();
    forward->set_name("F32Fprop1d")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);
    auto x = tensor(DY_UID, "X", dxDims(base), base);
    auto w = tensor(W_UID, "W", wDims(base), base);
    ConvFpropAttributes fprop;
    fprop.set_name("conv_fwd")
        .set_pre_padding(base.prePad)
        .set_post_padding(base.postPad)
        .set_stride(base.stride)
        .set_dilation(base.dilation)
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);
    auto y = forward->conv_fprop(x, w, fprop);
    y->set_uid(DX_UID).set_name("Y").set_output(true).set_data_type(DataType::FLOAT);
    EXPECT_FALSE(offered(*forward)) << "conv_fprop";
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_HIP_CONV_DGRAD_FMA_1D_RDNA
