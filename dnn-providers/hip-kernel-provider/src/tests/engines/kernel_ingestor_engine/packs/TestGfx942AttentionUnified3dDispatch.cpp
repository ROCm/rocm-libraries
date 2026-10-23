// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "TestDescriptorRoot.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dTestGraphs.hpp"

/**
 * @file TestGfx942AttentionUnified3dDispatch.cpp
 * @brief hipkernel:Gfx942AttentionUnified3d's dispatch, end to end on a gfx942 device:
 *        prepare() and launch() of a SHIPPED kernel, compared with CpuFpReferenceSdpa.
 *
 * Every case pre-fills O with 0xFF bytes (NaN in fp16 and bf16): O is written only by the
 * reduce launch, so a launch() that drops the reduce leaves NaN and every element fails
 * the comparison. Some cases also pre-fill the workspace with 0xFF, so a segment launch
 * that leaves any split-KV slot unwritten, or seqused_k / cu_seqlens_q not refreshed,
 * shows up as NaN or a wrong row.
 *
 * Skipped, with the reason, when there is no device or the device is not gfx942: the
 * shipped code objects are gfx942-only. On a gfx942 device the shipped descriptor set
 * MUST be discoverable (descriptorSearchDirectories()); its absence fails.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d
{
namespace
{

using hipdnn_data_sdk::utilities::ShallowTensor;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;

enum class KvLayout
{
    BHSD,
    BSHD,
    BHSD_PADDED_ROW, // row stride D + 8
};

struct DispatchCase
{
    std::string name;
    data_objects::DataType dataType;
    int64_t batch;
    int64_t numQueryHeads;
    int64_t numKvHeads;
    int64_t seqLenKv;
    int64_t headSize;
    KvLayout kLayout;
    KvLayout vLayout;
    std::string kernelName; // a shipped variant
    bool poisonWorkspace;
};

/// The cases, defined at the end of the file.
std::vector<DispatchCase> dispatchCases();

std::vector<int64_t> kvStrides(KvLayout layout, int64_t heads, int64_t seq, int64_t d)
{
    switch(layout)
    {
    case KvLayout::BSHD:
        return bshdStrides(heads, seq, d);
    case KvLayout::BHSD_PADDED_ROW:
    {
        const int64_t row = d + 8;
        return {heads * seq * row, seq * row, row, 1};
    }
    case KvLayout::BHSD:
    default:
        return bhsdStrides(heads, seq, d);
    }
}

/// Elements a strided view spans: 1 + sum((n - 1) * stride).
size_t spanElements(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    int64_t span = 1;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        span += (dims[i] - 1) * strides[i];
    }
    return static_cast<size_t>(span);
}

bool isGfx942(const hipdnn_plugin_sdk::ingestor::DeviceProperties& properties)
{
    return std::string_view(properties.gcnArchName).substr(0, 6) == "gfx942";
}

/// The shipped kernel named @p name, as the engine's catalog for @p context hands it to
/// dispatch; nullopt if the catalog for this graph and device does not hold it.
///
/// Loads the shipped descriptor tree itself, resolved from this binary the way
/// integration_tests/main.cpp resolves it. Process-wide discovery cannot serve: this
/// binary's main() points HIPDNN_DESCRIPTOR_DIR at the unit test set, which replaces the
/// provider tree, and discovery is memoized for the process. Loading here keeps the
/// shipped set out of every other unit test's discovery.
/// @throws std::runtime_error the shipped tree is absent or holds no set for this engine.
std::optional<KernelDefinition> shippedKernel(const MatchContext& context, const std::string& name)
{
    const auto root = hip_kernel_provider::testing::descriptorSetRoot(
        HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR);
    if(const auto unusable = hip_kernel_provider::testing::describeUnusableDescriptorRoot(root);
       !unusable.empty())
    {
        throw std::runtime_error(unusable);
    }

    // Validation drops a set whose native symbols are not registered yet.
    registerNativeIngestorSymbols();
    auto validated = hipdnn_plugin_sdk::ingestor::loadValidatedDescriptorSets<Handle>(root);
    const auto shipped = std::find_if(validated.begin(), validated.end(), [](const auto& entry) {
        return entry.set.engine.name == PACK.engineName;
    });
    if(shipped == validated.end())
    {
        throw std::runtime_error("no descriptor set for engine '" + std::string(PACK.engineName)
                                 + "' under '" + root.string() + "'");
    }
    for(auto& kernel : shipped->stateManager->unsortedDefinitions(context))
    {
        if(kernel.name == name)
        {
            return std::move(kernel);
        }
    }
    return std::nullopt;
}

/// A device buffer freed on scope exit.
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
        : _bytes(bytes)
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&_ptr, std::max<size_t>(bytes, 1)));
    }
    ~DeviceBuffer()
    {
        static_cast<void>(hipFree(_ptr));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const
    {
        return _ptr;
    }
    size_t bytes() const
    {
        return _bytes;
    }

private:
    void* _ptr = nullptr;
    size_t _bytes = 0;
};

template <typename T>
void runCase(const DispatchCase& c)
{
    const auto device = currentDeviceProperties();

    GraphSpec spec;
    spec.batch = c.batch;
    spec.numQueryHeads = c.numQueryHeads;
    spec.numKvHeads = c.numKvHeads;
    spec.seqLenKv = c.seqLenKv;
    spec.headSize = c.headSize;
    spec.dataType = c.dataType;
    spec.attnScaleValue = 1.0F / std::sqrt(static_cast<float>(c.headSize));
    spec.kStrides = kvStrides(c.kLayout, c.numKvHeads, c.seqLenKv, c.headSize);
    spec.vStrides = kvStrides(c.vLayout, c.numKvHeads, c.seqLenKv, c.headSize);
    const GraphFixture graph(buildGraph(spec), device);

    const auto bound = matchesGraph(PACK, graph.context());
    ASSERT_TRUE(bound.has_value()) << "graph_match refused the test graph";

    std::optional<KernelDefinition> kernel;
    try
    {
        kernel = shippedKernel(graph.context(), c.kernelName);
    }
    catch(const std::exception& e)
    {
        FAIL() << "gfx942 device but the shipped " << PACK.engineName
               << " descriptor set is not discoverable: " << e.what();
    }
    ASSERT_TRUE(kernel.has_value())
        << "kernel '" << c.kernelName << "' is not in the shipped catalog for this graph";
    ASSERT_TRUE(matchesKernel(PACK, graph.context(), *kernel, *bound));

    const auto& handler = dispatchHandler(PACK);
    const size_t workspaceBytes = handler.workspaceBytes(graph.context(), *bound, *kernel);
    const auto prepared = handler.prepare(graph.context(), *bound, *kernel);
    ASSERT_NE(prepared, nullptr);

    // Host tensors in the graph's own strides; the device copies are byte-identical.
    const std::vector<int64_t> qDims{c.batch, c.numQueryHeads, 1, c.headSize};
    const std::vector<int64_t> kvDims{c.batch, c.numKvHeads, c.seqLenKv, c.headSize};
    const auto qoStrides = decodeQoStrides(c.numQueryHeads, c.headSize);
    std::vector<T> q(spanElements(qDims, qoStrides));
    std::vector<T> k(spanElements(kvDims, *spec.kStrides));
    std::vector<T> v(spanElements(kvDims, *spec.vStrides));
    std::vector<T> expected(q.size());
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for(auto* buffer : {&q, &k, &v})
    {
        std::generate(buffer->begin(), buffer->end(), [&] { return static_cast<T>(dist(rng)); });
    }

    const ShallowTensor<T> qView(q.data(), qDims, qoStrides);
    const ShallowTensor<T> kView(k.data(), kvDims, *spec.kStrides);
    const ShallowTensor<T> vView(v.data(), kvDims, *spec.vStrides);
    ShallowTensor<T> oView(expected.data(), qDims, qoStrides);
    // Decode with a bottom-right causal mask sees every key.
    hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward<T, T, T, T, float>(
        qView,
        kView,
        vView,
        oView,
        spec.attnScaleValue,
        /*attnMask=*/nullptr,
        /*leftBound=*/-1,
        /*rightBound=*/0,
        /*topLeftAlignment=*/false);

    const DeviceBuffer dq(q.size() * sizeof(T));
    const DeviceBuffer dk(k.size() * sizeof(T));
    const DeviceBuffer dv(v.size() * sizeof(T));
    const DeviceBuffer dout(expected.size() * sizeof(T));
    const DeviceBuffer workspace(workspaceBytes);
    ASSERT_EQ(hipSuccess, hipMemcpy(dq.get(), q.data(), dq.bytes(), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemcpy(dk.get(), k.data(), dk.bytes(), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemcpy(dv.get(), v.data(), dv.bytes(), hipMemcpyHostToDevice));
    // NaN in both fp16 and bf16: survives only if the reduce never writes O.
    ASSERT_EQ(hipSuccess, hipMemset(dout.get(), 0xFF, dout.bytes()));
    ASSERT_EQ(hipSuccess,
              hipMemset(workspace.get(), c.poisonWorkspace ? 0xFF : 0x00, workspace.bytes()));

    const std::array<hipdnnPluginDeviceBuffer_t, 4> buffers{
        hipdnnPluginDeviceBuffer_t{Q_UID, dq.get()},
        hipdnnPluginDeviceBuffer_t{K_UID, dk.get()},
        hipdnnPluginDeviceBuffer_t{V_UID, dv.get()},
        hipdnnPluginDeviceBuffer_t{O_UID, dout.get()}};
    const Handle handle;
    // Twice: the second launch reuses the prepared plan and a workspace the first dirtied.
    for(int launch = 0; launch < 2; ++launch)
    {
        handler.launch(handle,
                       *prepared,
                       buffers.data(),
                       static_cast<uint32_t>(buffers.size()),
                       workspace.get());
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(handle.getStream()));
    }

    std::vector<T> actual(expected.size());
    ASSERT_EQ(hipSuccess,
              hipMemcpy(actual.data(), dout.get(), dout.bytes(), hipMemcpyDeviceToHost));

    const float tolerance = c.dataType == data_objects::DataType::BFLOAT16 ? 2.0e-2F : 5.0e-3F;
    size_t mismatches = 0;
    for(size_t i = 0; i < actual.size(); ++i)
    {
        const auto got = static_cast<float>(actual[i]);
        const auto want = static_cast<float>(expected[i]);
        if(!(std::abs(got - want) <= tolerance))
        {
            if(mismatches < 8)
            {
                ADD_FAILURE() << "O[" << i << "] = " << got << ", reference " << want;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u) << "of " << actual.size() << " output elements";
}

class TestGfx942AttentionUnified3dDispatch : public ::testing::TestWithParam<DispatchCase>
{
};

TEST_P(TestGfx942AttentionUnified3dDispatch, MatchesCpuReferenceOnDevice)
{
    SKIP_IF_NO_DEVICES();
    const auto device = currentDeviceProperties();
    if(!isGfx942(device))
    {
        GTEST_SKIP() << "needs a gfx942 device (the shipped code objects are gfx942-only); "
                        "this device is '"
                     << device.gcnArchName << "'";
    }
    const auto& c = GetParam();
    if(c.dataType == data_objects::DataType::BFLOAT16)
    {
        runCase<hipdnn_data_sdk::types::bfloat16>(c);
    }
    else
    {
        runCase<hipdnn_data_sdk::types::half>(c);
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGfx942AttentionUnified3dDispatch,
                         ::testing::ValuesIn(dispatchCases()),
                         [](const ::testing::TestParamInfo<DispatchCase>& info) {
                             return info.param.name;
                         });

// No device needed: every case is a graph the engine admits and a kernel record it accepts,
// so on gfx942 the cases exercise the launch, not the matchers.
TEST(TestGfx942AttentionUnified3dDispatchCases, EveryCaseIsAdmitted)
{
    for(const auto& c : dispatchCases())
    {
        GraphSpec spec;
        spec.batch = c.batch;
        spec.numQueryHeads = c.numQueryHeads;
        spec.numKvHeads = c.numKvHeads;
        spec.seqLenKv = c.seqLenKv;
        spec.headSize = c.headSize;
        spec.dataType = c.dataType;
        spec.kStrides = kvStrides(c.kLayout, c.numKvHeads, c.seqLenKv, c.headSize);
        spec.vStrides = kvStrides(c.vLayout, c.numKvHeads, c.seqLenKv, c.headSize);
        KernelSpec kernel;
        kernel.dtype = c.dataType == data_objects::DataType::BFLOAT16 ? "BF16" : "FP16";
        kernel.headSize = c.headSize;
        kernel.numQueryHeads = c.numQueryHeads;
        kernel.numKvHeads = c.numKvHeads;
        EXPECT_TRUE(kernelMatchesGraph(spec, kernel)) << c.name;
    }
}

std::vector<DispatchCase> dispatchCases()
{
    return {DispatchCase{"Fp16D128Gqa4Bhsd",
                         data_objects::DataType::HALF,
                         4,
                         32,
                         8,
                         512,
                         128,
                         KvLayout::BHSD,
                         KvLayout::BHSD,
                         "attention_unified_3d.fp16_d128_hq32_kv8_b16_t16_seg32.gfx942",
                         false},
            DispatchCase{"Bf16D128MixedKBshdVBhsdPoisonedWorkspace",
                         data_objects::DataType::BFLOAT16,
                         2,
                         32,
                         8,
                         1000,
                         128,
                         KvLayout::BSHD,
                         KvLayout::BHSD,
                         "attention_unified_3d.bf16_d128_hq32_kv8_b16_t16_seg38.gfx942",
                         true},
            DispatchCase{"Fp16D64MqaPaddedRowPoisonedWorkspace",
                         data_objects::DataType::HALF,
                         3,
                         8,
                         1,
                         33,
                         64,
                         KvLayout::BHSD_PADDED_ROW,
                         KvLayout::BHSD_PADDED_ROW,
                         "attention_unified_3d.fp16_d64_hq8_kv1_b16_t16_seg7.gfx942",
                         true},
            DispatchCase{"Fp16D256Block32Bshd",
                         data_objects::DataType::HALF,
                         1,
                         8,
                         8,
                         4096,
                         256,
                         KvLayout::BSHD,
                         KvLayout::BSHD,
                         "attention_unified_3d.fp16_d256_hq8_kv8_b32_t16_seg16.gfx942",
                         false}};
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
