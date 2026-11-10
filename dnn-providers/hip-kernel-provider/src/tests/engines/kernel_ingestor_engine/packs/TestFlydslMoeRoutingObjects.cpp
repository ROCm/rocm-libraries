// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeRouting.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslMoeRoutingTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"

/**
 * @file TestFlydslMoeRoutingObjects.cpp
 * @brief Every FlyDSL MoE routing object this device's shard carries, launched through its
 *        pack's dispatch handler and compared exactly with hipDNN's CPU reference
 *        (`CpuFpReferenceMoeRouting`).
 *
 * Each object runs decode and prefill shapes -- one block and many, ragged last tiles --
 * with INT32 and INT64 first_token_offset; ids outside `[0, num_experts)` (expert
 * parallelism: negative, `num_experts` and above, and every id out of range); one token
 * picking an expert more than once; every pair on one expert (the stable order across
 * waves and blocks); and num_experts at 1 and at the object's `e_max`. Outputs start as a
 * sentinel the reference leaves in place too, so a row the kernels write past
 * first_token_offset[E] fails, as does a routed row they miss. Every case is launched
 * three times from one prepared plan and must give the same bytes each time.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;
using hipdnn_data_sdk::utilities::Tensor;

/// Rows the reference and the kernels leave unwritten hold this.
constexpr int32_t SENTINEL = -0x5a5a5a5b;

int64_t intField(const KernelDefinition& kernel, const char* field)
{
    const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, field);
    return value == nullptr ? -1 : *value;
}

/// A device buffer freed on scope exit.
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
        : _bytes(std::max<size_t>(bytes, 1))
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&_pointer, _bytes));
    }

    ~DeviceBuffer()
    {
        static_cast<void>(hipFree(_pointer));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    void* get() const
    {
        return _pointer;
    }

    void upload(const void* host, size_t bytes) const
    {
        EXPECT_EQ(hipSuccess, hipMemcpy(_pointer, host, bytes, hipMemcpyHostToDevice));
    }

    std::vector<unsigned char> read(size_t bytes) const
    {
        std::vector<unsigned char> host(bytes);
        EXPECT_EQ(hipSuccess, hipMemcpy(host.data(), _pointer, bytes, hipMemcpyDeviceToHost));
        return host;
    }

private:
    size_t _bytes;
    void* _pointer = nullptr;
};

/// What one case routes.
struct RoutingCase
{
    std::string label;
    int64_t tokens = 0;
    int64_t topK = 0;
    int64_t experts = 0;
    /// Fraction of pairs given an id outside [0, experts).
    double outOfRange = 0.0;
    /// Every routed pair on expert 0.
    bool oneExpert = false;
};

std::vector<int32_t> makeIds(const RoutingCase& routing)
{
    const auto pairs = static_cast<size_t>(routing.tokens * routing.topK);
    std::vector<int32_t> ids(pairs);
    std::mt19937 generator(
        static_cast<uint32_t>(routing.tokens * 131 + routing.topK * 7 + routing.experts));
    std::uniform_int_distribution<int64_t> expert(0, routing.experts - 1);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for(size_t i = 0; i < pairs; ++i)
    {
        int64_t id = routing.oneExpert ? 0 : expert(generator);
        if(unit(generator) < routing.outOfRange)
        {
            // Below zero, at num_experts, or far above it.
            const auto kind = i % 3;
            if(kind == 0)
            {
                id = -1 - expert(generator);
            }
            else
            {
                id = kind == 1 ? routing.experts : routing.experts + expert(generator);
            }
        }
        ids[i] = static_cast<int32_t>(id);
    }
    return ids;
}

template <typename OffsetType>
void expectCaseMatchesReference(const KernelDefinition& kernel, const RoutingCase& routing)
{
    SCOPED_TRACE(kernel.name + " " + routing.label + (sizeof(OffsetType) == 8 ? " i64" : " i32"));

    const int64_t pairs = routing.tokens * routing.topK;
    const auto ids = makeIds(routing);

    // The reference, its outputs starting as the sentinel.
    const std::vector<int64_t> idDims{routing.tokens, routing.topK};
    const std::vector<int64_t> offsetDims{routing.experts + 1, 1, 1};
    const std::vector<int64_t> rowDims{1, pairs, 1};
    Tensor<int32_t> topkIds(idDims, hipdnn_data_sdk::utilities::generateStrides(idDims));
    Tensor<OffsetType> wantOffsets(offsetDims,
                                   hipdnn_data_sdk::utilities::generateStrides(offsetDims));
    Tensor<int32_t> wantIndex(rowDims, hipdnn_data_sdk::utilities::generateStrides(rowDims));
    Tensor<int32_t> wantKs(rowDims, hipdnn_data_sdk::utilities::generateStrides(rowDims));
    std::copy(ids.begin(), ids.end(), topkIds.memory().hostData());
    std::fill_n(wantOffsets.memory().hostData(), routing.experts + 1, OffsetType(SENTINEL));
    std::fill_n(wantIndex.memory().hostData(), pairs, SENTINEL);
    std::fill_n(wantKs.memory().hostData(), pairs, SENTINEL);
    hipdnn_test_sdk::utilities::CpuFpReferenceMoeRouting::route(
        topkIds, routing.experts, wantOffsets, wantIndex, wantKs);

    MoeRoutingGraphSpec spec;
    spec.tokens = routing.tokens;
    spec.topK = routing.topK;
    spec.experts = routing.experts;
    spec.offsetDataType = sizeof(OffsetType) == 8 ? DataType::INT64 : DataType::INT32;
    const GraphFixture fixture(buildFlydslMoeRoutingGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_MOE_ROUTING, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the graph this object is meant for is declined";
    // The object's own metadata has to admit the graph built for it.
    ASSERT_TRUE(matchesKernel(FLYDSL_MOE_ROUTING, fixture.context(), kernel, *bound));

    const auto& handler = dispatchHandler(FLYDSL_MOE_ROUTING);
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    ASSERT_NE(prepared, nullptr);

    const auto idBytes = static_cast<size_t>(pairs) * sizeof(int32_t);
    const auto offsetBytes = static_cast<size_t>(routing.experts + 1) * sizeof(OffsetType);
    const DeviceBuffer idsDevice(idBytes);
    idsDevice.upload(ids.data(), idBytes);
    const DeviceBuffer offsetsDevice(offsetBytes);
    const DeviceBuffer indexDevice(idBytes);
    const DeviceBuffer ksDevice(idBytes);
    const DeviceBuffer workspace(handler.workspaceBytes(fixture.context(), *bound, kernel));
    const std::vector<int32_t> sentinelRows(static_cast<size_t>(pairs), SENTINEL);
    const std::vector<OffsetType> sentinelOffsets(static_cast<size_t>(routing.experts + 1),
                                                  OffsetType(SENTINEL));

    const std::vector<hipdnnPluginDeviceBuffer_t> arguments{
        {FLYDSL_MOE_ROUTING_TOPK_IDS_UID, idsDevice.get()},
        {FLYDSL_MOE_ROUTING_OFFSETS_UID, offsetsDevice.get()},
        {FLYDSL_MOE_ROUTING_TOKEN_INDEX_UID, indexDevice.get()},
        {FLYDSL_MOE_ROUTING_TOKEN_KS_UID, ksDevice.get()}};
    const Handle handle;

    std::vector<unsigned char> firstOffsets;
    std::vector<unsigned char> firstIndex;
    std::vector<unsigned char> firstKs;
    for(int launch = 0; launch < 3; ++launch)
    {
        SCOPED_TRACE("launch " + std::to_string(launch));
        offsetsDevice.upload(sentinelOffsets.data(), offsetBytes);
        indexDevice.upload(sentinelRows.data(), idBytes);
        ksDevice.upload(sentinelRows.data(), idBytes);
        handler.launch(handle,
                       *prepared,
                       arguments.data(),
                       static_cast<uint32_t>(arguments.size()),
                       workspace.get());
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

        const auto offsets = offsetsDevice.read(offsetBytes);
        const auto index = indexDevice.read(idBytes);
        const auto ks = ksDevice.read(idBytes);
        if(launch == 0)
        {
            const auto* gotOffsets = reinterpret_cast<const OffsetType*>(offsets.data());
            const auto* want = wantOffsets.memory().hostData();
            for(int64_t e = 0; e <= routing.experts; ++e)
            {
                ASSERT_EQ(gotOffsets[e], want[e]) << "first_token_offset[" << e << "]";
            }
            const auto* gotIndex = reinterpret_cast<const int32_t*>(index.data());
            const auto* gotKs = reinterpret_cast<const int32_t*>(ks.data());
            const auto* wantIndexData = wantIndex.memory().hostData();
            const auto* wantKsData = wantKs.memory().hostData();
            for(int64_t row = 0; row < pairs; ++row)
            {
                ASSERT_EQ(gotIndex[row], wantIndexData[row])
                    << "token_index row " << row << " of " << pairs << " ("
                    << wantOffsets.memory().hostData()[routing.experts] << " routed)";
                ASSERT_EQ(gotKs[row], wantKsData[row]) << "token_ks row " << row << " of " << pairs;
            }
            firstOffsets = offsets;
            firstIndex = index;
            firstKs = ks;
        }
        else
        {
            EXPECT_EQ(offsets, firstOffsets) << "first_token_offset differs between launches";
            EXPECT_EQ(index, firstIndex) << "token_index differs between launches";
            EXPECT_EQ(ks, firstKs) << "token_ks differs between launches";
        }
    }
}

void expectObjectMatches(const KernelDefinition& kernel)
{
    const auto eMax = intField(kernel, FLYDSL_MOE_ROUTING_E_MAX_FIELD);
    ASSERT_GE(eMax, 512) << kernel.name;

    const std::vector<RoutingCase> cases{
        {"decode, one token", 1, 10, 512, 0.0, false},
        {"decode batch, EP ids", 16, 10, 512, 0.2, false},
        {"repeated experts in a token", 300, 8, 3, 0.0, false},
        {"one expert", 1, 1, 1, 0.0, false},
        {"every pair on one expert", 3000, 8, 64, 0.0, true},
        {"every id out of range", 77, 4, 300, 1.0, false},
        {"prefill, ragged tiles", 4097, 10, 512, 0.1, false},
        {"prefill, many blocks", 32768, 8, 256, 0.05, false},
        {"num_experts at e_max", 2049, 6, eMax, 0.1, false},
    };
    for(const auto& routing : cases)
    {
        expectCaseMatchesReference<int32_t>(kernel, routing);
        expectCaseMatchesReference<int64_t>(kernel, routing);
    }
}

TEST(TestGpuFlydslMoeRoutingObjects, EveryObjectMatchesTheCpuReference)
{
    SKIP_IF_NO_DEVICES();
    auto kernels = stagedKernels(FLYDSL_MOE_ROUTING.engineName);
    if(kernels.empty())
    {
        reportUnstaged("moe_routing", "this build staged no moe_routing objects for this device");
        return;
    }
    std::sort(kernels.begin(), kernels.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
    });
    for(const auto& kernel : kernels)
    {
        expectObjectMatches(kernel);
    }
    RecordProperty("objects", static_cast<int>(kernels.size()));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
