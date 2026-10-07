// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "PackedKernelSource.hpp"
#include "TestDescriptorRoot.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

// Plans built from the packed kpack sets that this build stages for the local device.
// Read the packed sets only through a copy in a scratch directory. Other suites corrupt
// or delete the staged archives. Each pack describes its packed set with a PackedPlanCase
// in its own header under packs/.
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

// The packed Pointwise descriptor with block size 256 (workspace 1024).
inline constexpr const char* POINTWISE_B256_DESCRIPTOR = "packed_pointwise_add_b256.ukd.json";
// The packed Pointwise pack, whose inline kernel has block size 64 (workspace 0).
inline constexpr const char* POINTWISE_B64_DESCRIPTOR = "packed_pointwise_add.kdp.json";
// The packed ConvFwd descriptor with block size 64.
inline constexpr const char* CONV_FWD_DESCRIPTOR = "conv_fwd_f16_block64.ukd.json";

// One packed set: where it is staged, the plan the tests build from it, and the exact
// result that plan computes.
struct PackedPlanCase
{
    // The test instance name.
    std::string_view name;
    // The engine that restores a plan of this set.
    std::string_view engineName;
    // The descriptor of the kernel the tests build a plan for.
    const char* descriptor;
    const PackSymbols* pack;
    int64_t outputUid;
    // The root that holds one packed directory per arch.
    const std::filesystem::path& (*root)();
    // The graph that the capture tests run.
    flatbuffers::FlatBufferBuilder (*captureGraph)();
    // The graph that the round-trip tests run.
    flatbuffers::FlatBufferBuilder (*roundTripGraph)();
    // Writes a fixed pattern to the inputs of roundTripGraph.
    void (*fillInputs)(hipdnn_test_sdk::utilities::GraphTensorBundle& tensors);
    // Checks `output` against the exact result of roundTripGraph for the fillInputs values.
    void (*expectExactOutput)(const std::vector<uint8_t>& output);
};

// The elements of `bytes` read as values of type T.
template <typename T>
std::vector<T> valuesOf(const std::vector<uint8_t>& bytes)
{
    std::vector<T> values(bytes.size() / sizeof(T));
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(T));
    return values;
}

// Copies the packed directory of device 0 under `root` into `scratch`. `copy` stays empty
// when the build packed nothing for the device. `deviceProperties` receives the properties
// of device 0, with the decorated arch name the device reports.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void copyPackedArchForDevice(const std::filesystem::path& root,
                                    const std::filesystem::path& scratch,
                                    hipdnn_plugin_sdk::ingestor::DeviceProperties& deviceProperties,
                                    std::filesystem::path& copy)
{
    hipDeviceProp_t properties{};
    std::string arch;
    std::filesystem::path packed;
    ASSERT_NO_FATAL_FAILURE(
        hip_kernel_provider::testing::findPackedArchDirectoryUnder(root, properties, arch, packed));

    deviceProperties.gcnArchName = properties.gcnArchName;
    deviceProperties.warpSize = properties.warpSize;
    deviceProperties.multiProcessorCount = properties.multiProcessorCount;

    if(!packed.empty())
    {
        copy = hip_kernel_provider::testing::copyPackedArchTree(packed, scratch);
    }
}

// A GenericPlan over the registered handler of `packedCase` for `kernel`, on `graph`.
// `tensors` receives the tensors of `graph` with every value zero and the output of
// `packedCase` marked.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void makePackedPlan(const PackedPlanCase& packedCase,
                           flatbuffers::FlatBufferBuilder graph,
                           const hipdnn_plugin_sdk::ingestor::DeviceProperties& deviceProperties,
                           const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                           std::unique_ptr<hipdnn_plugin_sdk::ingestor::GenericPlan<Handle>>& plan,
                           hipdnn_test_sdk::utilities::GraphTensorBundle& tensors)
{
    const PackSymbols& pack = *packedCase.pack;
    const GraphFixture fixture(std::move(graph), deviceProperties);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the test graph does not match its pack";
    plan = std::make_unique<hipdnn_plugin_sdk::ingestor::GenericPlan<Handle>>(
        hipdnn_plugin_sdk::ingestor::KernelDispatcher<Handle>{kernel, &dispatchHandler(pack)},
        fixture.context(),
        *bound);

    tensors = hipdnn_test_sdk::utilities::GraphTensorBundle(fixture.context().graph.getTensorMap());
    for(auto& entry : tensors.tensors)
    {
        entry.second->fillTensorWithValue(0.0F);
    }
    tensors.outputTensorIds = {packedCase.outputUid};
}

// A plan built from a copy of a packed set, the facts restore needs, and the tensors of
// its test graph.
struct PackedPlan
{
    std::filesystem::path copy;
    hipdnn_plugin_sdk::ingestor::DeviceProperties deviceProperties;
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    std::unique_ptr<hipdnn_plugin_sdk::ingestor::GenericPlan<Handle>> plan;
    hipdnn_test_sdk::utilities::GraphTensorBundle tensors;

    std::filesystem::path archive() const
    {
        return std::filesystem::weakly_canonical(kernel.originDirectory / kernel.source.library);
    }
};

// Copies the packed set of `packedCase` into `scratch` and builds a plan for the kernel of
// its descriptor on `graph`. `built.plan` stays null when nothing was packed for device 0.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void buildPackedPlan(const PackedPlanCase& packedCase,
                            flatbuffers::FlatBufferBuilder graph,
                            const std::filesystem::path& scratch,
                            PackedPlan& built)
{
    ASSERT_NO_FATAL_FAILURE(
        copyPackedArchForDevice(packedCase.root(), scratch, built.deviceProperties, built.copy));
    if(built.copy.empty())
    {
        return;
    }

    ASSERT_NO_FATAL_FAILURE(hip_kernel_provider::testing::readPackedKernelDefinition(
        built.copy, packedCase.descriptor, built.kernel));
    ASSERT_NO_FATAL_FAILURE(makePackedPlan(packedCase,
                                           std::move(graph),
                                           built.deviceProperties,
                                           built.kernel,
                                           built.plan,
                                           built.tensors));
}

// The device buffers of `tensors`, in the form that IPlan::execute takes.
inline std::vector<hipdnnPluginDeviceBuffer_t>
    deviceBuffersOf(hipdnn_test_sdk::utilities::GraphTensorBundle& tensors)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers;
    for(const auto& [uid, pointer] : tensors.toDeviceVariantPack())
    {
        buffers.push_back(hipdnnPluginDeviceBuffer_t{uid, pointer});
    }
    return buffers;
}

// The number of entries in the kpack module caches of all packs.
inline size_t packModuleCacheEntries()
{
    return pointwiseKpackModuleCache().size() + convFwdKpackModuleCache().size()
           + gfx950AttentionDenseKpackModuleCache().size();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
