// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"

/**
 * @file FlydslSdpaObjectTestSupport.hpp
 * @brief The SDPA device harnesses, reached with a named object, the split workspace both
 *        harnesses launch with, and what the per-object host walks read of a loaded object.
 *
 * The RDNA and CDNA dispatch suites each own a launch-and-compare harness that picks the
 * object to run itself. The per-object suite (TestFlydslSdpaObjects.cpp) names the object
 * instead, so these entry points run a given staged object through the same harness: the
 * graph @p spec describes is matched, the object's own kernel matcher must admit it, and
 * every output element (and the LSE when @p spec asks for one) is checked against the
 * harness's double-precision reference. Declared here, defined beside each harness.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// A split launch's workspace as a caller allocates it: exactly the bytes the engine's
/// workspaceBytes() reports, then a guard band holding a canary. The workspace comes from
/// the caller, so a launch that writes past the size the engine reported writes into the
/// caller's memory; here it changes the guard, which overwrittenGuardBytes() counts.
class GuardedWorkspace
{
public:
    /// Far past any shortfall a case's shapes could have (a stat per split row).
    static constexpr size_t GUARD_BYTES = size_t{1} << 20;
    static constexpr unsigned char CANARY = 0xA5;

    explicit GuardedWorkspace(size_t bytes)
        : _bytes(bytes)
    {
        if(bytes == 0)
        {
            return;
        }
        EXPECT_EQ(hipSuccess, hipMalloc(&_base, bytes + GUARD_BYTES));
        EXPECT_EQ(hipSuccess, hipMemset(static_cast<char*>(_base) + bytes, CANARY, GUARD_BYTES));
        EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
    }
    ~GuardedWorkspace()
    {
        static_cast<void>(hipFree(_base));
    }
    GuardedWorkspace(const GuardedWorkspace&) = delete;
    GuardedWorkspace& operator=(const GuardedWorkspace&) = delete;
    GuardedWorkspace(GuardedWorkspace&&) = delete;
    GuardedWorkspace& operator=(GuardedWorkspace&&) = delete;

    /// What the launch is given: null when the engine reported no workspace.
    void* get() const
    {
        return _base;
    }

    /// The guard bytes no longer holding the canary. Call after the launch has finished.
    size_t overwrittenGuardBytes() const
    {
        if(_base == nullptr)
        {
            return 0;
        }
        std::vector<unsigned char> guard(GUARD_BYTES);
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(guard.data(),
                            static_cast<const char*>(_base) + _bytes,
                            GUARD_BYTES,
                            hipMemcpyDeviceToHost));
        size_t overwritten = 0;
        for(const auto byte : guard)
        {
            overwritten += byte == CANARY ? 0 : 1;
        }
        return overwritten;
    }

private:
    size_t _bytes = 0;
    void* _base = nullptr;
};

/// The kernel @p kernel of @p pack as the engine's matchers read it, promoted from the loaded
/// descriptors. A kernel may narrow the arch list its pack claims but never reach outside it,
/// so an empty list on the kernel means "the pack's".
template <typename Pack, typename Kernel>
hipdnn_plugin_sdk::ingestor::KernelDefinition definitionOf(const Pack& pack, const Kernel& kernel)
{
    return hipdnn_plugin_sdk::ingestor::KernelDefinition{kernel.id,
                                                         pack.id,
                                                         pack.dispatchId,
                                                         kernel.source,
                                                         kernel.metadata,
                                                         kernel.priority,
                                                         kernel.arch.empty() ? pack.arch
                                                                             : kernel.arch,
                                                         kernel.originDirectory,
                                                         kernel.name,
                                                         kernel.treeRoot,
                                                         kernel.sources};
}

/// A metadata field of type @p T, or nullptr when absent or of another type.
template <typename T>
const T* metadataValue(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                       const char* field)
{
    const auto it = kernel.metadata.find(std::string(field));
    return it == kernel.metadata.end() ? nullptr : std::get_if<T>(&it->second);
}

/// A metadata integer, or @p absent when the object's KMD has no such field (each family
/// leaves out the fields its kernel fixes).
inline int64_t intField(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                        const char* field,
                        int64_t absent = 0)
{
    const auto* value = metadataValue<int64_t>(kernel, field);
    return value == nullptr ? absent : *value;
}

inline std::string stringField(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                               const char* field)
{
    const auto* value = metadataValue<std::string>(kernel, field);
    return value == nullptr ? std::string() : *value;
}

/// The device a host check stands in for when it matches an object of another arch.
inline hipdnn_plugin_sdk::ingestor::DeviceProperties devicePropertiesFor(const std::string& arch)
{
    hipdnn_plugin_sdk::ingestor::DeviceProperties properties;
    properties.gcnArchName = arch;
    properties.warpSize = arch.rfind("gfx9", 0) == 0 ? 64 : 32;
    properties.multiProcessorCount = 64;
    return properties;
}

/// The RDNA engine a forced object belongs to.
enum class RdnaObjectEngine
{
    PREFILL,
    DECODE,
    AIW,
};

/// Runs @p kernel, an object of @p engine, on the graph @p spec describes and checks it
/// against the RDNA harness's reference at the dispatch suite's tolerance for the dtype.
/// Defined in TestFlydslSdpaRdnaEngine.cpp.
void expectRdnaObjectMatchesReference(const SdpaGraphSpec& spec,
                                      RdnaObjectEngine engine,
                                      const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// Runs @p kernel, an object of any of the three CDNA engines, on the graph @p spec
/// describes and checks it against the CDNA harness's reference. Defined in
/// TestFlydslSdpaCdnaEngine.cpp.
void expectCdnaObjectMatchesReference(const SdpaGraphSpec& spec,
                                      const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
