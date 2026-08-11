// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/ExecutionContextBase.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

/// Smallest engine plugin that satisfies the three type contracts of
/// EnginePluginImpl.inl (container, handle, context) and nothing beyond them.
///
/// The fixture exists to pin one behaviour: CodegenFixtureContainer deliberately
/// does NOT declare the optional static getEngineName member, so the generated
/// hipdnnEnginePluginGetEngineName entry point takes its fallback branch and
/// reports HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE. That the target compiles at all
/// is the assertion that naming stayed optional for containers; the runtime half
/// is asserted from TestEnginePluginResourceManager.cpp, which loads this binary
/// by explicit path.
///
/// ⛔ Do not add getEngineName to CodegenFixtureContainer. Doing so silently
/// deletes the only in-tree coverage of the unnamed-container path.
///
/// The engine here computes nothing: it reports itself inapplicable to every
/// graph, so the backend never asks it for a plan.
namespace codegen_fixture
{

/// The single engine ID this fixture exposes.
///
/// Deliberately absent from the in-tree static engine name registry, whose IDs
/// are 64-bit FNV-1a hashes. Name resolution therefore falls all the way to
/// formatEngineIdHex, which renders this value as "0x000000000000C0DE".
inline constexpr int64_t K_FIXTURE_ENGINE_ID = 0xC0DE;

/// Execution settings.
///
/// Empty on purpose: the fixture has nothing to tune. The type still has to
/// exist because both EngineManager and ExecutionContextBase are parameterized
/// on a settings type.
struct CodegenFixtureSettings
{
};

struct CodegenFixtureHandle;

/// Execution context. The dual inheritance is required by the SDK: the first
/// base supplies opaque-pointer compatibility, the second the plan and settings
/// storage.
struct CodegenFixtureContext
    : HipdnnEnginePluginExecutionContext,
      hipdnn_plugin_sdk::ExecutionContextBase<CodegenFixtureHandle, CodegenFixtureSettings>
{
};

class CodegenFixtureContainer;

using CodegenFixtureEngineManager = hipdnn_plugin_sdk::
    EngineManager<CodegenFixtureHandle, CodegenFixtureSettings, CodegenFixtureContext>;

/// Plugin handle: stream, container, and detached engine-details buffers.
struct CodegenFixtureHandle : HipdnnEnginePluginHandle
{
    void setStream(hipStream_t stream)
    {
        _stream = stream;
    }

    hipStream_t getStream() const
    {
        return _stream;
    }

    std::shared_ptr<CodegenFixtureContainer> container;

    /// Defined out of line so this header does not need the container's
    /// definition, which in turn needs this handle's.
    CodegenFixtureEngineManager& getEngineManager() const;

    /// Keeps a serialized engine-details FlatBuffer alive until the backend
    /// releases it.
    void storeEngineDetailsDetachedBuffer(const void* ptr,
                                          std::unique_ptr<flatbuffers::DetachedBuffer> buffer)
    {
        _engineDetailsBuffers[ptr] = std::move(buffer);
    }

    void removeEngineDetailsDetachedBuffer(const void* ptr)
    {
        _engineDetailsBuffers.erase(ptr);
    }

private:
    hipStream_t _stream = nullptr;
    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>>
        _engineDetailsBuffers;
};

/// Plan stub. The fixture ships no kernels, so executing it is a caller error.
class CodegenFixturePlan : public hipdnn_plugin_sdk::IPlan<CodegenFixtureHandle>
{
public:
    size_t getWorkspaceSize(const CodegenFixtureHandle& handle) const override;

    void execute(const CodegenFixtureHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace) const override;
};

/// Engine stub. Applicable to no graph.
class CodegenFixtureEngine : public hipdnn_plugin_sdk::IEngine<CodegenFixtureHandle,
                                                               CodegenFixtureSettings,
                                                               CodegenFixtureContext>
{
public:
    int64_t id() const override;

    bool isApplicable(
        CodegenFixtureHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(CodegenFixtureHandle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t getMaxWorkspaceSize(const CodegenFixtureHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                   engineConfig) const override;

    void initializeExecutionContext(
        const CodegenFixtureHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        CodegenFixtureContext& executionContext) const override;
};

/// Container. Provides exactly the two members the container concept requires
/// and, pointedly, no getEngineName.
class CodegenFixtureContainer
{
public:
    CodegenFixtureContainer();

    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines);

    CodegenFixtureEngineManager& getEngineManager();

private:
    CodegenFixtureEngineManager _engineManager;
};

} // namespace codegen_fixture
