// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "ExampleProviderHandle.hpp"
#include "hip/IKernelCompiler.hpp"
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace example_provider
{

/// Type alias for engine pointers used in engine registration.
using ExampleProviderEnginePtr
    = std::unique_ptr<hipdnn_plugin_sdk::IEngine<ExampleProviderHandle,
                                                 ExampleProviderSettings,
                                                 ExampleProviderContext>>;

/// Container class that manages engine instantiation and ownership.
///
/// Creates DI dependencies (IKernelCompiler) at construction time
/// and passes them to engine factory functions.
class ExampleProviderContainer
{
public:
    ExampleProviderContainer();
    ~ExampleProviderContainer() noexcept;

    /// Copy engine IDs into a buffer.
    /// If maxEngines == 0: Does not copy, only queries total count.
    /// If maxEngines > 0: Copies up to maxEngines IDs into *engineIds, sets numEngines to number
    /// copied. Returns: Total number of available engines (regardless of maxEngines value).
    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines);

    /// Reports the canonical name of one engine.
    ///
    /// Optional member of the container concept: EnginePluginImpl.inl detects
    /// it and routes hipdnnEnginePluginGetEngineName here. A container that
    /// omits it still exports the entry point, which then reports
    /// HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE.
    ///
    /// The string written to *name must outlive the call; it points at the
    /// static storage created by HIPDNN_REGISTER_ENGINE.
    ///
    /// Returns HIPDNN_PLUGIN_STATUS_BAD_PARAM when engineId is not one of this
    /// plugin's engines.
    static hipdnnPluginStatus_t getEngineName(int64_t engineId, const char** name);

    hipdnn_plugin_sdk::
        EngineManager<ExampleProviderHandle, ExampleProviderSettings, ExampleProviderContext>&
        getEngineManager();

private:
    struct EngineDefinition
    {
        int64_t id;
        /// Points at the HIPDNN_REGISTER_ENGINE-generated _NAME constant, which
        /// has static storage duration and therefore satisfies the plugin API's
        /// lifetime requirement for the name it hands back to the host.
        const char* name;
        std::function<ExampleProviderEnginePtr(const IKernelCompiler&)> createEngine;
    };

    static const std::vector<EngineDefinition>& getEngineDefinitions();

    std::unique_ptr<IKernelCompiler> _kernelCompiler;

    std::unique_ptr<hipdnn_plugin_sdk::EngineManager<ExampleProviderHandle,
                                                     ExampleProviderSettings,
                                                     ExampleProviderContext>>
        _engineManager;
};

} // namespace example_provider
