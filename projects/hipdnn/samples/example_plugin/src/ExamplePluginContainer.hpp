// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "ExamplePluginHandle.hpp"
#include "hip/IDevicePropertyProvider.hpp"
#include "hip/IKernelCompiler.hpp"

namespace example_plugin
{

/// Container class that manages engine instantiation and ownership.
///
/// Creates DI dependencies (IKernelCompiler, IDevicePropertyProvider)
/// at construction time and passes them to engine factory functions.
class ExamplePluginContainer
{
public:
    ExamplePluginContainer();
    ~ExamplePluginContainer();

    /// Copy engine IDs into a buffer.
    /// If maxEngines == 0: Does not copy, only queries total count.
    /// If maxEngines > 0: Copies up to maxEngines IDs into *engineIds, sets numEngines to number
    /// copied. Returns: Total number of available engines (regardless of maxEngines value).
    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines);

    hipdnn_plugin_sdk::
        EngineManager<ExamplePluginHandle, ExamplePluginSettings, ExamplePluginContext>&
        getEngineManager();

private:
    struct EngineDefinition
    {
        int64_t id;
        std::function<std::unique_ptr<hipdnn_plugin_sdk::IEngine<ExamplePluginHandle,
                                                                 ExamplePluginSettings,
                                                                 ExamplePluginContext>>(
            const IKernelCompiler&, const IDevicePropertyProvider&)>
            createEngine;
    };

    static const std::vector<EngineDefinition>& getEngineDefinitions();

    std::unique_ptr<IKernelCompiler> _kernelCompiler;
    std::unique_ptr<IDevicePropertyProvider> _devicePropertyProvider;

    std::unique_ptr<hipdnn_plugin_sdk::EngineManager<ExamplePluginHandle,
                                                     ExamplePluginSettings,
                                                     ExamplePluginContext>>
        _engineManager;
};

} // namespace example_plugin
