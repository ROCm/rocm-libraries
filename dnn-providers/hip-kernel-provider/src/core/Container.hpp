// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "Handle.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "device/IDevicePropertyProvider.hpp"

namespace hip_kernel_provider::core
{

/*
 * Container class to manage the instantiation and ownership of all HIP kernel plan builders
 * and engines. The class designs use dependency injection to get the components they need
 * in order to function.
 *
 * The construction sequence should contain no logic other than the creation of various classes.
 * If logic is needed, it should be placed in a separate function that can be called after the
 * container has finished constructing all its components.
 */
class Container
{
public:
    Container();
    ~Container();

    // Copy engine IDs into a buffer.
    // If maxEngines == 0: Does not copy, only queries total count.
    // If maxEngines > 0: Copies up to maxEngines IDs into *engineIds, sets numEngines to number
    // copied. Returns: Total number of available engines (regardless of maxEngines value).
    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines);

    // The plugin SDK detects this member by callability and exports
    // hipdnnEnginePluginGetEngineName on its behalf; a signature it cannot call reads as
    // opting out and every engine falls back to a hexadecimal rendering of its ID.
    // Returns: BAD_PARAM if name is null, NOT_APPLICABLE for an unrecognized engineId, and
    // SUCCESS with *name pointing at a NUL-terminated name valid for the library's lifetime.
    static hipdnnPluginStatus_t getEngineName(int64_t engineId, const char** name);

    hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>& getEngineManager();

private:
    struct EngineDefinition
    {
        int64_t id;
        std::string name;
        std::function<std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>>(
            const device::IDevicePropertyProvider&)>
            createEngine;
    };

    static const std::vector<EngineDefinition>& getEngineDefinitions();

    std::unique_ptr<device::IDevicePropertyProvider> _devicePropertyProvider;
    std::unique_ptr<compilation::IKernelCompiler> _kernelCompiler;
    std::unique_ptr<hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>> _engineManager;
};

} // namespace hip_kernel_provider::core
