// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "HipKernelHandle.hpp"
#include "IDevicePropertyProvider.hpp"
#include "hip/IKernelCompiler.hpp"

namespace hip_kernel_provider
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
class HipKernelContainer
{
public:
    HipKernelContainer();
    ~HipKernelContainer();

    // Copy engine IDs into a buffer.
    // If maxEngines == 0: Does not copy, only queries total count.
    // If maxEngines > 0: Copies up to maxEngines IDs into *engineIds, sets numEngines to number
    // copied. Returns: Total number of available engines (regardless of maxEngines value).
    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines);

    hipdnn_plugin_sdk::EngineManager<HipKernelHandle, HipKernelSettings, HipKernelContext>&
        getEngineManager();

private:
    struct EngineDefinition
    {
        int64_t id;
        std::function<std::unique_ptr<
            hipdnn_plugin_sdk::IEngine<HipKernelHandle, HipKernelSettings, HipKernelContext>>(
            const IKernelCompiler&, const IDevicePropertyProvider&)>
            createEngine;
    };

    static const std::vector<EngineDefinition>& getEngineDefinitions();

    std::unique_ptr<IDevicePropertyProvider> _devicePropertyProvider;
    std::unique_ptr<IKernelCompiler> _kernelCompiler;
    std::unique_ptr<
        hipdnn_plugin_sdk::EngineManager<HipKernelHandle, HipKernelSettings, HipKernelContext>>
        _engineManager;
};

} // namespace hip_kernel_provider
