// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "HipdnnMiopenHandle.hpp"

namespace miopen_plugin
{

/// @brief Each MIOpen engine's declared L1 throughput model: architecture to UHD UUID.
///
/// RFC 0019 Open Question 7 (RESOLVED): MIOpen ships no UED, so it binds a
/// `predict_engine_tflops` model by naming that UHD's UUID in the provider's own engine
/// definition rather than through a UED role map. Defined in MiopenContainer.cpp beside
/// the engine table that consumes them, and declared here because the declaration is part
/// of what this provider promises: a deployer installs a model by publishing a UHD that
/// carries one of these ids.
extern const std::map<std::string, std::string> MIOPEN_ENGINE_L1_MODELS;
extern const std::map<std::string, std::string> MIOPEN_ENGINE_DETERMINISTIC_L1_MODELS;

/*
 * Container class to manage the intantiation and ownership of all MIOpen plan builders and engines.
 * The class designs use dependency injection to get the components they need in order to function.
 * This makes it easier to test and maintain the code as you can swap out implementations.
 *
 * The construction sequence should contain no logic other than the creation of various classes.
 * If logic is needed, it should be placed in a separate function that can be called after the
 * container has finished constructing all its components.
 */
class MiopenContainer
{
public:
    MiopenContainer();
    ~MiopenContainer();

    // Copy engine IDs into a buffer.
    // If maxEngines == 0: Does not copy, only queries total count.
    // If maxEngines > 0: Copies up to maxEngines IDs into *engineIds, sets numEngines to number copied.
    // Returns: Total number of available engines (regardless of maxEngines value).
    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines);

    hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>&
        getEngineManager();

private:
    struct EngineDefinition
    {
        int64_t id; // Set id using EngineNames.hpp.
        std::function<std::unique_ptr<hipdnn_plugin_sdk::IEngine<HipdnnMiopenHandle,
                                                                 HipdnnMiopenSettings,
                                                                 HipdnnMiopenContext>>()>
            createEngine;
    };

    static const std::vector<EngineDefinition>& getEngineDefinitions();

    std::unique_ptr<hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle,
                                                     HipdnnMiopenSettings,
                                                     HipdnnMiopenContext>>
        _engineManager;
};

}
