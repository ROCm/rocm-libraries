// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

namespace miopen_legacy_plugin
{

// ============================================================================
// Engine Registration
// ============================================================================
// For plugins that are not yet globally registered, use HIPDNN_REGISTER_ENGINE
// to register your engine names here. This will:
// 1. Create _NAME and _ID constants for the engine
// 2. Detect hash collisions with other formally-registered engines
//
// Example for new engines:
// HIPDNN_REGISTER_ENGINE(MY_CUSTOM_ENGINE, "MY_CUSTOM_ENGINE")
// HIPDNN_REGISTER_ENGINE(MY_OTHER_ENGINE, "MY_OTHER_ENGINE")
//
// Note: MIOPEN_ENGINE is already registered in EngineNames.hpp via
// HIPDNN_REGISTER_ENGINE(MIOPEN_ENGINE, "MIOPEN_ENGINE"), so we can use
// the MIOPEN_ENGINE_NAME and MIOPEN_ENGINE_ID constants directly from there.
// ============================================================================

class EngineManager;
class IEngine;

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
    // Always sets *numEngines to the total number of available engines.
    // If maxEngines > 0, copies up to maxEngines IDs into *engineIds.
    static void copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines);

    EngineManager& getEngineManager();

private:
    struct EngineDefinition
    {
        int64_t id; // Set id using EngineNames.hpp.
        std::function<std::unique_ptr<IEngine>()> createEngine;
    };

    static const std::vector<EngineDefinition>& getEngineDefinitions();

    std::unique_ptr<EngineManager> _engineManager;
};

}
