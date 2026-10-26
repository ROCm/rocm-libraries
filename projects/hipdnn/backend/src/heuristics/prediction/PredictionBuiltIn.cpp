// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PredictionBuiltIn.hpp"
#include "PredictionPolicy.hpp"

namespace hipdnn_backend::heuristics::prediction
{
hipdnn_backend::plugin::HeuristicPluginFunctionTable populateFunctionTable()
{
    hipdnn_backend::plugin::HeuristicPluginFunctionTable table;
    table.getName = getName;
    table.getVersion = getVersion;
    table.getApiVersion = getApiVersion;
    table.getType = getType;
    table.setLoggingCallback = setLoggingCallback;
    table.setLogLevel = setLogLevel;
    table.getLastErrorString = getLastErrorString;
    table.getAllPolicyIds = getAllPolicyIds;
    table.getPolicyName = getPolicyName;
    table.handleCreate = handleCreate;
    table.handleDestroy = handleDestroy;
    table.handleSetDeviceProperties = handleSetDeviceProperties;
    table.policyDescriptorCreate = policyDescriptorCreate;
    table.policyDescriptorDestroy = policyDescriptorDestroy;
    table.policySetEngineIds = policySetEngineIds;
    table.policySetSerializedGraph = policySetSerializedGraph;
    table.policyFinalize = policyFinalize;
    table.policyFinalizeWithHost = policyFinalizeWithHost;
    table.policyGetSortedEngineIds = policyGetSortedEngineIds;
    table.policyGetEngineConfig = policyGetEngineConfig;
    return table;
}
} // namespace hipdnn_backend::heuristics::prediction
