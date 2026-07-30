// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EngineHeuristicDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "EngineConfigDescriptor.hpp"
#include "EngineDescriptor.hpp"
#include "GraphDescriptor.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "ScopedDescriptor.hpp"
#include "handle/Handle.hpp"
#include "utilities/EngineOrdering.hpp"

// Heuristics framework
#include "heuristics/DeviceProperties.hpp"
#include "heuristics/SelectionHeuristic.hpp"
#include "logging/Logging.hpp"
#include "plugin/HeuristicPlugin.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <unordered_map>

namespace hipdnn_backend
{

std::vector<int64_t> EngineHeuristicDescriptor::resolveHeuristicPolicyOrder()
{
    // Policy order resolution.
    // Priority: env > descriptor attr > default
    // Storage and ABI are policy IDs (FNV-1a of the policy name); names are
    // hashed at the point they enter the system. The Config built-in
    // (HIPDNN_HEUR_CONFIG_PATH JSON rules) is a regular policy in this
    // list, not a precursor; it declines when no rule matches so subsequent
    // policies still run. The explicit Graph.preferred_engine_id setter is
    // handled by the frontend as a post-hoc reorder of the heuristic-ranked
    // engine configs.

    // 1. Environment variable HIPDNN_HEUR_POLICY_ORDER (highest priority)
    // Use the data_sdk getEnv() wrapper rather than std::getenv() so that this
    // reads the live process environment block on Windows.
    //
    // Tokens may be either policy names ("SelectionHeuristic::Config") or raw
    // int64 policy IDs (decimal, optionally signed). A token parses as an ID
    // only when std::strtoll consumes the *entire* trimmed token; anything
    // else — including names that happen to start with digits — is hashed
    // through policyNameToId.
    const std::string envStr = hipdnn_data_sdk::utilities::getEnv("HIPDNN_HEUR_POLICY_ORDER");
    if(!envStr.empty())
    {
        std::vector<int64_t> policyIds;
        std::istringstream iss(envStr);
        std::string token;
        while(std::getline(iss, token, ','))
        {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            if(token.empty())
            {
                continue;
            }

            char* end = nullptr;
            errno = 0;
            const int64_t asId = std::strtoll(token.c_str(), &end, 10);
            const bool fullyParsed = (end != nullptr) && (*end == '\0') && (errno == 0);
            policyIds.push_back(fullyParsed ? asId
                                            : hipdnn_data_sdk::utilities::policyNameToId(token));
        }
        HIPDNN_BACKEND_LOG_WARN("Using environment variable policy order: {} policies",
                                policyIds.size());
        return policyIds;
    }
    // 2. Descriptor attribute
    if(_policyOrderSet)
    {
        HIPDNN_BACKEND_LOG_DEBUG("Using descriptor-level policy order: {} policies",
                                 _policyOrder.size());
        return _policyOrder;
    }
    // 3. Default policy list — Config first so HIPDNN_HEUR_CONFIG_PATH
    // rules win when set; StaticOrdering is the canonical last-resort fallback
    // and always succeeds when there is at least one candidate. Vendor
    // heuristic plugins may be inserted via env or descriptor attribute above.
    std::vector<int64_t> policyIds = {
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::Config"),
        hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering"),
    };
    HIPDNN_BACKEND_LOG_WARN(
        "No heuristic policy order configured, falling back to built-in defaults "
        "[SelectionHeuristic::Config, SelectionHeuristic::StaticOrdering]. "
        "Set HIPDNN_HEUR_POLICY_ORDER or the descriptor attribute to silence "
        "this warning.");
    return policyIds;
}

void EngineHeuristicDescriptor::syncPolicySlots(const std::vector<int64_t>& orderedPolicyIds)
{
    // Ensure one SelectionHeuristic per policy slot.
    // If the policy list changed, recreate the slots.

    if(_orderedPolicyIds == orderedPolicyIds && !_policySlots.empty())
    {
        // Policy list unchanged and slots already created
        return;
    }

    _orderedPolicyIds = orderedPolicyIds;
    _policySlots.clear();

    auto handle = _graph->getHandle();
    auto heurRm = handle->getHeuristicPluginResourceManager();

    // Create one SelectionHeuristic per policy slot. The slot holds a
    // shared_ptr to the resource manager so the underlying plugin and handle
    // cannot be destroyed while the slot is alive; lookups happen by policy
    // ID inside SelectionHeuristic.
    for(const int64_t policyId : orderedPolicyIds)
    {
        if(heurRm->getHeuristicHandleForPolicyId(policyId) == nullptr)
        {
            // Policy not loaded - add null placeholder
            _policySlots.push_back(nullptr);
            continue;
        }

        // SelectionHeuristic's constructor calls into the plugin to create the
        // policy descriptor, which can throw. Treat a failed slot the same way
        // we treat a not-loaded policy: log and insert a null placeholder so
        // the policy loop in finalize() simply skips it via its existing
        // nullptr branch instead of aborting the whole descriptor.
        // HipdnnException derives from std::exception, so one catch covers both.
        try
        {
            _policySlots.push_back(
                std::make_unique<heuristics::SelectionHeuristic>(heurRm, policyId));
            continue;
        }
        catch(const std::exception& e)
        {
            HIPDNN_BACKEND_LOG_WARN("Failed to construct SelectionHeuristic for policy ID {}: {}. "
                                    "Slot will be skipped during finalize().",
                                    policyId,
                                    e.what());
        }
        catch(...)
        {
            HIPDNN_BACKEND_LOG_WARN(
                "Failed to construct SelectionHeuristic for policy ID {} (unknown exception). "
                "Slot will be skipped during finalize().",
                policyId);
        }
        _policySlots.push_back(nullptr);
    }
}

void EngineHeuristicDescriptor::finalize()
{
    // Outer loop policy selection
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineHeuristicDescriptor::finalize() failed: Already finalized.");

    THROW_IF_NULL(_graph,
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineHeuristicDescriptor::finalize() failed: Graph is not set.");

    THROW_IF_FALSE(_heuristicModeSet,
                   HIPDNN_STATUS_BAD_PARAM,
                   "EngineHeuristicDescriptor::finalize() failed: Heuristic mode is not set.");

    auto handle = _graph->getHandle();
    auto engineRm = handle->getPluginResourceManager();
    auto heurRm = handle->getHeuristicPluginResourceManager();

    // Get candidate engine IDs from engine plugins
    auto candidates = engineRm->getApplicableEngineIds(_graph.get(), _findFirst);

    // If no engines available, finalize with empty result (no need to invoke heuristics).
    // This is a valid state - not an error.
    if(candidates.empty())
    {
        _engineIds.clear();
        HipdnnBackendDescriptorImpl<EngineHeuristicDescriptor>::finalize();
        return;
    }

    // findFirst is a fast applicability probe (Graph::is_supported_ext) — the
    // caller only needs to know whether *any* engine can run the graph.
    if(_findFirst)
    {
        _engineIds = std::move(candidates);
        HipdnnBackendDescriptorImpl<EngineHeuristicDescriptor>::finalize();
        return;
    }

    // devicePropsSerialized must outlive devicePropsWrapper — the wrapper aliases its storage.
    const auto devProps = heuristics::queryDeviceProperties(handle);
    const auto devicePropsSerialized = heuristics::serializeDeviceProperties(devProps);
    const hipdnnPluginConstData_t devicePropsWrapper
        = heuristics::wrapSerializedDeviceProperties(devicePropsSerialized);

    // Get serialized graph from GraphDescriptor
    const hipdnnPluginConstData_t serializedGraph = _graph->getSerializedGraph();

    // Resolve ordered policy IDs
    auto orderedPolicyIds = resolveHeuristicPolicyOrder();

    // Ensure policy slots match the ordered policy list
    syncPolicySlots(orderedPolicyIds);

    // Set device properties on all distinct plugin handles (once per handle, not per slot)
    // NOTE: Multiple policies may share the same plugin handle if they come from the same
    // .so file (e.g., a single plugin providing multiple ordering strategies like Fast/Balanced/Accurate).
    // Use a map to deduplicate and call setDeviceProperties only once per unique handle.
    std::unordered_map<hipdnnHeuristicHandle_t, const plugin::HeuristicPlugin*> distinctHandles;
    for(const int64_t policyId : orderedPolicyIds)
    {
        auto pluginHandle = heurRm->getHeuristicHandleForPolicyId(policyId);
        if(pluginHandle != nullptr)
        {
            auto plugin = heurRm->getPluginForPolicyId(policyId);
            if(plugin != nullptr)
            {
                distinctHandles[pluginHandle] = plugin;
            }
        }
    }

    // Call SetDeviceProperties on each distinct handle.
    // Sort by handle pointer so the call order is stable *within this process
    // run* — std::unordered_map iteration order is otherwise unspecified, which
    // would scramble the order of any per-handle log lines emitted below and
    // make the fail-soft disable order non-reproducible from one finalize() to
    // the next on the same descriptor. Pointers vary across runs (ASLR), so
    // this is reproducible-per-run, not reproducible-across-runs.
    std::vector<std::pair<hipdnnHeuristicHandle_t, const plugin::HeuristicPlugin*>> sortedHandles(
        distinctHandles.begin(), distinctHandles.end());
    std::sort(sortedHandles.begin(), sortedHandles.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Mirror the policy loop's fail-soft contract below: a single plugin's
    // setDeviceProperties failure must not break the chain. Disable every slot
    // backed by a failed plugin handle so the policy loop skips it via the
    // existing nullptr branch.
    auto disableSlotsForHandle = [&](hipdnnHeuristicHandle_t failedHandle) {
        for(size_t i = 0; i < _policySlots.size(); ++i)
        {
            if(_policySlots[i] != nullptr
               && heurRm->getHeuristicHandleForPolicyId(_orderedPolicyIds[i]) == failedHandle)
            {
                _policySlots[i].reset();
            }
        }
    };

    for(const auto& [pluginHandle, plugin] : sortedHandles)
    {
        try
        {
            plugin->setDeviceProperties(pluginHandle, &devicePropsWrapper);
        }
        catch(const std::exception& e)
        {
            HIPDNN_BACKEND_LOG_WARN("setDeviceProperties failed for heuristic plugin '{}': {}. "
                                    "Disabling all policies provided by this plugin.",
                                    plugin->name(),
                                    e.what());
            disableSlotsForHandle(pluginHandle);
        }
        catch(...)
        {
            HIPDNN_BACKEND_LOG_WARN("setDeviceProperties threw unknown exception for heuristic "
                                    "plugin '{}'. Disabling all policies provided by this plugin.",
                                    plugin->name());
            disableSlotsForHandle(pluginHandle);
        }
    }

    // Outer policy loop: try each policy in order until one succeeds
    bool success = false;
    for(size_t i = 0; i < _policySlots.size(); ++i)
    {
        auto& selection = _policySlots[i];
        if(selection == nullptr)
        {
            // Policy plugin not loaded - continue to next policy
            continue;
        }

        try
        {
            // Set candidate engine IDs and serialized graph on the policy descriptor
            selection->setEngineIds(candidates);
            selection->setSerializedGraph(&serializedGraph);

            // Call finalize on this policy
            if(!selection->finalize())
            {
                // Policy declined or not applicable - continue to next policy
                continue;
            }

            // Policy succeeded! Get the sorted engine IDs
            candidates = selection->getSortedEngineIds();
            success = true;
            break;
        }
        catch(const HipdnnException& e)
        {
            // Policy threw an exception - log and continue to next policy
            HIPDNN_BACKEND_LOG_WARN("Heuristic policy at slot {} (ID {}) threw exception: {}. "
                                    "Continuing to next policy.",
                                    i,
                                    _orderedPolicyIds[i],
                                    e.what());
            continue;
        }
        catch(const std::exception& e)
        {
            // Plugin code is external and may throw any std-derived exception type.
            // Treat the same as HipdnnException: log and continue.
            HIPDNN_BACKEND_LOG_WARN("Heuristic policy at slot {} (ID {}) threw exception: {}. "
                                    "Continuing to next policy.",
                                    i,
                                    _orderedPolicyIds[i],
                                    e.what());
            continue;
        }
        catch(...)
        {
            // Plugin may throw a non-std-derived exception; never let it cross the C ABI.
            HIPDNN_BACKEND_LOG_WARN("Heuristic policy at slot {} (ID {}) threw unknown exception. "
                                    "Continuing to next policy.",
                                    i,
                                    _orderedPolicyIds[i]);
            continue;
        }
    }

    // If no policy succeeded, throw exception.
    // No hidden fallback to utilities::sortEngineIds.
    if(!success)
    {
        throw HipdnnException(
            HIPDNN_STATUS_INTERNAL_ERROR,
            "EngineHeuristicDescriptor::finalize() failed: No heuristic policy succeeded.");
    }

    _engineIds = candidates;

    HipdnnBackendDescriptorImpl<EngineHeuristicDescriptor>::finalize();
}

void EngineHeuristicDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                             hipdnnBackendAttributeType_t attributeType,
                                             int64_t requestedElementCount,
                                             int64_t* elementCount,
                                             void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "EngineHeuristicDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
        getGraph(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_MODE:
        getHeuristicMode(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_RESULTS:
        getEngineConfigs(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT:
        getFindFirst(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT:
        getPolicyOrder(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineHeuristicDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void EngineHeuristicDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                             hipdnnBackendAttributeType_t attributeType,
                                             int64_t elementCount,
                                             const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "EngineHeuristicDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINEHEUR_MODE:
        setHeuristicMode(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
        setGraph(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT:
        setFindFirst(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT:
        setPolicyOrder(attributeType, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineHeuristicDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void EngineHeuristicDescriptor::setHeuristicMode(hipdnnBackendAttributeType_t attributeType,
                                                 int64_t elementCount,
                                                 const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_HEUR_MODE,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set heuristic mode: Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set heuristic mode: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to set heuristic mode: Null pointer.");

    auto heurMode = static_cast<const hipdnnBackendHeurMode_t*>(arrayOfElements);
    THROW_IF_NULL(
        heurMode,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "EngineHeuristicDescriptor failed to set heuristic mode: Heuristic mode is null.");

    auto heurModeValue = *heurMode;
    if(heurModeValue != HIPDNN_HEUR_MODE_FALLBACK)
    {
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "EngineHeuristicDescriptor::setHeuristicMode() is not "
                              "supported for the given heuristic mode.");
    }

    _heuristicMode = heurModeValue;
    _heuristicModeSet = true;
}

void EngineHeuristicDescriptor::setGraph(hipdnnBackendAttributeType_t attributeType,
                                         int64_t elementCount,
                                         const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set graph: Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set graph: Invalid element count.");

    auto graph = HipdnnBackendDescriptor::unpackDescriptor<const GraphDescriptor>(
        arrayOfElements,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "EngineHeuristicDescriptor failed to set graph: Null pointer.");

    THROW_IF_FALSE(graph->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "EngineHeuristicDescriptor failed to set graph: Graph is not finalized.");

    _graph = graph;
}

void EngineHeuristicDescriptor::getGraph(hipdnnBackendAttributeType_t attributeType,
                                         int64_t requestedElementCount,
                                         int64_t* elementCount,
                                         void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get graph: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get graph: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to get graph: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    HipdnnBackendDescriptor::packDescriptor(_graph, arrayOfElements);
}

void EngineHeuristicDescriptor::getEngineConfigs(hipdnnBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get engine configs: Invalid attribute type.");

    // Return the number of engine configs if they aren't requesting any.
    if(requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineHeuristicDescriptor failed to get engine config count: Null pointer "
                      "for element count.");
        *elementCount = static_cast<int64_t>(_engineIds.size());
    }
    else
    {
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineHeuristicDescriptor failed to get engine configs: Null pointer.");

        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineHeuristicDescriptor failed to get engine config: Null pointer for "
                      "element count.");

        // Create engine config descriptors for each engine ID
        auto outputArray = static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements);
        for(size_t i = 0; i < _engineIds.size() && i < static_cast<size_t>(requestedElementCount);
            ++i)
        {
            auto config = HipdnnBackendDescriptor::unpackDescriptor<EngineConfigDescriptor>(
                outputArray[i],
                HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                "EngineHeuristicDescriptor failed to get engine config: Config "
                "descriptor is null.");

            auto engine = std::make_shared<EngineDescriptor>();

            engine->setAttribute(
                HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &_engineIds[i]);

            ScopedDescriptor graphDesc(HipdnnBackendDescriptor::packDescriptor(_graph));
            engine->setAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                 1,
                                 static_cast<const void*>(graphDesc.getPtr()));
            engine->finalize();

            ScopedDescriptor engineDesc(HipdnnBackendDescriptor::packDescriptor(engine));
            config->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                 1,
                                 static_cast<const void*>(engineDesc.getPtr()));
        }

        *elementCount = std::min(requestedElementCount, static_cast<int64_t>(_engineIds.size()));
    }
}

void EngineHeuristicDescriptor::getHeuristicMode(hipdnnBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_HEUR_MODE,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get heuristic mode: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get heuristic mode: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to get heuristic mode: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    auto heurModeOut = static_cast<hipdnnBackendHeurMode_t*>(arrayOfElements);
    *heurModeOut = _heuristicMode;
}

void EngineHeuristicDescriptor::setFindFirst(hipdnnBackendAttributeType_t attributeType,
                                             int64_t elementCount,
                                             const void* arrayOfElements)
{
    setScalar(_findFirst,
              HIPDNN_TYPE_BOOLEAN,
              attributeType,
              elementCount,
              arrayOfElements,
              "EngineHeuristicDescriptor::setAttribute()");
}

void EngineHeuristicDescriptor::getFindFirst(hipdnnBackendAttributeType_t attributeType,
                                             int64_t requestedElementCount,
                                             int64_t* elementCount,
                                             void* arrayOfElements) const
{
    getScalar(_findFirst,
              HIPDNN_TYPE_BOOLEAN,
              attributeType,
              requestedElementCount,
              elementCount,
              arrayOfElements,
              "EngineHeuristicDescriptor::getAttribute()");
}

std::shared_ptr<const GraphDescriptor> EngineHeuristicDescriptor::getGraph() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "EngineHeuristicDescriptor::getGraph() failed: Not finalized.");

    return _graph;
}

hipdnnBackendDescriptorType_t EngineHeuristicDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR;
}

void EngineHeuristicDescriptor::setPolicyOrder(hipdnnBackendAttributeType_t attributeType,
                                               int64_t elementCount,
                                               const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_INT64,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set policy order: Invalid attribute type.");

    THROW_IF_TRUE(elementCount < 0,
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineHeuristicDescriptor failed to set policy order: Negative element count.");

    THROW_IF_TRUE(elementCount > 0 && arrayOfElements == nullptr,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to set policy order: Null pointer.");

    if(elementCount == 0)
    {
        _policyOrder.clear();
        _policyOrderSet = true;
        HIPDNN_BACKEND_LOG_DEBUG("Set descriptor-level policy order: 0 policies");
        return;
    }

    const auto* data = static_cast<const int64_t*>(arrayOfElements);
    _policyOrder.assign(data, data + elementCount);
    _policyOrderSet = true;
    HIPDNN_BACKEND_LOG_DEBUG("Set descriptor-level policy order: {} policies", _policyOrder.size());
}

void EngineHeuristicDescriptor::getPolicyOrder(hipdnnBackendAttributeType_t attributeType,
                                               int64_t requestedElementCount,
                                               int64_t* elementCount,
                                               void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_INT64,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get policy order: Invalid attribute type.");

    THROW_IF_NULL(elementCount,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to get policy order: Null pointer for "
                  "element count.");

    THROW_IF_TRUE(requestedElementCount < 0,
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineHeuristicDescriptor failed to get policy order: Negative requested "
                  "element count.");

    THROW_IF_TRUE(requestedElementCount > 0 && arrayOfElements == nullptr,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to get policy order: Null pointer.");

    // The dispatcher requires isFinalized() before reaching here, so
    // _orderedPolicyIds reflects the resolved order (env > descriptor > default)
    // actually used during finalize(). See resolveHeuristicPolicyOrder().
    if(requestedElementCount == 0)
    {
        *elementCount = static_cast<int64_t>(_orderedPolicyIds.size());
        return;
    }

    auto* output = static_cast<int64_t*>(arrayOfElements);
    const auto count
        = std::min(static_cast<size_t>(requestedElementCount), _orderedPolicyIds.size());
    std::memcpy(output, _orderedPolicyIds.data(), count * sizeof(int64_t));
    *elementCount = static_cast<int64_t>(count);
}

std::string EngineHeuristicDescriptor::toString() const
{
    std::string str = "EngineHeuristicDescriptor: {heuristicMode=";
    str += _heuristicModeSet ? std::to_string(_heuristicMode) : "unset";
    str += _graph ? ", graph=" + fmt::format("{:p}", static_cast<const void*>(_graph.get()))
                  : ", graph=null";
    if(_policyOrderSet)
    {
        str += ", policyOrder=[";
        for(size_t i = 0; i < _policyOrder.size(); ++i)
        {
            if(i > 0)
            {
                str += ", ";
            }
            str += hipdnn_data_sdk::utilities::formatEngineIdHex(_policyOrder[i]);
        }
        str += ']';
    }
    str += '}';
    return str;
}

} // namespace hipdnn_backend
