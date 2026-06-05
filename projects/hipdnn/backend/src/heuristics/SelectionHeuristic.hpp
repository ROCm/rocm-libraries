// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hipdnn_backend::plugin
{
class HeuristicPlugin;
class HeuristicPluginResourceManager;
} // namespace hipdnn_backend::plugin

namespace hipdnn_backend::heuristics
{

/**
 * @brief C++ facade for one policy slot on EngineHeuristicDescriptor.
 *
 * This class wraps a hipdnnHeuristicPolicyDescriptor_t created with a
 * hipdnnHeuristicHandle_t for that policy's module. It provides a clean
 * C++ interface over the heuristic plugin C ABI.
 *
 * Session state (caches, tuning data, etc.) lives in the plugin behind the
 * handle, not in this wrapper.
 *
 * Device properties are NOT set on this facade. The host calls
 * hipdnnHeuristicHandleSetDeviceProperties on the handle BEFORE calling
 * finalize() on any descriptor created with that handle.
 *
 * Lifecycle: Owned by EngineHeuristicDescriptor, one per resolved policy slot.
 * Created when the policy list is established, destroyed with the descriptor.
 *
 * Ownership: Holds a shared_ptr to the HeuristicPluginResourceManager and a
 * policy ID rather than a raw HeuristicPlugin pointer. The shared_ptr keeps
 * the manager (and transitively the loaded plugin) alive for the descriptor's
 * lifetime; plugin and handle lookups happen by policy ID through the
 * manager. This protects against the manager being destroyed while a slot
 * still holds the descriptor.
 */
class SelectionHeuristic
{
public:
    /**
     * @brief Constructs a SelectionHeuristic for a given policy.
     *
     * Resolves the HeuristicPlugin and hipdnnHeuristicHandle_t through the
     * resource manager using the policy ID, and creates the underlying
     * hipdnnHeuristicPolicyDescriptor_t.
     *
     * @param resourceManager Shared pointer to the resource manager that owns
     *                        the plugin handles. Kept alive for the lifetime
     *                        of this object so the resolved plugin/handle
     *                        cannot dangle.
     * @param policyId Stable int64_t policy ID identifying the policy this
     *                 slot represents (policyNameToId hash).
     */
    SelectionHeuristic(std::shared_ptr<plugin::HeuristicPluginResourceManager> resourceManager,
                       int64_t policyId);

    /**
     * @brief Destroys the SelectionHeuristic and releases the policy descriptor.
     *
     * Calls hipdnnHeuristicPolicyDescriptorDestroy on the underlying descriptor.
     */
    ~SelectionHeuristic();

    // Prevent copying
    SelectionHeuristic(const SelectionHeuristic&) = delete;
    SelectionHeuristic& operator=(const SelectionHeuristic&) = delete;

    // Allow moving
    SelectionHeuristic(SelectionHeuristic&& other) noexcept;
    SelectionHeuristic& operator=(SelectionHeuristic&& other) noexcept;

    /**
     * @brief Sets the candidate engine IDs for this policy.
     *
     * Provides the list of candidate engine IDs from
     * EnginePluginResourceManager::getApplicableEngineIds.
     * The plugin must produce a reordered subset or permutation of these IDs.
     *
     * Mirrors hipdnnHeuristicPolicySetEngineIds (§8.8).
     *
     * @param engineIds Vector of candidate engine IDs.
     */
    void setEngineIds(const std::vector<int64_t>& engineIds);

    /**
     * @brief Sets the serialized operation graph for this policy.
     *
     * Provides the FlatBuffer-serialized operation graph from
     * GraphDescriptor::getSerializedGraph().
     *
     * Mirrors hipdnnHeuristicPolicySetSerializedGraph (§8.8).
     *
     * @param serializedGraph Pointer to hipdnnPluginConstData_t containing
     *                        the serialized graph buffer.
     */
    void setSerializedGraph(const hipdnnPluginConstData_t* serializedGraph);

    /**
     * @brief Executes the policy selection logic.
     *
     * Performs applicability checking and engine ordering based on the inputs
     * previously set via setEngineIds and setSerializedGraph.
     *
     * Device properties are queried from the bound hipdnnHeuristicHandle_t
     * (which received SetDeviceProperties earlier in the finalize() flow).
     *
     * Two-phase design: This function performs the selection work;
     * getSortedEngineIds retrieves the result.
     *
     * Mirrors hipdnnHeuristicPolicyFinalize (§8.9).
     *
     * @return true if policy succeeded (policy won the outer loop),
     *         false if not applicable or declined (host continues outer loop).
     */
    bool finalize();

    /**
     * @brief Retrieves the sorted engine IDs after successful finalize.
     *
     * Valid only after finalize() returned true. Returns the reordered
     * engine IDs produced by the policy.
     *
     * The output IDs are a permutation or subset of the input IDs from
     * setEngineIds. The host validates this constraint.
     *
     * Mirrors hipdnnHeuristicPolicyGetSortedEngineIds (§8.9).
     *
     * @return Vector of sorted engine IDs.
     */
    std::vector<int64_t> getSortedEngineIds();

private:
    // Look up the plugin for _policyId via _resourceManager. Returns nullptr
    // if the policy is no longer registered (e.g. plugin was removed). All
    // operations route through the resource manager so the plugin lookup
    // stays consistent with the manager's current state.
    const plugin::HeuristicPlugin* lookupPlugin() const;

    std::shared_ptr<plugin::HeuristicPluginResourceManager> _resourceManager;
    int64_t _policyId = 0;
    hipdnnHeuristicPolicyDescriptor_t _descriptor = nullptr;
    // Cached candidate engine IDs from the most recent setEngineIds call.
    // Used in getSortedEngineIds to validate that the plugin returned a
    // permutation or subset of the provided IDs (no duplicates, no fabrications).
    std::vector<int64_t> _inputEngineIds;
};

} // namespace hipdnn_backend::heuristics
