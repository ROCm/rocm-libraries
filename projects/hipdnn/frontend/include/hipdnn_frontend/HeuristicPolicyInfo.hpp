// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file HeuristicPolicyInfo.hpp
 * @brief Frontend helpers for querying loaded heuristic policies (RFC 0007 Section 16)
 *
 * This file provides frontend utilities to enumerate and inspect heuristic
 * policy plugins loaded by the backend. These policies determine the engine
 * selection order during graph compilation.
 *
 * @code{.cpp}
 * auto [handle, err] = hipdnn_frontend::createHipdnnHandle();
 * auto [infos, err2] = hipdnn_frontend::getLoadedHeuristicPolicyInfos(*handle);
 * for (const auto& info : infos) {
 *     std::cout << "Policy: " << info.policyName
 *               << " (ID: " << info.policyId << ")" << std::endl;
 * }
 * @endcode
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Handle.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

namespace hipdnn_frontend
{

/**
 * @struct HeuristicPolicyInfo
 * @brief Metadata for a loaded heuristic policy plugin
 *
 * Contains identifying information for one heuristic policy that has been
 * successfully loaded and validated by the backend.
 */
struct HeuristicPolicyInfo
{
    /// Policy ID (int64_t hash of policy name using policyNameToId)
    int64_t policyId = -1;

    /// Canonical policy name (e.g., "SelectionHeuristic::Config")
    std::string policyName;

    /// Plugin name reported by the heuristic plugin
    std::string pluginName;

    /// Plugin implementation version string
    std::string pluginVersion;

    /// Heuristic C ABI version string
    std::string apiVersion;
};

/**
 * @brief Get information about all loaded heuristic policy plugins
 *
 * RFC 0007 Section 16: Queries the backend for metadata about all heuristic
 * policy plugins that have been loaded and validated. The returned policies
 * are available for use in the outer loop engine selection.
 *
 * This is useful for diagnostics, logging, and understanding which policies
 * are influencing engine selection order.
 *
 * @param handle The hipDNN handle to query
 * @return Pair of (policy info vector, error); vector is empty on failure
 *
 * @code{.cpp}
 * auto [handle, err] = createHipdnnHandle();
 * auto [policies, err2] = getLoadedHeuristicPolicyInfos(*handle);
 * if (err2.is_bad()) {
 *     std::cerr << "Failed to query policies: " << err2.what() << std::endl;
 * } else {
 *     std::cout << "Loaded " << policies.size() << " heuristic policies" << std::endl;
 *     for (const auto& policy : policies) {
 *         std::cout << "  - " << policy.policyName
 *                   << " (v" << policy.pluginVersion << ")" << std::endl;
 *     }
 * }
 * @endcode
 */
inline std::pair<std::vector<HeuristicPolicyInfo>, Error>
    getLoadedHeuristicPolicyInfos(hipdnnHandle_t handle)
{
    if(handle == nullptr)
    {
        return {{}, {ErrorCode::INVALID_VALUE, "Cannot query policies from null handle"}};
    }

    auto backend = detail::hipdnnBackend();

    using hipdnn_data_sdk::utilities::bufferToString;

    // Get policy count
    size_t numPolicies = 0;
    auto status = backend->getHeuristicPolicyCount(handle, &numPolicies);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        return {{},
                {ErrorCode::HIPDNN_BACKEND_ERROR,
                 std::string{"Failed to get heuristic policy count: "}
                     + backend->getErrorString(status)}};
    }

    std::vector<HeuristicPolicyInfo> infos;
    infos.reserve(numPolicies);

    // Query each policy
    for(size_t i = 0; i < numPolicies; ++i)
    {
        HeuristicPolicyInfo info;

        // First call: query required buffer sizes
        size_t policyNameLen = 0;
        size_t pluginNameLen = 0;
        size_t pluginVersionLen = 0;
        size_t apiVersionLen = 0;

        // Size-query call: pass nullptr for every output to make the intent explicit.
        status = backend->getHeuristicPolicyInfo(handle,
                                                 i,
                                                 nullptr,
                                                 nullptr,
                                                 &policyNameLen,
                                                 nullptr,
                                                 &pluginNameLen,
                                                 nullptr,
                                                 &pluginVersionLen,
                                                 nullptr,
                                                 &apiVersionLen);

        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {{},
                    {ErrorCode::HIPDNN_BACKEND_ERROR,
                     "Failed to query policy info sizes for index " + std::to_string(i) + ": "
                         + backend->getErrorString(status)}};
        }

        // Allocate buffers. Use max(len, 1) so .data() is never null: passing a
        // null buffer to the second call would re-trigger "query mode" per the
        // hipdnnGetHeuristicPolicyInfo_ext contract, silently re-reading sizes
        // instead of filling the buffer.
        std::vector<char> policyNameBuf(std::max<size_t>(policyNameLen, 1));
        std::vector<char> pluginNameBuf(std::max<size_t>(pluginNameLen, 1));
        std::vector<char> pluginVersionBuf(std::max<size_t>(pluginVersionLen, 1));
        std::vector<char> apiVersionBuf(std::max<size_t>(apiVersionLen, 1));

        // Second call: retrieve actual strings
        status = backend->getHeuristicPolicyInfo(handle,
                                                 i,
                                                 &info.policyId,
                                                 policyNameBuf.data(),
                                                 &policyNameLen,
                                                 pluginNameBuf.data(),
                                                 &pluginNameLen,
                                                 pluginVersionBuf.data(),
                                                 &pluginVersionLen,
                                                 apiVersionBuf.data(),
                                                 &apiVersionLen);

        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {{},
                    {ErrorCode::HIPDNN_BACKEND_ERROR,
                     "Failed to retrieve policy info for index " + std::to_string(i) + ": "
                         + backend->getErrorString(status)}};
        }

        info.policyName = bufferToString(policyNameBuf, policyNameLen);
        info.pluginName = bufferToString(pluginNameBuf, pluginNameLen);
        info.pluginVersion = bufferToString(pluginVersionBuf, pluginVersionLen);
        info.apiVersion = bufferToString(apiVersionBuf, apiVersionLen);

        infos.emplace_back(std::move(info));
    }

    return {infos, {}};
}

/// @brief snake_case alias for HeuristicPolicyInfo
using heuristic_policy_info = HeuristicPolicyInfo;

/**
 * @brief snake_case alias for getLoadedHeuristicPolicyInfos()
 *
 * @param h The hipDNN handle to query.
 * @return Pair of (policy info vector, error); vector is empty on failure.
 */
inline auto
    get_loaded_heuristic_policy_infos(hipdnnHandle_t h) // NOLINT(readability-identifier-naming)
{
    return getLoadedHeuristicPolicyInfos(h);
}

} // namespace hipdnn_frontend
