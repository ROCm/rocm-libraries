// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_Heuristics.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_Heuristics.h
 * @brief cuDNN v8 `EngineHeuristics_v8` / `EngineHeuristicsBuilder_v8` and `get_heuristics_list`
 *        over the shim's v8 core.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfig.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Filters.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_OperationGraph.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/v8/boundary.h>
#include <hipdnn_compatibility/cudnn/detail/v8/engine_views.h>
#include <hipdnn_compatibility/cudnn/detail/v8/validate.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_Heuristics.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,modernize-redundant-void-arg,modernize-loop-convert,performance-unnecessary-value-param,performance-inefficient-vector-operation,readability-simplify-boolean-expr,misc-const-correctness,bugprone-macro-parentheses,bugprone-multi-level-implicit-pointer-conversion):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class EngineHeuristics_v8 : public BackendDescriptor
{
public:
    friend class EngineHeuristicsBuilder_v8;
    std::string describe() const override
    {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR :";
        return ss.str();
    }

    EngineHeuristics_v8(EngineHeuristics_v8&& from) = default;
    EngineHeuristics_v8& operator=(EngineHeuristics_v8&& from) = default;

    ~EngineHeuristics_v8() = default;

    // At most `count` results, in heuristic rank order; the same objects on every call.
    auto getEngineConfig(int64_t count = 1) -> std::vector<ManagedOpaqueDescriptor>&
    {
        if(!built)
        {
            set_error_and_throw_exception(this,
                                          CUDNN_STATUS_NOT_INITIALIZED,
                                          "EngineHeuristics: the heuristics are not built");
            return m_heuristic_results;
        }
        const auto kept = static_cast<std::size_t>(
            std::clamp<int64_t>(count, 0, static_cast<int64_t>(m_all_results.size())));
        m_heuristic_results.assign(m_all_results.begin(),
                                   m_all_results.begin() + static_cast<std::ptrdiff_t>(kept));
        return m_heuristic_results;
    }

    auto getEngineConfigCount(void) const -> int64_t
    {
        if(!built)
        {
            set_error_and_throw_exception(this,
                                          CUDNN_STATUS_NOT_INITIALIZED,
                                          "EngineHeuristics: the heuristics are not built");
            return -1;
        }
        return static_cast<int64_t>(m_all_results.size());
    }

private:
    EngineHeuristics_v8() = default;
    EngineHeuristics_v8(const EngineHeuristics_v8&) = delete;
    EngineHeuristics_v8& operator=(const EngineHeuristics_v8&) = delete;

    cudnnBackendHeurMode_t mode = CUDNN_HEUR_MODE_INSTANT;
    ManagedOpaqueDescriptor opGraph = nullptr;
    std::shared_ptr<const DeviceProperties> device_properties = nullptr;
    std::vector<ManagedOpaqueDescriptor> m_heuristic_results;
    std::string opGraphTag;
    int32_t target_sm_count = -1;
    // Every result of the build; m_heuristic_results is the prefix last requested.
    std::vector<ManagedOpaqueDescriptor> m_all_results;
    bool built = false;
};

class EngineHeuristicsBuilder_v8
{
public:
    auto setOperationGraph(OperationGraph_v8& opGraph_) -> EngineHeuristicsBuilder_v8&
    {
        m_heuristics.opGraph = opGraph_.get_desc();
        m_heuristics.opGraphTag = opGraph_.getTag();
        return *this;
    }
    auto setOperationGraph(ManagedOpaqueDescriptor opGraph,
                           std::string tag) -> EngineHeuristicsBuilder_v8&
    {
        m_heuristics.opGraph = opGraph;
        m_heuristics.opGraphTag = tag;
        return *this;
    }

    auto setDeviceProperties(std::shared_ptr<const DeviceProperties> device_properties)
        -> EngineHeuristicsBuilder_v8&
    {
        m_heuristics.device_properties = device_properties;
        return *this;
    }

    auto setHeurMode(cudnnBackendHeurMode_t mode_) -> EngineHeuristicsBuilder_v8&
    {
        m_heuristics.mode = mode_;
        return *this;
    }

    auto setSMCount(int32_t sm_count_) -> EngineHeuristicsBuilder_v8&
    {
        m_heuristics.target_sm_count = sm_count_;
        return *this;
    }

    EngineHeuristics_v8&& build()
    {
        if(m_heuristics.opGraph == nullptr)
        {
            set_error_and_throw_exception(
                &m_heuristics,
                CUDNN_STATUS_BAD_PARAM,
                "EngineHeuristicsBuilder: an operation graph must be set");
            return std::move(m_heuristics);
        };

        std::vector<ManagedOpaqueDescriptor> results;
        const detail::v8::Outcome outcome = detail::v8::guarded([&] {
            m_heuristics.initialize_managed_backend_pointer(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
            const std::shared_ptr<const detail::v8::GraphCore> graph
                = detail::v8::recordOf<detail::v8::GraphCore>(m_heuristics.opGraph);
            if(!graph)
            {
                throw detail::v8::ShimError(
                    CUDNN_STATUS_BAD_PARAM,
                    "EngineHeuristicsBuilder: the operation graph is not a built OperationGraph");
            }
            const cudnnBackendHeurMode_t mode = detail::v8::requireEnumerator(
                m_heuristics.mode, "EngineHeuristicsBuilder: heuristic mode");
            constexpr detail::v8::HeuristicLabels LABELS{
                "EngineHeuristicsBuilder: SM count target",
                "EngineHeuristicsBuilder: device properties"};
            // Every mode returns the graph's rank snapshot; no hipDNN heuristic runs per build.
            std::vector<std::shared_ptr<const detail::v8::EngineRecord>> engines
                = detail::v8::heuristicResults(
                    graph,
                    mode,
                    {m_heuristics.target_sm_count >= 0, m_heuristics.device_properties != nullptr},
                    LABELS);
            results.reserve(engines.size());
            for(std::shared_ptr<const detail::v8::EngineRecord>& engine : engines)
            {
                std::shared_ptr<const detail::v8::ConfigRecord> config
                    = detail::v8::makeConfigRecord(
                        std::move(engine), {}, "EngineHeuristicsBuilder");
                hipdnnBackendDescriptor_t view = config->hipdnnEngineConfig();
                results.push_back(detail::v8::makeManaged(
                    CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, std::move(config), view));
            }
        });
        if(!outcome)
        {
            set_error_and_throw_exception(&m_heuristics, outcome.status, outcome.message.c_str());
            return std::move(m_heuristics);
        }

        m_heuristics.m_all_results = std::move(results);
        m_heuristics.built = true;
        CUDNN_FE_LOG_LABEL_ENDL(m_heuristics);
        return std::move(m_heuristics);
    }

    explicit EngineHeuristicsBuilder_v8() = default;
    ~EngineHeuristicsBuilder_v8() = default;
    EngineHeuristicsBuilder_v8(EngineHeuristicsBuilder_v8&&) = delete;
    EngineHeuristicsBuilder_v8(const EngineHeuristicsBuilder_v8&) = delete;
    EngineHeuristicsBuilder_v8& operator=(const EngineHeuristicsBuilder_v8&) = delete;

private:
    EngineHeuristics_v8 m_heuristics;
};

template <std::size_t SIZE>
EngineConfigList get_heuristics_list(std::array<cudnnBackendHeurMode_t, SIZE> modes,
                                     OperationGraph_v8& opGraph,
                                     std::function<bool(cudnnBackendDescriptor_t)> filter_fn)
{
    CUDNN_FRONTEND_UNUSED(modes);
    EngineConfigList filtered_configs;

    for(auto mode : modes)
    {
        if(mode == CUDNN_HEUR_MODES_COUNT)
        {
            continue;
        }
        auto heuristics
            = EngineHeuristicsBuilder_v8().setOperationGraph(opGraph).setHeurMode(mode).build();
        CUDNN_FE_LOG_LABEL_ENDL("Heuristic Mode " << mode << " has "
                                                  << heuristics.getEngineConfigCount()
                                                  << " configurations.");
        auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
        cudnn_frontend::filter(engine_config, filtered_configs, filter_fn);
    }
    return filtered_configs;
}

#ifndef NV_CUDNN_DISABLE_EXCEPTION
#define NV_CUDNN_FE_TRY() \
    try                   \
    {
#else
#define NV_CUDNN_FE_TRY()
#endif

#ifndef NV_CUDNN_DISABLE_EXCEPTION
#define NV_CUDNN_FE_CATCH(...)                \
    }                                         \
    catch(cudnn_frontend::cudnnException & e) \
    {                                         \
        __VA_ARGS__;                          \
    }
#else
#define NV_CUDNN_FE_CATCH(...)
#endif

#define NV_CUDNN_RETURN_IF_ERROR(heuristics)                \
    do                                                      \
    {                                                       \
        if(heuristics.get_status() != CUDNN_STATUS_SUCCESS) \
        {                                                   \
            return heuristics.get_status();                 \
        }                                                   \
    } while(0);

#define NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(status, is_last_status) \
    if(is_last_status || status != CUDNN_STATUS_SUCCESS)              \
    {                                                                 \
        statuses.push_back(status);                                   \
        if(status == CUDNN_STATUS_SUCCESS || evaluate_all)            \
        {                                                             \
            continue;                                                 \
        }                                                             \
        break;                                                        \
    }

inline cudnnStatus_t
    get_heuristics_list_impl(cudnnBackendHeurMode_t heur_mode,
                             OperationGraph_v8& opGraph,
                             std::function<bool(cudnnBackendDescriptor_t)> filter_fn,
                             int32_t sm_count,
                             EngineConfigList& filtered_configs,
                             std::shared_ptr<const DeviceProperties> device_properties = nullptr)
{
    auto heuristics = EngineHeuristicsBuilder_v8()
                          .setDeviceProperties(device_properties)
                          .setOperationGraph(opGraph)
                          .setHeurMode(heur_mode)
                          .setSMCount(sm_count)
                          .build();
    NV_CUDNN_RETURN_IF_ERROR(heuristics);
    auto num_config = heuristics.getEngineConfigCount();
    NV_CUDNN_RETURN_IF_ERROR(heuristics);
    CUDNN_FE_LOG_LABEL_ENDL("Heuristic query for mode " << heur_mode << " has " << num_config
                                                        << " configurations.");
    auto& engine_config = heuristics.getEngineConfig(num_config);
    NV_CUDNN_RETURN_IF_ERROR(heuristics);
    CUDNN_FE_LOG_LABEL_ENDL("Backend heuristics recommendation count: " << engine_config.size());
    cudnn_frontend::filter(engine_config, filtered_configs, filter_fn);
    return CUDNN_STATUS_SUCCESS;
}

inline std::vector<cudnnStatus_t>
    get_heuristics_list(const std::vector<std::string>& modes,
                        OperationGraph_v8& opGraph,
                        std::function<bool(cudnnBackendDescriptor_t)> filter_fn,
                        EngineConfigList& filtered_configs,
                        bool evaluate_all = false,
                        int32_t sm_count = -1,
                        std::shared_ptr<const DeviceProperties> device_properties = nullptr)
{
    std::vector<cudnnStatus_t> statuses;

    for(auto& mode : modes)
    {
        if(mode.find("heuristics_instant") != std::string::npos
           || mode.find("heuristics_mode_a") != std::string::npos)
        {
            auto heur_mode = CUDNN_HEUR_MODE_A;
            NV_CUDNN_FE_TRY();
            auto status_l = get_heuristics_list_impl(
                heur_mode, opGraph, filter_fn, sm_count, filtered_configs, device_properties);
            NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(status_l, true);
            NV_CUDNN_FE_CATCH(NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(e.getCudnnStatus(), true));
        }
        else if(mode.find("heuristics_fallback") != std::string::npos)
        {
            NV_CUDNN_FE_TRY();
            auto status_l = get_heuristics_list_impl(CUDNN_HEUR_MODE_FALLBACK,
                                                     opGraph,
                                                     filter_fn,
                                                     sm_count,
                                                     filtered_configs,
                                                     device_properties);
            NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(status_l, true);
            NV_CUDNN_FE_CATCH(NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(e.getCudnnStatus(), true));
        }
        else if(mode.find("heuristics_mode_b") != std::string::npos)
        {
            auto heur_mode = CUDNN_HEUR_MODE_B;
            NV_CUDNN_FE_TRY();
            auto status_l = get_heuristics_list_impl(
                heur_mode, opGraph, filter_fn, sm_count, filtered_configs, device_properties);

#if(CUDNN_VERSION >= 8300) && (CUDNN_VERSION < 8600)
            if(status_l != CUDNN_STATUS_SUCCESS)
            {
                status_l = get_heuristics_list_impl(CUDNN_HEUR_MODE_INSTANT,
                                                    opGraph,
                                                    filter_fn,
                                                    sm_count,
                                                    filtered_configs,
                                                    device_properties);
            }
#endif
            NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(status_l, true);
#if(CUDNN_VERSION >= 8300) && (CUDNN_VERSION < 8600)
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        }
        catch(cudnn_frontend::cudnnException&)
        {
            NV_CUDNN_FE_TRY();
            auto status_ = get_heuristics_list_impl(
                heur_mode, opGraph, filter_fn, sm_count, filtered_configs, device_properties);
            statuses.push_back(status_);
            NV_CUDNN_FE_CATCH(NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(e.getCudnnStatus(), true));
        }
#endif
#else
            NV_CUDNN_FE_CATCH(NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(e.getCudnnStatus(), true));
#endif
    }
}
return statuses;
}

inline std::vector<cudnnStatus_t>
    get_heuristics_list(const std::vector<cudnn_frontend::HeurMode_t>& modes,
                        OperationGraph_v8& opGraph,
                        std::function<bool(cudnnBackendDescriptor_t)> filter_fn,
                        EngineConfigList& filtered_configs,
                        bool evaluate_all = false,
                        int32_t sm_count = -1,
                        std::shared_ptr<const DeviceProperties> device_properties = nullptr)
{
    std::unordered_map<HeurMode_t, std::string> mode_to_string = {
        {HeurMode_t::A, "heuristics_mode_a"},
        {HeurMode_t::B, "heuristics_mode_b"},
        {HeurMode_t::FALLBACK, "heuristics_fallback"},
    };

    std::vector<std::string> string_modes(modes.size());
    std::transform(modes.begin(),
                   modes.end(),
                   string_modes.begin(),
                   [&mode_to_string](const auto& mode) { return mode_to_string.at(mode); });

    return get_heuristics_list(string_modes,
                               opGraph,
                               filter_fn,
                               filtered_configs,
                               evaluate_all,
                               sm_count,
                               device_properties);
}

template <std::size_t SIZE>
std::vector<cudnnStatus_t>
    get_heuristics_list(std::array<std::string, SIZE> modes,
                        OperationGraph_v8& opGraph,
                        std::function<bool(cudnnBackendDescriptor_t)> filter_fn,
                        EngineConfigList& filtered_configs,
                        bool evaluate_all = false)
{
    std::vector<std::string> modes_vector(modes.begin(), modes.end());
    return get_heuristics_list(modes_vector, opGraph, filter_fn, filtered_configs, evaluate_all);
}

#undef NV_CUDNN_FE_TRY
#undef NV_CUDNN_FE_CATCH
#undef NV_CUDNN_RETURN_IF_ERROR
#undef NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE

using EngineHeuristicsBuilde = EngineHeuristicsBuilder_v8;
using EngineHeuristics = EngineHeuristics_v8;

// NOLINTEND(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,modernize-redundant-void-arg,modernize-loop-convert,performance-unnecessary-value-param,performance-inefficient-vector-operation,readability-simplify-boolean-expr,misc-const-correctness,bugprone-macro-parentheses,bugprone-multi-level-implicit-pointer-conversion)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
