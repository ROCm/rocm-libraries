// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend/graph_interface.h, include/cudnn_frontend/node_interface.h
//     Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file graph_wrapper.h
 * @brief cuDNN FE graph::Graph implemented over a hipDNN frontend graph.
 *
 * Include via cudnn_frontend.h; not a supported direct include.
 */

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_properties.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/sdpa_attributes.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/engine_filters.h>
#include <hipdnn_compatibility/cudnn/detail/error_recorder.h>
#include <hipdnn_compatibility/cudnn/detail/knob_wrapper.h>
#include <hipdnn_compatibility/cudnn/detail/node_wrappers/unsupported_nodes.h>
#include <hipdnn_compatibility/cudnn/detail/note_triage.h>
#include <hipdnn_frontend/Graph.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
// Node signatures mirror upstream graph_interface.h / node_interface.h at the pinned version.
// Upstream changes node arity between minor releases; re-diff them before bumping the pin.
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff every Graph node signature in this "
              "file against upstream graph_interface.h / node_interface.h at the new tag, "
              "update the upstream version and commit in frontend/THIRD_PARTY_LICENSES.md, "
              "then update this assert.");

// NOLINTBEGIN(readability-identifier-naming): the whole class mirrors cuDNN's
// snake_case public spelling for source compatibility.

#define HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR()       \
    do                                                  \
    {                                                   \
        if(auto err = getRecordedError(); err.is_bad()) \
        {                                               \
            return err;                                 \
        }                                               \
    } while(0)

#define HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH() \
    do                                                        \
    {                                                         \
        if(!hasOperationGraphState())                         \
        {                                                     \
            return noExecutionPlanError();                    \
        }                                                     \
    } while(0)

#define HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH() \
    do                                                   \
    {                                                    \
        if(!hasOperationGraphState())                    \
        {                                                \
            return {};                                   \
        }                                                \
    } while(0)

/// @brief cuDNN FE graph::Graph that builds and runs a wrapped hipDNN graph.
/// @note Copies share the native hipDNN graph and its tensors; build progress is tracked per copy.
class Graph : public ErrorRecorder<Graph>
{
public:
    Graph() = default;
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;

    // Copyable because cuDNN FE source returns graphs by value; copies alias the native graph.
    Graph(const Graph&) = default;
    Graph& operator=(const Graph&) = default;

    /// @brief Validates the graph; also returns any error recorded by an earlier unsupported call.
    error_t validate()
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        CHECK_CUDNN_FRONTEND_ERROR(validateOwnedTensors());
        if(hasOperationGraphState())
        {
            return _graph->validate();
        }

        return {};
    }

    /// @brief Lowers the graph to a hipDNN operation graph.
    error_t build_operation_graph(cudnnHandle_t handle)
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        CHECK_CUDNN_FRONTEND_ERROR(validateOwnedTensors());
        if(!hasOperationGraphState())
        {
            _stage = Stage::OpGraphBuilt;
            return {};
        }

        auto err = _graph->build_operation_graph(handle);
        if(err.is_good())
        {
            _stage = Stage::OpGraphBuilt;
        }
        return err;
    }

    /// @brief Deviceless build: succeeds only for an empty graph, otherwise returns INVALID_VALUE.
    error_t build_operation_graph()
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        CHECK_CUDNN_FRONTEND_ERROR(validateOwnedTensors());
        if(!hasOperationGraphState())
        {
            _stage = Stage::OpGraphBuilt;
            return {};
        }

        return unsupportedDevicelessBuildError();
    }

    /// @brief Creates execution plans and applies pending plan filters.
    /// @note Only HeurMode_t::FALLBACK is honored; other modes are accepted with a warning.
    error_t create_execution_plans(const std::vector<HeurMode_t>& modes = {HeurMode_t::FALLBACK})
    {
        warnUnhonoredHeuristicModes(modes);

        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH();

        auto err = _graph->create_execution_plans(modes);
        if(err.is_bad())
        {
            return err;
        }

        _planCreationModes = modes;
        CHECK_CUDNN_FRONTEND_ERROR(applyPendingPlanFilters(_planCreationModes));
        _stage = Stage::PlansCreated;
        return err;
    }

    /// @brief Checks that the created execution plans are supported.
    error_t check_support()
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH();

        CHECK_CUDNN_FRONTEND_ERROR(applyPendingFiltersForCreatedPlans());
        return _graph->check_support();
    }

    /// @brief Same as check_support(); the handle is unused.
    error_t check_support(cudnnHandle_t handle)
    {
        static_cast<void>(handle);
        return check_support();
    }

    /// @brief hipDNN extension: reports whether any engine can run the graph; creates no plans.
    /// @note Builds the operation graph as a side effect.
    error_t is_supported_ext(cudnnHandle_t handle,
                             const std::vector<HeurMode_t>& modes = {HeurMode_t::FALLBACK})
    {
        warnUnhonoredHeuristicModes(modes);

        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        CHECK_CUDNN_FRONTEND_ERROR(validateOwnedTensors());
        if(hasOperationGraphState())
        {
            CHECK_CUDNN_FRONTEND_ERROR(_graph->is_supported_ext(handle, modes));
        }

        if(!stageAtLeast(Stage::OpGraphBuilt))
        {
            _stage = Stage::OpGraphBuilt;
        }
        return {};
    }

    /// @brief Builds execution plans; `doMultithreadedBuilds` is logged and ignored.
    error_t build_plans(BuildPlanPolicy_t policy = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                        bool doMultithreadedBuilds = false)
    {
        if(doMultithreadedBuilds)
        {
            CUDNN_FE_LOG_LABEL("Ignoring multithreaded-build hint; this shim builds serially");
        }
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH();

        CHECK_CUDNN_FRONTEND_ERROR(applyPendingFiltersForCreatedPlans());

        // Native HEURISTICS_CHOICE only checks the active plan; cuDNN narrows the candidates,
        // so retarget onto the top-ranked plan this shim's filters did not bar.
        int64_t activeEngineId = -1;
        if(policy == BuildPlanPolicy_t::HEURISTICS_CHOICE && !_shimBarredEngineIds.empty()
           && _graph->get_execution_plan_engine_id(activeEngineId).is_good()
           && isShimBarred(activeEngineId))
        {
            int64_t survivingIndex = -1;
            CHECK_CUDNN_FRONTEND_ERROR(findTopSurvivingPlanIndex(survivingIndex));
            CHECK_CUDNN_FRONTEND_ERROR(_graph->build_plan_at_index(survivingIndex));
            _stage = Stage::PlansBuilt;
            return {};
        }

        auto err = _graph->build_plans(policy);
        if(err.is_good())
        {
            _stage = Stage::PlansBuilt;
        }
        return err;
    }

    /// @brief Same as build_plans(policy, doMultithreadedBuilds); the handle is unused.
    error_t build_plans(const cudnnHandle_t& handle,
                        BuildPlanPolicy_t policy = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                        bool doMultithreadedBuilds = false)
    {
        static_cast<void>(handle);
        return build_plans(policy, doMultithreadedBuilds);
    }

    /// @brief Builds the plan at `index`, creating plans first if needed.
    error_t build_plan_at_index(int64_t index)
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();

        if(!stageAtLeast(Stage::PlansCreated))
        {
            CHECK_CUDNN_FRONTEND_ERROR(create_execution_plans());
        }
        CHECK_CUDNN_FRONTEND_ERROR(applyPendingFiltersForCreatedPlans());
        CHECK_CUDNN_FRONTEND_ERROR(_graph->build_plan_at_index(index));
        _stage = Stage::PlansBuilt;
        return {};
    }

    /// @brief Same as build_plan_at_index(index); the handle is unused.
    error_t build_plan_at_index(const cudnnHandle_t& handle, int64_t index)
    {
        static_cast<void>(handle);
        return build_plan_at_index(index);
    }

    /// @brief Returns the number of created execution plans.
    int64_t get_execution_plan_count() const
    {
        if(hasOperationGraphState())
        {
            return _graph->get_execution_plan_count();
        }
        return stageAtLeast(Stage::PlansCreated) ? 1 : 0;
    }

    /// @brief Returns the number of applicable hipDNN engines, in hipDNN's ranking order.
    error_t get_engine_count(int64_t& count)
    {
        count = 0;
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH();

        CHECK_CUDNN_FRONTEND_ERROR(refreshEngineIndexMap());
        count = static_cast<int64_t>(_engineIndexToNativeEngineId.size());
        return {};
    }

    /// @brief Returns the engine's knobs; hipDNN knobs with no KnobType_t mapping are omitted.
    error_t get_knobs_for_engine(int64_t engineIndex, std::vector<Knob>& knobs)
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        if(!hasOperationGraphState())
        {
            knobs.clear();
            return noExecutionPlanError();
        }
        int64_t nativeEngineId = -1;
        CHECK_CUDNN_FRONTEND_ERROR(mapEngineIndex(engineIndex, nativeEngineId));

        std::vector<hipdnn_frontend::Knob> nativeKnobs;
        CHECK_CUDNN_FRONTEND_ERROR(_graph->get_knobs_for_engine(nativeEngineId, nativeKnobs));
        hipdnn_frontend::compatibility::cudnn_frontend::detail::projectNativeKnobs(nativeKnobs,
                                                                                   knobs);
        return {};
    }

    /// @brief Creates a plan for engine `engineIndex` with the given knob choices.
    error_t create_execution_plan(int64_t engineIndex,
                                  const std::unordered_map<KnobType_t, int64_t>& knobs)
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();

        int64_t nativeEngineId = -1;
        CHECK_CUDNN_FRONTEND_ERROR(mapEngineIndex(engineIndex, nativeEngineId));

        // Resolve the cuDNN knob enums against this engine's own knob ids; an
        // unresolved choice is an error, never a silently dropped setting.
        std::vector<hipdnn_frontend::KnobSetting> settings;
        if(!knobs.empty())
        {
            std::vector<hipdnn_frontend::Knob> nativeKnobs;
            CHECK_CUDNN_FRONTEND_ERROR(_graph->get_knobs_for_engine(nativeEngineId, nativeKnobs));
            CHECK_CUDNN_FRONTEND_ERROR(
                hipdnn_frontend::compatibility::cudnn_frontend::detail::makeNativeKnobSettings(
                    knobs, nativeKnobs, settings));
        }

        CHECK_CUDNN_FRONTEND_ERROR(_graph->create_execution_plan_ext(nativeEngineId, settings));
        CHECK_CUDNN_FRONTEND_ERROR(applyPendingPlanFilters());
        _stage = Stage::PlansCreated;
        return {};
    }

    /// @brief Returns the active plan's name.
    error_t get_plan_name(std::string& name) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph->get_plan_name(name);
    }

    /// @brief Returns the name of the plan at `index`.
    error_t get_plan_name_at_index(int64_t index, std::string& name) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph->get_plan_name_at_index(index, name);
    }

    /// @brief Returns the workspace size of the plan at `index`.
    error_t get_workspace_size_plan_at_index(int64_t index, int64_t& workspaceSize) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph->get_workspace_size_plan_at_index(index, workspaceSize);
    }

    /// @brief Returns the workspace size of the plan at `index`, or 0 on failure (logged).
    int64_t get_workspace_size_plan_at_index(int64_t index) const
    {
        int64_t workspaceSize = 0;
        auto err = get_workspace_size_plan_at_index(index, workspaceSize);
        if(err.is_bad())
        {
            CUDNN_FE_LOG_LABEL("ERROR: Querying workspace for plan failed: " << err.get_message());
            return 0;
        }
        return workspaceSize;
    }

    /// @brief Plan workspace query; non-empty shape overrides return GRAPH_NOT_SUPPORTED.
    error_t get_workspace_size_plan_at_index(
        cudnnHandle_t handle,
        int64_t index,
        int64_t& workspaceSize,
        const std::vector<int64_t>& overrideUids,
        const std::vector<std::vector<int64_t>>& overrideShapes,
        const std::vector<std::vector<int64_t>>& overrideStrides) const
    {
        static_cast<void>(handle);
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        if(overrideUids.empty() && overrideShapes.empty() && overrideStrides.empty())
        {
            return get_workspace_size_plan_at_index(index, workspaceSize);
        }
        return {error_code_t::GRAPH_NOT_SUPPORTED,
                "Runtime shape override workspace query is unsupported by this shim"};
    }

    /// @brief Plan workspace query, or 0 on failure (logged); non-empty overrides fail.
    int64_t get_workspace_size_plan_at_index(
        cudnnHandle_t handle,
        int64_t index,
        const std::vector<int64_t>& overrideUids,
        const std::vector<std::vector<int64_t>>& overrideShapes,
        const std::vector<std::vector<int64_t>>& overrideStrides) const
    {
        int64_t workspaceSize = 0;
        auto err = get_workspace_size_plan_at_index(
            handle, index, workspaceSize, overrideUids, overrideShapes, overrideStrides);
        if(err.is_bad())
        {
            CUDNN_FE_LOG_LABEL("ERROR: Querying workspace for plan failed: " << err.get_message());
            return 0;
        }
        return workspaceSize;
    }

    /// @brief Returns the behavior notes of the active plan's engine.
    error_t get_behavior_notes(std::vector<BehaviorNote_t>& notes) const
    {
        notes.clear();
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();

        int64_t engineId = -1;
        CHECK_CUDNN_FRONTEND_ERROR(_graph->get_execution_plan_engine_id(engineId));
        return _graph->get_behavior_notes_for_engine(engineId, notes);
    }

    /// @brief Returns the behavior notes of the plan at `index`.
    error_t get_behavior_notes_for_plan_at_index(int64_t index,
                                                 std::vector<BehaviorNote_t>& notes) const
    {
        notes.clear();
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();

        std::string planName;
        CHECK_CUDNN_FRONTEND_ERROR(_graph->get_plan_name_at_index(index, planName));
        const int64_t engineId = hipdnn_data_sdk::utilities::engineNameOrIdToId(planName);
        return _graph->get_behavior_notes_for_engine(engineId, notes);
    }

    /// @brief Runs the full validate-to-build_plans sequence.
    error_t build(const cudnnHandle_t& handle,
                  const std::vector<HeurMode_t>& modes = {HeurMode_t::FALLBACK},
                  BuildPlanPolicy_t policy = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                  bool doMultithreadedBuilds = false)
    {
        CHECK_CUDNN_FRONTEND_ERROR(validate());
        CHECK_CUDNN_FRONTEND_ERROR(build_operation_graph(handle));
        CHECK_CUDNN_FRONTEND_ERROR(create_execution_plans(modes));
        CHECK_CUDNN_FRONTEND_ERROR(check_support());
        return build_plans(policy, doMultithreadedBuilds);
    }

    /// @brief Deviceless build: succeeds only for an empty graph; the arguments are ignored.
    error_t build(const std::vector<HeurMode_t>& modes,
                  BuildPlanPolicy_t policy = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                  bool doMultithreadedBuilds = false)
    {
        static_cast<void>(modes);
        static_cast<void>(policy);
        static_cast<void>(doMultithreadedBuilds);
        CHECK_CUDNN_FRONTEND_ERROR(validate());
        if(hasOperationGraphState())
        {
            return unsupportedDevicelessBuildError();
        }
        return build_operation_graph();
    }

    /// @brief Sets the graph name.
    Graph& set_name(const std::string& name)
    {
        _graph->set_name(name);
        return *this;
    }

    /// @brief hipDNN extension: returns the graph name.
    const std::string& get_name() const
    {
        return _graph->get_name();
    }

    /// @brief Sets the default I/O data type.
    Graph& set_io_data_type(DataType_t type)
    {
        _graph->set_io_data_type(type);
        return *this;
    }

    /// @brief hipDNN extension: returns the default I/O data type.
    DataType_t get_io_data_type() const
    {
        return _graph->get_io_data_type();
    }

    /// @brief Sets the default compute data type.
    Graph& set_compute_data_type(DataType_t type)
    {
        _graph->set_compute_data_type(type);
        return *this;
    }

    /// @brief hipDNN extension: returns the default compute data type.
    DataType_t get_compute_data_type() const
    {
        return _graph->get_compute_data_type();
    }

    /// @brief Sets the default intermediate data type.
    Graph& set_intermediate_data_type(DataType_t type)
    {
        _graph->set_intermediate_data_type(type);
        return *this;
    }

    /// @brief hipDNN extension: returns the default intermediate data type.
    DataType_t get_intermediate_data_type() const
    {
        return _graph->get_intermediate_data_type();
    }

#ifdef HIPDNN_ENABLE_SDPA
    // Native set_override_shape_enabled exists only in SDPA-enabled builds.
    /// @brief Enables runtime shape overrides.
    Graph& set_override_shape_enabled(bool isEnabled)
    {
        _graph->set_override_shape_enabled(isEnabled);
        return *this;
    }
#endif // HIPDNN_ENABLE_SDPA

    // Droppable hints log and continue; requests that would change results record an error.
    /// @brief Accepted for source compatibility; logged and ignored.
    Graph& set_dynamic_shape_enabled(bool isEnabled)
    {
        static_cast<void>(isEnabled);
        CUDNN_FE_LOG_LABEL("Ignoring graph dynamic-shape hint; hipDNN has no graph-level setting");
        return *this;
    }

    /// @brief Accepted for source compatibility; logged and ignored.
    Graph& set_kernel_cache(const std::shared_ptr<KernelCache>& cache)
    {
        static_cast<void>(cache);
        CUDNN_FE_LOG_LABEL("Ignoring graph kernel cache hint; hipDNN selects kernels internally");
        return *this;
    }

    /// @brief Unsupported: records INVALID_VALUE, reported by the next validate() or build step.
    Graph& set_sm_count(int32_t count)
    {
        static_cast<void>(count);
        CUDNN_FE_LOG_LABEL("ERROR: Target SM count is unsupported by this shim");
        recordError(error_code_t::INVALID_VALUE, "Target SM count is unsupported by this shim");
        return *this;
    }

    /// @brief Unsupported: records INVALID_VALUE, reported by the next validate() or build step.
    Graph& set_sm_version(int32_t version)
    {
        static_cast<void>(version);
        CUDNN_FE_LOG_LABEL("ERROR: Target SM version is unsupported by this shim");
        recordError(error_code_t::INVALID_VALUE, "Target SM version is unsupported by this shim");
        return *this;
    }

    /// @brief Unsupported: records INVALID_VALUE, reported by the next validate() or build step.
    Graph& set_device_properties(const std::shared_ptr<const DeviceProperties>& deviceProperties)
    {
        static_cast<void>(deviceProperties);
        CUDNN_FE_LOG_LABEL("ERROR: Device properties are unsupported by this shim");
        recordError(error_code_t::INVALID_VALUE, "Device properties are unsupported by this shim");
        return *this;
    }

    // Per-note dispositions live in detail/note_triage.h.

    /// @brief Logged and ignored; hipDNN has no numerical-note metadata.
    /// @note STRICT_NAN_PROP records GRAPH_NOT_SUPPORTED.
    Graph& select_numeric_notes(const std::vector<NumericalNote_t>& notes)
    {
        for(const auto note : notes)
        {
            applyNote("select_numeric_notes",
                      note,
                      cudnn_frontend::detail::triageSelect(note),
                      [](NumericalNote_t) { return false; });
        }
        return *this;
    }

    /// @brief NONDETERMINISTIC restricts plans to engines that claim deterministic results.
    /// @note DOWN_CONVERT_INPUTS and REDUCED_PRECISION_REDUCTION record GRAPH_NOT_SUPPORTED;
    /// other notes are logged and ignored.
    Graph& deselect_numeric_notes(const std::vector<NumericalNote_t>& notes)
    {
        for(const auto note : notes)
        {
            applyNote("deselect_numeric_notes",
                      note,
                      cudnn_frontend::detail::triageDeselect(note),
                      [this](NumericalNote_t mapped) {
                          if(mapped != NumericalNote_t::NONDETERMINISTIC)
                          {
                              return false;
                          }
                          _requireDeterministic = true;
                          return true;
                      });
        }
        return *this;
    }

    /// @brief Restricts plans to engines that report these behavior notes.
    /// @note A cuDNN-only note, which no hipDNN engine reports, records GRAPH_NOT_SUPPORTED.
    Graph& select_behavior_notes(const std::vector<BehaviorNote_t>& notes)
    {
        for(const auto note : notes)
        {
            applyNote("select_behavior_notes",
                      note,
                      cudnn_frontend::detail::triageSelect(note),
                      [this](BehaviorNote_t mapped) {
                          _selectedBehaviorNotes.push_back(mapped);
                          return true;
                      });
        }
        return *this;
    }

    /// @brief Excludes engines that report these behavior notes.
    /// @note cuDNN-only notes are logged and ignored; no hipDNN engine reports them.
    Graph& deselect_behavior_notes(const std::vector<BehaviorNote_t>& notes)
    {
        for(const auto note : notes)
        {
            applyNote("deselect_behavior_notes",
                      note,
                      cudnn_frontend::detail::triageDeselect(note),
                      [this](BehaviorNote_t mapped) {
                          _deselectedBehaviorNotes.push_back(mapped);
                          return true;
                      });
        }
        return *this;
    }

    /// @brief Excludes plans whose workspace exceeds `workspace` bytes.
    Graph& deselect_workspace_greater_than(int64_t workspace)
    {
        _maxWorkspaceAllowed = workspace;
        _graph->deselect_workspace_greater_than(workspace);
        return *this;
    }

    /// @brief Unsupported for non-zero limits: records GRAPH_NOT_SUPPORTED; 0 is a no-op.
    Graph& deselect_shared_mem_greater_than(int64_t sharedMem)
    {
        if(sharedMem != 0)
        {
            recordError(error_code_t::GRAPH_NOT_SUPPORTED,
                        "deselect_shared_mem_greater_than requires per-plan shared-memory "
                        "metadata that hipDNN does not expose yet");
        }
        return *this;
    }

    /// @brief Excludes engines by name.
    /// @note Names are hipDNN engine names, not cuDNN engine names.
    Graph& deselect_engines(const std::vector<std::string>& engineNames)
    {
        _barredEngineNames.insert(_barredEngineNames.end(), engineNames.begin(), engineNames.end());
        _graph->deselect_engines(engineNames);
        return *this;
    }

    /// @brief hipDNN extension: excludes engines by index, as counted by get_engine_count().
    Graph& deselect_engines(const std::vector<int64_t>& engineIndices)
    {
        // Indices resolve to native engine IDs only after the operation graph is built;
        // applyPendingPlanFilters() does the translation.
        _barredEngineIndices.insert(
            _barredEngineIndices.end(), engineIndices.begin(), engineIndices.end());
        return *this;
    }

    /// @brief Creates a graph tensor from `tensorAttributes`.
    std::shared_ptr<Tensor_attributes> tensor(const Tensor_attributes& tensorAttributes)
    {
        auto tensorPtr = hipdnn_frontend::graph::Graph::tensor(tensorAttributes);
        _ownedTensors.emplace_back(tensorPtr);
        return tensorPtr;
    }

    /// @brief Creates a scalar tensor from a float value.
    std::shared_ptr<Tensor_attributes> tensor(const float& scalar, ScalarType scalarType)
    {
        return scalarTensor(scalar, scalarType);
    }

    /// @brief Creates a scalar tensor from a half value.
    std::shared_ptr<Tensor_attributes> tensor(const half& scalar, ScalarType scalarType)
    {
        return scalarTensor(scalar, scalarType);
    }

    /// @brief Creates a scalar tensor from a nv_bfloat16 value.
    std::shared_ptr<Tensor_attributes> tensor(const nv_bfloat16& scalar, ScalarType scalarType)
    {
        return scalarTensor(scalar, scalarType);
    }

    /// @brief Creates a scalar tensor from a int32_t value.
    std::shared_ptr<Tensor_attributes> tensor(const int32_t& scalar, ScalarType scalarType)
    {
        return scalarTensor(scalar, scalarType);
    }

    /// @brief Creates a scalar tensor from a int64_t value.
    std::shared_ptr<Tensor_attributes> tensor(const int64_t& scalar, ScalarType scalarType)
    {
        return scalarTensor(scalar, scalarType);
    }

    /// @brief Creates a scalar tensor from a double value.
    std::shared_ptr<Tensor_attributes> tensor(const double& scalar, ScalarType scalarType)
    {
        return scalarTensor(scalar, scalarType);
    }

    /// @brief Creates a tensor with the properties of `tensorAttributes` but no UID.
    std::shared_ptr<Tensor_attributes>
        tensor_like(const std::shared_ptr<Tensor_attributes>& tensorAttributes,
                    const std::string& name = std::string{})
    {
        auto tensorPtr = hipdnn_frontend::graph::Graph::tensor_like(tensorAttributes, name);
        _ownedTensors.emplace_back(tensorPtr);
        return tensorPtr;
    }

    /// @brief Copies the attributes of the tensor with `uid`; INVALID_VALUE if not found.
    error_t query_tensor_attributes_of_uid(int64_t uid, Tensor_attributes& tensorAttributes) const
    {
        if(auto tensorPtr = findOwnedTensorByUid(uid))
        {
            tensorAttributes = *tensorPtr;
            return {};
        }

        if(hasOperationGraphState())
        {
            auto nativeTensors = _graph->getTensorsByUid();
            auto it = nativeTensors.find(uid);
            if(it != nativeTensors.end() && it->second)
            {
                tensorAttributes = *it->second;
                return {};
            }
        }

        return {error_code_t::INVALID_VALUE, "Tensor UID was not found"};
    }

    // Node methods forward to the wrapped hipDNN graph and take *_attributes by value.

    /// @brief Adds a convolution forward node.
    std::shared_ptr<Tensor_attributes> conv_fprop(std::shared_ptr<Tensor_attributes> x,
                                                  std::shared_ptr<Tensor_attributes> w,
                                                  Conv_fprop_attributes attributes)
    {
        auto output = _graph->conv_fprop(std::move(x), std::move(w), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    /// @brief Adds a convolution data-gradient node.
    std::shared_ptr<Tensor_attributes> conv_dgrad(std::shared_ptr<Tensor_attributes> dy,
                                                  std::shared_ptr<Tensor_attributes> w,
                                                  Conv_dgrad_attributes attributes)
    {
        auto output = _graph->conv_dgrad(std::move(dy), std::move(w), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    /// @brief Adds a convolution weight-gradient node.
    std::shared_ptr<Tensor_attributes> conv_wgrad(std::shared_ptr<Tensor_attributes> dy,
                                                  std::shared_ptr<Tensor_attributes> x,
                                                  Conv_wgrad_attributes attributes)
    {
        auto output = _graph->conv_wgrad(std::move(dy), std::move(x), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    /// @brief Adds a batch normalization training node.
    std::array<std::shared_ptr<Tensor_attributes>, 5>
        batchnorm(std::shared_ptr<Tensor_attributes> x,
                  std::shared_ptr<Tensor_attributes> scale,
                  std::shared_ptr<Tensor_attributes> bias,
                  Batchnorm_attributes attributes)
    {
        auto outputs = _graph->batchnorm(
            std::move(x), std::move(scale), std::move(bias), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    /// @brief Adds a batch normalization backward node.
    std::array<std::shared_ptr<Tensor_attributes>, 3>
        batchnorm_backward(std::shared_ptr<Tensor_attributes> dy,
                           std::shared_ptr<Tensor_attributes> x,
                           std::shared_ptr<Tensor_attributes> scale,
                           Batchnorm_backward_attributes attributes)
    {
        auto outputs = _graph->batchnorm_backward(
            std::move(dy), std::move(x), std::move(scale), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    /// @brief Adds a batch normalization inference node.
    std::shared_ptr<Tensor_attributes>
        batchnorm_inference(std::shared_ptr<Tensor_attributes> x,
                            std::shared_ptr<Tensor_attributes> mean,
                            std::shared_ptr<Tensor_attributes> invVariance,
                            std::shared_ptr<Tensor_attributes> scale,
                            std::shared_ptr<Tensor_attributes> bias,
                            Batchnorm_inference_attributes attributes)
    {
        auto output = _graph->batchnorm_inference(std::move(x),
                                                  std::move(mean),
                                                  std::move(invVariance),
                                                  std::move(scale),
                                                  std::move(bias),
                                                  std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    /// @brief Adds a layer normalization node.
    std::array<std::shared_ptr<Tensor_attributes>, 3>
        layernorm(std::shared_ptr<Tensor_attributes> x,
                  std::shared_ptr<Tensor_attributes> scale,
                  std::shared_ptr<Tensor_attributes> bias,
                  Layernorm_attributes attributes)
    {
        auto outputs = _graph->layernorm(
            std::move(x), std::move(scale), std::move(bias), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    /// @brief Adds a layer normalization backward node.
    std::array<std::shared_ptr<Tensor_attributes>, 3>
        layernorm_backward(std::shared_ptr<Tensor_attributes> dy,
                           std::shared_ptr<Tensor_attributes> x,
                           std::shared_ptr<Tensor_attributes> scale,
                           Layernorm_backward_attributes attributes)
    {
        auto outputs = _graph->layernorm_backward(
            std::move(dy), std::move(x), std::move(scale), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    /// @brief Adds an RMS normalization node.
    std::array<std::shared_ptr<Tensor_attributes>, 2>
        rmsnorm(std::shared_ptr<Tensor_attributes> x,
                std::shared_ptr<Tensor_attributes> scale,
                Rmsnorm_attributes attributes)
    {
        auto outputs = _graph->rmsnorm(std::move(x), std::move(scale), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    /// @brief Adds an RMS normalization backward node.
    std::array<std::shared_ptr<Tensor_attributes>, 3>
        rmsnorm_backward(std::shared_ptr<Tensor_attributes> dy,
                         std::shared_ptr<Tensor_attributes> x,
                         std::shared_ptr<Tensor_attributes> scale,
                         std::shared_ptr<Tensor_attributes> invVariance,
                         Rmsnorm_backward_attributes attributes)
    {
        auto outputs = _graph->rmsnorm_backward(std::move(dy),
                                                std::move(x),
                                                std::move(scale),
                                                std::move(invVariance),
                                                std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    /// @brief Adds a matrix multiplication node.
    std::shared_ptr<Tensor_attributes> matmul(std::shared_ptr<Tensor_attributes> a,
                                              std::shared_ptr<Tensor_attributes> b,
                                              Matmul_attributes attributes)
    {
        auto output = _graph->matmul(std::move(a), std::move(b), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    /// @brief Adds a unary pointwise node.
    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes> a,
                                                 Pointwise_attributes attributes)
    {
        auto output = _graph->pointwise(std::move(a), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    /// @brief Adds a binary pointwise node.
    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes> a,
                                                 std::shared_ptr<Tensor_attributes> b,
                                                 Pointwise_attributes attributes)
    {
        auto output = _graph->pointwise(std::move(a), std::move(b), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    /// @brief Adds a ternary pointwise node.
    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes> a,
                                                 std::shared_ptr<Tensor_attributes> b,
                                                 std::shared_ptr<Tensor_attributes> c,
                                                 Pointwise_attributes attributes)
    {
        auto output
            = _graph->pointwise(std::move(a), std::move(b), std::move(c), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    /// @brief Adds a reduction node.
    std::shared_ptr<Tensor_attributes> reduction(std::shared_ptr<Tensor_attributes> a,
                                                 Reduction_attributes attributes)
    {
        auto output = _graph->reduction(std::move(a), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    /// @brief Adds a resample node.
    std::array<std::shared_ptr<Tensor_attributes>, 2> resample(std::shared_ptr<Tensor_attributes> x,
                                                               Resample_attributes attributes)
    {
        auto outputs = _graph->resample(std::move(x), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    /// @brief Adds a block-scale quantize node.
    std::array<std::shared_ptr<Tensor_attributes>, 2>
        block_scale_quantize(std::shared_ptr<Tensor_attributes> x,
                             Block_scale_quantize_attributes attributes)
    {
        auto outputs = _graph->block_scale_quantize(std::move(x), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    /// @brief Adds a block-scale dequantize node.
    std::shared_ptr<Tensor_attributes>
        block_scale_dequantize(std::shared_ptr<Tensor_attributes> x,
                               std::shared_ptr<Tensor_attributes> scale,
                               Block_scale_dequantize_attributes attributes)
    {
        auto output
            = _graph->block_scale_dequantize(std::move(x), std::move(scale), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    /// @brief Adds a mixture-of-experts grouped matmul node.
    std::shared_ptr<Tensor_attributes>
        moe_grouped_matmul(std::shared_ptr<Tensor_attributes> token,
                           std::shared_ptr<Tensor_attributes> weight,
                           std::shared_ptr<Tensor_attributes> firstTokenOffset,
                           std::shared_ptr<Tensor_attributes> tokenIndex,
                           std::shared_ptr<Tensor_attributes> tokenKs,
                           Moe_grouped_matmul_attributes attributes)
    {
        auto output = _graph->moe_grouped_matmul(std::move(token),
                                                 std::move(weight),
                                                 std::move(firstTokenOffset),
                                                 std::move(tokenIndex),
                                                 std::move(tokenKs),
                                                 std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    /// @brief Adds a mixture-of-experts grouped matmul backward node.
    std::shared_ptr<Tensor_attributes>
        moe_grouped_matmul_bwd(std::shared_ptr<Tensor_attributes> dOutput,
                               std::shared_ptr<Tensor_attributes> token,
                               std::shared_ptr<Tensor_attributes> firstTokenOffset,
                               Moe_grouped_matmul_bwd_attributes attributes)
    {
        auto dweight = _graph->moe_grouped_matmul_bwd(std::move(dOutput),
                                                      std::move(token),
                                                      std::move(firstTokenOffset),
                                                      std::move(attributes));
        _mode = Mode::Native;
        return dweight;
    }

    // Nodes with no hipDNN engine yet.

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(bn_finalize,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const BN_finalize_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 6>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(genstats,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const Genstats_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 2>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(dbn_weight,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const DBN_weight_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 5>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(instancenorm,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const Instancenorm_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 3>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(instancenorm_backward,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const Instancenorm_backward_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 3>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(adalayernorm,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const AdaLayernorm_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 3>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(adalayernorm_backward,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const AdaLayernorm_backward_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 3>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(rng,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const Rng_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(reshape,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const Reshape_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(transpose,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const Transpose_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(rope,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const RoPE_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(rope_backward,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const RoPE_backward_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    /// @brief Unsupported (FP8): records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(sdpa_fp8,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const SDPA_fp8_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 4>)

    /// @brief Unsupported (MXFP8): records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(sdpa_fp8,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const SDPA_fp8_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 3>)

    /// @brief Unsupported (FP8): records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(sdpa_fp8_backward,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const SDPA_fp8_backward_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 7>)

    /// @brief Unsupported (MXFP8): records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(sdpa_fp8_backward,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const SDPA_fp8_backward_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 6>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(diagonal_band_mask,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const DiagonalBandMask_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(slice,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const Slice_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    /// @brief Unsupported: records GRAPH_NOT_SUPPORTED, reported by the next validate() or
    /// build_operation_graph().
    HIPDNN_CUDNN_SHIM_FAIL_NODE(concatenate,
                                (const std::vector<std::shared_ptr<Tensor_attributes>>&,
                                 const Concatenate_attributes&),
                                std::shared_ptr<Tensor_attributes>)

#ifdef HIPDNN_ENABLE_SDPA
    /// @brief Adds an SDPA forward node; an unset mma_core_mode defaults to HALF.
    std::array<std::shared_ptr<Tensor_attributes>, 2> sdpa(std::shared_ptr<Tensor_attributes> q,
                                                           std::shared_ptr<Tensor_attributes> k,
                                                           std::shared_ptr<Tensor_attributes> v,
                                                           SDPA_attributes attributes)
    {
        // hipDNN leaves mma_core_mode NOT_SET; apply cuDNN FE's HALF default.
        if(attributes.mma_core_mode == DataType_t::NOT_SET)
        {
            attributes.set_mma_core_mode(DataType_t::HALF);
        }
        auto outputs
            = _graph->sdpa(std::move(q), std::move(k), std::move(v), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    /// @brief Adds an SDPA backward node.
    std::array<std::shared_ptr<Tensor_attributes>, 3>
        sdpa_backward(std::shared_ptr<Tensor_attributes> q,
                      std::shared_ptr<Tensor_attributes> k,
                      std::shared_ptr<Tensor_attributes> v,
                      std::shared_ptr<Tensor_attributes> o,
                      std::shared_ptr<Tensor_attributes> dO,
                      std::shared_ptr<Tensor_attributes> stats,
                      SDPA_backward_attributes attributes)
    {
        auto outputs = _graph->sdpa_backward(std::move(q),
                                             std::move(k),
                                             std::move(v),
                                             std::move(o),
                                             std::move(dO),
                                             std::move(stats),
                                             std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }
#endif // HIPDNN_ENABLE_SDPA

    /// @brief Executes the active plan with tensor-keyed device pointers.
    error_t
        execute(cudnnHandle_t handle,
                std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensorToPointerMap,
                void* workspace) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph->execute(handle, tensorToPointerMap, workspace);
    }

    /// @brief Executes the active plan with UID-keyed device pointers.
    error_t execute(cudnnHandle_t handle,
                    std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                    void* workspace) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph->execute(handle, tensorUidToPointerMap, workspace);
    }

    /// @brief Executes the active plan with runtime shape overrides.
    /// @note Without HIPDNN_ENABLE_SDPA, non-empty overrides return INVALID_VALUE.
    error_t execute(cudnnHandle_t handle,
                    std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                    void* workspace,
                    const std::vector<int64_t>& overrideUids,
                    const std::vector<std::vector<int64_t>>& overrideShapes,
                    const std::vector<std::vector<int64_t>>& overrideStrides) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();

#ifdef HIPDNN_ENABLE_SDPA
        return _graph->execute(handle,
                               tensorUidToPointerMap,
                               workspace,
                               overrideUids,
                               overrideShapes,
                               overrideStrides);
#else
        if(overrideUids.empty() && overrideShapes.empty() && overrideStrides.empty())
        {
            return _graph->execute(handle, tensorUidToPointerMap, workspace);
        }
        return {error_code_t::INVALID_VALUE,
                "Runtime shape override execute is unavailable in this build"};
#endif
    }

    /// @brief Unsupported: returns INVALID_VALUE (no UID mapping for a flat pointer array).
    error_t execute(cudnnHandle_t handle, void** sortedUserPtrs, int nUser, void* workspace) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        static_cast<void>(handle);
        static_cast<void>(sortedUserPtrs);
        static_cast<void>(nUser);
        static_cast<void>(workspace);
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return {error_code_t::INVALID_VALUE,
                "Flat pointer-array execute is unsupported by this shim"};
    }

    /// @brief Executes the plan at `planIndex`.
    error_t execute_plan_at_index(cudnnHandle_t handle,
                                  std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                                  void* workspace,
                                  int64_t planIndex) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph->execute_plan_at_index(handle, tensorUidToPointerMap, workspace, planIndex);
    }

    /// @brief Executes the plan at `planIndex`; INVALID_VALUE if any tensor has no UID.
    error_t execute_plan_at_index(
        cudnnHandle_t handle,
        std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensorMap,
        void* workspace,
        int64_t planIndex) const
    {
        std::unordered_map<int64_t, void*> uidMap;
        uidMap.reserve(tensorMap.size());
        for(const auto& [tensor, pointer] : tensorMap)
        {
            if(!tensor || !tensor->has_uid())
            {
                return {error_code_t::INVALID_VALUE,
                        "Tensor-keyed execute_plan_at_index requires every tensor to have a UID"};
            }
            uidMap.emplace(tensor->get_uid(), pointer);
        }
        return execute_plan_at_index(handle, uidMap, workspace, planIndex);
    }

    /// @brief Autotunes the built plans; `userImpl` is ignored.
    error_t autotune(cudnnHandle_t handle,
                     std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                     void* workspace,
                     void* userImpl = nullptr)
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph->autotune(handle, tensorUidToPointerMap, workspace, userImpl);
    }

    /// @brief Autotunes the built plans; `userImpl` is ignored.
    error_t autotune(cudnnHandle_t handle,
                     std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensorMap,
                     void* workspace,
                     void* userImpl = nullptr)
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph->autotune(handle, tensorMap, workspace, userImpl);
    }

    /// @brief Executes plan 0 once with the given buffers.
    /// @note Upstream takes only a handle; this shim needs the buffers to run the plan.
    // NOLINTNEXTLINE(readability-make-member-function-const)
    error_t warmup(cudnnHandle_t handle,
                   std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                   void* workspace)
    {
        return execute_plan_at_index(handle, tensorUidToPointerMap, workspace, 0);
    }

    /// @brief Executes plan 0 once with the given buffers.
    /// @note Upstream takes only a handle; this shim needs the buffers to run the plan.
    // NOLINTNEXTLINE(readability-make-member-function-const)
    error_t warmup(cudnnHandle_t handle,
                   std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensorMap,
                   void* workspace)
    {
        return execute_plan_at_index(handle, tensorMap, workspace, 0);
    }

    /// @brief Unsupported: returns GRAPH_NOT_SUPPORTED.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    error_t populate_cuda_graph(cudnnHandle_t handle,
                                std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                                void* workspace,
                                cudaGraph_t cudnnCudaGraph) const
    {
        static_cast<void>(handle);
        static_cast<void>(tensorUidToPointerMap);
        static_cast<void>(workspace);
        static_cast<void>(cudnnCudaGraph);
        return {error_code_t::GRAPH_NOT_SUPPORTED,
                "populate_cuda_graph is unsupported by this shim"};
    }

    /// @brief Unsupported: returns GRAPH_NOT_SUPPORTED.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    error_t populate_cuda_graph(
        cudnnHandle_t handle,
        std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensorMap,
        void* workspace,
        cudaGraph_t cudnnCudaGraph) const
    {
        static_cast<void>(handle);
        static_cast<void>(tensorMap);
        static_cast<void>(workspace);
        static_cast<void>(cudnnCudaGraph);
        return {error_code_t::GRAPH_NOT_SUPPORTED,
                "populate_cuda_graph is unsupported by this shim"};
    }

    /// @brief Unsupported: returns GRAPH_NOT_SUPPORTED.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    error_t update_cuda_graph(cudnnHandle_t handle,
                              std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                              void* workspace,
                              cudaGraph_t cudnnCudaGraph) const
    {
        static_cast<void>(handle);
        static_cast<void>(tensorUidToPointerMap);
        static_cast<void>(workspace);
        static_cast<void>(cudnnCudaGraph);
        return {error_code_t::GRAPH_NOT_SUPPORTED, "update_cuda_graph is unsupported by this shim"};
    }

    /// @brief Unsupported: returns GRAPH_NOT_SUPPORTED.
    // NOLINTBEGIN(readability-convert-member-functions-to-static)
    error_t
        update_cuda_graph(cudnnHandle_t handle,
                          std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensorMap,
                          void* workspace,
                          cudaGraph_t cudnnCudaGraph) const
    {
        static_cast<void>(handle);
        static_cast<void>(tensorMap);
        static_cast<void>(workspace);
        static_cast<void>(cudnnCudaGraph);
        return {error_code_t::GRAPH_NOT_SUPPORTED, "update_cuda_graph is unsupported by this shim"};
    }
    // NOLINTEND(readability-convert-member-functions-to-static)

    /// @brief Returns the active plan's workspace size; 0 for an empty graph.
    error_t get_workspace_size(int64_t& workspaceSize) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        if(!hasOperationGraphState())
        {
            if(stageAtLeast(Stage::OpGraphBuilt))
            {
                workspaceSize = 0;
                return {};
            }
            return noExecutionPlanError();
        }

        return _graph->get_workspace_size(workspaceSize);
    }

    /// @brief Returns the workspace size, or 0 on failure (logged).
    /// @note Use the error_t overload to tell a failure from a zero-size workspace.
    int64_t get_workspace_size() const
    {
        int64_t workspaceSize = 0;
        auto err = get_workspace_size(workspaceSize);
        if(err.is_bad())
        {
            CUDNN_FE_LOG_LABEL("ERROR: Querying workspace failed: " << err.get_message());
            return 0;
        }
        return workspaceSize;
    }

    /// @brief Workspace query; non-empty shape overrides return GRAPH_NOT_SUPPORTED.
    error_t get_workspace_size(cudnnHandle_t handle,
                               int64_t& workspaceSize,
                               const std::vector<int64_t>& overrideUids,
                               const std::vector<std::vector<int64_t>>& overrideShapes,
                               const std::vector<std::vector<int64_t>>& overrideStrides) const
    {
        static_cast<void>(handle);
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        if(overrideUids.empty() && overrideShapes.empty() && overrideStrides.empty())
        {
            return get_workspace_size(workspaceSize);
        }
        return {error_code_t::GRAPH_NOT_SUPPORTED,
                "Runtime shape override workspace query is unsupported by this shim"};
    }

    /// @brief Workspace query, or 0 on failure (logged); non-empty overrides fail.
    int64_t get_workspace_size(cudnnHandle_t handle,
                               const std::vector<int64_t>& overrideUids,
                               const std::vector<std::vector<int64_t>>& overrideShapes,
                               const std::vector<std::vector<int64_t>>& overrideStrides) const
    {
        int64_t workspaceSize = 0;
        auto err = get_workspace_size(
            handle, workspaceSize, overrideUids, overrideShapes, overrideStrides);
        if(err.is_bad())
        {
            CUDNN_FE_LOG_LABEL("ERROR: Querying workspace failed: " << err.get_message());
            return 0;
        }
        return workspaceSize;
    }

    /// @brief Returns the workspace autotune() needs; 0 for an empty graph.
    int64_t get_autotune_workspace_size() const
    {
        if(!hasOperationGraphState())
        {
            return 0;
        }
        return _graph->get_autotune_workspace_size();
    }

    /// @brief Returns the lowered graph as hipDNN JSON, or "{}" before build_operation_graph().
    std::string print() const
    {
        if(!hasOperationGraphState())
        {
            return "{}";
        }
        // as_const: the non-const native to_json() lowers the graph as a side effect.
        auto [text, err] = std::as_const(*_graph).to_json();
        if(err.is_bad())
        {
            CUDNN_FE_LOG_LABEL("ERROR: Serializing graph to JSON failed: " << err.get_message());
            return "{}";
        }
        return text;
    }

    /// @brief Serializes the graph without lowering it; INVALID_VALUE if it has not been lowered.
    error_t serialize(std::vector<uint8_t>& data) const
    {
        if(!hasOperationGraphState())
        {
            return serializeWithoutOperationGraphError();
        }
        // as_const: see print().
        return std::as_const(*_graph).serialize(data);
    }

    /// @brief Validates, lowers if needed, and serializes the graph.
    error_t serialize(std::vector<uint8_t>& data)
    {
        CHECK_CUDNN_FRONTEND_ERROR(validate());
        if(!hasOperationGraphState())
        {
            return serializeWithoutOperationGraphError();
        }
        // Non-const: keep the native auto-lowering overload, so serializing a
        // described-but-unbuilt graph still works.
        return _graph->serialize(data);
    }

    /// @brief Restores a serialized graph, and its plan if the blob carries one.
    /// @note `enforcePrecompiled` is ignored; upstream has no such parameter.
    error_t deserialize(cudnnHandle_t handle,
                        const std::vector<uint8_t>& data,
                        bool enforcePrecompiled = false)
    {
        static_cast<void>(enforcePrecompiled);
        auto err = _graph->deserialize(handle, data);
        if(err.is_good())
        {
            clearWrapperGraphState();
            _mode = Mode::Native;
            // A handle-bearing deserialize installs a plan only if the blob carried one;
            // otherwise the graph is finalized but planless.
            _stage
                = _graph->get_execution_plan_count() > 0 ? Stage::PlansBuilt : Stage::OpGraphBuilt;
        }
        return err;
    }

    /// @brief hipDNN extension: restores a serialized graph without a handle.
    /// @note `enforcePrecompiled` is ignored.
    error_t deserialize(const std::vector<uint8_t>& data, bool enforcePrecompiled = false)
    {
        static_cast<void>(enforcePrecompiled);
        auto err = _graph->deserialize(data);
        if(err.is_good())
        {
            clearWrapperGraphState();
            _mode = Mode::Native;
        }
        return err;
    }

private:
    // Graph lifecycle as one source discriminator + one monotonic stage, instead
    // of a set of interdependent booleans. Mode::Empty is a node-less graph the
    // shim handles locally (deviceless build, workspace 0); Mode::Native forwards
    // to the wrapped hipDNN graph. Stage advances Described -> OpGraphBuilt ->
    // PlansCreated -> PlansBuilt; a re-build resets it to an earlier stage.
    enum class Mode
    {
        Empty,
        Native
    };

    enum class Stage
    {
        Described,
        OpGraphBuilt,
        PlansCreated,
        PlansBuilt
    };

    // shared_ptr, not a value: the native graph is move-only (its backend
    // descriptors are unique_ptr), so sharing it is what makes the wrapper
    // copyable. Never null.
    std::shared_ptr<hipdnn_frontend::graph::Graph> _graph
        = std::make_shared<hipdnn_frontend::graph::Graph>();
    std::vector<std::shared_ptr<Tensor_attributes>> _ownedTensors;
    std::optional<int64_t> _maxWorkspaceAllowed;
    std::vector<int64_t> _barredEngineIndices;
    std::vector<HeurMode_t> _planCreationModes = {HeurMode_t::FALLBACK};
    std::vector<std::string> _barredEngineNames;
    std::vector<int64_t> _engineIndexToNativeEngineId;
    std::vector<BehaviorNote_t> _selectedBehaviorNotes;
    std::vector<BehaviorNote_t> _deselectedBehaviorNotes;
    bool _requireDeterministic = false;
    std::unordered_set<int64_t> _shimBarredEngineIds;

    Mode _mode = Mode::Empty;
    Stage _stage = Stage::Described;

    bool hasOperationGraphState() const
    {
        return _mode == Mode::Native;
    }

    bool stageAtLeast(Stage stage) const
    {
        return static_cast<int>(_stage) >= static_cast<int>(stage);
    }

    // Engine indices are a property of the operation graph, as in cuDNN: they always follow
    // the FALLBACK ranking, whatever heuristic modes created the plans, so get_engine_count,
    // get_knobs_for_engine, create_execution_plan and deselect_engines agree on every index.
    error_t refreshEngineIndexMap()
    {
        _engineIndexToNativeEngineId.clear();
        HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH();
        return _graph->get_ranked_engine_ids(_engineIndexToNativeEngineId, {HeurMode_t::FALLBACK});
    }

    error_t mapEngineIndex(int64_t engineIndex, int64_t& nativeEngineId)
    {
        CHECK_CUDNN_FRONTEND_ERROR(refreshEngineIndexMap());
        if(engineIndex < 0
           || static_cast<size_t>(engineIndex) >= _engineIndexToNativeEngineId.size())
        {
            return {error_code_t::INVALID_VALUE,
                    "Engine index " + std::to_string(engineIndex)
                        + " is out of bounds for this operation graph (have "
                        + std::to_string(_engineIndexToNativeEngineId.size()) + " engines)"};
        }

        nativeEngineId = _engineIndexToNativeEngineId[static_cast<size_t>(engineIndex)];
        return {};
    }

    error_t mapEngineIndices(const std::vector<int64_t>& engineIndices,
                             std::vector<int64_t>& nativeEngineIds)
    {
        nativeEngineIds.clear();
        nativeEngineIds.reserve(engineIndices.size());
        for(const auto engineIndex : engineIndices)
        {
            int64_t nativeEngineId = -1;
            CHECK_CUDNN_FRONTEND_ERROR(mapEngineIndex(engineIndex, nativeEngineId));
            nativeEngineIds.push_back(nativeEngineId);
        }
        return {};
    }

    // Filters set after plan creation take effect here, on the next
    // check_support(), build_plans() or build_plan_at_index().
    error_t applyPendingFiltersForCreatedPlans()
    {
        if(!stageAtLeast(Stage::PlansCreated))
        {
            return {};
        }
        return applyPendingPlanFilters(_planCreationModes);
    }

    error_t applyPendingPlanFilters(const std::vector<HeurMode_t>& modes = {HeurMode_t::FALLBACK})
    {
        // create_execution_plans() clears the native graph's filter state, so the
        // shim's record of what it barred is rebuilt on every pass rather than
        // accumulated.
        _shimBarredEngineIds.clear();

        if(_maxWorkspaceAllowed.has_value())
        {
            _graph->deselect_workspace_greater_than(*_maxWorkspaceAllowed);
        }
        if(!_barredEngineNames.empty())
        {
            _graph->deselect_engines(_barredEngineNames);
            for(const auto& name : _barredEngineNames)
            {
                _shimBarredEngineIds.insert(hipdnn_data_sdk::utilities::engineNameOrIdToId(name));
            }
        }
        if(!_barredEngineIndices.empty())
        {
            std::vector<int64_t> nativeEngineIds;
            CHECK_CUDNN_FRONTEND_ERROR(mapEngineIndices(_barredEngineIndices, nativeEngineIds));
            _graph->deselect_engines(nativeEngineIds);
            _shimBarredEngineIds.insert(nativeEngineIds.begin(), nativeEngineIds.end());
        }
        CHECK_CUDNN_FRONTEND_ERROR(applyEngineMetadataFilters(modes));

        if(_shimBarredEngineIds.empty() || _graph->get_execution_plan_count() == 0)
        {
            return {};
        }
        int64_t survivingIndex = -1;
        return findTopSurvivingPlanIndex(survivingIndex);
    }

    // Behavior-note and determinism filters bar engines by per-engine metadata.
    error_t applyEngineMetadataFilters(const std::vector<HeurMode_t>& modes)
    {
        const bool filterBehavior
            = !_selectedBehaviorNotes.empty() || !_deselectedBehaviorNotes.empty();
        if(!filterBehavior && !_requireDeterministic)
        {
            return {};
        }
        HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH();

        std::vector<int64_t> engineIds;
        CHECK_CUDNN_FRONTEND_ERROR(_graph->get_ranked_engine_ids(engineIds, modes));

        std::vector<int64_t> enginesToBar;
        enginesToBar.reserve(engineIds.size());
        for(const auto engineId : engineIds)
        {
            if(isShimBarred(engineId))
            {
                continue;
            }
            if(_requireDeterministic && !cudnn_frontend::detail::isDeterminismClaimed(engineId))
            {
                enginesToBar.push_back(engineId);
                continue;
            }
            if(filterBehavior)
            {
                std::vector<BehaviorNote_t> engineNotes;
                CHECK_CUDNN_FRONTEND_ERROR(
                    _graph->get_behavior_notes_for_engine(engineId, engineNotes));
                if(!cudnn_frontend::detail::behaviorNotesMatch(
                       _selectedBehaviorNotes, _deselectedBehaviorNotes, engineNotes))
                {
                    enginesToBar.push_back(engineId);
                }
            }
        }

        if(!enginesToBar.empty())
        {
            _graph->deselect_engines(enginesToBar);
            _shimBarredEngineIds.insert(enginesToBar.begin(), enginesToBar.end());
        }
        return {};
    }

    bool isShimBarred(int64_t engineId) const
    {
        return _shimBarredEngineIds.count(engineId) != 0;
    }

    // Survival is judged against everything this shim barred, so an engine
    // excluded by deselect_engines cannot count as a survivor of a metadata
    // filter. Plan order is not guaranteed to match ranked-engine order, so each
    // plan's engine is resolved through its own name.
    error_t findTopSurvivingPlanIndex(int64_t& index) const
    {
        const int64_t planCount = _graph->get_execution_plan_count();
        for(index = 0; index < planCount; ++index)
        {
            std::string planName;
            if(_graph->get_plan_name_at_index(index, planName).is_bad())
            {
                continue;
            }
            // Plugin engines come back as their hex id rather than a registered
            // name; engineNameOrIdToId resolves both forms.
            if(!isShimBarred(hipdnn_data_sdk::utilities::engineNameOrIdToId(planName)))
            {
                return {};
            }
        }

        index = -1;
        return {error_code_t::GRAPH_NOT_SUPPORTED,
                "No created hipDNN execution plan survives the plan filters ("
                    + activePlanFilterNames() + ")"};
    }

    std::string activePlanFilterNames() const
    {
        std::string names;
        const auto append = [&names](const char* name) {
            if(!names.empty())
            {
                names += ", ";
            }
            names += name;
        };
        if(!_barredEngineNames.empty() || !_barredEngineIndices.empty())
        {
            append("deselect_engines");
        }
        if(!_selectedBehaviorNotes.empty() || !_deselectedBehaviorNotes.empty())
        {
            append("behavior notes");
        }
        if(_requireDeterministic)
        {
            append("deselect_numeric_notes(NONDETERMINISTIC)");
        }
        return names;
    }

    // onMap returns false when this setter cannot honor the mapped note; that fails closed.
    template <typename Note, typename OnMap>
    void applyNote(const char* method,
                   Note note,
                   cudnn_frontend::detail::NoteAction action,
                   OnMap&& onMap)
    {
        switch(action)
        {
        case cudnn_frontend::detail::NoteAction::NO_OP:
            return;
        case cudnn_frontend::detail::NoteAction::WARN:
            HIPDNN_FE_LOG_WARN("[cudnn_frontend] "
                               << cudnn_frontend::detail::noteMessage(method, note, action));
            return;
        case cudnn_frontend::detail::NoteAction::MAP:
            if(std::forward<OnMap>(onMap)(note))
            {
                CUDNN_FE_LOG_LABEL(cudnn_frontend::detail::noteMessage(method, note, action));
                return;
            }
            break;
        case cudnn_frontend::detail::NoteAction::RECORD_ERROR:
        default: // unknown actions fail closed
            break;
        }
        recordError(error_code_t::GRAPH_NOT_SUPPORTED,
                    cudnn_frontend::detail::noteMessage(
                        method, note, cudnn_frontend::detail::NoteAction::RECORD_ERROR));
    }

    template <typename T>
    std::shared_ptr<Tensor_attributes> scalarTensor(const T& scalar, ScalarType scalarType)
    {
        auto tensorPtr = std::make_shared<Tensor_attributes>(scalar, scalarType);
        _ownedTensors.emplace_back(tensorPtr);
        return tensorPtr;
    }

    error_t validateOwnedTensors()
    {
        std::unordered_map<int64_t, const Tensor_attributes*> uidMap;
        for(const auto& tensorPtr : _ownedTensors)
        {
            if(!tensorPtr)
            {
                return {error_code_t::INVALID_VALUE, "Owned tensor is null"};
            }

            if(tensorPtr->has_uid())
            {
                const auto uid = tensorPtr->get_uid();
                if(uidMap.find(uid) != uidMap.end())
                {
                    return {error_code_t::INVALID_VALUE, "Duplicate tensor UID in graph"};
                }
                uidMap.emplace(uid, tensorPtr.get());
            }

            hipdnn_frontend::graph::GraphAttributes context;
            context.set_name(_graph->get_name())
                .set_compute_data_type(_graph->get_compute_data_type())
                .set_intermediate_data_type(_graph->get_intermediate_data_type())
                .set_io_data_type(_graph->get_io_data_type());
            tensorPtr->fill_from_context(context);
            CHECK_CUDNN_FRONTEND_ERROR(tensorPtr->validate());
        }
        return {};
    }

    std::shared_ptr<Tensor_attributes> findOwnedTensorByUid(int64_t uid) const
    {
        for(const auto& tensorPtr : _ownedTensors)
        {
            if(tensorPtr && tensorPtr->has_uid() && tensorPtr->get_uid() == uid)
            {
                return tensorPtr;
            }
        }
        return {};
    }

    // A and B select hipDNN's ModeA/ModeB prediction policies; OPENSOURCE has no hipDNN
    // counterpart and runs fallback selection.
    static void warnUnhonoredHeuristicModes(const std::vector<HeurMode_t>& modes)
    {
        for(const auto mode : modes)
        {
            if(mode == HeurMode_t::OPENSOURCE)
            {
                HIPDNN_FE_LOG_WARN("[cudnn_frontend] cuDNN heuristic mode "
                                   << hipdnn_frontend::to_string(mode)
                                   << " is accepted but not honored; hipDNN uses fallback "
                                      "selection. Plan choice may differ from cuDNN.");
            }
        }
    }

    static error_t unsupportedDevicelessBuildError()
    {
        return {error_code_t::INVALID_VALUE,
                "Deviceless build is unsupported for non-empty graphs by this shim"};
    }

    static error_t noExecutionPlanError()
    {
        return {error_code_t::INVALID_VALUE, "Graph has no compiled execution plan"};
    }

    static error_t serializeWithoutOperationGraphError()
    {
        return {error_code_t::INVALID_VALUE,
                "Serializing a graph without a compiled operation graph is unsupported"};
    }

    void clearWrapperGraphState()
    {
        _ownedTensors.clear();
        _maxWorkspaceAllowed.reset();
        _barredEngineIndices.clear();
        _planCreationModes = {HeurMode_t::FALLBACK};
        _barredEngineNames.clear();
        _engineIndexToNativeEngineId.clear();
        _selectedBehaviorNotes.clear();
        _deselectedBehaviorNotes.clear();
        _requireDeterministic = false;
        _shimBarredEngineIds.clear();
        _recordedError.reset();
        _mode = Mode::Empty;
        _stage = Stage::Described;
    }
};

/// @brief Writes print() to `os`.
inline std::ostream& operator<<(std::ostream& os, const Graph& graph)
{
    return os << graph.print();
}

// NOLINTEND(readability-identifier-naming)

#undef HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH
#undef HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH
#undef HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
