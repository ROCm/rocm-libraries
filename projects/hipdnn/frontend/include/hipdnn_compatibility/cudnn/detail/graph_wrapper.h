// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend
// (include/cudnn_frontend/graph_interface.h), used under the MIT license.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
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
#include <hipdnn_compatibility/cudnn/detail/error_recorder.h>
#include <hipdnn_compatibility/cudnn/detail/knob_wrapper.h>
#include <hipdnn_compatibility/cudnn/detail/node_wrappers/unsupported_nodes.h>
#include <hipdnn_frontend/Graph.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
// Node signatures in this file mirror upstream cudnn-frontend at the version
// pinned in cudnn_frontend_version.h (its graph_interface.h and
// node_interface.h). Upstream grows node arity between minor releases, so
// bumping the pin without re-diffing the signatures silently breaks source
// compatibility for hipified consumers; this assert forces that re-diff.
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff every Graph node signature in this "
              "file against upstream graph_interface.h / node_interface.h at the new tag, "
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

class Graph : public ErrorRecorder<Graph>
{
public:
    Graph() = default;
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    error_t validate()
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        CHECK_CUDNN_FRONTEND_ERROR(validateOwnedTensors());
        if(hasOperationGraphState())
        {
            return _graph.validate();
        }

        return {};
    }

    error_t build_operation_graph(cudnnHandle_t handle)
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        CHECK_CUDNN_FRONTEND_ERROR(validateOwnedTensors());
        if(!hasOperationGraphState())
        {
            _stage = Stage::OpGraphBuilt;
            return {};
        }

        auto err = _graph.build_operation_graph(handle);
        if(err.is_good())
        {
            _stage = Stage::OpGraphBuilt;
        }
        return err;
    }

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

    error_t create_execution_plans(const std::vector<HeurMode_t>& modes = {HeurMode_t::FALLBACK})
    {
        for(const auto mode : modes)
        {
            if(mode != HeurMode_t::FALLBACK)
            {
                HIPDNN_FE_LOG_WARN("[cudnn_frontend] cuDNN heuristic mode "
                                   << hipdnn_frontend::to_string(mode)
                                   << " is accepted but not honored; hipDNN uses fallback "
                                      "selection. Plan choice may differ from cuDNN.");
            }
        }

        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH();

        auto err = _graph.create_execution_plans(modes);
        if(err.is_bad())
        {
            return err;
        }

        _planCreationModes = modes;
        CHECK_CUDNN_FRONTEND_ERROR(applyPendingPlanFilters(_planCreationModes));
        _stage = Stage::PlansCreated;
        return err;
    }

    error_t check_support()
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH();

        return _graph.check_support();
    }

    error_t check_support(cudnnHandle_t handle)
    {
        static_cast<void>(handle);
        return check_support();
    }

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

        // Native build_plans(HEURISTICS_CHOICE) only ever inspects the active
        // plan, so a filter that bars the top-ranked engine fails the build
        // outright. cuDNN instead narrows the candidate set, so retarget onto the
        // top-ranked survivor first.
        if(policy == BuildPlanPolicy_t::HEURISTICS_CHOICE)
        {
            int64_t survivingIndex = -1;
            CHECK_CUDNN_FRONTEND_ERROR(findPlanIndexIfActiveBarred(survivingIndex));
            if(survivingIndex >= 0)
            {
                CHECK_CUDNN_FRONTEND_ERROR(_graph.build_plan_at_index(survivingIndex));
                _stage = Stage::PlansBuilt;
                return {};
            }
        }

        auto err = _graph.build_plans(policy);
        if(err.is_good())
        {
            _stage = Stage::PlansBuilt;
        }
        return err;
    }

    error_t build_plans(const cudnnHandle_t& handle,
                        BuildPlanPolicy_t policy = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                        bool doMultithreadedBuilds = false)
    {
        static_cast<void>(handle);
        return build_plans(policy, doMultithreadedBuilds);
    }

    error_t build_plan_at_index(int64_t index)
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();

        if(!stageAtLeast(Stage::PlansCreated))
        {
            CHECK_CUDNN_FRONTEND_ERROR(create_execution_plans());
        }
        CHECK_CUDNN_FRONTEND_ERROR(applyPendingFiltersForCreatedPlans());
        CHECK_CUDNN_FRONTEND_ERROR(_graph.build_plan_at_index(index));
        _stage = Stage::PlansBuilt;
        return {};
    }

    error_t build_plan_at_index(const cudnnHandle_t& handle, int64_t index)
    {
        static_cast<void>(handle);
        return build_plan_at_index(index);
    }

    int64_t get_execution_plan_count() const
    {
        if(hasOperationGraphState())
        {
            return _graph.get_execution_plan_count();
        }
        return stageAtLeast(Stage::PlansCreated) ? 1 : 0;
    }

    error_t get_engine_count(int64_t& count)
    {
        count = 0;
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH();

        CHECK_CUDNN_FRONTEND_ERROR(refreshEngineIndexMap());
        count = static_cast<int64_t>(_engineIndexToNativeEngineId.size());
        return {};
    }

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
        CHECK_CUDNN_FRONTEND_ERROR(_graph.get_knobs_for_engine(nativeEngineId, nativeKnobs));
        hipdnn_frontend::compatibility::cudnn_frontend::detail::projectNativeKnobs(nativeKnobs,
                                                                                   knobs);
        return {};
    }

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
            CHECK_CUDNN_FRONTEND_ERROR(_graph.get_knobs_for_engine(nativeEngineId, nativeKnobs));
            CHECK_CUDNN_FRONTEND_ERROR(
                hipdnn_frontend::compatibility::cudnn_frontend::detail::makeNativeKnobSettings(
                    knobs, nativeKnobs, settings));
        }

        CHECK_CUDNN_FRONTEND_ERROR(_graph.create_execution_plan_ext(nativeEngineId, settings));
        CHECK_CUDNN_FRONTEND_ERROR(applyPendingPlanFilters());
        _stage = Stage::PlansCreated;
        return {};
    }

    error_t get_plan_name(std::string& name) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph.get_plan_name(name);
    }

    error_t get_plan_name_at_index(int64_t index, std::string& name) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph.get_plan_name_at_index(index, name);
    }

    error_t get_workspace_size_plan_at_index(int64_t index, int64_t& workspaceSize) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph.get_workspace_size_plan_at_index(index, workspaceSize);
    }

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

    error_t get_behavior_notes(std::vector<BehaviorNote_t>& notes) const
    {
        notes.clear();
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();

        int64_t engineId = -1;
        CHECK_CUDNN_FRONTEND_ERROR(_graph.get_execution_plan_engine_id(engineId));
        return _graph.get_behavior_notes_for_engine(engineId, notes);
    }

    error_t get_behavior_notes_for_plan_at_index(int64_t index,
                                                 std::vector<BehaviorNote_t>& notes) const
    {
        notes.clear();
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();

        std::string planName;
        CHECK_CUDNN_FRONTEND_ERROR(_graph.get_plan_name_at_index(index, planName));
        if(!hipdnn_data_sdk::utilities::isEngineNameRegistered(planName))
        {
            return {error_code_t::INVALID_VALUE,
                    "Cannot resolve engine id for plan '" + planName
                        + "'; per-plan behavior-note query is unsupported for unknown engines"};
        }

        const int64_t engineId = hipdnn_data_sdk::utilities::engineNameToId(planName);
        return _graph.get_behavior_notes_for_engine(engineId, notes);
    }

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

    Graph& set_name(const std::string& name)
    {
        _graph.set_name(name);
        return *this;
    }

    const std::string& get_name() const
    {
        return _graph.get_name();
    }

    Graph& set_io_data_type(DataType_t type)
    {
        _graph.set_io_data_type(type);
        return *this;
    }

    DataType_t get_io_data_type() const
    {
        return _graph.get_io_data_type();
    }

    Graph& set_compute_data_type(DataType_t type)
    {
        _graph.set_compute_data_type(type);
        return *this;
    }

    DataType_t get_compute_data_type() const
    {
        return _graph.get_compute_data_type();
    }

    Graph& set_intermediate_data_type(DataType_t type)
    {
        _graph.set_intermediate_data_type(type);
        return *this;
    }

    DataType_t get_intermediate_data_type() const
    {
        return _graph.get_intermediate_data_type();
    }

#ifdef HIPDNN_ENABLE_SDPA
    // Native set_override_shape_enabled is SDPA-gated (see Graph.hpp); mirror
    // that gating here so the shim never calls a method the frontend omits.
    // Not part of cuDNN's graph::Graph surface, so nothing cuDNN-spelled relies
    // on it in a compat-only (SDPA-off) build.
    Graph& set_override_shape_enabled(bool isEnabled)
    {
        _graph.set_override_shape_enabled(isEnabled);
        return *this;
    }
#endif // HIPDNN_ENABLE_SDPA

    // Setter triage: hints that are safe to drop (dynamic-shape, kernel-cache)
    // log and continue; requests the shim cannot honor without changing results
    // (SM targeting, device properties) record an error that surfaces at the next
    // validate()/build. The asymmetry is deliberate — do not "normalize" one
    // group to the other.
    Graph& set_dynamic_shape_enabled(bool isEnabled)
    {
        static_cast<void>(isEnabled);
        CUDNN_FE_LOG_LABEL("Ignoring graph dynamic-shape hint; hipDNN has no graph-level setting");
        return *this;
    }

    Graph& set_kernel_cache(const std::shared_ptr<KernelCache>& cache)
    {
        static_cast<void>(cache);
        CUDNN_FE_LOG_LABEL("Ignoring graph kernel cache hint; hipDNN selects kernels internally");
        return *this;
    }

    // SM targeting / device properties are unsupported (see triage note above):
    // record so the next validate()/build surfaces the error.
    Graph& set_sm_count(int32_t count)
    {
        static_cast<void>(count);
        CUDNN_FE_LOG_LABEL("ERROR: Target SM count is unsupported by this shim");
        recordError(error_code_t::INVALID_VALUE, "Target SM count is unsupported by this shim");
        return *this;
    }

    Graph& set_sm_version(int32_t version)
    {
        static_cast<void>(version);
        CUDNN_FE_LOG_LABEL("ERROR: Target SM version is unsupported by this shim");
        recordError(error_code_t::INVALID_VALUE, "Target SM version is unsupported by this shim");
        return *this;
    }

    Graph& set_device_properties(const std::shared_ptr<const DeviceProperties>& deviceProperties)
    {
        static_cast<void>(deviceProperties);
        CUDNN_FE_LOG_LABEL("ERROR: Device properties are unsupported by this shim");
        recordError(error_code_t::INVALID_VALUE, "Device properties are unsupported by this shim");
        return *this;
    }

    // --- Plan-selection note filters ---------------------------------------
    //
    // Inline triage: advisory filters warn-and-ignore, while exclusions that
    // request a numerical guarantee hipDNN cannot prove record an error.

    Graph& select_numeric_notes(const std::vector<NumericalNote_t>& notes)
    {
        for(const auto note : notes)
        {
            if(note != NumericalNote_t::NOT_SET)
            {
                HIPDNN_FE_LOG_WARN("[cudnn_frontend] Ignoring select_numeric_notes("
                                   << hipdnn_frontend::to_string(note)
                                   << "); hipDNN exposes no per-plan numerical-note metadata.");
            }
        }
        return *this;
    }

    Graph& deselect_numeric_notes(const std::vector<NumericalNote_t>& notes)
    {
        for(const auto note : notes)
        {
            if(note == NumericalNote_t::NOT_SET)
            {
                continue;
            }

            if(note == NumericalNote_t::NONDETERMINISTIC
               || note == NumericalNote_t::REDUCED_PRECISION_REDUCTION)
            {
                recordError(error_code_t::GRAPH_NOT_SUPPORTED,
                            std::string{"deselect_numeric_notes("}
                                + hipdnn_frontend::to_string(note)
                                + ") requests a guarantee this shim cannot enforce; refusing to "
                                  "run rather than return a plan the caller excluded");
                continue;
            }

            HIPDNN_FE_LOG_WARN("[cudnn_frontend] Ignoring deselect_numeric_notes("
                               << hipdnn_frontend::to_string(note)
                               << "); hipDNN exposes no per-plan numerical-note metadata.");
        }
        return *this;
    }

    Graph& select_behavior_notes(const std::vector<BehaviorNote_t>& notes)
    {
        for(const auto note : notes)
        {
            if(note == BehaviorNote_t::NOT_SET)
            {
                continue;
            }
            if(hipdnn_frontend::isKnownBehaviorNote(note))
            {
                _selectedBehaviorNotes.push_back(note);
                HIPDNN_FE_LOG_WARN("[cudnn_frontend] select_behavior_notes("
                                   << hipdnn_frontend::to_string(note)
                                   << ") will filter hipDNN engines by behavior metadata.");
            }
            else
            {
                // Selecting a note no hipDNN engine can emit is unsatisfiable:
                // record an error rather than return a plan that ignores the
                // request. Deselecting the same note is a safe no-op (below).
                recordError(error_code_t::GRAPH_NOT_SUPPORTED,
                            "select_behavior_notes requested a behavior note that no hipDNN "
                            "engine reports; the request cannot be satisfied");
            }
        }
        return *this;
    }

    Graph& deselect_behavior_notes(const std::vector<BehaviorNote_t>& notes)
    {
        for(const auto note : notes)
        {
            if(note == BehaviorNote_t::NOT_SET)
            {
                continue;
            }
            if(hipdnn_frontend::isKnownBehaviorNote(note))
            {
                _deselectedBehaviorNotes.push_back(note);
                HIPDNN_FE_LOG_WARN("[cudnn_frontend] deselect_behavior_notes("
                                   << hipdnn_frontend::to_string(note)
                                   << ") will filter hipDNN engines by behavior metadata.");
            }
            else
            {
                HIPDNN_FE_LOG_WARN("[cudnn_frontend] Ignoring deselect_behavior_notes("
                                   << hipdnn_frontend::to_string(note)
                                   << "); hipDNN engines do not report this cuDNN behavior note.");
            }
        }
        return *this;
    }

    Graph& deselect_workspace_greater_than(int64_t workspace)
    {
        _maxWorkspaceAllowed = workspace;
        _graph.deselect_workspace_greater_than(workspace);
        return *this;
    }

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

    Graph& deselect_engines(const std::vector<std::string>& engineNames)
    {
        _barredEngineNames.insert(_barredEngineNames.end(), engineNames.begin(), engineNames.end());
        _graph.deselect_engines(engineNames);
        return *this;
    }

    Graph& deselect_engines(const std::vector<int64_t>& engineIndices)
    {
        // Store cuDNN-shaped indices only. Translation to native engine IDs needs
        // the ranked engine list, which does not exist until the operation graph
        // is built, so it is deferred to applyPendingPlanFilters() after build.
        _barredEngineIndices.insert(
            _barredEngineIndices.end(), engineIndices.begin(), engineIndices.end());
        return *this;
    }

    std::shared_ptr<Tensor_attributes> tensor(const Tensor_attributes& tensorAttributes)
    {
        auto tensorPtr = hipdnn_frontend::graph::Graph::tensor(tensorAttributes);
        _ownedTensors.emplace_back(tensorPtr);
        return tensorPtr;
    }

    std::shared_ptr<Tensor_attributes> tensor(const float& scalar, ScalarType scalarType)
    {
        return scalarTensor(scalar, scalarType);
    }

    std::shared_ptr<Tensor_attributes> tensor(const half& scalar, ScalarType scalarType)
    {
        return scalarTensor(scalar, scalarType);
    }

    std::shared_ptr<Tensor_attributes> tensor(const nv_bfloat16& scalar, ScalarType scalarType)
    {
        return scalarTensor(scalar, scalarType);
    }

    std::shared_ptr<Tensor_attributes> tensor(const int32_t& scalar, ScalarType scalarType)
    {
        return scalarTensor(scalar, scalarType);
    }

    std::shared_ptr<Tensor_attributes> tensor(const int64_t& scalar, ScalarType scalarType)
    {
        return scalarTensor(scalar, scalarType);
    }

    std::shared_ptr<Tensor_attributes> tensor(const double& scalar, ScalarType scalarType)
    {
        return scalarTensor(scalar, scalarType);
    }

    std::shared_ptr<Tensor_attributes>
        tensor_like(const std::shared_ptr<Tensor_attributes>& tensorAttributes,
                    const std::string& name = std::string{})
    {
        auto tensorPtr = hipdnn_frontend::graph::Graph::tensor_like(tensorAttributes, name);
        _ownedTensors.emplace_back(tensorPtr);
        return tensorPtr;
    }

    error_t query_tensor_attributes_of_uid(int64_t uid, Tensor_attributes& tensorAttributes) const
    {
        if(auto tensorPtr = findOwnedTensorByUid(uid))
        {
            tensorAttributes = *tensorPtr;
            return {};
        }

        if(hasOperationGraphState())
        {
            auto nativeTensors = _graph.getTensorsByUid();
            auto it = nativeTensors.find(uid);
            if(it != nativeTensors.end() && it->second)
            {
                tensorAttributes = *it->second;
                return {};
            }
        }

        return {error_code_t::INVALID_VALUE, "Tensor UID was not found"};
    }

    // --- Node-adding methods -----------------------------------------------
    //
    // Tier-1 nodes with a 1:1 hipDNN engine forward straight to the wrapped
    // graph and flip the graph into Native mode so the plan lifecycle runs
    // against hipDNN. Nodes take their *_attributes BY VALUE, matching cuDNN FE.
    // Tier-2 nodes with no hipDNN equivalent are stamped by
    // HIPDNN_CUDNN_SHIM_FAIL_NODE: they record GRAPH_NOT_SUPPORTED (surfaced at
    // the next validate()/build_operation_graph()) and return a live,
    // graph-registered placeholder tensor so the consumer's fluent chain (e.g.
    // ->set_output(...)) does not dereference null before the error surfaces.

    std::shared_ptr<Tensor_attributes> conv_fprop(std::shared_ptr<Tensor_attributes> x,
                                                  std::shared_ptr<Tensor_attributes> w,
                                                  Conv_fprop_attributes attributes)
    {
        auto output = _graph.conv_fprop(std::move(x), std::move(w), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    std::shared_ptr<Tensor_attributes> conv_dgrad(std::shared_ptr<Tensor_attributes> dy,
                                                  std::shared_ptr<Tensor_attributes> w,
                                                  Conv_dgrad_attributes attributes)
    {
        auto output = _graph.conv_dgrad(std::move(dy), std::move(w), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    std::shared_ptr<Tensor_attributes> conv_wgrad(std::shared_ptr<Tensor_attributes> dy,
                                                  std::shared_ptr<Tensor_attributes> x,
                                                  Conv_wgrad_attributes attributes)
    {
        auto output = _graph.conv_wgrad(std::move(dy), std::move(x), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 5>
        batchnorm(std::shared_ptr<Tensor_attributes> x,
                  std::shared_ptr<Tensor_attributes> scale,
                  std::shared_ptr<Tensor_attributes> bias,
                  Batchnorm_attributes attributes)
    {
        auto outputs = _graph.batchnorm(
            std::move(x), std::move(scale), std::move(bias), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 3>
        batchnorm_backward(std::shared_ptr<Tensor_attributes> dy,
                           std::shared_ptr<Tensor_attributes> x,
                           std::shared_ptr<Tensor_attributes> scale,
                           Batchnorm_backward_attributes attributes)
    {
        auto outputs = _graph.batchnorm_backward(
            std::move(dy), std::move(x), std::move(scale), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    std::shared_ptr<Tensor_attributes>
        batchnorm_inference(std::shared_ptr<Tensor_attributes> x,
                            std::shared_ptr<Tensor_attributes> mean,
                            std::shared_ptr<Tensor_attributes> invVariance,
                            std::shared_ptr<Tensor_attributes> scale,
                            std::shared_ptr<Tensor_attributes> bias,
                            Batchnorm_inference_attributes attributes)
    {
        auto output = _graph.batchnorm_inference(std::move(x),
                                                 std::move(mean),
                                                 std::move(invVariance),
                                                 std::move(scale),
                                                 std::move(bias),
                                                 std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 3>
        layernorm(std::shared_ptr<Tensor_attributes> x,
                  std::shared_ptr<Tensor_attributes> scale,
                  std::shared_ptr<Tensor_attributes> bias,
                  Layernorm_attributes attributes)
    {
        auto outputs = _graph.layernorm(
            std::move(x), std::move(scale), std::move(bias), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 3>
        layernorm_backward(std::shared_ptr<Tensor_attributes> dy,
                           std::shared_ptr<Tensor_attributes> x,
                           std::shared_ptr<Tensor_attributes> scale,
                           Layernorm_backward_attributes attributes)
    {
        auto outputs = _graph.layernorm_backward(
            std::move(dy), std::move(x), std::move(scale), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 2>
        rmsnorm(std::shared_ptr<Tensor_attributes> x,
                std::shared_ptr<Tensor_attributes> scale,
                Rmsnorm_attributes attributes)
    {
        auto outputs = _graph.rmsnorm(std::move(x), std::move(scale), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 3>
        rmsnorm_backward(std::shared_ptr<Tensor_attributes> dy,
                         std::shared_ptr<Tensor_attributes> x,
                         std::shared_ptr<Tensor_attributes> scale,
                         std::shared_ptr<Tensor_attributes> invVariance,
                         Rmsnorm_backward_attributes attributes)
    {
        auto outputs = _graph.rmsnorm_backward(std::move(dy),
                                               std::move(x),
                                               std::move(scale),
                                               std::move(invVariance),
                                               std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    std::shared_ptr<Tensor_attributes> matmul(std::shared_ptr<Tensor_attributes> a,
                                              std::shared_ptr<Tensor_attributes> b,
                                              Matmul_attributes attributes)
    {
        auto output = _graph.matmul(std::move(a), std::move(b), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes> a,
                                                 Pointwise_attributes attributes)
    {
        auto output = _graph.pointwise(std::move(a), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes> a,
                                                 std::shared_ptr<Tensor_attributes> b,
                                                 Pointwise_attributes attributes)
    {
        auto output = _graph.pointwise(std::move(a), std::move(b), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes> a,
                                                 std::shared_ptr<Tensor_attributes> b,
                                                 std::shared_ptr<Tensor_attributes> c,
                                                 Pointwise_attributes attributes)
    {
        auto output
            = _graph.pointwise(std::move(a), std::move(b), std::move(c), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    std::shared_ptr<Tensor_attributes> reduction(std::shared_ptr<Tensor_attributes> a,
                                                 Reduction_attributes attributes)
    {
        auto output = _graph.reduction(std::move(a), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 2> resample(std::shared_ptr<Tensor_attributes> x,
                                                               Resample_attributes attributes)
    {
        auto outputs = _graph.resample(std::move(x), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 2>
        block_scale_quantize(std::shared_ptr<Tensor_attributes> x,
                             Block_scale_quantize_attributes attributes)
    {
        auto outputs = _graph.block_scale_quantize(std::move(x), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    std::shared_ptr<Tensor_attributes>
        block_scale_dequantize(std::shared_ptr<Tensor_attributes> x,
                               std::shared_ptr<Tensor_attributes> scale,
                               Block_scale_dequantize_attributes attributes)
    {
        auto output
            = _graph.block_scale_dequantize(std::move(x), std::move(scale), std::move(attributes));
        _mode = Mode::Native;
        return output;
    }

    // --- Tier-2 fail-stub nodes (no hipDNN engine yet) ---------------------

    HIPDNN_CUDNN_SHIM_FAIL_NODE(bn_finalize,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const BN_finalize_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 6>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(genstats,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const Genstats_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 2>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(dbn_weight,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const DBN_weight_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 5>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(instancenorm,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const Instancenorm_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 3>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(instancenorm_backward,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const Instancenorm_backward_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 3>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(adalayernorm,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const AdaLayernorm_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 3>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(adalayernorm_backward,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const AdaLayernorm_backward_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 3>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(rng,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const Rng_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(reshape,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const Reshape_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(transpose,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const Transpose_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(rope,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const RoPE_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(rope_backward,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const RoPE_backward_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    // FP8 version
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

    // MXFP8 version
    HIPDNN_CUDNN_SHIM_FAIL_NODE(sdpa_fp8,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const SDPA_fp8_attributes&),
                                std::array<std::shared_ptr<Tensor_attributes>, 3>)

    // FP8 version
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

    // MXFP8 version
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

    HIPDNN_CUDNN_SHIM_FAIL_NODE(diagonal_band_mask,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const DiagonalBandMask_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(slice,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const Slice_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(concatenate,
                                (const std::vector<std::shared_ptr<Tensor_attributes>>&,
                                 const Concatenate_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(moe_grouped_matmul,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const Moe_grouped_matmul_attributes&),
                                std::shared_ptr<Tensor_attributes>)

    HIPDNN_CUDNN_SHIM_FAIL_NODE(moe_grouped_matmul_bwd,
                                (const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const std::shared_ptr<Tensor_attributes>&,
                                 const Moe_grouped_matmul_bwd_attributes&),
                                std::shared_ptr<Tensor_attributes>)

#ifdef HIPDNN_ENABLE_SDPA
    std::array<std::shared_ptr<Tensor_attributes>, 2> sdpa(std::shared_ptr<Tensor_attributes> q,
                                                           std::shared_ptr<Tensor_attributes> k,
                                                           std::shared_ptr<Tensor_attributes> v,
                                                           SDPA_attributes attributes)
    {
        // cuDNN's Graph::sdpa defaults mma_core_mode to HALF when unset; hipDNN
        // leaves it NOT_SET and omits the attribute, so replicate the default.
        if(attributes.mma_core_mode == DataType_t::NOT_SET)
        {
            attributes.set_mma_core_mode(DataType_t::HALF);
        }
        auto outputs = _graph.sdpa(std::move(q), std::move(k), std::move(v), std::move(attributes));
        _mode = Mode::Native;
        return outputs;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 3>
        sdpa_backward(std::shared_ptr<Tensor_attributes> q,
                      std::shared_ptr<Tensor_attributes> k,
                      std::shared_ptr<Tensor_attributes> v,
                      std::shared_ptr<Tensor_attributes> o,
                      std::shared_ptr<Tensor_attributes> dO,
                      std::shared_ptr<Tensor_attributes> stats,
                      SDPA_backward_attributes attributes)
    {
        auto outputs = _graph.sdpa_backward(std::move(q),
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

    error_t
        execute(cudnnHandle_t handle,
                std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensorToPointerMap,
                void* workspace) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph.execute(handle, tensorToPointerMap, workspace);
    }

    error_t execute(cudnnHandle_t handle,
                    std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                    void* workspace) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();

        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph.execute(handle, tensorUidToPointerMap, workspace);
    }

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
        return _graph.execute(handle,
                              tensorUidToPointerMap,
                              workspace,
                              overrideUids,
                              overrideShapes,
                              overrideStrides);
#else
        if(overrideUids.empty() && overrideShapes.empty() && overrideStrides.empty())
        {
            return _graph.execute(handle, tensorUidToPointerMap, workspace);
        }
        return {error_code_t::INVALID_VALUE,
                "Runtime shape override execute is unavailable in this build"};
#endif
    }

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

    error_t execute_plan_at_index(cudnnHandle_t handle,
                                  std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                                  void* workspace,
                                  int64_t planIndex) const
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph.execute_plan_at_index(handle, tensorUidToPointerMap, workspace, planIndex);
    }

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

    error_t autotune(cudnnHandle_t handle,
                     std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                     void* workspace,
                     void* userImpl = nullptr)
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph.autotune(handle, tensorUidToPointerMap, workspace, userImpl);
    }

    error_t autotune(cudnnHandle_t handle,
                     std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensorMap,
                     void* workspace,
                     void* userImpl = nullptr)
    {
        HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR();
        HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH();
        return _graph.autotune(handle, tensorMap, workspace, userImpl);
    }

    // NOLINTNEXTLINE(readability-make-member-function-const)
    error_t warmup(cudnnHandle_t handle,
                   std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                   void* workspace)
    {
        return execute_plan_at_index(handle, tensorUidToPointerMap, workspace, 0);
    }

    // NOLINTNEXTLINE(readability-make-member-function-const)
    error_t warmup(cudnnHandle_t handle,
                   std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensorMap,
                   void* workspace)
    {
        return execute_plan_at_index(handle, tensorMap, workspace, 0);
    }

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

        return _graph.get_workspace_size(workspaceSize);
    }

    // cuDNN keeps this fallible-to-0 overload for source compatibility. Note it
    // cannot distinguish a legitimate zero-workspace graph from a failed query:
    // both return 0 (the failure is logged). Prefer the error_t& overload when
    // the distinction matters.
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

    error_t serialize(std::vector<uint8_t>& data) const
    {
        if(!hasOperationGraphState())
        {
            return {error_code_t::INVALID_VALUE,
                    "Serializing a graph without a compiled operation graph is unsupported"};
        }
        return _graph.serialize(data);
    }

    error_t serialize(std::vector<uint8_t>& data)
    {
        CHECK_CUDNN_FRONTEND_ERROR(validate());
        return std::as_const(*this).serialize(data);
    }

    error_t deserialize(cudnnHandle_t handle,
                        const std::vector<uint8_t>& data,
                        bool enforcePrecompiled = false)
    {
        static_cast<void>(enforcePrecompiled);
        auto err = _graph.deserialize(handle, data);
        if(err.is_good())
        {
            clearWrapperGraphState();
            _mode = Mode::Native;
            // Trust the native graph on whether a compiled plan was actually
            // embedded: a handle-bearing deserialize only installs one when the
            // blob carried an execution plan. Without it the graph is described
            // and finalized but planless, so do not claim PlansBuilt.
            _stage
                = _graph.get_execution_plan_count() > 0 ? Stage::PlansBuilt : Stage::OpGraphBuilt;
        }
        return err;
    }

    error_t deserialize(const std::vector<uint8_t>& data, bool enforcePrecompiled = false)
    {
        static_cast<void>(enforcePrecompiled);
        auto err = _graph.deserialize(data);
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

    hipdnn_frontend::graph::Graph _graph;
    std::vector<std::shared_ptr<Tensor_attributes>> _ownedTensors;
    std::optional<int64_t> _maxWorkspaceAllowed;
    std::vector<int64_t> _barredEngineIndices;
    std::vector<HeurMode_t> _planCreationModes = {HeurMode_t::FALLBACK};
    std::vector<std::string> _barredEngineNames;
    std::vector<int64_t> _engineIndexToNativeEngineId;
    std::vector<BehaviorNote_t> _selectedBehaviorNotes;
    std::vector<BehaviorNote_t> _deselectedBehaviorNotes;
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

    error_t refreshEngineIndexMap(const std::vector<HeurMode_t>& modes = {HeurMode_t::FALLBACK})
    {
        _engineIndexToNativeEngineId.clear();
        HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH();
        return _graph.get_ranked_engine_ids(_engineIndexToNativeEngineId, modes);
    }

    error_t mapEngineIndex(int64_t engineIndex,
                           int64_t& nativeEngineId,
                           const std::vector<HeurMode_t>& modes = {HeurMode_t::FALLBACK})
    {
        CHECK_CUDNN_FRONTEND_ERROR(refreshEngineIndexMap(modes));
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
                             std::vector<int64_t>& nativeEngineIds,
                             const std::vector<HeurMode_t>& modes = {HeurMode_t::FALLBACK})
    {
        nativeEngineIds.clear();
        nativeEngineIds.reserve(engineIndices.size());
        for(const auto engineIndex : engineIndices)
        {
            int64_t nativeEngineId = -1;
            CHECK_CUDNN_FRONTEND_ERROR(mapEngineIndex(engineIndex, nativeEngineId, modes));
            nativeEngineIds.push_back(nativeEngineId);
        }
        return {};
    }

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
            _graph.deselect_workspace_greater_than(*_maxWorkspaceAllowed);
        }
        if(!_barredEngineNames.empty())
        {
            _graph.deselect_engines(_barredEngineNames);
            for(const auto& name : _barredEngineNames)
            {
                _shimBarredEngineIds.insert(hipdnn_data_sdk::utilities::engineNameToId(name));
            }
        }
        if(!_barredEngineIndices.empty())
        {
            std::vector<int64_t> nativeEngineIds;
            CHECK_CUDNN_FRONTEND_ERROR(
                mapEngineIndices(_barredEngineIndices, nativeEngineIds, modes));
            _graph.deselect_engines(nativeEngineIds);
            _shimBarredEngineIds.insert(nativeEngineIds.begin(), nativeEngineIds.end());
        }

        return applyBehaviorNoteFilters(modes);
    }

    error_t applyBehaviorNoteFilters(const std::vector<HeurMode_t>& modes)
    {
        if(_selectedBehaviorNotes.empty() && _deselectedBehaviorNotes.empty())
        {
            return {};
        }
        HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH();

        std::vector<int64_t> engineIds;
        CHECK_CUDNN_FRONTEND_ERROR(_graph.get_ranked_engine_ids(engineIds, modes));
        if(engineIds.empty())
        {
            return {};
        }

        std::vector<int64_t> enginesToBar;
        enginesToBar.reserve(engineIds.size());
        for(const auto engineId : engineIds)
        {
            std::vector<BehaviorNote_t> engineNotes;
            CHECK_CUDNN_FRONTEND_ERROR(_graph.get_behavior_notes_for_engine(engineId, engineNotes));
            if(!behaviorNotesMatch(engineNotes))
            {
                enginesToBar.push_back(engineId);
            }
        }

        if(enginesToBar.size() == engineIds.size())
        {
            return {error_code_t::GRAPH_NOT_SUPPORTED,
                    "Behavior-note filters removed every applicable hipDNN engine"};
        }
        if(!enginesToBar.empty())
        {
            _graph.deselect_engines(enginesToBar);
            _shimBarredEngineIds.insert(enginesToBar.begin(), enginesToBar.end());
        }
        return {};
    }

    // Report the top-ranked plan that survives this shim's own engine filters, or
    // -1 when the active plan already survives and native can build it directly.
    // Plan order is not guaranteed to match ranked-engine order, so each plan's
    // engine is resolved through its own name.
    error_t findPlanIndexIfActiveBarred(int64_t& survivingIndex)
    {
        survivingIndex = -1;
        if(_shimBarredEngineIds.empty())
        {
            return {};
        }

        int64_t activeEngineId = -1;
        if(_graph.get_execution_plan_engine_id(activeEngineId).is_bad()
           || _shimBarredEngineIds.count(activeEngineId) == 0)
        {
            return {};
        }

        const int64_t planCount = _graph.get_execution_plan_count();
        for(int64_t index = 0; index < planCount; ++index)
        {
            std::string planName;
            if(_graph.get_plan_name_at_index(index, planName).is_bad())
            {
                continue;
            }
            // An engine whose name does not resolve cannot be matched against the
            // barred set. Treat it as a survivor: only ids this shim explicitly
            // barred are known to be excluded, and native re-checks its own filter
            // before building.
            if(!hipdnn_data_sdk::utilities::isEngineNameRegistered(planName)
               || _shimBarredEngineIds.count(hipdnn_data_sdk::utilities::engineNameToId(planName))
                      == 0)
            {
                survivingIndex = index;
                return {};
            }
        }

        return {error_code_t::GRAPH_NOT_SUPPORTED,
                "Engine filters removed every applicable hipDNN execution plan"};
    }

    bool behaviorNotesMatch(const std::vector<BehaviorNote_t>& engineNotes) const
    {
        for(const auto required : _selectedBehaviorNotes)
        {
            if(required != BehaviorNote_t::NOT_SET
               && std::find(engineNotes.begin(), engineNotes.end(), required) == engineNotes.end())
            {
                return false;
            }
        }
        for(const auto excluded : _deselectedBehaviorNotes)
        {
            if(excluded != BehaviorNote_t::NOT_SET
               && std::find(engineNotes.begin(), engineNotes.end(), excluded) != engineNotes.end())
            {
                return false;
            }
        }
        return true;
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
            context.set_name(_graph.get_name())
                .set_compute_data_type(_graph.get_compute_data_type())
                .set_intermediate_data_type(_graph.get_intermediate_data_type())
                .set_io_data_type(_graph.get_io_data_type());
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

    static error_t unsupportedDevicelessBuildError()
    {
        return {error_code_t::INVALID_VALUE,
                "Deviceless build is unsupported for non-empty graphs by this shim"};
    }

    static error_t noExecutionPlanError()
    {
        return {error_code_t::INVALID_VALUE, "Graph has no compiled execution plan"};
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
        _recordedError.reset();
        _mode = Mode::Empty;
        _stage = Stage::Described;
    }
};

// NOLINTEND(readability-identifier-naming)

#undef HIPDNN_CUDNN_SHIM_RETURN_OK_IF_NO_NATIVE_GRAPH
#undef HIPDNN_CUDNN_SHIM_RETURN_NO_PLAN_IF_NO_NATIVE_GRAPH
#undef HIPDNN_CUDNN_SHIM_RETURN_RECORDED_ERROR

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
