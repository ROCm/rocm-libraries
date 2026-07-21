// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend
// (include/cudnn_frontend/graph_interface.h), used under the MIT license.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_properties.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/sdpa_attributes.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/detail/error_recorder.h>
#include <hipdnn_frontend/Graph.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
// NOLINTBEGIN(readability-identifier-naming): the whole class mirrors cuDNN's
// snake_case public spelling for source compatibility.

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
        if(auto err = getRecordedError(); err.is_bad())
        {
            return err;
        }

        CHECK_CUDNN_FRONTEND_ERROR(validateOwnedTensors());
        if(hasOperationGraphState())
        {
            return _graph.validate();
        }

        return {};
    }

    error_t build_operation_graph(cudnnHandle_t handle)
    {
        if(auto err = getRecordedError(); err.is_bad())
        {
            return err;
        }

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
        if(auto err = getRecordedError(); err.is_bad())
        {
            return err;
        }

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
        if(auto err = getRecordedError(); err.is_bad())
        {
            return err;
        }

        if(!hasOperationGraphState())
        {
            return {};
        }

        auto err = _graph.create_execution_plans(modes);
        if(err.is_good())
        {
            _stage = Stage::PlansCreated;
        }
        return err;
    }

    error_t check_support()
    {
        if(auto err = getRecordedError(); err.is_bad())
        {
            return err;
        }

        if(!hasOperationGraphState())
        {
            return {};
        }

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
        if(auto err = getRecordedError(); err.is_bad())
        {
            return err;
        }

        if(policy == BuildPlanPolicy_t::ALL)
        {
            recordError(error_code_t::INVALID_VALUE,
                        "Building all execution plans is unsupported by this shim");
            return getRecordedError();
        }

        if(!hasOperationGraphState())
        {
            return {};
        }

        auto err = _graph.build_plans();
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
        if(auto err = getRecordedError(); err.is_bad())
        {
            return err;
        }

        if(!hasOperationGraphState())
        {
            return noExecutionPlanError();
        }

        if(index != 0)
        {
            return {error_code_t::INVALID_VALUE, "Execution plan index is invalid"};
        }

        if(stageAtLeast(Stage::PlansBuilt))
        {
            return {};
        }

        if(!stageAtLeast(Stage::PlansCreated))
        {
            CHECK_CUDNN_FRONTEND_ERROR(create_execution_plans());
        }
        return build_plans();
    }

    error_t build_plan_at_index(const cudnnHandle_t& handle, int64_t index)
    {
        static_cast<void>(handle);
        return build_plan_at_index(index);
    }

    int64_t get_execution_plan_count() const
    {
        return hasOperationGraphState() && stageAtLeast(Stage::PlansCreated) ? 1 : 0;
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
        if(auto err = getRecordedError(); err.is_bad())
        {
            return err;
        }

        if(!hasOperationGraphState())
        {
            return noExecutionPlanError();
        }
        return _graph.execute(handle, tensorToPointerMap, workspace);
    }

    error_t execute(cudnnHandle_t handle,
                    std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                    void* workspace) const
    {
        if(auto err = getRecordedError(); err.is_bad())
        {
            return err;
        }

        if(!hasOperationGraphState())
        {
            return noExecutionPlanError();
        }
        return _graph.execute(handle, tensorUidToPointerMap, workspace);
    }

    error_t execute(cudnnHandle_t handle,
                    std::unordered_map<int64_t, void*>& tensorUidToPointerMap,
                    void* workspace,
                    const std::vector<int64_t>& overrideUids,
                    const std::vector<std::vector<int64_t>>& overrideShapes,
                    const std::vector<std::vector<int64_t>>& overrideStrides) const
    {
        if(auto err = getRecordedError(); err.is_bad())
        {
            return err;
        }

        if(!hasOperationGraphState())
        {
            return noExecutionPlanError();
        }

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
        if(auto err = getRecordedError(); err.is_bad())
        {
            return err;
        }

        static_cast<void>(handle);
        static_cast<void>(sortedUserPtrs);
        static_cast<void>(nUser);
        static_cast<void>(workspace);
        if(!hasOperationGraphState())
        {
            return noExecutionPlanError();
        }
        return {error_code_t::INVALID_VALUE,
                "Flat pointer-array execute is unsupported by this shim"};
    }

    error_t get_workspace_size(int64_t& workspaceSize) const
    {
        if(auto err = getRecordedError(); err.is_bad())
        {
            return err;
        }

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
        _recordedError.reset();
        _mode = Mode::Empty;
        _stage = Stage::Described;
    }
};

// NOLINTEND(readability-identifier-naming)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
