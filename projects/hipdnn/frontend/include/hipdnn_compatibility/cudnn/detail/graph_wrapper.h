// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend
// (include/cudnn_frontend/graph_interface.h), used under the MIT license.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_properties.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/sdpa_attributes.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_frontend/Graph.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
// NOLINTBEGIN(readability-identifier-naming)

class Graph
{
public:
    Graph() = default;
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    error_t validate() // NOLINT(readability-identifier-naming)
    {
        if(auto err = checkRecordedError(); err.is_bad())
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

    error_t build_operation_graph(cudnnHandle_t handle) // NOLINT(readability-identifier-naming)
    {
        if(auto err = checkRecordedError(); err.is_bad())
        {
            return err;
        }

        CHECK_CUDNN_FRONTEND_ERROR(validateOwnedTensors());
        if(!hasOperationGraphState())
        {
            _builtEmpty = true;
            _executionPlanCreated = false;
            _executionPlanBuilt = false;
            return {};
        }

        auto err = _graph.build_operation_graph(handle);
        if(err.is_good())
        {
            _operationGraphBuilt = true;
            _executionPlanCreated = false;
            _executionPlanBuilt = false;
        }
        return err;
    }

    error_t build_operation_graph() // NOLINT(readability-identifier-naming)
    {
        if(auto err = checkRecordedError(); err.is_bad())
        {
            return err;
        }

        CHECK_CUDNN_FRONTEND_ERROR(validateOwnedTensors());
        if(!hasOperationGraphState())
        {
            _builtEmpty = true;
            _executionPlanCreated = false;
            _executionPlanBuilt = false;
            return {};
        }

        return unsupportedDevicelessBuildError();
    }

    error_t create_execution_plans(const std::vector<HeurMode_t>& modes = {HeurMode_t::FALLBACK})
    // NOLINT(readability-identifier-naming)
    {
        if(auto err = checkRecordedError(); err.is_bad())
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
            _executionPlanCreated = true;
            _executionPlanBuilt = false;
        }
        return err;
    }

    error_t check_support() // NOLINT(readability-identifier-naming)
    {
        if(auto err = checkRecordedError(); err.is_bad())
        {
            return err;
        }

        if(!hasOperationGraphState())
        {
            return {};
        }

        return _graph.check_support();
    }

    error_t check_support(cudnnHandle_t handle) // NOLINT(readability-identifier-naming)
    {
        static_cast<void>(handle);
        return check_support();
    }

    error_t build_plans(BuildPlanPolicy_t policy = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                        bool doMultithreadedBuilds = false) // NOLINT(readability-identifier-naming)
    {
        static_cast<void>(doMultithreadedBuilds);
        if(auto err = checkRecordedError(); err.is_bad())
        {
            return err;
        }

        if(policy == BuildPlanPolicy_t::ALL)
        {
            return recordError(error_code_t::INVALID_VALUE,
                               "Building all execution plans is unsupported by this shim");
        }

        if(!hasOperationGraphState())
        {
            return {};
        }

        auto err = _graph.build_plans();
        if(err.is_good())
        {
            _executionPlanCreated = true;
            _executionPlanBuilt = true;
        }
        return err;
    }

    error_t build_plans(const cudnnHandle_t& handle,
                        BuildPlanPolicy_t policy = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                        bool doMultithreadedBuilds = false) // NOLINT(readability-identifier-naming)
    {
        static_cast<void>(handle);
        return build_plans(policy, doMultithreadedBuilds);
    }

    error_t build_plan_at_index(int64_t index) // NOLINT(readability-identifier-naming)
    {
        if(auto err = checkRecordedError(); err.is_bad())
        {
            return err;
        }

        if(index != 0 || !hasOperationGraphState())
        {
            return {error_code_t::INVALID_VALUE, "Execution plan index is invalid"};
        }

        if(_executionPlanBuilt)
        {
            return {};
        }

        if(!_executionPlanCreated)
        {
            CHECK_CUDNN_FRONTEND_ERROR(create_execution_plans());
        }
        return build_plans();
    }

    error_t build_plan_at_index(const cudnnHandle_t& handle, int64_t index)
    // NOLINT(readability-identifier-naming)
    {
        static_cast<void>(handle);
        return build_plan_at_index(index);
    }

    int64_t get_execution_plan_count() const // NOLINT(readability-identifier-naming)
    {
        return hasOperationGraphState() && _executionPlanCreated ? 1 : 0;
    }

    error_t build(const cudnnHandle_t& handle,
                  const std::vector<HeurMode_t>& modes = {HeurMode_t::FALLBACK},
                  BuildPlanPolicy_t policy = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                  bool doMultithreadedBuilds = false) // NOLINT(readability-identifier-naming)
    {
        CHECK_CUDNN_FRONTEND_ERROR(validate());
        CHECK_CUDNN_FRONTEND_ERROR(build_operation_graph(handle));
        CHECK_CUDNN_FRONTEND_ERROR(create_execution_plans(modes));
        CHECK_CUDNN_FRONTEND_ERROR(check_support());
        return build_plans(policy, doMultithreadedBuilds);
    }

    error_t build(const std::vector<HeurMode_t>& modes,
                  BuildPlanPolicy_t policy = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                  bool doMultithreadedBuilds = false) // NOLINT(readability-identifier-naming)
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

    Graph& set_name(const std::string& name) // NOLINT(readability-identifier-naming)
    {
        _graph.set_name(name);
        return *this;
    }

    const std::string& get_name() const // NOLINT(readability-identifier-naming)
    {
        return _graph.get_name();
    }

    Graph& set_io_data_type(DataType_t type) // NOLINT(readability-identifier-naming)
    {
        _graph.set_io_data_type(type);
        return *this;
    }

    DataType_t get_io_data_type() const // NOLINT(readability-identifier-naming)
    {
        return _graph.get_io_data_type();
    }

    Graph& set_compute_data_type(DataType_t type) // NOLINT(readability-identifier-naming)
    {
        _graph.set_compute_data_type(type);
        return *this;
    }

    DataType_t get_compute_data_type() const // NOLINT(readability-identifier-naming)
    {
        return _graph.get_compute_data_type();
    }

    Graph& set_intermediate_data_type(DataType_t type) // NOLINT(readability-identifier-naming)
    {
        _graph.set_intermediate_data_type(type);
        return *this;
    }

    DataType_t get_intermediate_data_type() const // NOLINT(readability-identifier-naming)
    {
        return _graph.get_intermediate_data_type();
    }

    Graph& set_override_shape_enabled(bool isEnabled) // NOLINT(readability-identifier-naming)
    {
        _graph.set_override_shape_enabled(isEnabled);
        return *this;
    }

    // Setter triage: hints that are safe to drop
    // (dynamic-shape, kernel-cache) log and continue; requests the shim cannot
    // honor without changing results (SM targeting, device properties) record an
    // error that surfaces at the next validate()/build. The asymmetry is
    // deliberate — do not "normalize" one group to the other.
    Graph& set_dynamic_shape_enabled(bool isEnabled) // NOLINT(readability-identifier-naming)
    {
        static_cast<void>(isEnabled);
        CUDNN_FE_LOG_LABEL("Ignoring graph dynamic-shape hint; hipDNN has no graph-level setting");
        return *this;
    }

    Graph& set_kernel_cache(const std::shared_ptr<KernelCache>& cache)
    // NOLINT(readability-identifier-naming)
    {
        static_cast<void>(cache);
        CUDNN_FE_LOG_LABEL("Ignoring graph kernel cache hint; hipDNN selects kernels internally");
        return *this;
    }

    Graph& set_sm_count(int32_t count) // NOLINT(readability-identifier-naming)
    {
        static_cast<void>(count);
        recordError(error_code_t::INVALID_VALUE, "Target SM count is unsupported by this shim");
        return *this;
    }

    Graph& set_sm_version(int32_t version) // NOLINT(readability-identifier-naming)
    {
        static_cast<void>(version);
        recordError(error_code_t::INVALID_VALUE, "Target SM version is unsupported by this shim");
        return *this;
    }

    Graph& set_device_properties(const std::shared_ptr<const DeviceProperties>& deviceProperties)
    // NOLINT(readability-identifier-naming)
    {
        static_cast<void>(deviceProperties);
        recordError(error_code_t::INVALID_VALUE, "Device properties are unsupported by this shim");
        return *this;
    }

    std::shared_ptr<Tensor_attributes> tensor(const Tensor_attributes& tensorAttributes)
    // NOLINT(readability-identifier-naming)
    {
        auto tensorPtr = hipdnn_frontend::graph::Graph::tensor(tensorAttributes);
        _ownedTensors.emplace_back(tensorPtr);
        return tensorPtr;
    }

    std::shared_ptr<Tensor_attributes> tensor(const float& scalar, ScalarType scalarType)
    // NOLINT(readability-identifier-naming)
    {
        return scalarTensor(scalar, scalarType);
    }

    std::shared_ptr<Tensor_attributes> tensor(const half& scalar, ScalarType scalarType)
    // NOLINT(readability-identifier-naming)
    {
        return scalarTensor(scalar, scalarType);
    }

    std::shared_ptr<Tensor_attributes> tensor(const nv_bfloat16& scalar, ScalarType scalarType)
    // NOLINT(readability-identifier-naming)
    {
        return scalarTensor(scalar, scalarType);
    }

    std::shared_ptr<Tensor_attributes> tensor(const int32_t& scalar, ScalarType scalarType)
    // NOLINT(readability-identifier-naming)
    {
        return scalarTensor(scalar, scalarType);
    }

    std::shared_ptr<Tensor_attributes> tensor(const int64_t& scalar, ScalarType scalarType)
    // NOLINT(readability-identifier-naming)
    {
        return scalarTensor(scalar, scalarType);
    }

    std::shared_ptr<Tensor_attributes> tensor(const double& scalar, ScalarType scalarType)
    // NOLINT(readability-identifier-naming)
    {
        return scalarTensor(scalar, scalarType);
    }

    std::shared_ptr<Tensor_attributes>
        tensor_like(const std::shared_ptr<Tensor_attributes>& tensorAttributes,
                    const std::string& name
                    = std::string{}) // NOLINT(readability-identifier-naming)
    {
        auto tensorPtr = hipdnn_frontend::graph::Graph::tensor_like(tensorAttributes, name);
        _ownedTensors.emplace_back(tensorPtr);
        return tensorPtr;
    }

    error_t query_tensor_attributes_of_uid(int64_t uid, Tensor_attributes& tensorAttributes) const
    // NOLINT(readability-identifier-naming)
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
    std::array<std::shared_ptr<Tensor_attributes>, 2>
        sdpa(std::shared_ptr<Tensor_attributes> q, // NOLINT(readability-identifier-naming)
             std::shared_ptr<Tensor_attributes> k,
             std::shared_ptr<Tensor_attributes> v,
             SDPA_attributes attributes)
    {
        if(attributes._recordedError.has_value())
        {
            recordError(*attributes._recordedError);
        }
        auto outputs
            = _graph.sdpa(std::move(q), std::move(k), std::move(v), std::move(attributes._attrs));
        _hasNodes = true;
        return outputs;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 3>
        sdpa_backward(std::shared_ptr<Tensor_attributes> q, // NOLINT(readability-identifier-naming)
                      std::shared_ptr<Tensor_attributes> k,
                      std::shared_ptr<Tensor_attributes> v,
                      std::shared_ptr<Tensor_attributes> o,
                      std::shared_ptr<Tensor_attributes> dO,
                      std::shared_ptr<Tensor_attributes> stats,
                      SDPA_backward_attributes attributes)
    {
        if(attributes._recordedError.has_value())
        {
            recordError(*attributes._recordedError);
        }
        auto outputs = _graph.sdpa_backward(std::move(q),
                                            std::move(k),
                                            std::move(v),
                                            std::move(o),
                                            std::move(dO),
                                            std::move(stats),
                                            std::move(attributes._attrs));
        _hasNodes = true;
        return outputs;
    }
#endif // HIPDNN_ENABLE_SDPA

    error_t
        execute(cudnnHandle_t handle,
                std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensorToPointerMap,
                void* workspace) const
    {
        if(auto err = checkRecordedError(); err.is_bad())
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
        if(auto err = checkRecordedError(); err.is_bad())
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
        if(auto err = checkRecordedError(); err.is_bad())
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
        if(auto err = checkRecordedError(); err.is_bad())
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

    error_t
        get_workspace_size(int64_t& workspaceSize) const // NOLINT(readability-identifier-naming)
    {
        if(auto err = checkRecordedError(); err.is_bad())
        {
            return err;
        }

        if(!hasOperationGraphState())
        {
            if(_builtEmpty)
            {
                workspaceSize = 0;
                return {};
            }
            return noExecutionPlanError();
        }

        return _graph.get_workspace_size(workspaceSize);
    }

    int64_t get_workspace_size() const // NOLINT(readability-identifier-naming)
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
        if(hasOperationGraphState())
        {
            return _graph.serialize(data);
        }

        data.clear();
        appendMagic(data);
        appendValue<uint32_t>(data, kBlobVersion);
        appendString(data, _graph.get_name());
        appendEnum(data, _graph.get_compute_data_type());
        appendEnum(data, _graph.get_intermediate_data_type());
        appendEnum(data, _graph.get_io_data_type());
        appendValue<uint64_t>(data, static_cast<uint64_t>(_ownedTensors.size()));
        for(const auto& tensorPtr : _ownedTensors)
        {
            appendTensor(data, tensorPtr);
        }
        return {};
    }

    error_t serialize(std::vector<uint8_t>& data)
    {
        CHECK_CUDNN_FRONTEND_ERROR(validate());
        if(!hasOperationGraphState())
        {
            _builtEmpty = true;
        }
        return std::as_const(*this).serialize(data);
    }

    error_t deserialize(cudnnHandle_t handle,
                        const std::vector<uint8_t>& data,
                        bool enforcePrecompiled = false)
    {
        if(isShimBlob(data))
        {
            static_cast<void>(handle);
            return deserializeShimBlob(data, enforcePrecompiled);
        }

        auto err = _graph.deserialize(handle, data);
        if(err.is_good())
        {
            clearWrapperGraphState();
            _hasNativeGraphState = true;
            _operationGraphBuilt = handle != nullptr;
        }
        return err;
    }

    error_t deserialize(const std::vector<uint8_t>& data, bool enforcePrecompiled = false)
    {
        if(isShimBlob(data))
        {
            return deserializeShimBlob(data, enforcePrecompiled);
        }

        auto err = _graph.deserialize(data);
        if(err.is_good())
        {
            clearWrapperGraphState();
            _hasNativeGraphState = true;
            _operationGraphBuilt = false;
        }
        return err;
    }

private:
    enum class ScalarTag : uint8_t
    {
        None = 0,
        Double,
        Float,
        Half,
        Bfloat16,
        Uint8,
        Int32,
        Int64,
        Bool
    };

    class BlobReader
    {
    public:
        explicit BlobReader(const std::vector<uint8_t>& data)
            : _current(data.data())
            , _end(data.data() + data.size())
        {
        }

        bool consume(const uint8_t*& ptr, size_t size)
        {
            if(size > static_cast<size_t>(_end - _current))
            {
                return false;
            }
            ptr = _current;
            _current += size;
            return true;
        }

        template <typename T>
        bool read(T& value)
        {
            const uint8_t* ptr = nullptr;
            if(!consume(ptr, sizeof(T)))
            {
                return false;
            }
            std::memcpy(&value, ptr, sizeof(T));
            return true;
        }

        bool empty() const
        {
            return _current == _end;
        }

    private:
        const uint8_t* _current;
        const uint8_t* _end;
    };

    static constexpr std::array<uint8_t, 20> kBlobMagic
        = {'H', 'I', 'P', 'D', 'N', 'N', '_', 'C', 'U', 'D',
           'N', 'N', '_', 'G', 'R', 'A', 'P', 'H', '_', '1'};
    static constexpr uint32_t kBlobVersion = 1;

    hipdnn_frontend::graph::Graph _graph;
    std::optional<error_t> _recordedError;
    std::vector<std::shared_ptr<Tensor_attributes>> _ownedTensors;
    bool _hasNativeGraphState = false;
    bool _hasNodes = false;
    bool _operationGraphBuilt = false;
    bool _executionPlanCreated = false;
    bool _executionPlanBuilt = false;
    bool _builtEmpty = false;

    bool hasOperationGraphState() const
    {
        return _hasNativeGraphState || _hasNodes;
    }

    error_t recordError(error_t err)
    {
        if(err.is_bad() && !_recordedError.has_value())
        {
            _recordedError = std::move(err);
        }
        return *_recordedError;
    }

    error_t recordError(error_code_t code, const char* message)
    {
        return recordError({code, message});
    }

    error_t checkRecordedError() const
    {
        return _recordedError.value_or(error_t{});
    }

    template <typename T>
    std::shared_ptr<Tensor_attributes> scalarTensor(const T& scalar, ScalarType scalarType)
    {
        if(scalarType == ScalarType::COMPILE_TIME_CONST)
        {
            recordError(error_code_t::INVALID_VALUE,
                        "Compile-time scalar tensors are unsupported by this shim");
        }

        auto tensorPtr = std::make_shared<Tensor_attributes>(scalar);
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
        _hasNativeGraphState = false;
        _hasNodes = false;
        _operationGraphBuilt = false;
        _executionPlanCreated = false;
        _executionPlanBuilt = false;
        _builtEmpty = false;
    }

    static void appendMagic(std::vector<uint8_t>& data)
    {
        data.insert(data.end(), kBlobMagic.begin(), kBlobMagic.end());
    }

    template <typename T>
    static void appendValue(std::vector<uint8_t>& data, const T& value)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
        data.insert(data.end(), bytes, bytes + sizeof(T));
    }

    static void appendEnum(std::vector<uint8_t>& data, DataType_t value)
    {
        appendValue<int32_t>(data, static_cast<int32_t>(value));
    }

    static void appendString(std::vector<uint8_t>& data, const std::string& value)
    {
        appendValue<uint64_t>(data, static_cast<uint64_t>(value.size()));
        data.insert(data.end(), value.begin(), value.end());
    }

    static void appendInt64Vector(std::vector<uint8_t>& data, const std::vector<int64_t>& values)
    {
        appendValue<uint64_t>(data, static_cast<uint64_t>(values.size()));
        for(const auto value : values)
        {
            appendValue<int64_t>(data, value);
        }
    }

    static void appendTensor(std::vector<uint8_t>& data,
                             const std::shared_ptr<Tensor_attributes>& tensorPtr)
    {
        const uint8_t present = tensorPtr ? 1 : 0;
        appendValue<uint8_t>(data, present);
        if(!tensorPtr)
        {
            return;
        }

        appendValue<uint8_t>(data, tensorPtr->has_uid() ? 1 : 0);
        appendValue<int64_t>(data, tensorPtr->get_uid());
        appendString(data, tensorPtr->get_name());
        appendEnum(data, tensorPtr->get_data_type());
        appendInt64Vector(data, tensorPtr->get_dim());
        appendInt64Vector(data, tensorPtr->get_stride());
        appendValue<uint8_t>(data, tensorPtr->get_is_virtual() ? 1 : 0);
        appendScalar(data, *tensorPtr);
    }

    static void appendScalar(std::vector<uint8_t>& data, const Tensor_attributes& tensor)
    {
        const auto& value = tensor.get_value_variant();
        if(const auto* doubleScalar = std::get_if<double>(&value))
        {
            appendValue<uint8_t>(data, static_cast<uint8_t>(ScalarTag::Double));
            appendValue<double>(data, *doubleScalar);
        }
        else if(const auto* floatScalar = std::get_if<float>(&value))
        {
            appendValue<uint8_t>(data, static_cast<uint8_t>(ScalarTag::Float));
            appendValue<float>(data, *floatScalar);
        }
        else if(const auto* halfScalar = std::get_if<half>(&value))
        {
            appendValue<uint8_t>(data, static_cast<uint8_t>(ScalarTag::Half));
            appendValue<uint16_t>(data, halfScalar->data);
        }
        else if(const auto* bfloatScalar = std::get_if<nv_bfloat16>(&value))
        {
            appendValue<uint8_t>(data, static_cast<uint8_t>(ScalarTag::Bfloat16));
            appendValue<uint16_t>(data, bfloatScalar->data);
        }
        else if(const auto* uint8Scalar = std::get_if<uint8_t>(&value))
        {
            appendValue<uint8_t>(data, static_cast<uint8_t>(ScalarTag::Uint8));
            appendValue<uint8_t>(data, *uint8Scalar);
        }
        else if(const auto* int32Scalar = std::get_if<int32_t>(&value))
        {
            appendValue<uint8_t>(data, static_cast<uint8_t>(ScalarTag::Int32));
            appendValue<int32_t>(data, *int32Scalar);
        }
        else if(const auto* int64Scalar = std::get_if<int64_t>(&value))
        {
            appendValue<uint8_t>(data, static_cast<uint8_t>(ScalarTag::Int64));
            appendValue<int64_t>(data, *int64Scalar);
        }
        else if(const auto* boolScalar = std::get_if<bool>(&value))
        {
            appendValue<uint8_t>(data, static_cast<uint8_t>(ScalarTag::Bool));
            appendValue<uint8_t>(data, *boolScalar ? 1 : 0);
        }
        else
        {
            appendValue<uint8_t>(data, static_cast<uint8_t>(ScalarTag::None));
        }
    }

    static bool isShimBlob(const std::vector<uint8_t>& data)
    {
        return data.size() >= kBlobMagic.size()
               && std::equal(kBlobMagic.begin(), kBlobMagic.end(), data.begin());
    }

    static error_t readString(BlobReader& reader, std::string& value)
    {
        uint64_t size = 0;
        if(!reader.read(size) || size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        {
            return {error_code_t::INVALID_VALUE, "Invalid serialized graph string"};
        }

        const uint8_t* ptr = nullptr;
        if(!reader.consume(ptr, static_cast<size_t>(size)))
        {
            return {error_code_t::INVALID_VALUE, "Truncated serialized graph string"};
        }

        value.assign(reinterpret_cast<const char*>(ptr), static_cast<size_t>(size));
        return {};
    }

    static error_t readInt64Vector(BlobReader& reader, std::vector<int64_t>& values)
    {
        uint64_t size = 0;
        if(!reader.read(size) || size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        {
            return {error_code_t::INVALID_VALUE, "Invalid serialized graph vector"};
        }

        values.clear();
        values.reserve(static_cast<size_t>(size));
        for(uint64_t i = 0; i < size; ++i)
        {
            int64_t value = 0;
            if(!reader.read(value))
            {
                return {error_code_t::INVALID_VALUE, "Truncated serialized graph vector"};
            }
            values.emplace_back(value);
        }
        return {};
    }

    static error_t readScalar(BlobReader& reader, Tensor_attributes& tensor)
    {
        uint8_t tagValue = 0;
        if(!reader.read(tagValue))
        {
            return {error_code_t::INVALID_VALUE, "Truncated serialized scalar tensor"};
        }

        switch(static_cast<ScalarTag>(tagValue))
        {
        case ScalarTag::None:
            return {};
        case ScalarTag::Double:
        {
            double value = 0.0;
            if(!reader.read(value))
            {
                return {error_code_t::INVALID_VALUE, "Truncated double scalar tensor"};
            }
            tensor.set_value(value);
            return {};
        }
        case ScalarTag::Float:
        {
            float value = 0.0F;
            if(!reader.read(value))
            {
                return {error_code_t::INVALID_VALUE, "Truncated float scalar tensor"};
            }
            tensor.set_value(value);
            return {};
        }
        case ScalarTag::Half:
        {
            uint16_t value = 0;
            if(!reader.read(value))
            {
                return {error_code_t::INVALID_VALUE, "Truncated half scalar tensor"};
            }
            tensor.set_value(half::from_bits(value));
            return {};
        }
        case ScalarTag::Bfloat16:
        {
            uint16_t value = 0;
            if(!reader.read(value))
            {
                return {error_code_t::INVALID_VALUE, "Truncated bfloat16 scalar tensor"};
            }
            tensor.set_value(nv_bfloat16::from_bits(value));
            return {};
        }
        case ScalarTag::Uint8:
        {
            uint8_t value = 0;
            if(!reader.read(value))
            {
                return {error_code_t::INVALID_VALUE, "Truncated uint8 scalar tensor"};
            }
            tensor.set_value(value);
            return {};
        }
        case ScalarTag::Int32:
        {
            int32_t value = 0;
            if(!reader.read(value))
            {
                return {error_code_t::INVALID_VALUE, "Truncated int32 scalar tensor"};
            }
            tensor.set_value(value);
            return {};
        }
        case ScalarTag::Int64:
        {
            int64_t value = 0;
            if(!reader.read(value))
            {
                return {error_code_t::INVALID_VALUE, "Truncated int64 scalar tensor"};
            }
            tensor.set_value(value);
            return {};
        }
        case ScalarTag::Bool:
        {
            uint8_t value = 0;
            if(!reader.read(value))
            {
                return {error_code_t::INVALID_VALUE, "Truncated bool scalar tensor"};
            }
            tensor.set_value(value != 0);
            return {};
        }
        default:
            return {error_code_t::INVALID_VALUE, "Invalid serialized scalar tensor tag"};
        }
    }

    static error_t readTensor(BlobReader& reader, std::shared_ptr<Tensor_attributes>& tensorPtr)
    {
        uint8_t present = 0;
        if(!reader.read(present))
        {
            return {error_code_t::INVALID_VALUE, "Truncated serialized tensor"};
        }
        if(present == 0)
        {
            tensorPtr.reset();
            return {};
        }

        uint8_t hasUid = 0;
        int64_t uid = 0;
        std::string name;
        int32_t dataType = 0;
        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
        uint8_t isVirtual = 0;
        if(!reader.read(hasUid) || !reader.read(uid))
        {
            return {error_code_t::INVALID_VALUE, "Truncated serialized tensor UID"};
        }
        CHECK_CUDNN_FRONTEND_ERROR(readString(reader, name));
        if(!reader.read(dataType))
        {
            return {error_code_t::INVALID_VALUE, "Truncated serialized tensor data type"};
        }
        CHECK_CUDNN_FRONTEND_ERROR(readInt64Vector(reader, dims));
        CHECK_CUDNN_FRONTEND_ERROR(readInt64Vector(reader, strides));
        if(!reader.read(isVirtual))
        {
            return {error_code_t::INVALID_VALUE, "Truncated serialized tensor virtual flag"};
        }

        auto tensor = std::make_shared<Tensor_attributes>();
        CHECK_CUDNN_FRONTEND_ERROR(readScalar(reader, *tensor));
        tensor->set_name(name)
            .set_data_type(static_cast<DataType_t>(dataType))
            .set_dim(dims)
            .set_stride(strides)
            .set_is_virtual(isVirtual != 0);
        if(hasUid != 0)
        {
            tensor->set_uid(uid);
        }
        tensorPtr = std::move(tensor);
        return {};
    }

    error_t deserializeShimBlob(const std::vector<uint8_t>& data, bool enforcePrecompiled)
    {
        if(enforcePrecompiled)
        {
            return {error_code_t::INVALID_VALUE,
                    "Shim empty-graph blobs do not contain a precompiled execution plan"};
        }

        BlobReader reader(data);
        const uint8_t* magic = nullptr;
        if(!reader.consume(magic, kBlobMagic.size()))
        {
            return {error_code_t::INVALID_VALUE, "Truncated serialized graph"};
        }

        uint32_t version = 0;
        if(!reader.read(version) || version != kBlobVersion)
        {
            return {error_code_t::UNSUPPORTED_GRAPH_FORMAT, "Unsupported shim graph blob version"};
        }

        std::string name;
        int32_t computeType = 0;
        int32_t intermediateType = 0;
        int32_t ioType = 0;
        uint64_t tensorCount = 0;
        CHECK_CUDNN_FRONTEND_ERROR(readString(reader, name));
        if(!reader.read(computeType) || !reader.read(intermediateType) || !reader.read(ioType)
           || !reader.read(tensorCount))
        {
            return {error_code_t::INVALID_VALUE, "Truncated serialized graph attributes"};
        }
        if(tensorCount > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        {
            return {error_code_t::INVALID_VALUE, "Serialized graph has too many tensors"};
        }

        std::vector<std::shared_ptr<Tensor_attributes>> tensors;
        tensors.reserve(static_cast<size_t>(tensorCount));
        for(uint64_t i = 0; i < tensorCount; ++i)
        {
            std::shared_ptr<Tensor_attributes> tensorPtr;
            CHECK_CUDNN_FRONTEND_ERROR(readTensor(reader, tensorPtr));
            if(tensorPtr)
            {
                tensors.emplace_back(std::move(tensorPtr));
            }
        }
        if(!reader.empty())
        {
            return {error_code_t::INVALID_VALUE, "Serialized graph has trailing data"};
        }

        _graph = hipdnn_frontend::graph::Graph{};
        _graph.set_name(name)
            .set_compute_data_type(static_cast<DataType_t>(computeType))
            .set_intermediate_data_type(static_cast<DataType_t>(intermediateType))
            .set_io_data_type(static_cast<DataType_t>(ioType));
        clearWrapperGraphState();
        _ownedTensors = std::move(tensors);
        return {};
    }
};

// NOLINTEND(readability-identifier-naming)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
