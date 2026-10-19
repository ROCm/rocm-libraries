// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Mock-backend fixture for the cuDNN v8 backend shim: records every hipDNN
// descriptor it hands out with its type, every SetAttribute with a copy of its
// payload, and executions, and can play back a heuristic result list,
// per-engine knobs, behavior notes and names, plan workspace and serialization,
// profiling times, and execute failures.
#pragma once

#include "MockBackendFixture.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hipdnn_shim_test
{

struct CapturedSetAttribute
{
    hipdnnBackendDescriptor_t descriptor;
    hipdnnBackendDescriptorType_t descriptorType;
    hipdnnBackendAttributeName_t name;
    hipdnnBackendAttributeType_t type;
    std::int64_t count;
    std::vector<std::byte> payload;

    // T may be a handle or descriptor pointer; its own size is meant.
    // NOLINTBEGIN(bugprone-sizeof-expression)
    template <typename T>
    std::vector<T> values() const
    {
        std::vector<T> result(payload.size() / sizeof(T));
        if(!result.empty())
        {
            std::memcpy(
                static_cast<void*>(result.data()), payload.data(), result.size() * sizeof(T));
        }
        return result;
    }

    // Read directly, so T = bool works (std::vector<bool> has no data()).
    template <typename T>
    T value() const
    {
        EXPECT_EQ(payload.size(), sizeof(T));
        T result{};
        if(payload.size() == sizeof(T))
        {
            std::memcpy(static_cast<void*>(&result), payload.data(), sizeof(T));
        }
        return result;
    }
    // NOLINTEND(bugprone-sizeof-expression)
};

struct CapturedExecute
{
    hipdnnHandle_t handle;
    hipdnnBackendDescriptor_t plan;
    hipdnnBackendDescriptor_t variantPack;
};

// An INT64 hipDNN knob as the backend's KNOB_INFO descriptor reports it.
struct MockKnob
{
    std::string id;
    std::int64_t defaultValue = 0;
    std::optional<std::int64_t> minValue;
    std::optional<std::int64_t> maxValue;
    std::optional<std::int64_t> stride;
    std::vector<std::int64_t> validValues;
};

class ShimV8MockBackendFixture : public ShimMockBackendFixture
{
protected:
    static constexpr std::size_t DESCRIPTOR_POOL_SIZE = 8192;
    static constexpr hipdnnStatus_t FAILED_CONFIG_STATUS = HIPDNN_STATUS_NOT_SUPPORTED;

    void SetUp() override
    {
        using ::testing::_;

        ShimMockBackendFixture::SetUp();
        _pool.assign(DESCRIPTOR_POOL_SIZE, 0);

        ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
            .WillByDefault([this](hipdnnBackendDescriptorType_t type,
                                  hipdnnBackendDescriptor_t* descriptor) {
                const std::lock_guard<std::mutex> lock(_mutex);
                if(type == HIPDNN_BACKEND_PROFILING_CONTROL_EXT && _failProfilingCreate)
                {
                    *descriptor = nullptr;
                    _profilingTimeline.emplace_back("create failed");
                    return HIPDNN_STATUS_ALLOC_FAILED;
                }
                *descriptor = allocate(type);
                if(type == HIPDNN_BACKEND_PROFILING_CONTROL_EXT)
                {
                    _profilingTimeline.emplace_back("create");
                }
                return *descriptor == nullptr ? HIPDNN_STATUS_ALLOC_FAILED : HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendDestroyDescriptor(_))
            .WillByDefault([this](hipdnnBackendDescriptor_t descriptor) {
                const std::lock_guard<std::mutex> lock(_mutex);
                if(isProfiling(descriptor))
                {
                    _profilingTimeline.emplace_back("destroy");
                }
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
            .WillByDefault([this](hipdnnBackendDescriptor_t descriptor,
                                  hipdnnBackendAttributeName_t name,
                                  hipdnnBackendAttributeType_t type,
                                  std::int64_t count,
                                  const void* elements) {
                const std::lock_guard<std::mutex> lock(_mutex);
                capture(descriptor, name, type, count, elements);
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendFinalize(_))
            .WillByDefault([this](hipdnnBackendDescriptor_t descriptor) {
                const std::lock_guard<std::mutex> lock(_mutex);
                _finalized.push_back(descriptor);
                _setCountAtFinalize.emplace(descriptor, _setAttributes.size());
                if(isProfiling(descriptor))
                {
                    _profilingTimeline.emplace_back("finalize");
                }
                const auto config = _configIds.find(descriptor);
                if(config != _configIds.end() && _failingConfigIds.count(config->second) != 0)
                {
                    _lastError = "mock engine config " + std::to_string(config->second)
                                 + " failed to finalize";
                    return FAILED_CONFIG_STATUS;
                }
                const auto plan = _planIds.find(descriptor);
                if(plan != _planIds.end() && _failingPlanIds.count(plan->second) != 0)
                {
                    _lastError = "mock plan for engine " + std::to_string(plan->second)
                                 + " failed to finalize";
                    return FAILED_CONFIG_STATUS;
                }
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendGetAttribute(_, _, _, _, _, _))
            .WillByDefault([this](hipdnnBackendDescriptor_t descriptor,
                                  hipdnnBackendAttributeName_t name,
                                  hipdnnBackendAttributeType_t,
                                  std::int64_t requested,
                                  std::int64_t* count,
                                  void* elements) {
                const std::lock_guard<std::mutex> lock(_mutex);
                return answerGet(descriptor, name, requested, count, elements);
            });
        ON_CALL(*_mockBackend, backendExecute(_, _, _))
            .WillByDefault([this](hipdnnHandle_t handle,
                                  hipdnnBackendDescriptor_t plan,
                                  hipdnnBackendDescriptor_t variantPack) {
                const std::lock_guard<std::mutex> lock(_mutex);
                _executions.push_back({handle, plan, variantPack});
                _profilingTimeline.emplace_back("execute");
                return answerExecute(plan);
            });
        ON_CALL(*_mockBackend, getLastErrorString(_, _))
            .WillByDefault([this](char* message, std::size_t size) {
                const std::lock_guard<std::mutex> lock(_mutex);
                if(message != nullptr && size > 0)
                {
                    const std::size_t length = std::min(_lastError.size(), size - 1);
                    std::memcpy(message, _lastError.data(), length);
                    message[length] = '\0';
                }
                _lastError.clear();
            });
        ON_CALL(*_mockBackend, getErrorString(_)).WillByDefault(::testing::Return("mock status"));
    }

    // ENGINEHEUR RESULTS yields one ENGINECFG per ID, in this (rank) order.
    // Finalizing the config of an ID in `failingConfigIds` fails.
    void installHeuristicResults(std::vector<std::int64_t> rankedIds,
                                 std::set<std::int64_t> failingConfigIds = {})
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _rankedIds = std::move(rankedIds);
        _failingConfigIds = std::move(failingConfigIds);
    }

    // ENGINE_KNOB_INFO of the engine `engineId` yields one KNOB_INFO per knob.
    void installEngineKnobs(std::int64_t engineId, std::vector<MockKnob> knobs)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _engineKnobs[engineId] = std::move(knobs);
    }

    void installBehaviorNotes(std::int64_t engineId, std::vector<hipdnnBackendBehaviorNote_t> notes)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _behaviorNotes[engineId] = std::move(notes);
    }

    void installEngineName(std::int64_t engineId, std::string name)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _engineNames[engineId] = std::move(name);
    }

    // Every hipDNN plan reports this workspace size.
    void installPlanWorkspace(std::int64_t bytes)
    {
        using ::testing::_;

        ON_CALL(*_mockBackend,
                backendGetAttribute(_, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, _, _, _, _))
            .WillByDefault([bytes](hipdnnBackendDescriptor_t,
                                   hipdnnBackendAttributeName_t,
                                   hipdnnBackendAttributeType_t,
                                   std::int64_t,
                                   std::int64_t* count,
                                   void* elements) {
                *static_cast<std::int64_t*>(elements) = bytes;
                if(count != nullptr)
                {
                    *count = 1;
                }
                return HIPDNN_STATUS_SUCCESS;
            });
    }

    // Every hipDNN plan serializes to `bytes`; deserializing gives a new plan
    // for the engine of the plan serialized last, and a plan reports its
    // engine ID through EXECUTION_PLAN_ENGINE_GLOBAL_INDEX_EXT.
    void installSerializedPlan(std::vector<std::uint8_t> bytes)
    {
        using ::testing::_;

        {
            const std::lock_guard<std::mutex> lock(_mutex);
            _serializedPlan = std::move(bytes);
        }
        ON_CALL(*_mockBackend, backendGetSerializedExecutionPlanExt(_, _, _, _))
            .WillByDefault([this](hipdnnBackendDescriptor_t plan,
                                  std::size_t requested,
                                  std::size_t* size,
                                  std::uint8_t* serialized) {
                const std::lock_guard<std::mutex> lock(_mutex);
                const auto engine = _planIds.find(plan);
                _lastSerializedEngineId = engine == _planIds.end() ? -1 : engine->second;
                *size = _serializedPlan.size();
                if(serialized != nullptr && requested >= _serializedPlan.size())
                {
                    std::memcpy(serialized, _serializedPlan.data(), _serializedPlan.size());
                }
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendCreateAndDeserializeExecutionPlanExt(_, _, _, _))
            .WillByDefault([this](hipdnnHandle_t,
                                  hipdnnBackendDescriptor_t* descriptor,
                                  const std::uint8_t*,
                                  std::size_t) {
                const std::lock_guard<std::mutex> lock(_mutex);
                *descriptor = allocate(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
                if(*descriptor == nullptr)
                {
                    return HIPDNN_STATUS_ALLOC_FAILED;
                }
                _planIds[*descriptor] = _lastSerializedEngineId;
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(
            *_mockBackend,
            backendGetAttribute(_, HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_GLOBAL_INDEX_EXT, _, _, _, _))
            .WillByDefault([this](hipdnnBackendDescriptor_t plan,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  std::int64_t,
                                  std::int64_t* count,
                                  void* elements) {
                const std::lock_guard<std::mutex> lock(_mutex);
                const auto engine = _planIds.find(plan);
                if(engine == _planIds.end())
                {
                    return HIPDNN_STATUS_BAD_PARAM;
                }
                *static_cast<std::int64_t*>(elements) = engine->second;
                if(count != nullptr)
                {
                    *count = 1;
                }
                return HIPDNN_STATUS_SUCCESS;
            });
    }

    // Successive PROFILING_ELAPSED_MS_EXT reads return these times in order
    // (any float, NaN and negatives included), then 0.
    void installProfilingTimes(std::vector<float> times)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _profilingTimes = std::move(times);
        _nextProfilingTime = 0;
    }

    // Finalizing a hipDNN plan for `engineId` fails, as a plan whose context
    // cannot be created does.
    void failPlanFinalizeFor(std::int64_t engineId)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _failingPlanIds.insert(engineId);
    }

    // Executing a plan of `engineId` fails on its nth (1-based) execution, or
    // on every execution when `nth` is 0.
    void failExecuteFor(std::int64_t engineId, std::size_t nth)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _executeFailures[engineId] = nth;
    }

    // Creating a PROFILING_CONTROL_EXT descriptor fails (recorded as "create failed").
    void failProfilingCreate()
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _failProfilingCreate = true;
    }

    // Profiling-control calls ("create", "reset", "handle", "start", "stop",
    // "finalize", "elapsed", "sync", "destroy") interleaved with "execute".
    std::vector<std::string> profilingTimeline() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _profilingTimeline;
    }

    // Engine ID of every executed plan, in execution order (-1 if unknown).
    std::vector<std::int64_t> executedEngineIds() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        std::vector<std::int64_t> ids;
        ids.reserve(_executions.size());
        for(const CapturedExecute& execution : _executions)
        {
            const auto engine = _planIds.find(execution.plan);
            ids.push_back(engine == _planIds.end() ? -1 : engine->second);
        }
        return ids;
    }

    // How many SetAttribute calls had been recorded when `descriptor` was
    // first finalized, or nullopt if it never was.
    std::optional<std::size_t> setCountAtFinalize(hipdnnBackendDescriptor_t descriptor) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        const auto found = _setCountAtFinalize.find(descriptor);
        if(found == _setCountAtFinalize.end())
        {
            return std::nullopt;
        }
        return found->second;
    }

    std::vector<CapturedSetAttribute> setAttributes() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _setAttributes;
    }

    std::vector<CapturedSetAttribute> setAttributesOn(hipdnnBackendDescriptorType_t type) const
    {
        std::vector<CapturedSetAttribute> all = setAttributes();
        all.erase(std::remove_if(all.begin(),
                                 all.end(),
                                 [type](const CapturedSetAttribute& call) {
                                     return call.descriptorType != type;
                                 }),
                  all.end());
        return all;
    }

    std::vector<CapturedSetAttribute> setAttributesOf(hipdnnBackendDescriptor_t descriptor) const
    {
        std::vector<CapturedSetAttribute> all = setAttributes();
        all.erase(std::remove_if(all.begin(),
                                 all.end(),
                                 [descriptor](const CapturedSetAttribute& call) {
                                     return call.descriptor != descriptor;
                                 }),
                  all.end());
        return all;
    }

    // Descriptors of `type` in creation order.
    std::vector<hipdnnBackendDescriptor_t> created(hipdnnBackendDescriptorType_t type) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        std::vector<hipdnnBackendDescriptor_t> result;
        for(const auto& [descriptor, descriptorType] : _creationOrder)
        {
            if(descriptorType == type)
            {
                result.push_back(descriptor);
            }
        }
        return result;
    }

    hipdnnBackendDescriptorType_t typeOf(hipdnnBackendDescriptor_t descriptor) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        const auto found = _types.find(descriptor);
        EXPECT_NE(found, _types.end()) << "not a mock descriptor";
        return found == _types.end() ? HIPDNN_INVALID_TYPE_EXT : found->second;
    }

    std::vector<hipdnnBackendDescriptor_t> finalized() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _finalized;
    }

    std::vector<hipdnnBackendAttributeName_t> getAttributeNames() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _getAttributeNames;
    }

    std::vector<CapturedExecute> executions() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _executions;
    }

    // Engine ID behind a mock ENGINE, ENGINECFG or EXECUTION_PLAN descriptor, or -1.
    std::int64_t engineIdOf(hipdnnBackendDescriptor_t descriptor) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        const auto engine = _engineIds.find(descriptor);
        if(engine != _engineIds.end())
        {
            return engine->second;
        }
        const auto config = _configIds.find(descriptor);
        if(config != _configIds.end())
        {
            return config->second;
        }
        const auto plan = _planIds.find(descriptor);
        return plan == _planIds.end() ? -1 : plan->second;
    }

    // Hands out a typed mock descriptor, for tests that replace the create action.
    hipdnnBackendDescriptor_t newMockDescriptor(hipdnnBackendDescriptorType_t type)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return allocate(type);
    }

private:
    hipdnnBackendDescriptor_t allocate(hipdnnBackendDescriptorType_t type)
    {
        if(_nextDescriptor >= _pool.size())
        {
            ADD_FAILURE() << "mock descriptor pool exhausted";
            return nullptr;
        }
        auto* descriptor = reinterpret_cast<hipdnnBackendDescriptor_t>(&_pool[_nextDescriptor++]);
        _types[descriptor] = type;
        _creationOrder.emplace_back(descriptor, type);
        return descriptor;
    }

    bool isProfiling(hipdnnBackendDescriptor_t descriptor) const
    {
        const auto found = _types.find(descriptor);
        return found != _types.end() && found->second == HIPDNN_BACKEND_PROFILING_CONTROL_EXT;
    }

    static const char* profilingEventName(hipdnnBackendAttributeName_t name)
    {
        switch(name)
        {
        case HIPDNN_ATTR_PROFILING_RESET_EXT:
            return "reset";
        case HIPDNN_ATTR_PROFILING_HANDLE_EXT:
            return "handle";
        case HIPDNN_ATTR_PROFILING_START_EXT:
            return "start";
        case HIPDNN_ATTR_PROFILING_STOP_EXT:
            return "stop";
        case HIPDNN_ATTR_PROFILING_DEVICE_SYNC_EXT:
            return "sync";
        default:
            return "set";
        }
    }

    hipdnnStatus_t answerExecute(hipdnnBackendDescriptor_t plan)
    {
        const auto engine = _planIds.find(plan);
        if(engine == _planIds.end())
        {
            return HIPDNN_STATUS_SUCCESS;
        }
        const auto failure = _executeFailures.find(engine->second);
        if(failure == _executeFailures.end())
        {
            return HIPDNN_STATUS_SUCCESS;
        }
        const std::size_t run = ++_executeCounts[engine->second];
        if(failure->second != 0 && run != failure->second)
        {
            return HIPDNN_STATUS_SUCCESS;
        }
        _lastError = "mock engine " + std::to_string(engine->second) + " failed to execute";
        return HIPDNN_STATUS_EXECUTION_FAILED;
    }

    static std::size_t elementSize(hipdnnBackendAttributeType_t type)
    {
        switch(type)
        {
        case HIPDNN_TYPE_HANDLE:
        case HIPDNN_TYPE_VOID_PTR:
        case HIPDNN_TYPE_BACKEND_DESCRIPTOR:
            return sizeof(void*);
        case HIPDNN_TYPE_BOOLEAN:
            return sizeof(bool);
        case HIPDNN_TYPE_INT64:
        case HIPDNN_TYPE_DOUBLE:
            return sizeof(std::int64_t);
        case HIPDNN_TYPE_CHAR:
        case HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT:
            return 1;
        case HIPDNN_TYPE_FRACTION:
            return 2 * sizeof(std::int64_t);
        default:
            // FLOAT, INT32 and every enum type.
            return sizeof(std::int32_t);
        }
    }

    void capture(hipdnnBackendDescriptor_t descriptor,
                 hipdnnBackendAttributeName_t name,
                 hipdnnBackendAttributeType_t type,
                 std::int64_t count,
                 const void* elements)
    {
        CapturedSetAttribute call{descriptor, {}, name, type, count, {}};
        const auto found = _types.find(descriptor);
        call.descriptorType = found == _types.end() ? HIPDNN_INVALID_TYPE_EXT : found->second;
        if(elements != nullptr && count > 0)
        {
            call.payload.resize(static_cast<std::size_t>(count) * elementSize(type));
            std::memcpy(call.payload.data(), elements, call.payload.size());
        }
        // An ENGINE built by the shim names its engine through GLOBAL_INDEX.
        if(name == HIPDNN_ATTR_ENGINE_GLOBAL_INDEX && call.payload.size() == sizeof(std::int64_t))
        {
            std::int64_t id = 0;
            std::memcpy(&id, call.payload.data(), sizeof(id));
            _engineIds[descriptor] = id;
        }
        // A config built by the shim names its engine through ENGINECFG_ENGINE.
        if(name == HIPDNN_ATTR_ENGINECFG_ENGINE
           && call.payload.size() == sizeof(hipdnnBackendDescriptor_t))
        {
            hipdnnBackendDescriptor_t engine = nullptr;
            std::memcpy(static_cast<void*>(&engine),
                        call.payload.data(),
                        sizeof(hipdnnBackendDescriptor_t));
            const auto engineId = _engineIds.find(engine);
            if(engineId != _engineIds.end())
            {
                _configIds[descriptor] = engineId->second;
            }
        }
        // A plan built by the shim names its engine through its hipDNN config.
        if(name == HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG
           && call.payload.size() == sizeof(hipdnnBackendDescriptor_t))
        {
            hipdnnBackendDescriptor_t config = nullptr;
            std::memcpy(static_cast<void*>(&config),
                        call.payload.data(),
                        sizeof(hipdnnBackendDescriptor_t));
            const auto engineId = _configIds.find(config);
            if(engineId != _configIds.end())
            {
                _planIds[descriptor] = engineId->second;
            }
        }
        if(call.descriptorType == HIPDNN_BACKEND_PROFILING_CONTROL_EXT)
        {
            _profilingTimeline.emplace_back(profilingEventName(name));
        }
        _setAttributes.push_back(std::move(call));
    }

    hipdnnStatus_t answerGet(hipdnnBackendDescriptor_t descriptor,
                             hipdnnBackendAttributeName_t name,
                             std::int64_t requested,
                             std::int64_t* count,
                             void* elements)
    {
        _getAttributeNames.push_back(name);
        switch(name)
        {
        case HIPDNN_ATTR_ENGINEHEUR_RESULTS:
        {
            const auto available = static_cast<std::int64_t>(_rankedIds.size());
            if(requested == 0 || elements == nullptr)
            {
                *count = available;
                return HIPDNN_STATUS_SUCCESS;
            }
            const std::int64_t written = std::min(requested, available);
            const auto* configs = static_cast<const hipdnnBackendDescriptor_t*>(elements);
            for(std::int64_t i = 0; i < written; ++i)
            {
                _configIds[configs[i]] = _rankedIds[static_cast<std::size_t>(i)];
            }
            if(count != nullptr)
            {
                *count = written;
            }
            return HIPDNN_STATUS_SUCCESS;
        }
        case HIPDNN_ATTR_ENGINECFG_ENGINE:
        {
            const auto config = _configIds.find(descriptor);
            if(config == _configIds.end())
            {
                return HIPDNN_STATUS_BAD_PARAM;
            }
            hipdnnBackendDescriptor_t engine = allocate(HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
            _engineIds[engine] = config->second;
            *static_cast<hipdnnBackendDescriptor_t*>(elements) = engine;
            return HIPDNN_STATUS_SUCCESS;
        }
        case HIPDNN_ATTR_ENGINE_GLOBAL_INDEX:
        {
            const auto engine = _engineIds.find(descriptor);
            if(engine == _engineIds.end())
            {
                return HIPDNN_STATUS_BAD_PARAM;
            }
            *static_cast<std::int64_t*>(elements) = engine->second;
            return HIPDNN_STATUS_SUCCESS;
        }
        case HIPDNN_ATTR_ENGINE_KNOB_INFO:
            return answerKnobInfo(descriptor, requested, count, elements);
        case HIPDNN_ATTR_KNOB_INFO_TYPE:
        case HIPDNN_ATTR_KNOB_INFO_DESCRIPTION:
        case HIPDNN_ATTR_KNOB_INFO_DEPRECATED:
        case HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE:
        case HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE:
        case HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE:
        case HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE:
        case HIPDNN_ATTR_KNOB_INFO_STRIDE:
        case HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT:
            return answerKnobField(descriptor, name, requested, count, elements);
        case HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
        {
            const auto engine = _engineIds.find(descriptor);
            if(engine == _engineIds.end())
            {
                return HIPDNN_STATUS_SUCCESS;
            }
            const auto notes = _behaviorNotes.find(engine->second);
            return answerArray(notes == _behaviorNotes.end()
                                   ? std::vector<hipdnnBackendBehaviorNote_t>{}
                                   : notes->second,
                               requested,
                               count,
                               elements);
        }
        case HIPDNN_ATTR_ENGINE_NAME_EXT:
        {
            const auto engine = _engineIds.find(descriptor);
            const auto engineName = engine == _engineIds.end() ? _engineNames.end()
                                                               : _engineNames.find(engine->second);
            if(engineName == _engineNames.end())
            {
                return HIPDNN_STATUS_NOT_SUPPORTED;
            }
            return answerString(engineName->second, requested, count, elements);
        }
        case HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT:
        {
            _profilingTimeline.emplace_back("elapsed");
            const float time = _nextProfilingTime < _profilingTimes.size()
                                   ? _profilingTimes[_nextProfilingTime++]
                                   : 0.0F;
            *static_cast<float*>(elements) = time;
            if(count != nullptr)
            {
                *count = 1;
            }
            return HIPDNN_STATUS_SUCCESS;
        }
        default:
            return HIPDNN_STATUS_SUCCESS;
        }
    }

    template <typename T>
    static hipdnnStatus_t answerArray(const std::vector<T>& values,
                                      std::int64_t requested,
                                      std::int64_t* count,
                                      void* elements)
    {
        const auto available = static_cast<std::int64_t>(values.size());
        if(requested == 0 || elements == nullptr)
        {
            if(count != nullptr)
            {
                *count = available;
            }
            return HIPDNN_STATUS_SUCCESS;
        }
        const std::int64_t written = std::min(requested, available);
        if(written > 0)
        {
            std::memcpy(elements, values.data(), static_cast<std::size_t>(written) * sizeof(T));
        }
        if(count != nullptr)
        {
            *count = written;
        }
        return HIPDNN_STATUS_SUCCESS;
    }

    static hipdnnStatus_t answerString(const std::string& value,
                                       std::int64_t requested,
                                       std::int64_t* count,
                                       void* elements)
    {
        std::vector<char> chars(value.begin(), value.end());
        chars.push_back('\0');
        return answerArray(chars, requested, count, elements);
    }

    template <typename T>
    static hipdnnStatus_t answerOptional(const std::optional<T>& value,
                                         std::int64_t requested,
                                         std::int64_t* count,
                                         void* elements)
    {
        return answerArray(
            value ? std::vector<T>{*value} : std::vector<T>{}, requested, count, elements);
    }

    hipdnnStatus_t answerKnobInfo(hipdnnBackendDescriptor_t engine,
                                  std::int64_t requested,
                                  std::int64_t* count,
                                  void* elements)
    {
        const auto engineId = _engineIds.find(engine);
        const auto knobs = engineId == _engineIds.end() ? _engineKnobs.end()
                                                        : _engineKnobs.find(engineId->second);
        const std::size_t available = knobs == _engineKnobs.end() ? 0 : knobs->second.size();
        if(requested == 0 || elements == nullptr)
        {
            if(count != nullptr)
            {
                *count = static_cast<std::int64_t>(available);
            }
            return HIPDNN_STATUS_SUCCESS;
        }
        const auto written = std::min(static_cast<std::size_t>(requested), available);
        auto* infos = static_cast<hipdnnBackendDescriptor_t*>(elements);
        for(std::size_t i = 0; i < written; ++i)
        {
            infos[i] = allocate(HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR);
            _knobInfos[infos[i]] = knobs->second[i];
        }
        if(count != nullptr)
        {
            *count = static_cast<std::int64_t>(written);
        }
        return HIPDNN_STATUS_SUCCESS;
    }

    hipdnnStatus_t answerKnobField(hipdnnBackendDescriptor_t info,
                                   hipdnnBackendAttributeName_t name,
                                   std::int64_t requested,
                                   std::int64_t* count,
                                   void* elements)
    {
        const auto found = _knobInfos.find(info);
        if(found == _knobInfos.end())
        {
            return HIPDNN_STATUS_BAD_PARAM;
        }
        const MockKnob& knob = found->second;
        switch(name)
        {
        case HIPDNN_ATTR_KNOB_INFO_TYPE:
            return answerString(knob.id, requested, count, elements);
        case HIPDNN_ATTR_KNOB_INFO_DESCRIPTION:
            return answerString("mock knob " + knob.id, requested, count, elements);
        case HIPDNN_ATTR_KNOB_INFO_DEPRECATED:
            // One byte, the size of the bool the reader passes.
            return answerArray(std::vector<char>{0}, requested, count, elements);
        case HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE:
            return answerArray(
                std::vector<std::int64_t>{static_cast<std::int64_t>(HIPDNN_TYPE_INT64)},
                requested,
                count,
                elements);
        case HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE:
            return answerArray(
                std::vector<std::int64_t>{knob.defaultValue}, requested, count, elements);
        case HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE:
            return answerOptional(knob.minValue, requested, count, elements);
        case HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE:
            return answerOptional(knob.maxValue, requested, count, elements);
        case HIPDNN_ATTR_KNOB_INFO_STRIDE:
            return answerOptional(knob.stride, requested, count, elements);
        default:
            return answerArray(knob.validValues, requested, count, elements);
        }
    }

    mutable std::mutex _mutex;
    std::vector<char> _pool;
    std::size_t _nextDescriptor = 0;
    std::unordered_map<hipdnnBackendDescriptor_t, hipdnnBackendDescriptorType_t> _types;
    std::vector<std::pair<hipdnnBackendDescriptor_t, hipdnnBackendDescriptorType_t>> _creationOrder;
    std::vector<CapturedSetAttribute> _setAttributes;
    std::vector<hipdnnBackendDescriptor_t> _finalized;
    std::vector<hipdnnBackendAttributeName_t> _getAttributeNames;
    std::vector<CapturedExecute> _executions;
    std::vector<std::int64_t> _rankedIds;
    std::set<std::int64_t> _failingConfigIds;
    std::unordered_map<hipdnnBackendDescriptor_t, std::int64_t> _configIds;
    std::unordered_map<hipdnnBackendDescriptor_t, std::int64_t> _engineIds;
    std::unordered_map<hipdnnBackendDescriptor_t, std::size_t> _setCountAtFinalize;
    std::unordered_map<std::int64_t, std::vector<MockKnob>> _engineKnobs;
    std::unordered_map<hipdnnBackendDescriptor_t, MockKnob> _knobInfos;
    std::unordered_map<std::int64_t, std::vector<hipdnnBackendBehaviorNote_t>> _behaviorNotes;
    std::unordered_map<std::int64_t, std::string> _engineNames;
    std::unordered_map<hipdnnBackendDescriptor_t, std::int64_t> _planIds;
    std::vector<std::uint8_t> _serializedPlan;
    std::int64_t _lastSerializedEngineId = -1;
    std::vector<float> _profilingTimes;
    std::size_t _nextProfilingTime = 0;
    bool _failProfilingCreate = false;
    std::vector<std::string> _profilingTimeline;
    std::map<std::int64_t, std::size_t> _executeFailures;
    std::map<std::int64_t, std::size_t> _executeCounts;
    std::set<std::int64_t> _failingPlanIds;
    std::string _lastError;
};

} // namespace hipdnn_shim_test
