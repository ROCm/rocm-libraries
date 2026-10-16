// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Mock-backend fixture for the cuDNN v8 backend shim: records every hipDNN
// descriptor it hands out with its type, every SetAttribute with a copy of its
// payload, and executions, and can play back a heuristic result list.
#pragma once

#include "MockBackendFixture.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
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
};

struct CapturedExecute
{
    hipdnnHandle_t handle;
    hipdnnBackendDescriptor_t plan;
    hipdnnBackendDescriptor_t variantPack;
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
                *descriptor = allocate(type);
                return *descriptor == nullptr ? HIPDNN_STATUS_ALLOC_FAILED : HIPDNN_STATUS_SUCCESS;
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
                const auto config = _configIds.find(descriptor);
                if(config != _configIds.end() && _failingConfigIds.count(config->second) != 0)
                {
                    _lastError = "mock engine config " + std::to_string(config->second)
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
                return HIPDNN_STATUS_SUCCESS;
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

    // Engine ID behind a mock ENGINE or ENGINECFG descriptor, or -1.
    std::int64_t engineIdOf(hipdnnBackendDescriptor_t descriptor) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        const auto engine = _engineIds.find(descriptor);
        if(engine != _engineIds.end())
        {
            return engine->second;
        }
        const auto config = _configIds.find(descriptor);
        return config == _configIds.end() ? -1 : config->second;
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
        default:
            return HIPDNN_STATUS_SUCCESS;
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
    std::string _lastError;
};

} // namespace hipdnn_shim_test
