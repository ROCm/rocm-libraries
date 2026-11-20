// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct EngineInfo
{
    int64_t engineId;
    std::string engineName;
    std::string pluginName;
    std::string version;
    std::string type;
};

class HandleWrapper
{
private:
    hipdnn_frontend::HipdnnHandlePtr _handle;

    void checkNotDestroyed() const
    {
        if(!_handle)
        {
            throw std::runtime_error("Handle has been destroyed");
        }
    }

public:
    HandleWrapper()
    {
        const auto error = hipdnn_frontend::createHipdnnHandle(_handle);
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to create hipdnn handle: " + error.get_message());
        }
    }

    explicit HandleWrapper(uintptr_t streamPtr)
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto stream = reinterpret_cast<hipStream_t>(streamPtr);
        const auto error = hipdnn_frontend::createHipdnnHandle(_handle, stream);
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to create hipdnn handle: " + error.get_message());
        }
    }

    hipdnnHandle_t get() const
    {
        checkNotDestroyed();
        return *_handle;
    }

    bool isValid() const
    {
        return _handle != nullptr;
    }

    void destroy()
    {
        _handle.reset();
    }

    void setStream(uintptr_t streamPtr)
    {
        checkNotDestroyed();
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto stream = reinterpret_cast<hipStream_t>(streamPtr);
        const auto error = hipdnn_frontend::setHipdnnHandleStream(_handle, stream);
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to set stream on hipdnn handle: "
                                     + error.get_message());
        }
    }

    uintptr_t getStream() const
    {
        checkNotDestroyed();
        hipStream_t stream = nullptr;
        const auto error = hipdnn_frontend::getHipdnnHandleStream(_handle, &stream);
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to get stream from hipdnn handle: "
                                     + error.get_message());
        }
        return reinterpret_cast<uintptr_t>(stream);
    }
    std::string engineIdToName(int64_t engineId) const
    {
        checkNotDestroyed();
        size_t engineNameLen = 0;
        if(hipdnnGetEngineNameById_ext(get(), engineId, nullptr, &engineNameLen)
           != HIPDNN_STATUS_SUCCESS)
        {
            throw std::out_of_range("Engine ID is not loaded");
        }
        std::vector<char> engineName(engineNameLen);
        if(hipdnnGetEngineNameById_ext(get(), engineId, engineName.data(), &engineNameLen)
           != HIPDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error("Failed to resolve engine name");
        }
        return {engineName.data()};
    }

    EngineInfo getEngineInfo(int64_t engineId) const
    {
        checkNotDestroyed();
        size_t count = 0;
        if(hipdnnGetEngineCount_ext(get(), &count) != HIPDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error("Failed to query loaded engine count");
        }
        for(size_t index = 0; index < count; ++index)
        {
            int64_t candidateId = 0;
            size_t engineNameLen = 0;
            size_t pluginNameLen = 0;
            size_t versionLen = 0;
            size_t typeLen = 0;
            if(hipdnnGetEngineInfo_ext(get(),
                                       index,
                                       &candidateId,
                                       nullptr,
                                       &engineNameLen,
                                       nullptr,
                                       &pluginNameLen,
                                       nullptr,
                                       &versionLen,
                                       nullptr,
                                       &typeLen)
               != HIPDNN_STATUS_SUCCESS)
            {
                throw std::runtime_error("Failed to query loaded engine metadata sizes");
            }
            if(candidateId != engineId)
            {
                continue;
            }
            std::vector<char> engineName(engineNameLen);
            std::vector<char> pluginName(pluginNameLen);
            std::vector<char> version(versionLen);
            std::vector<char> type(typeLen);
            if(hipdnnGetEngineInfo_ext(get(),
                                       index,
                                       &candidateId,
                                       engineName.data(),
                                       &engineNameLen,
                                       pluginName.data(),
                                       &pluginNameLen,
                                       version.data(),
                                       &versionLen,
                                       type.data(),
                                       &typeLen)
               != HIPDNN_STATUS_SUCCESS)
            {
                throw std::runtime_error("Failed to query loaded engine metadata");
            }

            return {candidateId,
                    std::string(engineName.data()),
                    std::string(pluginName.data()),
                    std::string(version.data()),
                    std::string(type.data())};
        }
        throw std::out_of_range("Engine ID is not loaded");
    }
};
