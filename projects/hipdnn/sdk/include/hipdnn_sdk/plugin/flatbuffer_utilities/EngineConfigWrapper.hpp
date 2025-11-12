// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>

#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/plugin/PluginException.hpp>

namespace hipdnn_plugin
{

class IEngineConfig
{
public:
    virtual ~IEngineConfig() = default;

    virtual const hipdnn_sdk::data_objects::EngineConfig& getEngineConfig() const = 0;
    virtual bool isValid() const = 0;
    virtual int64_t engineId() const = 0;
};

class EngineConfigWrapper : public IEngineConfig
{
public:
    explicit EngineConfigWrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_sdk::data_objects::EngineConfig>())
            {
                _shallowEngineConfig
                    = flatbuffers::GetRoot<hipdnn_sdk::data_objects::EngineConfig>(buffer);
            }
        }
    }

    const hipdnn_sdk::data_objects::EngineConfig& getEngineConfig() const override
    {
        throwIfNotValid();

        return *_shallowEngineConfig;
    }

    bool isValid() const override
    {
        return _shallowEngineConfig != nullptr;
    }

    int64_t engineId() const override
    {
        throwIfNotValid();

        return _shallowEngineConfig->engine_id();
    }

private:
    void throwIfNotValid() const
    {
        if(!isValid())
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Engine config is not valid");
        }
    }

    // Pointer to the flatbuffer representation of the engine config. We do not own this memory
    // as were just reading from the buffer passed during construction.
    const hipdnn_sdk::data_objects::EngineConfig* _shallowEngineConfig = nullptr;
};

}
