// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>

#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/plugin/PluginException.hpp>

namespace hipdnn_plugin
{

class IEngineDetails
{
public:
    virtual ~IEngineDetails() = default;

    virtual const hipdnn_sdk::data_objects::EngineDetails& getEngineDetails() const = 0;
    virtual bool isValid() const = 0;
    virtual int64_t engineId() const = 0;
};

class EngineDetailsWrapper : public IEngineDetails
{
public:
    explicit EngineDetailsWrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_sdk::data_objects::EngineDetails>())
            {
                _shallowEngineDetails
                    = flatbuffers::GetRoot<hipdnn_sdk::data_objects::EngineDetails>(buffer);
            }
        }
    }

    const hipdnn_sdk::data_objects::EngineDetails& getEngineDetails() const override
    {
        throwIfNotValid();
        return *_shallowEngineDetails;
    }

    bool isValid() const override
    {
        return _shallowEngineDetails != nullptr;
    }

    int64_t engineId() const override
    {
        throwIfNotValid();

        return _shallowEngineDetails->engine_id();
    }

private:
    void throwIfNotValid() const
    {
        if(!isValid())
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Engine details is not valid");
        }
    }

    // Pointer to the flatbuffer representation of the engine details. We do not own this memory
    // as were just reading from the buffer passed during construction.
    const hipdnn_sdk::data_objects::EngineDetails* _shallowEngineDetails = nullptr;
};

}
