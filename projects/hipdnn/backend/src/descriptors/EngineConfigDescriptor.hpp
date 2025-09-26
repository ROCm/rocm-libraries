// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"
#include <flatbuffers/detached_buffer.h>
#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>

namespace hipdnn_backend
{

class EngineDescriptor;

class EngineConfigDescriptor : public HipdnnBackendDescriptorImpl<EngineConfigDescriptor>
{
private:
    std::shared_ptr<const EngineDescriptor> _engine;
    std::unique_ptr<hipdnn_sdk::data_objects::EngineConfigT> _engineConfigData;
    mutable flatbuffers::DetachedBuffer _engineConfigSerializedBuffer;
    int64_t _maxWorkspaceSize = INVALID_WORKSPACE_SIZE;

    void setEngine(hipdnnBackendAttributeType_t attributeType,
                   int64_t elementCount,
                   const void* arrayOfElements);

    void getEngine(hipdnnBackendAttributeType_t attributeType,
                   int64_t requestedElementCount,
                   int64_t* elementCount,
                   void* arrayOfElements) const;

    void getMaxWorkspaceSize(hipdnnBackendAttributeType_t attributeType,
                             int64_t requestedElementCount,
                             int64_t* elementCount,
                             void* arrayOfElements) const;

public:
    EngineConfigDescriptor();
    static constexpr int64_t INVALID_WORKSPACE_SIZE = -1;

    void finalize() override;

    void getAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) const override;

    void setAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements) override;

    static hipdnnBackendDescriptorType_t getStaticType();

    // Throws an exception if the descriptor is not finalized before calling these.
    virtual std::shared_ptr<const EngineDescriptor> getEngine() const;

    virtual hipdnnPluginConstData_t getSerializedEngineConfig() const;
};

} // namespace hipdnn_backend
