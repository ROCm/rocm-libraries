// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"

namespace hipdnn_backend
{

class GraphDescriptor;

namespace plugin
{
class EngineDetailsWrapper;
}

class EngineDescriptor : public HipdnnBackendDescriptorImpl<EngineDescriptor>
{
private:
    std::shared_ptr<const GraphDescriptor> _graph;
    int64_t _engineId;
    bool _engineIdSet = false;
    std::shared_ptr<const plugin::EngineDetailsWrapper> _engineDetails;

    void setGraph(hipdnnBackendAttributeType_t attributeType,
                  int64_t elementCount,
                  const void* arrayOfElements);

    void getGraph(hipdnnBackendAttributeType_t attributeType,
                  int64_t requestedElementCount,
                  int64_t* elementCount,
                  void* arrayOfElements) const;

    void setGlobalId(hipdnnBackendAttributeType_t attributeType,
                     int64_t elementCount,
                     const void* arrayOfElements);

    void getGlobalId(hipdnnBackendAttributeType_t attributeType,
                     int64_t requestedElementCount,
                     int64_t* elementCount,
                     void* arrayOfElements) const;

public:
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

    // These getters throw an exception if the descriptor is not finalized.
    virtual std::shared_ptr<const GraphDescriptor> getGraph() const;
    virtual int64_t getEngineId() const;

    static hipdnnBackendDescriptorType_t getStaticType();
};

} // namespace hipdnn_backend
