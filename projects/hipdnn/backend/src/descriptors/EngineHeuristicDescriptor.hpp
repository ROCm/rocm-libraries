// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>

#include "BackendDescriptor.hpp"

namespace hipdnn_backend
{

class GraphDescriptor;

class EngineHeuristicDescriptor : public HipdnnBackendDescriptorImpl<EngineHeuristicDescriptor>
{
private:
    std::shared_ptr<const GraphDescriptor> _graph;
    std::vector<int64_t> _engineIds;
    hipdnnBackendHeurMode_t _heuristicMode = HIPDNN_HEUR_MODE_FALLBACK;
    bool _heuristicModeSet = false;

    void setGraph(hipdnnBackendAttributeType_t attributeType,
                  int64_t elementCount,
                  const void* arrayOfElements);

    void setHeuristicMode(hipdnnBackendAttributeType_t attributeType,
                          int64_t elementCount,
                          const void* arrayOfElements);

    void getGraph(hipdnnBackendAttributeType_t attributeType,
                  int64_t requestedElementCount,
                  int64_t* elementCount,
                  void* arrayOfElements) const;

    void getEngineConfigs(hipdnnBackendAttributeType_t attributeType,
                          int64_t requestedElementCount,
                          int64_t* elementCount,
                          void* arrayOfElements) const;

    void getHeuristicMode(hipdnnBackendAttributeType_t attributeType,
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

    // Throws an exception if the descriptor is not finalized.
    std::shared_ptr<const GraphDescriptor> getGraph() const;

    static hipdnnBackendDescriptorType_t getStaticType();
};

} // namespace hipdnn_backend
