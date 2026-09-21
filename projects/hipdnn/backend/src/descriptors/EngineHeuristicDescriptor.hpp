// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <vector>

#include "BackendDescriptor.hpp"

namespace hipdnn_backend
{

class GraphDescriptor;

namespace heuristics
{
class SelectionHeuristic;
}

class EngineHeuristicDescriptor : public HipdnnBackendDescriptorImpl<EngineHeuristicDescriptor>
{
private:
    std::shared_ptr<const GraphDescriptor> _graph;
    std::vector<int64_t> _engineIds;
    hipdnnBackendHeurMode_t _heuristicMode = HIPDNN_HEUR_MODE_FALLBACK;
    bool _heuristicModeSet = false;
    bool _findFirst = false;

    // Heuristics framework integration
    std::vector<int64_t> _orderedPolicyIds;
    std::vector<std::unique_ptr<heuristics::SelectionHeuristic>> _policySlots;
    std::vector<int64_t> _policyOrder; // descriptor-level policy IDs
    bool _policyOrderSet = false;

    // Resolve policy order from descriptor/handle/env/default
    std::vector<int64_t> resolveHeuristicPolicyOrder();

    // Ensure policy slots match orderedPolicyIds
    void syncPolicySlots(const std::vector<int64_t>& orderedPolicyIds);

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

    void setFindFirst(hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements);

    void getFindFirst(hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) const;

    void setPolicyOrder(hipdnnBackendAttributeType_t attributeType,
                        int64_t elementCount,
                        const void* arrayOfElements);

    void getPolicyOrder(hipdnnBackendAttributeType_t attributeType,
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

    std::string toString() const override;
};

} // namespace hipdnn_backend
