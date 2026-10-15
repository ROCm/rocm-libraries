// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"

#include <HipdnnBackendBehaviorNote.h>
#include <flatbuffers/detached_buffer.h>

namespace hipdnn_backend
{

class GraphDescriptor;
class KnobDescriptor;

namespace plugin
{
class EngineDetailsWrapper;
class EnginePluginResourceManager;
}

class EngineDescriptor : public HipdnnBackendDescriptorImpl<EngineDescriptor>
{
private:
    std::shared_ptr<const GraphDescriptor> _graph;
    int64_t _engineId;
    bool _engineIdSet = false;
    std::shared_ptr<const plugin::EngineDetailsWrapper> _engineDetails;
    std::vector<flatbuffers::DetachedBuffer> _knobSerializedBuffers;
    std::vector<hipdnnBackendBehaviorNote_t> _behaviorNotes;

    /// Name resolved during finalize() by
    /// EnginePluginResourceManager::resolveEngineName(); empty until then. Either
    /// a declared name, which hashes to _engineId, or that ID rendered as hex.
    std::string _engineName;

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

    void getKnobInfo(hipdnnBackendAttributeType_t attributeType,
                     int64_t requestedElementCount,
                     int64_t* elementCount,
                     void* arrayOfElements) const;

    void getKnobInfoDescriptors(hipdnnBackendAttributeType_t attributeType,
                                int64_t requestedElementCount,
                                int64_t* elementCount,
                                void* arrayOfElements) const;

    void getBehaviorNotes(hipdnnBackendAttributeType_t attributeType,
                          int64_t requestedElementCount,
                          int64_t* elementCount,
                          void* arrayOfElements) const;

    /// Populated during finalize() from _knobSerializedBuffers.
    std::vector<std::shared_ptr<KnobDescriptor>> _knobDescriptors;

    /// Loads engine details, behavior notes, knobs and the resolved name for
    /// _engineId, then marks the descriptor finalized. Requires _graph and
    /// _engineId to be set and _engineId to be applicable to _graph.
    void loadEngineDetailsAndFinalize(
        const std::shared_ptr<plugin::EnginePluginResourceManager>& pluginResourceManager);

public:
    void finalize() override;

    /// Builds a finalized engine for an ID the caller has just obtained from
    /// EnginePluginResourceManager::getApplicableEngineIds() for this finalized
    /// graph, so the applicability probe that finalize() runs is skipped. Not
    /// reachable via the C API.
    static std::shared_ptr<EngineDescriptor>
        createForApplicableEngine(std::shared_ptr<const GraphDescriptor> graph, int64_t engineId);

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

    std::string toString() const override;
};

} // namespace hipdnn_backend
