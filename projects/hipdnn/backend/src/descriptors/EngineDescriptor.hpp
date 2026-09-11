// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"

#include <HipdnnBackendBehaviorNote.h>
#include <atomic>
#include <flatbuffers/detached_buffer.h>
#include <mutex>

namespace hipdnn_backend
{

class GraphDescriptor;
class KnobDescriptor;
class KnobSettingDescriptor;

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
    mutable std::shared_ptr<const plugin::EngineDetailsWrapper> _engineDetails;
    mutable std::vector<flatbuffers::DetachedBuffer> _knobSerializedBuffers;
    mutable std::vector<hipdnnBackendBehaviorNote_t> _behaviorNotes;
    mutable std::once_flag _detailsOnce;
    mutable std::atomic<bool> _detailsLoaded{false};

    /// Generation-tool inspection inputs, frozen at finalize.
    int64_t _candidateOffset = 0;
    int64_t _candidateLimit = MAX_CANDIDATE_LIMIT;
    bool _predictionEvaluate = true;
    std::vector<std::shared_ptr<const KnobSettingDescriptor>> _candidateScope;

    /// Computed on first read and owned by this descriptor, each behind its own flag so
    /// ordinary details caching is untouched.
    mutable std::vector<uint8_t> _candidatePage;
    mutable std::once_flag _candidatesOnce;
    mutable flatbuffers::DetachedBuffer _prediction;
    mutable std::once_flag _predictionOnce;

    /// Resolved with details; publication is guarded by _detailsOnce/_detailsLoaded.
    mutable std::string _engineName;

    void ensureDetailsLoaded(const std::shared_ptr<plugin::EnginePluginResourceManager>& manager
                             = nullptr) const;

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

    /// Enumerates the matched catalog once. Engines that cannot enumerate propagate
    /// HIPDNN_STATUS_NOT_SUPPORTED; an empty page is never substituted.
    const std::vector<uint8_t>& ensureCandidates() const;

    void getCandidates(hipdnnBackendAttributeType_t attributeType,
                       int64_t requestedElementCount,
                       int64_t* elementCount,
                       void* arrayOfElements) const;

    /// Packs the engine-kind prediction once.
    const flatbuffers::DetachedBuffer& ensurePrediction() const;

    void getPrediction(hipdnnBackendAttributeType_t attributeType,
                       int64_t requestedElementCount,
                       int64_t* elementCount,
                       void* arrayOfElements) const;

    void setCandidateScope(hipdnnBackendAttributeType_t attributeType,
                           int64_t elementCount,
                           const void* arrayOfElements);

    void setInspectionScalar(hipdnnBackendAttributeName_t attributeName,
                             hipdnnBackendAttributeType_t attributeType,
                             int64_t elementCount,
                             const void* arrayOfElements);

    /// Immutable after the first successful details query.
    mutable std::vector<std::shared_ptr<KnobDescriptor>> _knobDescriptors;

public:
    /// Largest catalog page an engine will return in one read, per
    /// HIPDNN_ATTR_ENGINE_CANDIDATE_LIMIT_EXT. Also the default page size.
    static constexpr int64_t MAX_CANDIDATE_LIMIT = 10000;

    void finalize() override;

    /// Internal policy materialization for an engine already proven applicable.
    /// Defers provider metadata; ordinary public finalize retains applicability validation.
    void initializeHeuristicResult(std::shared_ptr<const GraphDescriptor> graph, int64_t engineId);

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
