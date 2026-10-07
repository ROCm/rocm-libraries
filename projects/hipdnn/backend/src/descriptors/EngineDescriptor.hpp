// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"

#include <HipdnnBackendBehaviorNote.h>
#include <flatbuffers/detached_buffer.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>

#include <memory>
#include <mutex>
#include <vector>

namespace hipdnn_backend
{

class GraphDescriptor;
class KnobDescriptor;

namespace plugin
{
class EngineDetailsWrapper;
}

class EngineDescriptor : public HipdnnBackendDescriptorImpl<EngineDescriptor>,
                         public std::enable_shared_from_this<EngineDescriptor>
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

    void getKnobCatalogDescriptors(hipdnnBackendAttributeType_t attributeType,
                                   int64_t requestedElementCount,
                                   int64_t* elementCount,
                                   void* arrayOfElements) const;

    /// Fetches the catalog from the plugin, validates it and stores the entries in
    /// _knobCatalog. Runs at most once successfully, from the first read of
    /// HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT.
    void loadKnobCatalog() const;

    /// Populated during finalize() from _knobSerializedBuffers.
    std::vector<std::shared_ptr<KnobDescriptor>> _knobDescriptors;

    /// Validated catalog entries, populated lazily by loadKnobCatalog(). Kept as plain
    /// data rather than finalized descriptors: a config descriptor holds its engine, so
    /// owning them here would be a reference cycle.
    mutable std::once_flag _knobCatalogOnce;
    mutable std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::EngineConfigT>>
        _knobCatalog;

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

    std::string toString() const override;
};

} // namespace hipdnn_backend
