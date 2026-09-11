// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"
#include <flatbuffers/detached_buffer.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <mutex>
#include <vector>

namespace hipdnn_backend
{

class EngineDescriptor;
class GraphDescriptor;

class EngineConfigDescriptor : public HipdnnBackendDescriptorImpl<EngineConfigDescriptor>
{
private:
    std::shared_ptr<const EngineDescriptor> _engine;
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::EngineConfigT> _engineConfigData;
    mutable flatbuffers::DetachedBuffer _engineConfigSerializedBuffer;
    mutable int64_t _maxWorkspaceSize = INVALID_WORKSPACE_SIZE;
    mutable std::once_flag _workspaceOnce;
    bool _deferWorkspace = false;
    bool _predictionEvaluate = true;
    mutable flatbuffers::DetachedBuffer _predictionBuffer;
    /// Buffers already handed out. HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT is readable
    /// while the descriptor still accepts setAttribute, and its bytes are documented to
    /// live until the descriptor dies, so an invalidated buffer is retired, not freed.
    mutable std::vector<flatbuffers::DetachedBuffer> _retiredPredictions;
    mutable std::mutex _predictionMutex;

    void ensureWorkspaceSize() const;

    /// Packs the configuration-kind prediction once per input state; setAttribute retires
    /// the previous buffer so a caller that still holds it keeps reading valid bytes.
    const flatbuffers::DetachedBuffer& ensurePrediction() const;

    /// Retires any packed prediction, under the prediction lock.
    void invalidatePrediction();

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

    void getPrediction(hipdnnBackendAttributeType_t attributeType,
                       int64_t requestedElementCount,
                       int64_t* elementCount,
                       void* arrayOfElements) const;

    void setKnobChoice(hipdnnBackendAttributeType_t attributeType,
                       int64_t elementCount,
                       const void* arrayOfElements);

    void setKnobSettingDescriptor(hipdnnBackendAttributeType_t attributeType,
                                  int64_t elementCount,
                                  const void* arrayOfElements);

public:
    EngineConfigDescriptor();
    static constexpr int64_t INVALID_WORKSPACE_SIZE = -1;

    /// Maximum number of knob choices that can be set on a single engine config.
    static constexpr int64_t MAX_KNOB_CHOICES = 1024;

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

    /// Copies constraints without engine initialization or selector work.
    hipdnn_flatbuffers_sdk::data_objects::EngineConfigT
        getEngineConfigForPrediction(int64_t engineId, const GraphDescriptor* graph) const;

    virtual hipdnnPluginConstData_t getSerializedEngineConfig() const;

    /// Copies the complete configuration after the matching engine is assigned.
    /// Heuristic results may defer workspace selection until explicitly requested.
    void setEngineConfig(const hipdnn_flatbuffers_sdk::data_objects::EngineConfigT& config,
                         bool deferWorkspace = false);

    static void
        validateEngineConfig(const hipdnn_flatbuffers_sdk::data_objects::EngineConfigT& config,
                             int64_t engineId);

    std::string toString() const override;
};

} // namespace hipdnn_backend
