// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::detail
{

// Forwards every call to the wrapped engine. Records the engine config's ID on each context
// that the wrapped engine initializes. The save hook reads this ID to find the engine.
class EngineIdRecordingEngine final : public hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>
{
public:
    // Throws HipdnnPluginException with INTERNAL_ERROR when engine is null.
    explicit EngineIdRecordingEngine(
        std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> engine)
        : _engine(std::move(engine))
    {
        if(_engine == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "an engine ID recording engine needs an engine to wrap");
        }
    }

    int64_t id() const override
    {
        return _engine->id();
    }

    bool isApplicable(
        Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override
    {
        return _engine->isApplicable(handle, opGraph);
    }

    void getDetails(Handle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override
    {
        _engine->getDetails(handle, opGraph, detailsOut);
    }

    size_t getMaxWorkspaceSize(const Handle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                   engineConfig) const override
    {
        return _engine->getMaxWorkspaceSize(handle, opGraph, engineConfig);
    }

    // Records the engine ID after the wrapped engine returns. A build that throws leaves the
    // context without an engine ID.
    void initializeExecutionContext(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        Context& executionContext) const override
    {
        _engine->initializeExecutionContext(handle, opGraph, engineConfig, executionContext);
        executionContext.setEngineId(engineConfig.engineId());
    }

private:
    std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> _engine;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
