// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "Context.hpp"
#include "Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanCapture.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanRestore.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"

// Saves only the plans that an ingestor engine built. A context without a plan gets the
// non-ingestor refusal.
hipdnnPluginStatus_t hipdnnEnginePluginSerializeExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext,
    hipdnnPluginConstData_t* serializedContext)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", executionContext=" << static_cast<void*>(executionContext)
                            << ", serializedContext=" << static_cast<void*>(serializedContext));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(handle);
        hipdnn_plugin_sdk::throwIfNull(executionContext);
        hipdnn_plugin_sdk::throwIfNull(serializedContext);

        const auto& typedHandle = *static_cast<const Handle*>(handle);
        const auto& typedContext = *static_cast<const Context*>(executionContext);

        namespace kie = hip_kernel_provider::kernel_ingestor_engine;
        if(!typedContext.hasValidPlan())
        {
            kie::refuseNonIngestorPlanSave();
        }

        const std::vector<uint8_t> encoded = kie::serialization::encodeIngestorPlan(
            kie::captureIngestorPlan(typedContext.plan(), typedHandle));

        // The destroy hook frees this buffer with delete[].
        // NOLINTNEXTLINE(modernize-avoid-c-arrays)
        auto bytes = std::make_unique<uint8_t[]>(encoded.size());
        std::memcpy(bytes.get(), encoded.data(), encoded.size());

        serializedContext->size = encoded.size();
        serializedContext->ptr = bytes.release();

        LOG_API_SUCCESS(apiName, "serializedContext->size=" << serializedContext->size);
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginDestroySerializedExecutionContext(hipdnnEnginePluginHandle_t handle,
                                                        hipdnnPluginConstData_t* serializedContext)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", serializedContext=" << static_cast<void*>(serializedContext));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(handle);
        hipdnn_plugin_sdk::throwIfNull(serializedContext);
        hipdnn_plugin_sdk::throwIfNull(serializedContext->ptr);

        delete[] static_cast<const uint8_t*>(serializedContext->ptr);
        serializedContext->ptr = nullptr;
        serializedContext->size = 0;

        LOG_API_SUCCESS(apiName, "destroyed serializedContext");
    });
}

// The bytes hold GPU code, and the restored plan executes that code. Load only bytes from a
// trusted source.
hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContextFromSerialized(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t* serializedContext,
    hipdnnEnginePluginExecutionContext_t* executionContext)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", serializedContext=" << static_cast<const void*>(serializedContext)
                            << ", executionContext=" << static_cast<void*>(executionContext));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(handle);
        hipdnn_plugin_sdk::throwIfNull(serializedContext);
        hipdnn_plugin_sdk::throwIfNull(serializedContext->ptr);
        hipdnn_plugin_sdk::throwIfNull(executionContext);

        const auto& typedHandle = *static_cast<const Handle*>(handle);
        auto plan = hip_kernel_provider::kernel_ingestor_engine::restoreIngestorPlan(
            static_cast<const uint8_t*>(serializedContext->ptr),
            serializedContext->size,
            typedHandle);

        auto* context = new Context;
        try
        {
            context->setPlan(std::move(plan));
        }
        catch(...)
        {
            delete context;
            throw;
        }

        *executionContext = static_cast<hipdnnEnginePluginExecutionContext_t>(context);

        LOG_API_SUCCESS(apiName,
                        "createdExecutionContext=" << static_cast<void*>(*executionContext));
    });
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
