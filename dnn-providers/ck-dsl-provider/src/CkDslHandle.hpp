// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <memory>
#include <unordered_map>

#include "CkDslContext.hpp"
#include "CkDslSettings.hpp"

namespace ck_dsl_provider {
class CkDslContainer;
}

/// Plugin handle for the CK DSL provider.
///
/// Inherits from HipdnnEnginePluginHandle for opaque-pointer
/// compatibility. The handle owns the HIP stream, a shared pointer to
/// the plugin container (so multiple handles can share engine state),
/// and a map of detached FlatBuffer buffers backing per-call
/// hipdnnPluginConstData_t payloads handed back to the SDK.
///
/// Sits at namespace scope (not inside ck_dsl_provider) to match the
/// EnginePluginImpl.inl convention: the SDK casts the opaque
/// HipdnnEnginePluginHandle* directly to HIPDNN_PLUGIN_HANDLE_TYPE
/// without a namespace qualifier.
struct CkDslHandle : HipdnnEnginePluginHandle {
    CkDslHandle() = default;

    ~CkDslHandle() override = default;

    CkDslHandle(const CkDslHandle&) = delete;
    CkDslHandle& operator=(const CkDslHandle&) = delete;
    CkDslHandle(CkDslHandle&&) = delete;
    CkDslHandle& operator=(CkDslHandle&&) = delete;

    void setStream(hipStream_t stream) {
        _stream = stream;
    }

    hipStream_t getStream() const {
        return _stream;
    }

    std::shared_ptr<ck_dsl_provider::CkDslContainer> container;

    /// Returns the engine manager owned by the container. Defined in
    /// CkDslHandle.cpp so the header does not need the full container
    /// type (breaks the include cycle between Handle and Container).
    hipdnn_plugin_sdk::EngineManager<::CkDslHandle, ck_dsl_provider::CkDslSettings,
                                     ck_dsl_provider::CkDslContext>&
    getEngineManager() const;

    /// Stash a detached FlatBuffer keyed by the raw pointer the SDK
    /// will hand back later via removeEngineDetailsDetachedBuffer.
    /// Lifetime: until the user releases the engine details.
    void storeEngineDetailsDetachedBuffer(const void* ptr,
                                          std::unique_ptr<flatbuffers::DetachedBuffer> buffer) {
        HIPDNN_PLUGIN_LOG_INFO("Storing detached buffer at address: " << ptr);
        _engineDetailsBuffers[ptr] = std::move(buffer);
    }

    void removeEngineDetailsDetachedBuffer(const void* ptr) {
        HIPDNN_PLUGIN_LOG_INFO("Removing detached buffer at address: " << ptr);

        auto it = _engineDetailsBuffers.find(ptr);
        if (it != _engineDetailsBuffers.end()) {
            _engineDetailsBuffers.erase(it);
        } else {
            HIPDNN_PLUGIN_LOG_WARN("No detached buffer found at address: "
                                   << ptr
                                   << ". Could not remove engine details. Ensure you are using the "
                                      "same hipdnn handle you used for engine details creation");
        }
    }

   private:
    hipStream_t _stream = nullptr;
    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>>
        _engineDetailsBuffers;
};
