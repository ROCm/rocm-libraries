// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>
#include <memory>
#include <string>
#include <unordered_map>

#include "CkDslSettings.hpp"
#include "ck_dsl_runtime/ml_heuristic.hpp"
#include "ck_dsl_runtime/runtime.hpp"

struct CkDslContext;

namespace ck_dsl_plugin {
class CkDslContainer;
}  // namespace ck_dsl_plugin

// Per-device plugin handle: detects the gfx arch, indexes the shipped per-arch
// kernel bundle into a ck_dsl::ArtifactStore, and exposes a ck_dsl::Dispatcher.
struct CkDslHandle : HipdnnEnginePluginHandle {
    CkDslHandle();
    ~CkDslHandle() override;

    CkDslHandle(const CkDslHandle&) = delete;
    CkDslHandle& operator=(const CkDslHandle&) = delete;

    void setStream(hipStream_t s) {
        stream_ = s;
    }
    hipStream_t getStream() const {
        return stream_;
    }

    std::shared_ptr<ck_dsl_plugin::CkDslContainer> container;

    hipdnn_plugin_sdk::EngineManager<CkDslHandle, ck_dsl_plugin::CkDslSettings, CkDslContext>&
    getEngineManager();

    void storeEngineDetailsDetachedBuffer(const void* ptr,
                                          std::unique_ptr<flatbuffers::DetachedBuffer> buffer) {
        engine_details_buffers_[ptr] = std::move(buffer);
    }
    void removeEngineDetailsDetachedBuffer(const void* ptr) {
        engine_details_buffers_.erase(ptr);
    }

    const std::string& gfxArch() const {
        return gfx_arch_;
    }
    const std::string& isa() const {
        return isa_;
    }
    const ck_dsl::ArtifactStore& store() const {
        return *store_;
    }
    const ck_dsl::Dispatcher& dispatcher() const {
        return *dispatcher_;
    }

   private:
    hipStream_t stream_ = nullptr;
    std::string gfx_arch_;
    std::string isa_;
    std::unique_ptr<ck_dsl::ArtifactStore> store_;
    std::unique_ptr<ck_dsl::Dispatcher> dispatcher_;
    std::unique_ptr<ck_dsl::DslMlHeuristic> ml_heuristic_;  // trained-model ranker (optional)
    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>>
        engine_details_buffers_;
};
