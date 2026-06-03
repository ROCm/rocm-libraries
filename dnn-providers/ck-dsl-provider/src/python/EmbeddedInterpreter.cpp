// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EmbeddedInterpreter.hpp"

#include <atomic>
#include <cstdlib>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <mutex>

extern "C" {
#include "port/micropython_embed.h"
}

namespace ck_dsl_provider {

namespace {

std::once_flag _initFlag;
std::atomic<unsigned> _initializationCount{0};
std::atomic<bool> _initialized{false};

// Process-lifetime GC heap. Sized generously for the JIT compile workload
// (the conv spike peaked well under this); intentionally never freed.
// TODO(perf): measure the real per-compile peak and/or expose as a provider
// config knob instead of a fixed reservation.
constexpr std::size_t kHeapBytes = 512u * 1024u * 1024u;

void doInitialize() {
    void* heap = std::malloc(kHeapBytes);
    if (heap == nullptr) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "EmbeddedInterpreter: failed to allocate MicroPython GC heap");
    }
    // Record a stack top from this frame. Stack checking is disabled in the
    // embed mpconfigport (see header), so the exact value is not load-bearing;
    // mp_embed_init still requires the argument.
    int stackTop = 0;
    mp_embed_init(heap, kHeapBytes, &stackTop);
    // Intentionally never mp_embed_deinit() -- see header.

    _initialized.store(true, std::memory_order_release);
    _initializationCount.fetch_add(1, std::memory_order_relaxed);
    HIPDNN_PLUGIN_LOG_INFO("EmbeddedInterpreter: MicroPython embed init complete ("
                           << (kHeapBytes >> 20) << " MiB heap)");
}

}  // namespace

void EmbeddedInterpreter::ensureInitialized() {
    std::call_once(_initFlag, &doInitialize);
}

bool EmbeddedInterpreter::isInitialized() noexcept {
    return _initialized.load(std::memory_order_acquire);
}

unsigned EmbeddedInterpreter::initializationCount() noexcept {
    return _initializationCount.load(std::memory_order_relaxed);
}

std::mutex& EmbeddedInterpreter::interpreterMutex() noexcept {
    // Meyers singleton: one process-wide lock guarding the single MicroPython
    // runtime state. Constructed on first use, before any bridge call.
    static std::mutex mutex;
    return mutex;
}

}  // namespace ck_dsl_provider
