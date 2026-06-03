// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EmbeddedInterpreter.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <mutex>
#include <string>

extern "C" {
#include "port/micropython_embed.h"
#include "py/stackctrl.h"
}

namespace ck_dsl_provider {

namespace {

std::once_flag _initFlag;
std::atomic<unsigned> _initializationCount{0};
std::atomic<bool> _initialized{false};

// GC heap for the embedded interpreter. Sized generously because a single
// implicit-GEMM conv lowering builds a large LLVM-IR text plus the codegen's
// transient Python objects. The heap is a one-time process reservation that is
// never freed (the interpreter is never deinitialised).
constexpr std::size_t kHeapBytes = 512ULL * 1024 * 1024;

void doInitialize() {
    char* heap = static_cast<char*>(std::malloc(kHeapBytes));
    if (heap == nullptr) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "EmbeddedInterpreter: failed to allocate the MicroPython GC heap");
    }
    // The stack-top set here is provisional: mp_embed_init records this frame,
    // but every Python-running entry point resets it to its own frame via
    // setCallStackTop() because calls may arrive on other host threads.
    int stackTopMarker = 0;
    mp_embed_init(heap, kHeapBytes, &stackTopMarker);

    _initialized.store(true, std::memory_order_release);
    _initializationCount.fetch_add(1, std::memory_order_relaxed);
    HIPDNN_PLUGIN_LOG_INFO("EmbeddedInterpreter: MicroPython initialised (heap="
                           << (kHeapBytes / (1024 * 1024)) << " MiB)");
    // Intentionally never mp_embed_deinit() -- see header.
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
    static std::mutex mutex;
    return mutex;
}

void EmbeddedInterpreter::setCallStackTop(void* frame) noexcept {
    mp_stack_set_top(frame);
}

}  // namespace ck_dsl_provider
