// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EmbeddedInterpreter.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <mutex>
#include <string>

#if defined(CKDSL_ON_DISK) && CKDSL_ON_DISK
#include <dlfcn.h>
#include <sys/stat.h>
#endif

extern "C" {
#include "port/micropython_embed.h"
#include "py/stackctrl.h"
#if defined(CKDSL_ON_DISK) && CKDSL_ON_DISK
#include "py/objlist.h"
#include "py/runtime.h"
#endif
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

#if defined(CKDSL_ON_DISK) && CKDSL_ON_DISK
// Resolve the on-disk module bundle. Prefer the installed layout: a directory
// named CKDSL_BUNDLE_INSTALL_DIRNAME next to this plugin (located at runtime via
// dladdr on a symbol in this binary). Fall back to the build-time path
// CKDSL_BUNDLE_DIR for the dev / build-tree case -- the tests link the static
// lib, so dladdr resolves to the test binary, which has no bundle beside it.
std::string resolveBundleDir() {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&resolveBundleDir), &info) != 0 &&
        info.dli_fname != nullptr) {
        const std::string path = info.dli_fname;
        const std::string::size_type slash = path.find_last_of('/');
        if (slash != std::string::npos) {
            const std::string candidate =
                path.substr(0, slash) + "/" + CKDSL_BUNDLE_INSTALL_DIRNAME;
            struct stat st;
            if (stat(candidate.c_str(), &st) == 0 && (st.st_mode & S_IFDIR) != 0) {
                return candidate;
            }
        }
    }
    return CKDSL_BUNDLE_DIR;
}
#endif

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

#if defined(CKDSL_ON_DISK) && CKDSL_ON_DISK
    // On-disk modes: the ck_dsl/shims tree ships beside the plugin and is loaded
    // from the filesystem. Resolve it (installed dir next to the .so, else the
    // build-time path) and put it on sys.path so `import ck_dsl_provider` resolves.
    {
        const std::string bundle = resolveBundleDir();
        nlr_buf_t nlr;
        if (nlr_push(&nlr) == 0) {
            mp_obj_list_append(mp_sys_path, mp_obj_new_str(bundle.c_str(), bundle.size()));
            nlr_pop();
        } else {
            HIPDNN_PLUGIN_LOG_WARN("EmbeddedInterpreter: failed to add on-disk bundle to sys.path");
        }
        HIPDNN_PLUGIN_LOG_INFO("EmbeddedInterpreter: on-disk module bundle " << bundle);
    }
#endif

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
