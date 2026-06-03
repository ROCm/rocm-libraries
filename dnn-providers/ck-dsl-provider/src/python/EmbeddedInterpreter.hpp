// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <mutex>

namespace ck_dsl_provider {

/// Per-process embedded MicroPython interpreter wrapper.
///
/// Initialises MicroPython lazily via ``mp_embed_init`` over a single
/// process-lifetime GC heap. ck_dsl ships *inside* the plugin: frozen
/// modules in the default (release) build, or an on-disk bundle in the
/// dev/loose build (the ``CKDSL_MICROPYTHON_FROZEN`` toggle). Either way
/// there is no dependency on a system Python install, and the native
/// ``comgr`` module (see ``micropython/modcomgr.c``) is registered at
/// link time so ``ck_dsl.runtime.comgr`` can call libamd_comgr.
///
/// The interpreter is intentionally never deinitialised (no
/// ``mp_embed_deinit``): like the previous CPython embedder, tearing it
/// down at process exit from a thread that does not own the runtime is
/// unsafe, and hipDNN keeps the plugin resident.
///
/// Threading: MicroPython has a single global runtime state and (in this
/// embed config) no GIL. hipDNN may call the provider from multiple
/// threads, so every interpreter access MUST be serialised under
/// ``interpreterMutex()``. This replaces pybind's ``gil_scoped_acquire``.
/// `CompileServiceBridge` takes that lock around each call.
///
/// C-stack: ``mp_embed_init`` records a stack top for overflow checks. A
/// process-singleton interpreter is later entered from varying stack
/// frames, so stack checking is disabled in the embed ``mpconfigport.h``
/// (MICROPY_STACK_CHECK off); the compile workload is bounded and runs
/// under our own large heap.
class EmbeddedInterpreter {
   public:
    /// Initialise the embedded interpreter if it has not already been
    /// initialised in this process. Subsequent calls are cheap no-ops.
    /// Thread-safe via std::call_once.
    static void ensureInitialized();

    /// Returns true after the interpreter has been initialised.
    static bool isInitialized() noexcept;

    /// Number of times initialization actually ran (0 or 1). Test helper.
    static unsigned initializationCount() noexcept;

    /// Process-wide lock serialising all interpreter access. Hold it for
    /// the entire duration of any mp_* call sequence (import, call,
    /// result marshalling). Replaces the CPython GIL.
    static std::mutex& interpreterMutex() noexcept;

   private:
    EmbeddedInterpreter() = delete;
};

}  // namespace ck_dsl_provider
