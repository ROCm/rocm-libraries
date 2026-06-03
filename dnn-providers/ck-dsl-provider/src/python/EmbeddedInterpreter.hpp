// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <mutex>

namespace ck_dsl_provider {

/// Per-process embedded MicroPython interpreter wrapper.
///
/// Initialises MicroPython lazily via ``mp_embed_init`` over a large
/// process-owned GC heap (no system Python, no filesystem, no
/// environment). The ck_dsl / ck_dsl_provider Python modules are frozen
/// into the plugin (or loaded from an on-disk .mpy/.py bundle in the dev
/// build), so there is no sys.path to harden or shadow: ``import
/// ck_dsl_provider`` resolves only to the frozen tree. This closes the
/// channel through which a host could shadow ``import ck_dsl`` by setting
/// PYTHONPATH before loading the plugin -- the attack surface that the
/// previous CPython embedding had to defend against does not exist here.
///
/// MicroPython has a single global runtime state and NO GIL. All
/// interaction with the interpreter must therefore be serialised on
/// ``interpreterMutex()``. The interpreter is intentionally never
/// deinitialised: hipDNN may host multiple plugins and the GC heap /
/// runtime state outlive any single compile.
///
/// GC note: MicroPython scans the C stack for roots between the current
/// stack pointer and ``stack_top`` (set at init). Because compile calls
/// may arrive on different host threads, every entry point that runs
/// Python code must reset the stack top to its own frame via
/// ``setCallStackTop`` while holding ``interpreterMutex()``.
class EmbeddedInterpreter {
   public:
    /// Initialise the embedded interpreter if it has not already been
    /// initialised in this process. Subsequent calls are cheap no-ops.
    /// Thread-safe via std::call_once.
    static void ensureInitialized();

    /// Returns true after the interpreter has been initialised.
    static bool isInitialized() noexcept;

    /// Returns how many times ensureInitialized() has actually performed
    /// initialization (always 0 or 1 in a healthy process). Test helper.
    static unsigned initializationCount() noexcept;

    /// The single lock serialising all interpreter access (no GIL).
    static std::mutex& interpreterMutex() noexcept;

    /// Reset MicroPython's C-stack root-scanning top to the caller's
    /// frame. Call once, holding interpreterMutex(), at the top of every
    /// entry point that runs Python, passing ``__builtin_frame_address(0)``
    /// so the argument is evaluated in the caller's frame:
    ///   std::lock_guard<std::mutex> lock(interpreterMutex());
    ///   EmbeddedInterpreter::setCallStackTop(__builtin_frame_address(0));
    static void setCallStackTop(void* frame) noexcept;

   private:
    EmbeddedInterpreter() = delete;
};

}  // namespace ck_dsl_provider
