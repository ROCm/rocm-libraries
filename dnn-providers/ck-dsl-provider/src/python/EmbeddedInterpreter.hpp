// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <string_view>

namespace ck_dsl_provider {

/// Per-process embedded CPython interpreter wrapper.
///
/// Initialises Python lazily via ``Py_InitializeFromConfig`` with
/// ``PyConfig_InitIsolatedConfig``. Isolated config rejects the host
/// process's PYTHONPATH / PYTHONHOME / PYTHONSTARTUP / PYTHONUSERBASE
/// environment variables, disables user-site-packages, and sets
/// safe_path -- closing the channel through which a host could
/// shadow `import ck_dsl` by setting PYTHONPATH before loading the
/// plugin. The provider's own ck_dsl_provider + ck_dsl trees are
/// brought onto sys.path explicitly inside
/// ``CompileServiceBridge::ctor``; no other paths need to be searched
/// for the JIT compile path.
///
/// The interpreter is intentionally never finalised. hipDNN may host
/// multiple plugins, any of which may also embed CPython, and calling
/// Py_Finalize() from this plugin would tear down the interpreter
/// state shared with those siblings.
///
/// If another in-process embedder has initialised CPython before this
/// plugin loads, ``ensureInitialized`` skips the init path and reuses
/// the existing interpreter -- the PyConfig hardening only applies if
/// this plugin is the first embedder to run.
class EmbeddedInterpreter {
   public:
    /// Initialise the embedded interpreter if it has not already been
    /// initialised in this process. Subsequent calls are cheap no-ops.
    /// Thread-safe via std::call_once.
    static void ensureInitialized();

    /// Returns true after the interpreter has been initialised.
    /// Mainly useful for unit tests; production callers should rely on
    /// ensureInitialized() and the GIL helpers.
    static bool isInitialized() noexcept;

    /// Returns how many times ensureInitialized() has actually performed
    /// initialization (always 0 or 1 in a healthy process). Test helper.
    static unsigned initializationCount() noexcept;

    /// Convenience smoke helper: import a Python module by name with
    /// the GIL held, returning the imported py::module_ as a py::object.
    /// Throws py::error_already_set on failure (caller decides whether
    /// to catch). Used by the unit tests; production code uses pybind11
    /// directly.
    static pybind11::object importCheck(std::string_view moduleName);

   private:
    EmbeddedInterpreter() = delete;
};

}  // namespace ck_dsl_provider
