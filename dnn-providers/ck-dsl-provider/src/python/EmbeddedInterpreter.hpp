// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <string_view>

namespace ck_dsl_provider {

/// Per-process embedded CPython interpreter wrapper.
///
/// Owns a pybind11::scoped_interpreter that is initialised lazily on the
/// first call to ensureInitialized() and intentionally never finalised:
/// hipDNN may host multiple plugins, any of which may also embed
/// CPython, and calling Py_Finalize() from this plugin would tear down
/// the interpreter state shared with those siblings (plan §3.4 risk
/// register). The scoped_interpreter is therefore allocated on the heap
/// and leaked at plugin unload.
///
/// The class is callable from any plugin entry point; the per-process
/// natural hook is the CkDslContainer constructor, because hipDNN's
/// SharedContainerManager makes that constructor run exactly once per
/// process even when several handles are created.
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
    /// to catch). Used by the I-2 unit tests; production code uses
    /// pybind11 directly.
    static pybind11::object importCheck(std::string_view moduleName);

   private:
    EmbeddedInterpreter() = delete;
};

}  // namespace ck_dsl_provider
