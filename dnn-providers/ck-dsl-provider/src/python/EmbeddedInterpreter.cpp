// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EmbeddedInterpreter.hpp"

#include <atomic>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <mutex>
#include <string>

namespace py = pybind11;

namespace ck_dsl_provider {

namespace {

// Heap-allocated, intentionally leaked: see EmbeddedInterpreter.hpp
// header comment. The CK DSL provider plugin cannot call Py_Finalize()
// at unload because hipDNN may also host sibling plugins that embed
// CPython; finalising would tear down the interpreter state shared
// across all of them (plan §3.4 risk register; spike notes in
// WIP/pybind11_rtld_local_spike/).
py::scoped_interpreter* _instance = nullptr;

std::once_flag _initFlag;
std::atomic<unsigned> _initializationCount{0};

void doInitialize() {
    // Guard against another in-process embedder having initialised
    // CPython before this plugin loaded. If the interpreter is already
    // up, we record the fact and skip scoped_interpreter (constructing
    // one when Py_IsInitialized() is true asserts inside pybind11).
    if (Py_IsInitialized() == 0) {
        _instance = new py::scoped_interpreter{};
    }
    _initializationCount.fetch_add(1, std::memory_order_relaxed);

    try {
        py::gil_scoped_acquire gil;
        py::module_ sys = py::module_::import("sys");
        std::string version = sys.attr("version").cast<std::string>();
        std::string executable = sys.attr("executable").cast<std::string>();
        HIPDNN_PLUGIN_LOG_INFO("EmbeddedInterpreter: Py_Initialize complete, Python="
                               << version << ", sys.executable=" << executable);
    } catch (const py::error_already_set& e) {
        // Initialisation succeeded but introspection failed; log and
        // carry on. The interpreter itself is up and usable.
        HIPDNN_PLUGIN_LOG_WARN("EmbeddedInterpreter: post-init introspection failed: " << e.what());
    }
}

}  // namespace

void EmbeddedInterpreter::ensureInitialized() {
    // Subsequent calls are intentionally silent: HIPDNN_PLUGIN_LOG_TRACE
    // in the current Plugin SDK actually routes to INFO severity, so any
    // log here would spam every per-handle path that touches Python.
    std::call_once(_initFlag, &doInitialize);
}

bool EmbeddedInterpreter::isInitialized() noexcept {
    return Py_IsInitialized() != 0;
}

unsigned EmbeddedInterpreter::initializationCount() noexcept {
    return _initializationCount.load(std::memory_order_relaxed);
}

py::object EmbeddedInterpreter::importCheck(std::string_view moduleName) {
    ensureInitialized();
    py::gil_scoped_acquire gil;
    return py::module_::import(std::string(moduleName).c_str());
}

}  // namespace ck_dsl_provider
