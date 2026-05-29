// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EmbeddedInterpreter.hpp"

#include <atomic>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <mutex>
#include <string>

namespace py = pybind11;

namespace ck_dsl_provider {

namespace {

std::once_flag _initFlag;
std::atomic<unsigned> _initializationCount{0};

void doInitialize() {
    // Guard against another in-process embedder having initialised
    // CPython before this plugin loaded. If the interpreter is already
    // up, we cannot retroactively apply isolated config -- we use what
    // is there.
    if (Py_IsInitialized() == 0) {
        PyConfig config;
        PyConfig_InitIsolatedConfig(&config);
        // Isolated config defaults to:
        //   * use_environment = 0    (ignore PYTHONPATH / PYTHONHOME /
        //                             PYTHONSTARTUP / PYTHONUSERBASE)
        //   * user_site_directory = 0
        //   * safe_path = 1          (do not auto-prepend the program
        //                             directory to sys.path)
        // The provider's required search paths are injected explicitly
        // inside CompileServiceBridge::ctor after init.
        PyStatus status = Py_InitializeFromConfig(&config);
        PyConfig_Clear(&config);
        if (PyStatus_Exception(status) != 0) {
            std::string msg = "EmbeddedInterpreter: Py_InitializeFromConfig failed";
            if (status.err_msg != nullptr) {
                msg += std::string(": ") + status.err_msg;
            }
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           msg);
        }
        // Intentionally never Py_Finalize() -- see header.
    }
    _initializationCount.fetch_add(1, std::memory_order_relaxed);

    try {
        py::gil_scoped_acquire gil;
        py::module_ sys = py::module_::import("sys");
        std::string version = sys.attr("version").cast<std::string>();
        // sys.executable is intentionally NOT logged: it leaks the
        // host process's Python install path, which is mildly
        // fingerprinty in shipped logs. The Python version alone is
        // sufficient for diagnostic value.
        HIPDNN_PLUGIN_LOG_INFO("EmbeddedInterpreter: Py_Initialize complete, Python=" << version);
    } catch (const py::error_already_set& e) {
        // Initialisation succeeded but introspection failed; log and
        // carry on. The interpreter itself is up and usable.
        HIPDNN_PLUGIN_LOG_WARN("EmbeddedInterpreter: post-init introspection failed: " << e.what());
    }
}

}  // namespace

void EmbeddedInterpreter::ensureInitialized() {
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
