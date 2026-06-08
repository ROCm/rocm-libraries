// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EmbeddedInterpreter.hpp"

#include <atomic>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <mutex>
#include <string>

#include "ckdsl_provider_paths.h"

namespace py = pybind11;

namespace ck_dsl_provider {

namespace {

std::once_flag _initFlag;
std::atomic<unsigned> _initializationCount{0};

/// Resolve the CPython prefix the embedded interpreter should run from.
///
/// The interpreter is a self-contained python-build-standalone prefix
/// bundled with the provider. In a dev/build tree this is the baked
/// absolute path; an installed plugin will resolve it relative to its
/// own .so (added in the install-staging step). Returning the baked
/// path here keeps the build-tree path working today.
std::string resolvePythonHome() {
    return std::string(kCkDslPythonHome);
}

/// The interpreter executable within a python-build-standalone prefix.
/// Layout differs by platform: Linux keeps it under bin/ (python3 is a
/// version-agnostic symlink); Windows places python.exe at the prefix
/// root. Used only for sys.executable provenance.
/// NOTE: the Windows branch is written but not yet verified on Windows.
std::string pythonExecutable(const std::string& home) {
#ifdef _WIN32
    return home + "/python.exe";
#else
    return home + "/bin/python3";
#endif
}

/// Set an isolated-config string field, raising on failure.
void setConfigString(PyConfig& config, wchar_t** field, const std::string& value,
                     const char* what) {
    PyStatus status = PyConfig_SetBytesString(&config, field, value.c_str());
    if (PyStatus_Exception(status) != 0) {
        std::string msg = std::string("EmbeddedInterpreter: failed to set ") + what;
        if (status.err_msg != nullptr) {
            msg += std::string(": ") + status.err_msg;
        }
        PyConfig_Clear(&config);
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, msg);
    }
}

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

        // Pin home + executable to the bundled python-build-standalone
        // prefix so the bundled stdlib loads deterministically rather
        // than from whatever interpreter happens to host the process.
        // `home` drives the stdlib search; `executable` gives a correct
        // sys.executable pointing at the bundled interpreter.
        const std::string pythonHome = resolvePythonHome();
        setConfigString(config, &config.home, pythonHome, "PyConfig.home");
        setConfigString(config, &config.executable, pythonExecutable(pythonHome),
                        "PyConfig.executable");

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

}  // namespace ck_dsl_provider
