// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CompileServiceBridge.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <string>

#include "EmbeddedInterpreter.hpp"
#include "PythonError.hpp"
#include "ckdsl_provider_paths.h"

namespace py = pybind11;

namespace ck_dsl_provider {

CompileServiceBridge::CompileServiceBridge() {
    EmbeddedInterpreter::ensureInitialized();

    try {
        py::gil_scoped_acquire gil;
        py::module_ sys = py::module_::import("sys");

        prependSysPathIdempotent(sys, kCkDslProviderPythonPackagePath);
        prependSysPathIdempotent(sys, kCkDslPythonPackagePath);

        _module = py::module_::import("ck_dsl_provider.compile_service");

        // Resolve ck_dsl.__file__ for the one-shot INFO log so the
        // operator can see exactly which source tree the embedded
        // interpreter actually imported (not just which path CMake
        // baked in — sys.path could be shadowed by an earlier sibling).
        std::string ckDslFile;
        try {
            py::module_ ckDsl = py::module_::import("ck_dsl");
            ckDslFile = ckDsl.attr("__file__").cast<std::string>();
        } catch (const py::error_already_set&) {
            ckDslFile = "<ck_dsl import failed>";
        }
        std::string moduleFile;
        try {
            moduleFile = _module.attr("__file__").cast<std::string>();
        } catch (const py::error_already_set&) {
            moduleFile = "<unknown>";
        }

        HIPDNN_PLUGIN_LOG_INFO(
            "CompileServiceBridge: imported ck_dsl_provider.compile_service from "
            << moduleFile << ", ck_dsl from " << ckDslFile);
    } catch (const py::error_already_set& error) {
        PythonError::raise(error, "CompileServiceBridge::ctor");
    }
}

bool CompileServiceBridge::prependSysPathIdempotent(py::module_& sys, std::string_view path) {
    py::list sysPath = sys.attr("path").cast<py::list>();
    py::str candidate(path.data(), path.size());

    for (py::handle entry : sysPath) {
        // Compare as Python str values; entries may be PosixPath in
        // pathological setups but pure CPython startup uses str.
        try {
            if (py::isinstance<py::str>(entry) && entry.cast<std::string>() == std::string(path)) {
                return false;
            }
        } catch (const py::error_already_set&) {
            // Non-comparable entry: skip and keep scanning.
            continue;
        }
    }

    sysPath.attr("insert")(0, candidate);
    return true;
}

py::dict CompileServiceBridge::noopSmoke() {
    try {
        py::gil_scoped_acquire gil;
        py::object result = _module.attr("noop_smoke")();
        return result.cast<py::dict>();
    } catch (const py::error_already_set& error) {
        PythonError::raise(error, "CompileServiceBridge::noopSmoke");
    }
}

}  // namespace ck_dsl_provider
