// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CompileServiceBridge.hpp"

#include <pybind11/stl.h>

#include <cstddef>
#include <cstring>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../runtime/KernelArtifact.hpp"
#include "EmbeddedInterpreter.hpp"
#include "PythonError.hpp"
#include "ckdsl_provider_paths.h"

namespace py = pybind11;

namespace ck_dsl_provider {

CompileServiceBridge::~CompileServiceBridge() noexcept {
    // Drop the py::module_ reference while the GIL is held. Member
    // destructors run after this body, but by then _module is empty
    // and its default destructor touches no Python state.
    try {
        py::gil_scoped_acquire gil;
        _module = py::module_();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
        // ~noexcept; if GIL acquisition itself throws (interpreter
        // already torn down by a sibling plugin), we have nothing left
        // to do but let _module's no-op destructor run.
    }
}

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

namespace {

/// Decode a Python int into uint32_t with a clear error context if it
/// is out of range. Used to translate the (gx, gy, gz) / (bx, by, bz)
/// tuples emitted by compile_smoke into ``KernelArtifact``'s
/// GridSpec / BlockSpec fields.
std::uint32_t castU32(const py::handle& obj, const char* fieldName) {
    auto wide = obj.cast<long long>();
    if (wide < 0 || wide > static_cast<long long>(std::numeric_limits<std::uint32_t>::max())) {
        std::ostringstream oss;
        oss << "CompileServiceBridge: compile_smoke '" << fieldName << "' value " << wide
            << " does not fit in uint32_t";
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       oss.str());
    }
    return static_cast<std::uint32_t>(wide);
}

KernelArtifact::GridSpec gridFromPy(const py::handle& tup) {
    auto seq = tup.cast<py::sequence>();
    if (seq.size() != 3) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "CompileServiceBridge: compile_smoke 'grid' must be a 3-tuple");
    }
    return KernelArtifact::GridSpec{castU32(seq[0], "grid.x"), castU32(seq[1], "grid.y"),
                                    castU32(seq[2], "grid.z")};
}

KernelArtifact::BlockSpec blockFromPy(const py::handle& tup) {
    auto seq = tup.cast<py::sequence>();
    if (seq.size() != 3) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "CompileServiceBridge: compile_smoke 'block' must be a 3-tuple");
    }
    return KernelArtifact::BlockSpec{castU32(seq[0], "block.x"), castU32(seq[1], "block.y"),
                                     castU32(seq[2], "block.z")};
}

std::vector<ArgSchema> argSchemaFromPy(const py::handle& list) {
    auto seq = list.cast<py::sequence>();
    std::vector<ArgSchema> out;
    out.reserve(seq.size());
    for (std::size_t i = 0; i < seq.size(); ++i) {
        auto entry = seq[i].cast<py::dict>();
        if (!entry.contains("kind")) {
            std::ostringstream oss;
            oss << "CompileServiceBridge: compile_smoke arg_schema entry " << i
                << " missing 'kind'";
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           oss.str());
        }
        ArgSchema slot;
        slot.name = entry.contains("name") ? entry["name"].cast<std::string>() : std::string{};
        slot.kind = parseArgKind(entry["kind"].cast<std::string>());
        if (entry.contains("size")) {
            slot.size = static_cast<std::uint16_t>(entry["size"].cast<long>());
        }
        if (entry.contains("align")) {
            slot.align = static_cast<std::uint16_t>(entry["align"].cast<long>());
        }
        out.push_back(std::move(slot));
    }
    return out;
}

/// Translate a Python dict (the on-wire shape returned by either
/// compile_service.compile_smoke or compile_service.compile) into a
/// C++ ``KernelArtifact``. Caller must hold the GIL.
///
/// ``contextTag`` shows up in any thrown HipdnnPluginException so the
/// operator can tell which entry point produced the malformed dict
/// (the same dict-shape contract covers both smoke and production).
KernelArtifact dictToArtifact(const py::dict& resultDict, const char* contextTag) {
    const char* requiredFields[] = {"hsaco", "kernel_name", "kind",      "grid",
                                    "block", "lds_bytes",   "arg_schema"};
    for (const char* field : requiredFields) {
        if (!resultDict.contains(field)) {
            std::ostringstream oss;
            oss << contextTag << ": returned dict is missing '" << field << "'";
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           oss.str());
        }
    }

    KernelArtifact artifact;
    artifact.kernelName = resultDict["kernel_name"].cast<std::string>();
    artifact.kind = resultDict["kind"].cast<std::string>();
    artifact.grid = gridFromPy(resultDict["grid"]);
    artifact.block = blockFromPy(resultDict["block"]);
    artifact.ldsBytes = castU32(resultDict["lds_bytes"], "lds_bytes");
    artifact.argSchema = argSchemaFromPy(resultDict["arg_schema"]);
    if (resultDict.contains("isa")) {
        artifact.isa = resultDict["isa"].cast<std::string>();
    }

    // Copy the HSACO bytes out of the py::bytes payload. The caller
    // holds the GIL so PyBytes_AsStringAndSize is safe; the resulting
    // std::vector<std::byte> outlives any Python state since it
    // carries its own storage.
    auto hsacoBytes = resultDict["hsaco"].cast<py::bytes>();
    char* buf = nullptr;
    Py_ssize_t len = 0;
    if (PyBytes_AsStringAndSize(hsacoBytes.ptr(), &buf, &len) != 0) {
        // PyBytes_AsStringAndSize sets a Python exception on
        // failure; convert to py::error_already_set so callers can
        // funnel through the PythonError translation.
        throw py::error_already_set();
    }
    // Hard cap: a runaway compile path could otherwise request a
    // multi-GB allocation that succeeds host-side and then fails
    // inside hipModuleLoadData -- OOM rather than fail-fast. 256 MB
    // is two orders of magnitude above the largest realistic HSACO
    // we expect from ck_dsl (~few MB).
    constexpr Py_ssize_t kHsacoMaxBytes = 256LL * 1024 * 1024;
    if (len > kHsacoMaxBytes) {
        std::ostringstream oss;
        oss << contextTag << ": HSACO blob is " << len << " bytes; rejecting (max "
            << kHsacoMaxBytes << ")";
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       oss.str());
    }
    artifact.hsaco.resize(static_cast<std::size_t>(len));
    if (len > 0) {
        std::memcpy(artifact.hsaco.data(), buf, static_cast<std::size_t>(len));
    }
    return artifact;
}

}  // namespace

KernelArtifact CompileServiceBridge::compileSmoke() {
    try {
        py::gil_scoped_acquire gil;
        py::object result = _module.attr("compile_smoke")();
        KernelArtifact artifact =
            dictToArtifact(result.cast<py::dict>(), "CompileServiceBridge::compileSmoke");

        HIPDNN_PLUGIN_LOG_INFO("CompileServiceBridge::compileSmoke produced kernel='"
                               << artifact.kernelName << "' kind='" << artifact.kind
                               << "' hsaco_bytes=" << artifact.hsaco.size() << " grid=("
                               << artifact.grid.x << "," << artifact.grid.y << ","
                               << artifact.grid.z << ") block=(" << artifact.block.x << ","
                               << artifact.block.y << "," << artifact.block.z << ")");

        return artifact;
    } catch (const py::error_already_set& error) {
        PythonError::raise(error, "CompileServiceBridge::compileSmoke");
    }
}

KernelArtifact CompileServiceBridge::compile(std::string_view opKind, const py::dict& payload) {
    try {
        py::gil_scoped_acquire gil;
        py::str opKindStr(opKind.data(), opKind.size());
        py::object result = _module.attr("compile")(opKindStr, payload);
        KernelArtifact artifact =
            dictToArtifact(result.cast<py::dict>(), "CompileServiceBridge::compile");

        HIPDNN_PLUGIN_LOG_INFO(
            "CompileServiceBridge::compile op_kind='"
            << std::string(opKind) << "' kernel='" << artifact.kernelName << "' kind='"
            << artifact.kind << "' hsaco_bytes=" << artifact.hsaco.size() << " grid=("
            << artifact.grid.x << "," << artifact.grid.y << "," << artifact.grid.z << ") block=("
            << artifact.block.x << "," << artifact.block.y << "," << artifact.block.z << ")");

        return artifact;
    } catch (const py::error_already_set& error) {
        PythonError::raise(error, "CompileServiceBridge::compile");
    }
}

}  // namespace ck_dsl_provider
