// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PythonError.hpp"

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <string>

namespace py = pybind11;

namespace ck_dsl_provider {

namespace {

/// Best-effort Python exception type-name lookup. error.type() is the
/// PyObject* the interpreter recorded; .__name__ exists for every
/// real Python exception class.
std::string typeName(const py::error_already_set& error) {
    try {
        if (!error.type().is_none()) {
            return error.type().attr("__name__").cast<std::string>();
        }
    } catch (const py::error_already_set&) {
        // Suppressed: introspection itself failed. Fall through to the
        // generic label below.
    } catch (const std::exception&) {
    }
    return "PythonError";
}

}  // namespace

void PythonError::raise(const py::error_already_set& error, std::string_view contextTag) {
    // py::error_already_set::what() already serialises both the type
    // and the message in a readable form; use it as the message
    // baseline and prepend the type explicitly so callers that lex the
    // string can find the type without re-parsing what().
    std::string message;
    message.reserve(contextTag.size() + 64);
    message.append(contextTag.data(), contextTag.size());
    message.append(": ");
    message.append(typeName(error));
    message.append(": ");

    // error.what() can itself throw if the interpreter is in a bad
    // state. Guard against that by falling back to a constant.
    try {
        message.append(error.what());
    } catch (...)  // NOLINT(bugprone-empty-catch)
    {
        message.append("<failed to render Python exception message>");
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   std::move(message));
}

}  // namespace ck_dsl_provider
