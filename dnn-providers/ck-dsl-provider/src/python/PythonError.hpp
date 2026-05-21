// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <pybind11/pybind11.h>

#include <string_view>

namespace ck_dsl_provider {

/// Helpers that translate pybind11::error_already_set into a
/// hipdnn_plugin_sdk::HipdnnPluginException. Keeps the Python exception
/// type name and message in the resulting .what() string so plugin
/// callers see the original Python diagnostic when the error surfaces
/// in hipDNN's logs.
///
/// Use the static raise(...) helper from inside a catch block; calling
/// it outside a catch is a programmer error and triggers an undefined-
/// state translation that simply forwards a generic internal-error
/// message.
class PythonError {
   public:
    /// Convert a live pybind11::error_already_set into a
    /// HipdnnPluginException with status HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR.
    /// The .what() of the thrown exception is:
    ///   "<contextTag>: <PythonExceptionTypeName>: <PythonExceptionMessage>"
    ///
    /// [[noreturn]] — always throws.
    [[noreturn]] static void raise(const pybind11::error_already_set& error,
                                   std::string_view contextTag);

   private:
    PythonError() = delete;
};

}  // namespace ck_dsl_provider
