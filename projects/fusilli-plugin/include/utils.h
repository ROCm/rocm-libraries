// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the fusilli plugin utils and macros.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_PLUGIN_SRC_UTILS_H
#define FUSILLI_PLUGIN_SRC_UTILS_H

#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/plugin/PluginLastErrorManager.hpp>

// Checks for null, sets the plugin last error manager and returns error if
// null.
//
// SIDE EFFECT: any util function returning an `hipdnnPluginStatus_t` is
// intended for error checking and reporting, and therefore sets
// PluginLastErrorManager::setLastError to an appropriate error on the unhappy
// path.
template <typename T> hipdnnPluginStatus_t isNull(T *value) {
  if (value == nullptr) {
    return hipdnn_plugin::PluginLastErrorManager::setLastError(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        std::string(typeid(T).name()) + " is nullptr");
  }
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// If null, set plugin error manager last error to
// HIPDNN_PLUGIN_STATUS_BAD_PARAM and return said error from the enclosing
// scope.
#define FUSILLI_PLUGIN_CHECK_NULL(X)                                           \
  do {                                                                         \
    if (hipdnnPluginStatus_t status = isNull(X);                               \
        status != HIPDNN_PLUGIN_STATUS_SUCCESS) {                              \
      return status;                                                           \
    }                                                                          \
  } while (false)

// LOG_API_SUCCESS from hipDNN, but deducing the enclosing function rather than
// passing the function name.
#define LOG_API_SUCCESS_AUTO(format, ...)                                      \
  LOG_API_SUCCESS(__func__, format, __VA_ARGS__)

// Unwrap the value returned from an expression that evaluates to a
// fusilli::ErrorOr. In the unhappy path set plugin error manager last error to
//  HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR and return said error from the enclosing
//  scope.
//
// Usage:
//   fusilli::ErrorOr<std::string> getString();
//
//   hipdnnPluginStatus_t processString() {
//     // Either gets the string or returns error.
//     std::string str = FUSILLI_PLUGIN_TRY(getString());
//     doSomethingImportant(str);
//     return HIPDNN_PLUGIN_STATUS_SUCCESS;
//   }
#define FUSILLI_PLUGIN_TRY(expr)                                               \
  ({                                                                           \
    auto errorOr = (expr);                                                     \
    if (fusilli::isError(errorOr)) {                                           \
      return hipdnn_plugin::PluginLastErrorManager::setLastError(              \
          HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,                                 \
          fusilli::ErrorObject(errorOr).getMessage());                         \
    }                                                                          \
    std ::move(*errorOr);                                                      \
  })

// Set plugin error manager last error and return failed status from enclosing
// scope if expression evaluates to a fusilli::ErrorObject in an error state; or
// in the case of fusilli::ErrorOr<T> is convertible to an fusilli::ErrorObject
// in an error state.
//
// Usage:
//   fusilli::ErrorObject doBar();
//
//   hipdnnPluginStatus_t doFoo() {
//     // Returns error if doBar() fails
//     FUSILLI_PLUGIN_CHECK_ERROR(doBar());
//     return HIPDNN_PLUGIN_STATUS_SUCCESS;
//   }
#define FUSILLI_PLUGIN_CHECK_ERROR(expr)                                       \
  do {                                                                         \
    fusilli::ErrorObject err = (expr);                                         \
    if (isError(err)) {                                                        \
      return hipdnn_plugin::PluginLastErrorManager::setLastError(              \
          HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, err.getMessage());              \
    }                                                                          \
  } while (false)

#endif // FUSILLI_PLUGIN_SRC_UTILS_H
