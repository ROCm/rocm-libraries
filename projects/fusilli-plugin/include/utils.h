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

#include <fusilli.h>
#include <hip/hip_runtime.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/plugin/PluginLastErrorManager.hpp>
#include <iree/hal/buffer_view.h>

#include <type_traits>

namespace fusilli_plugin {

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

// Find deviceBuffer with UID.
inline fusilli::ErrorOr<hipdnnPluginDeviceBuffer_t>
findDeviceBuffer(int64_t uid, const hipdnnPluginDeviceBuffer_t *deviceBuffers,
                 uint32_t numDeviceBuffers) {
  for (uint32_t i = 0; i < numDeviceBuffers; i++) {
    if (uid == deviceBuffers[i].uid) {
      return fusilli::ok(deviceBuffers[i]);
    }
  }

  return fusilli::error(fusilli::ErrorCode::AttributeNotSet,
                        "Device buffer with the uid: " + std::to_string(uid) +
                            " not found in the provided device buffers.");
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

template <typename T> fusilli::ErrorObject convertToErrorObject(T &&error) {
  using DecayT = std::decay_t<T>;
  if constexpr (std::is_convertible_v<DecayT, fusilli::ErrorObject>) {
    return std::forward<T>(error);
  } else if constexpr (std::is_same_v<DecayT, hipError_t>) {
    // Convert HIP error to fusilli ErrorObject
    if (error != hipSuccess) {
      return fusilli::error(fusilli::ErrorCode::InternalError,
                            hipGetErrorString(error));
    }
    return fusilli::ok();
  } else {
    static_assert(
        std::is_convertible_v<DecayT, fusilli::ErrorObject> ||
            std::is_same_v<DecayT, hipError_t>,
        "convertToErrorObject requires fusilli::ErrorObject or hipError_t");
    // Unreachable
    return fusilli::error(fusilli::ErrorCode::InternalError,
                          "Unknown error type");
  }
}

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
    fusilli::ErrorObject err = convertToErrorObject(expr);                     \
    if (isError(err)) {                                                        \
      return hipdnn_plugin::PluginLastErrorManager::setLastError(              \
          HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, err.getMessage());              \
    }                                                                          \
  } while (false)

// Convert from fusilli DataType to iree hal data type.
inline fusilli::ErrorOr<iree_hal_element_type_t>
fusilliDataTypeToIreeHalDataType(fusilli::DataType fusilliDataType) {
  switch (fusilliDataType) {
  case fusilli::DataType::Half:
    return fusilli::ok(IREE_HAL_ELEMENT_TYPE_FLOAT_16);
  case fusilli::DataType::BFloat16:
    return fusilli::ok(IREE_HAL_ELEMENT_TYPE_BFLOAT_16);
  case fusilli::DataType::Float:
    return fusilli::ok(IREE_HAL_ELEMENT_TYPE_FLOAT_32);
  case fusilli::DataType::Double:
    return fusilli::ok(IREE_HAL_ELEMENT_TYPE_FLOAT_64);
  case fusilli::DataType::Uint8:
    return fusilli::ok(IREE_HAL_ELEMENT_TYPE_UINT_8);
  case fusilli::DataType::Int8:
    return fusilli::ok(IREE_HAL_ELEMENT_TYPE_INT_8);
  case fusilli::DataType::Int16:
    return fusilli::ok(IREE_HAL_ELEMENT_TYPE_INT_16);
  case fusilli::DataType::Int32:
    return fusilli::ok(IREE_HAL_ELEMENT_TYPE_INT_32);
  case fusilli::DataType::Int64:
    return fusilli::ok(IREE_HAL_ELEMENT_TYPE_INT_64);
  case fusilli::DataType::Boolean:
    return fusilli::ok(IREE_HAL_ELEMENT_TYPE_BOOL_8);
  case fusilli::DataType::FP8E5M2:
    return fusilli::ok(IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2);
  case fusilli::DataType::NotSet:
  default:
    return fusilli::error(
        fusilli::ErrorCode::InvalidAttribute,
        "unknown data type in fusilli -> iree runtime data type conversion");
  }
}

} // namespace fusilli_plugin

#endif // FUSILLI_PLUGIN_SRC_UTILS_H
