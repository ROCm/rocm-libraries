// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

#include <exception>
#include <string>

namespace mosaic {

namespace error {

typedef hipError_t error_t;

/**
 * @brief Exception class for errors in device code.
 */
struct exception_t : std::exception {
    std::string report;

    exception_t(error_t status, std::string message = "") {
        report = hipGetErrorString(status) + std::string("\t: ") + message;
    }

    exception_t(std::string message = "") {
        report = message;
    }

    virtual const char* what() const noexcept {
        return report.c_str();
    }
};

/**
 * @brief Throw an exception if the given HIP error code is not hipSuccess.
 */
inline void throw_if_exception(error_t status, std::string message = "") {
    if (status != hipSuccess) throw exception_t(status, message);
}

inline void throw_if_exception(bool is_exception, std::string message = "") {
    if (is_exception) throw exception_t(message);
}

}  // namespace error
}  // namespace mosaic

// Wrap a HIP runtime call and throw on failure. Captures the call expression
// (stringified) plus source location as the error message.
#define MOSAIC_CHECK(call)               \
    ::mosaic::error::throw_if_exception( \
        (call), std::string(#call " @ ") + __FILE__ + ":" + std::to_string(__LINE__))
