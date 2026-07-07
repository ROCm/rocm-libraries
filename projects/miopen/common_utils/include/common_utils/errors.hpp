// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Lightweight throw helper for the Core Utilities layer. This is intentionally
// NOT a move of miopen/errors.hpp: it has no dependency on miopenStatus_t, the
// logger, or the object system, so common_utils stays free of MIOpen coupling.
// At the C API boundary a std::runtime_error is caught by the std::exception
// handler in miopen/errors.hpp (try_) and mapped to miopenStatusUnknownError,
// which is the same status the single-argument MIOPEN_THROW produced.
#ifndef GUARD_COMMON_UTILS_ERRORS_HPP
#define GUARD_COMMON_UTILS_ERRORS_HPP

#include <stdexcept>
#include <string>

namespace common_utils {
[[noreturn]] inline void Throw(const std::string& file, int line, const std::string& msg)
{
    throw std::runtime_error(file + ":" + std::to_string(line) + ": " + msg);
}
} // namespace common_utils

#define COMMON_THROW(msg)                             \
    do                                                \
    {                                                 \
        common_utils::Throw(__FILE__, __LINE__, msg); \
    } while(false)

#endif // GUARD_COMMON_UTILS_ERRORS_HPP
