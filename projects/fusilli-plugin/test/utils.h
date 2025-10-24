// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains utilities for fusilli plugin tests.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_PLUGIN_TESTS_UTILS_H
#define FUSILLI_PLUGIN_TESTS_UTILS_H

// Unwrap the type returned from an expression that evaluates to an ErrorOr,
// fail the test using GTest's EXPECT_TRUE if the result is an ErrorObject.
//
// This is very similar to FUSILLI_TRY, but FUSILLI_TRY propagates an error to
// callers on the error path, this fails the test on the error path. The two
// macros are analogous to rust's `?` (try) operator and `.unwrap()` call.
#define FUSILLI_PLUGIN_EXPECT_UNWRAP(expr)                                     \
  ({                                                                           \
    auto _errorOr = (expr);                                                    \
    EXPECT_TRUE(isOk(_errorOr));                                               \
    std::move(*_errorOr);                                                      \
  })

#endif // FUSILLI_PLUGIN_TESTS_UTILS_H
