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

#define FUSILLI_PLUGIN_CONCAT_IMPL(a, b, c) a##_##b##_##c
#define FUSILLI_PLUGIN_ERROR_VAR(name, line, counter)                          \
  FUSILLI_PLUGIN_CONCAT_IMPL(name, line, counter)

#define FUSILLI_PLUGIN_EXPECT_OR_ASSIGN_IMPL(errorOr, var, expr)               \
  auto errorOr = (expr);                                                       \
  EXPECT_TRUE(!isError(errorOr));                                              \
  var = std::move(*errorOr);

// Assigns the result of an expression that evaluates to an ErrorOr<T> to a
// variable, or returns the error from the enclosing function if the result is
// an error.
#define FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(varDecl, expr)                         \
  FUSILLI_DISABLE_COUNTER_WARNING                                              \
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN_IMPL(                                        \
      FUSILLI_PLUGIN_ERROR_VAR(_errorOr, __LINE__, __COUNTER__), varDecl,      \
      expr)                                                                    \
  FUSILLI_RESTORE_COUNTER_WARNING

#endif // FUSILLI_PLUGIN_TESTS_UTILS_H
