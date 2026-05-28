// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Compiler portability macros for LLVM/Clang/GCC and MSVC.
// C++23 will provide std::unreachable(): https://en.cppreference.com/w/cpp/utility/unreachable

#pragma once

#include <ck_common/platform.hpp>

// Backward-compatible alias — canonical definition is CK_COMMON_UNREACHABLE().
#ifndef ROCM_CK_UNREACHABLE
#define ROCM_CK_UNREACHABLE() CK_COMMON_UNREACHABLE()
#endif
