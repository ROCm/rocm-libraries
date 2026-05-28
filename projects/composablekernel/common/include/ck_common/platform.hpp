// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Compiler portability macros for LLVM/Clang/GCC and MSVC.
// C++23 will provide std::unreachable(): https://en.cppreference.com/w/cpp/utility/unreachable

#pragma once

#ifdef _MSC_VER
#define CK_COMMON_UNREACHABLE() __assume(false)
#else
#define CK_COMMON_UNREACHABLE() __builtin_unreachable()
#endif
