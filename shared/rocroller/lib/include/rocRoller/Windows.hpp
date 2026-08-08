// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef _MSC_VER

#include <io.h>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#define popen _popen
#define pclose _pclose

inline char* mkdtemp(char* tmpl) {
    if (tmpl == nullptr) {
        return nullptr;
    }
    if (const auto err{_mktemp_s(tmpl, strlen(tmpl) + 1)}; err != 0) {
        return nullptr;
    }
    if (CreateDirectory(tmpl, nullptr) == FALSE) {
        return nullptr;
    }
    return tmpl;
}

inline char* mkdtemp(std::string_view s) {
    return mkdtemp(s.data());
}

#endif // _MSC_VER
