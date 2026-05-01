/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#pragma once

#ifdef _WIN32
#ifdef STINKYTOFU_STATIC
#define STINKYTOFU_EXPORT
#define STINKYTOFU_UNIQUE
#elif defined(stinkytofu_EXPORTS)
#define STINKYTOFU_EXPORT __declspec(dllexport)
#define STINKYTOFU_UNIQUE __declspec(dllexport)
#else
#define STINKYTOFU_EXPORT __declspec(dllimport)
#define STINKYTOFU_UNIQUE __declspec(dllimport)
#endif
#else
#ifdef stinkytofu_EXPORTS
#define STINKYTOFU_EXPORT __attribute__((visibility("default")))
#else
#define STINKYTOFU_EXPORT
#endif
// On Linux, unique symbols (like AnalysisKey) must be visible from BOTH
// the library and its consumers so the linker merges them into one copy.
#define STINKYTOFU_UNIQUE __attribute__((visibility("default")))
#endif
