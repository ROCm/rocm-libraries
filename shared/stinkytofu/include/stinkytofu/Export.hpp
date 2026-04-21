/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#pragma once

#ifdef _WIN32
#ifdef STINKYTOFU_STATIC
#define STINKYTOFU_EXPORT
#elif defined(stinkytofu_EXPORTS)
#define STINKYTOFU_EXPORT __declspec(dllexport)
#else
#define STINKYTOFU_EXPORT __declspec(dllimport)
#endif
#else
#define STINKYTOFU_EXPORT
#endif
