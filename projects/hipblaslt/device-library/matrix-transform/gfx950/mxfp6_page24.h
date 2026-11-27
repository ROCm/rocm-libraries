// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The launch contract of the near-die kernels in mxfp6_page24.cpp, shared with the host that
// launches them.

#pragma once

#include <cstdint>

// Work-items per work-group. The kernels stride by this count and never read blockDim, so a
// launch with any other size skips or repeats bytes.
constexpr uint32_t mxfp6Page24Threads = 256;
