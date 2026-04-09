// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Example shared header for GPU kernels.
// This example header demonstrates the kernel include embedding system.
//
// See:
// * samples/example_provider/kernels/CMakeLists.txt
// * samples/example_provider/kernels/cmake/EmbedKernelSources.cmake.
//
// Files listed in KERNEL_FILES with .h/.hpp extensions are embedded in their
// entirety as string literals and made available to HIPRTC at runtime. Kernels
// that #include this IndexType.hpp file will have it included automatically
// during runtime compilation of the kernel.

/// For example: define a type for thread-indexing used by GPU kernels for
/// grid/block index arithmetic.
using IndexType = unsigned int;
