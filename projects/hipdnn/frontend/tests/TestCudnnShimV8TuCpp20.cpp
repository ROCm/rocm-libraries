// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Compiles TestCudnnShimV8Tu.cpp as C++20. The wrapper gives the C++20 compile a path of its
// own: clang-tidy run with -p lints every compile_commands.json entry for a source, so sharing
// one path with the C++17 unit binary would lint the shim as C++20 there too.
#include "TestCudnnShimV8Tu.cpp" // NOLINT(bugprone-suspicious-include)
