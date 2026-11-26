// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>

// The Smoke tier is the quick tier of hipdnn_gpu_ref_tests (see test_categories_gpu_ref.yaml):
// one fp32 case per kernel family, so that `ctest -L quick` compiles and launches every GPU
// reference kernel entry point once and compares it against the CPU reference.
//
// A parameterized suite instantiates Smoke with the first case of its small catalog. The full
// catalog stays in the StandardBasic instantiation, which runs the Smoke case again; it costs
// milliseconds, because the kernel is already compiled in that process.
template <typename T>
std::vector<T> smokeCase(const std::vector<T>& cases)
{
    return {cases.front()};
}
