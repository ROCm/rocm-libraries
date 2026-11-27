// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>

// The Quick tier of hipdnn_gpu_ref_tests (see test_categories_gpu_ref.yaml) runs one fp32 case
// per kernel family, so that `ctest -L quick` compiles and launches every GPU reference kernel
// entry point once and compares it against the CPU reference.
//
// A parameterized suite instantiates Quick with the first case of its small catalog
// (quickCase). The Standard instantiation of the same fixture runs the small catalog again,
// followed by the medium catalog (concatCases), because GTest allows one instantiation per
// prefix and fixture. The repeated case costs milliseconds: its kernel is already compiled in
// that process.
template <typename T>
std::vector<T> quickCase(const std::vector<T>& cases)
{
    return {cases.front()};
}

// The cases of `first`, then the cases of `second`.
template <typename T>
std::vector<T> concatCases(std::vector<T> first, const std::vector<T>& second)
{
    first.insert(first.end(), second.begin(), second.end());
    return first;
}
