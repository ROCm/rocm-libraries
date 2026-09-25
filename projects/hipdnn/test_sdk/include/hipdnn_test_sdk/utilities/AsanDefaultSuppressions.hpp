// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// The suppression text that __asan_default_suppressions() returns, defined in
// src/AsanDefaultSuppressions.cpp.
//
// It lives in a header so the tests can name it without restating the pattern, and is deliberately
// not guarded by ADDRESS_SANITIZER so they build in every configuration.
namespace hipdnn_test_sdk::utilities::asan
{

// Upstream rocBLAS/Tensile data race on the lazy placeholder-library load: a solution matching
// table is read while an std::async loader thread deserializes into it and reallocates the backing
// storage. AIBTINFRA-48, ROCm/rocm-libraries#8869.
inline constexpr const char* K_DEFAULT_SUPPRESSIONS = "interceptor_via_fun:*findBestKeyMatch*\n";

} // namespace hipdnn_test_sdk::utilities::asan
