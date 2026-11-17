// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <unordered_map>

namespace hipdnn_integration_tests
{

/// uid -> the buffer an executor reads or writes for that tensor.
///
/// Named rather than spelled out because the raw type contains a comma, which a
/// MOCK_METHOD parameter list cannot take without extra parentheses. Its own header
/// so the engine-side and reference-side interfaces can share it without one of
/// them including the other.
using VariantPack = std::unordered_map<int64_t, void*>;

} // namespace hipdnn_integration_tests
