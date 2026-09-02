// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "harness/bundle/HarnessPolicy.hpp"

namespace hipdnn_integration_tests::bundle
{

/// TestConfig, read once, into a value.
///
/// Reads only TestConfig and the current platform, neither of which touches a
/// device or the frontend graph, so this lives in its own translation unit rather
/// than in HarnessDependencies.cpp: every unit test would otherwise hand-build a
/// HarnessPolicy with nothing to catch a transposed field against this, the one
/// place production actually assembles one.
HarnessPolicy productionPolicy(TensorPlacement placement);

} // namespace hipdnn_integration_tests::bundle
