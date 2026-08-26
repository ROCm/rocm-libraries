// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// The registry moved to the plugin SDK so the kernel ingestor's plan-build path can reach
// it (RFC 0019 §5). This shim keeps the backend's spelling working; it will go away once
// the policy path is rewritten against the plugin_sdk names directly.
#include <hipdnn_plugin_sdk/ingestor/uhd/NativeScorerRegistry.hpp>

namespace hipdnn_backend::heuristics::uhd
{

using hipdnn_plugin_sdk::ingestor::uhd::NativeScorerRegistry;
using hipdnn_plugin_sdk::ingestor::uhd::ScopedNativeScorer;
using hipdnn_plugin_sdk::ingestor::uhd::UhdScoreFn;

} // namespace hipdnn_backend::heuristics::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
