// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// The implementation moved to the plugin SDK so the kernel ingestor's plan-build path can
// reach it (RFC 0019 §5). This shim keeps the backend's spelling working; it will go away
// once the policy path is rewritten against the plugin_sdk names directly.
#include <hipdnn_plugin_sdk/ingestor/uhd/JsonLogicEvaluator.hpp>

namespace hipdnn_backend::heuristics::uhd
{

using hipdnn_plugin_sdk::ingestor::uhd::JsonLogicError;
using hipdnn_plugin_sdk::ingestor::uhd::JsonLogicEvaluator;
using hipdnn_plugin_sdk::ingestor::uhd::UndefinedVariableError;
using hipdnn_plugin_sdk::ingestor::uhd::VariableContext;

} // namespace hipdnn_backend::heuristics::uhd
