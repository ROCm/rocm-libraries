// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Moved to the test SDK so the backend and plugin SDK suites share one builder. Two copies
// would be free to drift on the very thing they assert -- the model artifact's layout.
#include <hipdnn_test_sdk/utilities/GbdtModelTestBuilder.hpp>

namespace hipdnn_backend::heuristics::uhd::testing
{

using hipdnn_test_sdk::utilities::GbdtModelTestBuilder;
using hipdnn_test_sdk::utilities::makeLeafTreeSpec;

} // namespace hipdnn_backend::heuristics::uhd::testing
