// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Arch matching lives in the Plugin SDK so providers and tests share one
// implementation; hipdnn_test_sdk links hipdnn_plugin_sdk (INTERFACE), so the
// header is always available here.
#include <hipdnn_plugin_sdk/ArchMatch.hpp>

namespace hipdnn_test_sdk::utilities
{

using hipdnn_plugin_sdk::archMatches;
using hipdnn_plugin_sdk::ArchMatchMode;

} // namespace hipdnn_test_sdk::utilities
