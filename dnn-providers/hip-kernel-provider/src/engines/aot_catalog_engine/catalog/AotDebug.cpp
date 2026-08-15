// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "catalog/AotDebug.hpp"

#include <cstdio>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

namespace aot_catalog_engine
{

bool aotDebugEnabled()
{
    // Cached: the environment does not change mid-run, and this is hit on every
    // graph match. Treat empty/0/false/off as disabled; any other value enables.
    static const bool s_enabled = [] {
        const std::string value = hipdnn_data_sdk::utilities::getEnv("HIPDNN_AOT_DEBUG");
        return !value.empty() && value != "0" && value != "false" && value != "off";
    }();
    return s_enabled;
}

void aotDebugEmit(const std::string& message)
{
    // Straight to stderr, deliberately independent of HIPDNN_LOG_LEVEL (which
    // defaults to off) so the diagnostic is visible the moment the KA opts in.
    std::fprintf(stderr, "[hipdnn aot-catalog] %s\n", message.c_str());
}

} // namespace aot_catalog_engine
