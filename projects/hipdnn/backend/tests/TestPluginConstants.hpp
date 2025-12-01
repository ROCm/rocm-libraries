// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

namespace hipdnn_backend::plugin_constants
{
// Test plugin directory constants relative to backend library location
inline const std::string& getTestPluginDefaultDir()
{

    static std::string s_defaultDir
#if defined(_WIN32)
        = "bin/test_plugins";
#else
        = "lib/test_plugins";
#endif
    return s_defaultDir;
}

} // namespace hipdnn_backend::plugin_constants
