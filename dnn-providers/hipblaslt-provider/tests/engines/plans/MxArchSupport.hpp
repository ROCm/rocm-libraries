// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string_view>

#include <hipdnn_test_sdk/utilities/ArchMatch.hpp>

namespace hipblaslt_plugin_test
{

// hipBLASLt only ships VEC32_UE8M0 MX GEMM kernels for gfx950 and gfx1250.
inline bool isMxSupportedArch(std::string_view archName)
{
    using hipdnn_test_sdk::utilities::archMatches;
    using hipdnn_test_sdk::utilities::ArchMatchMode;
    return archMatches(archName, "gfx950", ArchMatchMode::PREFIX)
           || archMatches(archName, "gfx1250", ArchMatchMode::PREFIX);
}

} // namespace hipblaslt_plugin_test
