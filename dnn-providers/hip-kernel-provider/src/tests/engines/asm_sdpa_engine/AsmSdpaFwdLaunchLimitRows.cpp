// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Kept apart from AsmSdpaConfigHelpers.cpp, which the integration test binary also
// compiles: that binary links only the SDKs, not the engine, so it cannot resolve
// fwdConfigBlockDimX.

#include "../../../engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"
#include "AsmSdpaConfigHelpers.hpp"
#include "AsmSdpaTestSupport.hpp"

namespace asm_sdpa_engine
{

std::vector<AsmFwdLaunchLimitRow> getFwdLaunchLimitRows(const std::string& arch)
{
    std::vector<AsmFwdLaunchLimitRow> rows;
    for(const auto& [key, config] : cfg_fmha_fwd)
    {
        if(config.arch != arch)
        {
            continue;
        }
        rows.push_back({config.knl_name,
                        getFwdKernelTocKeys(config, arch),
                        config.max_flat_workgroup_size,
                        fwdConfigBlockDimX(config)});
    }
    return rows;
}

} // namespace asm_sdpa_engine
