// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <vector>

#include "common/GemmCall.hpp"
#include "engines/plans/PlanInterface.hpp"

namespace hipblaslt_plugin
{

/// An ordered list of GEMM calls run through rocBLAS on the handle's stream. rocBLAS takes its
/// scratch from its own handle, so the plan needs no workspace.
class RocblasGemmPlan : public IPlan
{
public:
    /// Throws when rocBLAS would round a narrow result more than once on @p gcnArchName (by
    /// default the current device's): a reduction it splits into a non-fp32 output
    /// (rocblasRoundsReductionChunks), or bf16 products it rounds in a GEMV kernel
    /// (rocblasGemvRoundsBf16Products).
    explicit RocblasGemmPlan(std::vector<GemmCall> calls, const std::string& gcnArchName = {});

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

    const std::vector<GemmCall>& calls() const
    {
        return _calls;
    }

private:
    std::vector<GemmCall> _calls;
};

} // namespace hipblaslt_plugin
