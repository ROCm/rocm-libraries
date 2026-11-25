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
    explicit RocblasGemmPlan(const std::vector<GemmCall>& calls,
                             const std::string& gcnArchName = {});

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

    const std::vector<GemmCall>& calls() const
    {
        return _calls;
    }

    /// Runs every call with rocBLAS atomics not allowed (rocblas_backend::ScopedNoAtomics), so
    /// the plan gives the same bits on every execute. The convolution engines set it: they claim
    /// bitwise-reproducible results (RFC 0025).
    void requireBitwiseReproducible()
    {
        _bitwiseReproducible = true;
    }

    bool bitwiseReproducible() const
    {
        return _bitwiseReproducible;
    }

private:
    std::vector<GemmCall> _calls;
    bool _bitwiseReproducible = false;
};

/// Throws when rocBLAS on @p gcnArchName (by default the current device's) has only untuned
/// kernels for one of @p calls (rocblasFp32OutputIsUntuned: a 16-bit GEMM into fp32 on gfx11,
/// 25-180x behind): the explicit-GEMM convolution engines decline such a plan, so neither their
/// default nor autotune runs it.
void requireTunedRocblasKernels(const std::vector<GemmCall>& calls,
                                const std::string& gcnArchName = {});

} // namespace hipblaslt_plugin
