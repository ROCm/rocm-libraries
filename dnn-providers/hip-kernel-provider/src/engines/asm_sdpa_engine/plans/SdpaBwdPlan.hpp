// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "SdpaBwdParams.hpp"
#include "SdpaKernelResolvers.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <memory>
#include <optional>

namespace asm_sdpa_engine
{

/**
 * @brief SDPA backward kernel plan.
 *
 * Orchestrates 2 or 3 ASM kernels for the backward pass:
 *   1. ODO        — D reduction: D[b,h,i] = sum_j(O * dO)
 *   2. DQDKDV     — Main gradients: dQ (FP32 via A32, or direct BF16 via A16), dK, dV
 *   3. DQ_CONVERT — Post-processing: FP32 dQ → BF16 (A32 path only)
 *
 * Accumulator type selects the path:
 *   - A32 (3-kernel path): ODO → DQDKDV → DQ_CONVERT
 *   - A16 (2-kernel path): ODO → DQDKDV (dQ written directly in BF16)
 *
 * When A16, the resolved modules carry no DQ_CONVERT stage and the DQ_CONVERT
 * launch + dq_acc workspace are both skipped.
 */
class SdpaBwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    /// @param kernels Answers for the device each execute() runs on; plan build seeds it
    ///                with the build device's modules. Its DQ_CONVERT stage is present
    ///                exactly when @p params selects the A32 path.
    SdpaBwdPlan(std::unique_ptr<SdpaBwdKernels> kernels, SdpaBwdParams params);

    ~SdpaBwdPlan() override = default;

    SdpaBwdPlan(const SdpaBwdPlan&) = delete;
    SdpaBwdPlan& operator=(const SdpaBwdPlan&) = delete;
    SdpaBwdPlan(SdpaBwdPlan&&) noexcept = default;
    SdpaBwdPlan& operator=(SdpaBwdPlan&&) noexcept = default;

    size_t getWorkspaceSize(const Handle& handle) const override;

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    std::unique_ptr<SdpaBwdKernels> _kernels;
    SdpaBwdParams _params;
};

} // namespace asm_sdpa_engine
