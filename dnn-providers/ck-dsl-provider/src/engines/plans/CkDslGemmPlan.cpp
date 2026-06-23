// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslGemmPlan.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "ck_dsl_runtime/timing.hpp"

namespace ck_dsl_plugin {

CkDslGemmPlan::CkDslGemmPlan(CkDslParamParser::ParsedGemmParams params,
                             std::unique_ptr<ck_dsl::Kernel> kernel)
    : params_(std::move(params)), kernel_(std::move(kernel)) {}

void* CkDslGemmPlan::findBuffer(int64_t uid, const hipdnnPluginDeviceBuffer_t* bufs,
                                uint32_t count) {
    for (uint32_t i = 0; i < count; ++i)
        if (bufs[i].uid == uid) return bufs[i].ptr;
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "CkDslGemmPlan: buffer uid not found");
}

void CkDslGemmPlan::execute(const CkDslHandle& handle,
                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                            uint32_t numDeviceBuffers, void* /*workspace*/) const {
    void* A = findBuffer(params_.a_uid, deviceBuffers, numDeviceBuffers);
    void* B = findBuffer(params_.b_uid, deviceBuffers, numDeviceBuffers);
    void* C = findBuffer(params_.c_uid, deviceBuffers, numDeviceBuffers);

    auto grid = kernel_->gemm_grid(params_.M, params_.N);
    unsigned block = static_cast<unsigned>(kernel_->manifest().threads_per_block);
    // Launch (kernarg pack + hipModuleLaunchKernel). Under CK_DSL_TIME=1 the
    // stream is synced so launchUs reflects end-to-end kernel time.
    ck_dsl::ScopedTimer t("gemm", ck_dsl::ScopedTimer::Unit::Us);
    kernel_->launch({{"A", A}, {"B", B}, {"C", C}},
                    {{"M", static_cast<uint64_t>(params_.M)},
                     {"N", static_cast<uint64_t>(params_.N)},
                     {"K", static_cast<uint64_t>(params_.K)}},
                    grid, block, handle.getStream());
    if (ck_dsl::timing_enabled())
        ck_dsl::hip_check(hipStreamSynchronize(handle.getStream()), "gemm sync");
}

}  // namespace ck_dsl_plugin
