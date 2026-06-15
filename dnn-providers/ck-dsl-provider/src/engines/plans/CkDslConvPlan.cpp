// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslConvPlan.hpp"

#include <algorithm>
#include <cstring>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <unordered_map>

#include "ck_dsl_runtime/timing.hpp"

namespace ck_dsl_plugin {

CkDslConvPlan::CkDslConvPlan(CkDslConvParamParser::ParsedConvParams params,
                             std::unique_ptr<ck_dsl::Kernel> kernel)
    : params_(std::move(params)), kernel_(std::move(kernel)) {}

void* CkDslConvPlan::findBuffer(int64_t uid, const hipdnnPluginDeviceBuffer_t* bufs,
                                uint32_t count) {
    for (uint32_t i = 0; i < count; ++i)
        if (bufs[i].uid == uid) return bufs[i].ptr;
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "CkDslConv: buffer uid not found");
}

void CkDslConvPlan::execute(const CkDslHandle& handle,
                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                            uint32_t numDeviceBuffers, void* /*workspace*/) const {
    void* A = findBuffer(params_.x_uid, deviceBuffers, numDeviceBuffers);  // NHWC input
    void* B = findBuffer(params_.w_uid, deviceBuffers, numDeviceBuffers);  // KRSC weights
    void* D = findBuffer(params_.y_uid, deviceBuffers, numDeviceBuffers);  // NHWK output

    const auto& p = params_;
    int Ho = p.Ho(), Wo = p.Wo();
    int elt = 2;  // fp16/bf16
    uint64_t a_bytes = (uint64_t)p.N * p.Hi * p.Wi * p.C * elt;
    uint64_t b_bytes = (uint64_t)p.K * p.R * p.S * (p.C / std::max(p.G, 1)) * elt;
    uint64_t d_bytes = (uint64_t)p.N * Ho * Wo * p.K * elt;

    const auto& m = kernel_->manifest();
    long M = (long)p.N * Ho * Wo;
    unsigned m_tiles = (unsigned)((M + m.block_m - 1) / m.block_m);
    unsigned n_tiles = (unsigned)((p.K + m.block_n - 1) / m.block_n);
    // grid_order NM: block.x = N-tile, block.y = M-tile.
    std::array<unsigned, 3> grid = (m.grid_order == "NM")
                                       ? std::array<unsigned, 3>{n_tiles, m_tiles, 1}
                                       : std::array<unsigned, 3>{m_tiles, n_tiles, 1};
    unsigned block = (unsigned)m.threads_per_block;

    // Launch, timed under CK_DSL_TIME=1 (launchUs, stream-synced).
    ck_dsl::ScopedTimer t("conv", ck_dsl::ScopedTimer::Unit::Us);
    kernel_->launch({{"A", A}, {"B", B}, {"D", D}},
                    {{"A_bytes", a_bytes}, {"B_bytes", b_bytes}, {"D_bytes", d_bytes}}, grid, block,
                    handle.getStream());
    if (ck_dsl::timing_enabled())
        ck_dsl::hip_check(hipStreamSynchronize(handle.getStream()), "conv sync");
}

}  // namespace ck_dsl_plugin
