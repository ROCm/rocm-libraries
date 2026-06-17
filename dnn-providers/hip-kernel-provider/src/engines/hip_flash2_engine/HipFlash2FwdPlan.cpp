// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipFlash2FwdPlan.hpp"

#include <cmath>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <stdexcept>
#include <unordered_map>

namespace hip_flash2_engine {

HipFlash2FwdPlan::HipFlash2FwdPlan(HipModuleGuard kernel, Flash2FwdParams params)
    : _kernel(std::move(kernel)), _params(std::move(params)) {}

size_t HipFlash2FwdPlan::getWorkspaceSize(const Handle& /*handle*/) const {
    // Flash-Attention 2 V7 uses only registers and LDS — zero global workspace.
    return 0;
}

void HipFlash2FwdPlan::execute(const Handle& handle,
                               const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                               uint32_t numDeviceBuffers, void* /*workspace*/) const {
    // ── 1. Build UID → device pointer map ────────────────────────────────────
    std::unordered_map<int64_t, void*> uidToPtrMap;
    uidToPtrMap.reserve(numDeviceBuffers);
    for (uint32_t i = 0; i < numDeviceBuffers; ++i) {
        uidToPtrMap[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    auto findPtr = [&](int64_t uid, const char* name) -> void* {
        auto it = uidToPtrMap.find(uid);
        if (it == uidToPtrMap.end()) {
            HIPDNN_PLUGIN_LOG_ERROR("HipFlash2FwdPlan::execute — missing buffer for tensor '"
                                    << name << "' (uid=" << uid << ")");
            throw std::runtime_error(std::string("HipFlash2FwdPlan: missing tensor buffer '") +
                                     name + "'");
        }
        return it->second;
    };

    void* Q = findPtr(_params.qUid, "Q");
    void* K = findPtr(_params.kUid, "K");
    void* V = findPtr(_params.vUid, "V");
    void* O = findPtr(_params.oUid, "O");

    // ── 2. Populate kernel argument struct ───────────────────────────────────
    Flash2KernelArgs args{};
    args.ptr_q = Q;
    args.ptr_k = K;
    args.ptr_v = V;
    args.ptr_o = O;

    args.batch = _params.batch;
    args.num_heads_q = _params.num_heads_q;
    args.num_heads_k = _params.num_heads_k;
    args.seq_len_q = _params.seq_len_q;
    args.seq_len_kv = _params.seq_len_kv;
    args.head_dim = _params.head_dim;
    args.causal = _params.causal ? 1 : 0;

    // Attention scale: use provided value or default to 1/sqrt(head_dim)
    args.scale = (_params.attn_scale != 0.0f)
                     ? _params.attn_scale
                     : 1.0f / std::sqrt(static_cast<float>(_params.head_dim));

    // Strides (in elements, BHSD layout)
    args.q_stride_batch = static_cast<int>(_params.q_stride_batch);
    args.q_stride_head = static_cast<int>(_params.q_stride_head);
    args.q_stride_seq = static_cast<int>(_params.q_stride_seq);
    args.k_stride_batch = static_cast<int>(_params.k_stride_batch);
    args.k_stride_head = static_cast<int>(_params.k_stride_head);
    args.k_stride_seq = static_cast<int>(_params.k_stride_seq);
    args.v_stride_batch = static_cast<int>(_params.v_stride_batch);
    args.v_stride_head = static_cast<int>(_params.v_stride_head);
    args.v_stride_seq = static_cast<int>(_params.v_stride_seq);
    args.o_stride_batch = static_cast<int>(_params.o_stride_batch);
    args.o_stride_head = static_cast<int>(_params.o_stride_head);
    args.o_stride_seq = static_cast<int>(_params.o_stride_seq);

    // ── 3. Grid dimensions ────────────────────────────────────────────────────
    // V7 uses BQ=64 tile — one CTA per (tile_q, head, batch)
    constexpr unsigned int K_BQ = 64;
    const unsigned int gridX = (static_cast<unsigned>(_params.seq_len_q) + K_BQ - 1u) / K_BQ;
    const unsigned int gridY = static_cast<unsigned>(_params.num_heads_q);
    const unsigned int gridZ = static_cast<unsigned>(_params.batch);

    // Block dim: 4 warps × 64 threads/warp = 256 threads per CTA
    constexpr unsigned int K_BLOCK_DIM = 256;

    // ── 4. Dispatch ───────────────────────────────────────────────────────────
    const bool ok = launchFlash2Kernel(_kernel.function(), args, gridX, gridY, gridZ, K_BLOCK_DIM,
                                       handle.getStream());
    if (!ok) {
        HIPDNN_PLUGIN_LOG_ERROR("HipFlash2FwdPlan::execute — kernel launch failed");
    }
}

}  // namespace hip_flash2_engine
