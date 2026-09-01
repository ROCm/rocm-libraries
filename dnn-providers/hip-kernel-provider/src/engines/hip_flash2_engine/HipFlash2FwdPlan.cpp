// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipFlash2FwdPlan.hpp"

#include <cmath>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace hip_flash2_engine
{

HipFlash2FwdPlan::HipFlash2FwdPlan(HipModuleGuard kernel, Flash2FwdParams params)
    : _kernel(std::move(kernel))
    , _params(std::move(params))
{
}

size_t HipFlash2FwdPlan::getWorkspaceSize(const Handle& /*handle*/) const
{
    // Flash-Attention 2 V7 uses only registers and LDS -- zero global workspace.
    return 0;
}

void HipFlash2FwdPlan::execute(const Handle& handle,
                               const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                               uint32_t numDeviceBuffers,
                               void* /*workspace*/) const
{
    // -- 1. Build UID -> device pointer map ------------------------------------
    std::unordered_map<int64_t, void*> uidToPtrMap;
    uidToPtrMap.reserve(numDeviceBuffers);
    for(uint32_t i = 0; i < numDeviceBuffers; ++i)
    {
        uidToPtrMap[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    auto findPtr = [&](int64_t uid, const char* name) -> void* {
        auto it = uidToPtrMap.find(uid);
        if(it == uidToPtrMap.end())
        {
            HIPDNN_PLUGIN_LOG_ERROR("HipFlash2FwdPlan::execute -- missing buffer for tensor '"
                                    << name << "' (uid=" << uid << ")");
            throw std::runtime_error(std::string("HipFlash2FwdPlan: missing tensor buffer '") + name
                                     + "'");
        }
        return it->second;
    };

    void* Q = findPtr(_params.qUid, "Q");
    void* K = findPtr(_params.kUid, "K");
    void* V = findPtr(_params.vUid, "V");
    void* O = findPtr(_params.oUid, "O");

    // -- 2. Populate kernel argument struct -----------------------------------
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

    // Strides (in elements, BHSD layout).
    // Guard against int64_t -> int truncation (I9): strides must fit in int.
    // For the FP16 shapes this engine accepts (seq <= 131072, D <= 128, H <= 128,
    // B <= 32768) the largest possible batch stride is ~32768x128x131072x128
    // which overflows int.  Log and abort if any stride exceeds INT_MAX.
    auto checkedStride = [&](int64_t s, const char* name) -> int {
        if(s > static_cast<int64_t>(std::numeric_limits<int>::max()) || s < 0)
        {
            HIPDNN_PLUGIN_LOG_ERROR("HipFlash2FwdPlan::execute -- stride '" << name << "'=" << s
                                                                            << " out of int range");
            throw std::overflow_error(std::string("HipFlash2FwdPlan: stride overflow '") + name
                                      + "'");
        }
        return static_cast<int>(s);
    };
    args.q_stride_batch = checkedStride(_params.q_stride_batch, "q_stride_batch");
    args.q_stride_head = checkedStride(_params.q_stride_head, "q_stride_head");
    args.q_stride_seq = checkedStride(_params.q_stride_seq, "q_stride_seq");
    args.k_stride_batch = checkedStride(_params.k_stride_batch, "k_stride_batch");
    args.k_stride_head = checkedStride(_params.k_stride_head, "k_stride_head");
    args.k_stride_seq = checkedStride(_params.k_stride_seq, "k_stride_seq");
    args.v_stride_batch = checkedStride(_params.v_stride_batch, "v_stride_batch");
    args.v_stride_head = checkedStride(_params.v_stride_head, "v_stride_head");
    args.v_stride_seq = checkedStride(_params.v_stride_seq, "v_stride_seq");
    args.o_stride_batch = checkedStride(_params.o_stride_batch, "o_stride_batch");
    args.o_stride_head = checkedStride(_params.o_stride_head, "o_stride_head");
    args.o_stride_seq = checkedStride(_params.o_stride_seq, "o_stride_seq");

    // -- 3. Grid dimensions -------------------------------?

    // -- 3. Grid dimensions ----------------------------------------------------
    // Tile size is a property of the SELECTED variant, not a constant.
    const unsigned int qPerCta = _params.qPerCta;
    const unsigned int gridX = (static_cast<unsigned>(_params.seq_len_q) + qPerCta - 1u) / qPerCta;
    // Finding 3 fix: kernel decodes blockIdx.y=batch, blockIdx.z=head_q
    const unsigned int gridY = static_cast<unsigned>(_params.batch);
    const unsigned int gridZ = static_cast<unsigned>(_params.num_heads_q);

    // Block dim must match the selected variant's __launch_bounds__. A
    // mismatch is not benign: too few threads silently computes wrong results,
    // too many fails with hipErrorLaunchFailure (719).
    const unsigned int blockDim = _params.blockDim;

    // -- 4. Dispatch -----------------------------------------------------------
    // I5: propagate launch failure so callers see a hard error.
    const bool ok = launchFlash2Kernel(
        _kernel.function(), args, gridX, gridY, gridZ, blockDim, handle.getStream());
    if(!ok)
    {
        HIPDNN_PLUGIN_LOG_ERROR("HipFlash2FwdPlan::execute -- kernel launch failed");
        throw std::runtime_error("HipFlash2FwdPlan::execute: hipModuleLaunchKernel failed");
    }
}

} // namespace hip_flash2_engine
