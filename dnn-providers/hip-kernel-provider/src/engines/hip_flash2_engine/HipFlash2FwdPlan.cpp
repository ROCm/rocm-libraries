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
    // Single-pass uses only registers and LDS. Split-K needs fp32 partials:
    // po[B*H][splitK][Sq][D] plus per-chunk m and l, sized by the same helper
    // the builder used, so the two can never disagree.
    return _params.workspaceBytes;
}

void HipFlash2FwdPlan::execute(const Handle& handle,
                               const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                               uint32_t numDeviceBuffers,
                               void* workspace) const
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

    void* q = findPtr(_params.qUid, "Q");
    void* k = findPtr(_params.kUid, "K");
    void* v = findPtr(_params.vUid, "V");
    void* o = findPtr(_params.oUid, "O");

    // -- 2. Populate kernel argument struct -----------------------------------
    Flash2KernelArgs args{};
    args.ptrQ = q;
    args.ptrK = k;
    args.ptrV = v;
    args.ptrO = o;

    args.batch = _params.batch;
    args.numHeadsQ = _params.numHeadsQ;
    args.numHeadsK = _params.numHeadsK;
    args.seqLenQ = _params.seqLenQ;
    args.seqLenKv = _params.seqLenKv;
    args.headDim = _params.headDim;
    args.causal = _params.causal ? 1 : 0;

    // Attention scale: use provided value or default to 1/sqrt(headDim)
    args.scale = (_params.attnScale != 0.0f)
                     ? _params.attnScale
                     : 1.0f / std::sqrt(static_cast<float>(_params.headDim));

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
    args.qStrideBatch = checkedStride(_params.qStrideBatch, "qStrideBatch");
    args.qStrideHead = checkedStride(_params.qStrideHead, "qStrideHead");
    args.qStrideSeq = checkedStride(_params.qStrideSeq, "qStrideSeq");
    args.kStrideBatch = checkedStride(_params.kStrideBatch, "kStrideBatch");
    args.kStrideHead = checkedStride(_params.kStrideHead, "kStrideHead");
    args.kStrideSeq = checkedStride(_params.kStrideSeq, "kStrideSeq");
    args.vStrideBatch = checkedStride(_params.vStrideBatch, "vStrideBatch");
    args.vStrideHead = checkedStride(_params.vStrideHead, "vStrideHead");
    args.vStrideSeq = checkedStride(_params.vStrideSeq, "vStrideSeq");
    args.oStrideBatch = checkedStride(_params.oStrideBatch, "oStrideBatch");
    args.oStrideHead = checkedStride(_params.oStrideHead, "oStrideHead");
    args.oStrideSeq = checkedStride(_params.oStrideSeq, "oStrideSeq");

    // -- 3. Grid dimensions ----------------------------------------------------
    // Tile size is a property of the SELECTED variant, not a constant.
    const unsigned int qPerCta = _params.qPerCta;
    const unsigned int gridX = (static_cast<unsigned>(_params.seqLenQ) + qPerCta - 1u) / qPerCta;
    // Finding 3 fix: kernel decodes blockIdx.y=batch, blockIdx.z=head_q
    const auto gridY = static_cast<unsigned>(_params.batch);
    const auto gridZ = static_cast<unsigned>(_params.numHeadsQ);

    // Block dim must match the selected variant's __launch_bounds__. A
    // mismatch is not benign: too few threads silently computes wrong results,
    // too many fails with hipErrorLaunchFailure (719).
    const unsigned int blockDim = _params.blockDim;

    // -- 4. Split-K path -------------------------------------------------------
    // Two launches on one stream: the split pass writes fp32 partials per KV
    // chunk, the merge pass combines them with the exact online-softmax
    // rescale. Same-stream ordering is the synchronisation -- no explicit
    // barrier needed (SdpaBwdPlan relies on the same property).
    if(_params.splitK > 1)
    {
        if(workspace == nullptr)
        {
            const std::string msg
                = "HipFlash2FwdPlan::execute -- split-K requires a workspace of "
                  + std::to_string(_params.workspaceBytes)
                  + " bytes but the caller passed nullptr. Query getWorkspaceSize() "
                    "and pass the allocation on the variant pack.";
            HIPDNN_PLUGIN_LOG_ERROR(msg);
            throw std::runtime_error(msg);
        }
        if(_kernel.mergeFunction() == nullptr)
        {
            const std::string msg
                = "HipFlash2FwdPlan::execute -- split-K selected but the merge kernel "
                  "was not loaded";
            HIPDNN_PLUGIN_LOG_ERROR(msg);
            throw std::runtime_error(msg);
        }

        // Carve the workspace exactly as the merge kernel indexes it. No
        // zeroing: chunks with no work still write m = -inf and l = 0, so
        // every slot the merge reads has been initialised by the split pass.
        const size_t rows
            = static_cast<size_t>(_params.batch) * static_cast<size_t>(_params.num_heads_q)
              * static_cast<size_t>(_params.splitK) * static_cast<size_t>(_params.seq_len_q);
        auto* po = static_cast<float*>(workspace);
        auto* pm = po + rows * static_cast<size_t>(_params.head_dim);
        auto* pl = pm + rows;

        Flash2SplitKernelArgs sargs{};
        sargs.ptr_q = Q;
        sargs.ptr_k = K;
        sargs.ptr_v = V;
        sargs.ptr_po = po;
        sargs.ptr_pm = pm;
        sargs.ptr_pl = pl;
        sargs.batch = _params.batch;
        sargs.num_heads_q = _params.num_heads_q;
        sargs.num_heads_k = _params.num_heads_k;
        sargs.seq_len_q = _params.seq_len_q;
        sargs.seq_len_kv = _params.seq_len_kv;
        sargs.head_dim = _params.head_dim;
        sargs.scale = args.scale;
        sargs.causal = args.causal;
        sargs.nsplit = _params.splitK;
        sargs.q_stride_batch = args.q_stride_batch;
        sargs.q_stride_head = args.q_stride_head;
        sargs.q_stride_seq = args.q_stride_seq;
        sargs.k_stride_batch = args.k_stride_batch;
        sargs.k_stride_head = args.k_stride_head;
        sargs.k_stride_seq = args.k_stride_seq;
        sargs.v_stride_batch = args.v_stride_batch;
        sargs.v_stride_head = args.v_stride_head;
        sargs.v_stride_seq = args.v_stride_seq;

        // Split grid: (query tiles, batch*heads, chunks). Note blockIdx.y is
        // the fused batch-head pair here, unlike the single-pass kernel which
        // takes batch and head as separate grid dimensions.
        const unsigned int splitGridX
            = (static_cast<unsigned>(_params.seq_len_q) + _params.qPerCta - 1u) / _params.qPerCta;
        const unsigned int bh
            = static_cast<unsigned>(_params.batch) * static_cast<unsigned>(_params.num_heads_q);
        if(!launchFlash2SplitKernel(_kernel.function(),
                                    sargs,
                                    splitGridX,
                                    bh,
                                    static_cast<unsigned>(_params.splitK),
                                    _params.blockDim,
                                    handle.getStream()))
        {
            throw std::runtime_error("HipFlash2FwdPlan::execute: split kernel launch failed");
        }

        Flash2MergeKernelArgs margs{};
        margs.ptr_po = po;
        margs.ptr_pm = pm;
        margs.ptr_pl = pl;
        margs.ptr_o = O;
        margs.batch = _params.batch;
        margs.num_heads_q = _params.num_heads_q;
        margs.seq_len_q = _params.seq_len_q;
        margs.head_dim = _params.head_dim;
        margs.nsplit = _params.splitK;
        margs.o_stride_batch = args.o_stride_batch;
        margs.o_stride_head = args.o_stride_head;
        margs.o_stride_seq = args.o_stride_seq;

        // Merge: 4 query rows per 256-thread CTA.
        constexpr unsigned int K_MERGE_ROWS_PER_CTA = 4;
        const unsigned int mergeGridX
            = (static_cast<unsigned>(_params.seq_len_q) + K_MERGE_ROWS_PER_CTA - 1u)
              / K_MERGE_ROWS_PER_CTA;
        if(!launchFlash2MergeKernel(
               _kernel.mergeFunction(), margs, mergeGridX, bh, handle.getStream()))
        {
            throw std::runtime_error("HipFlash2FwdPlan::execute: merge kernel launch failed");
        }
        return;
    }

    // -- 5. Single-pass dispatch -----------------------------------------------
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
