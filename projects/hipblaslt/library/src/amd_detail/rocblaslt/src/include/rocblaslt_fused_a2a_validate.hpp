// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "handle.h"

#if HIPBLASLT_HAS_GEMM_A2A_FUSION

inline rocblaslt_status validate_a2a_gemm(const _rocblaslt_handle*           handle,
                                          const RocblasltContractionProblem& problem,
                                          const RocblasltFusedEpilogueInfo&  info)
{
    const int64_t world = handle->device_comm_world;
    if(info.a2aPeerBPtrs == nullptr || info.a2aPeerBCount != (size_t)world
       || info.a2aGatherBuffer == nullptr)
        return rocblaslt_status_invalid_value;

    const int64_t k = problem.k;
    const int64_t n = problem.n;
    if(info.a2aExtent != 0 && info.a2aExtent != k)
        return rocblaslt_status_invalid_value;
    if(k <= 0 || k % world != 0)
        return rocblaslt_status_invalid_value;
    if(problem.trans_b != HIPBLAS_OP_N || problem.swizzleB || problem.b_type != HIP_R_16BF
       || problem.col_stride_b != (size_t)k)
        return rocblaslt_status_invalid_value;

    // A gathered row is one copy, with a 14-bit width field in 16-byte units;
    // each block starts 128-byte aligned in the gather buffer.
    const int64_t rowBytes = (k / world) * (int64_t)sizeof(hip_bfloat16);
    if(rowBytes / 16 > (int64_t{1} << 14) || (n * rowBytes) % 128 != 0)
        return rocblaslt_status_invalid_value;

    return rocblaslt_status_success;
}

// Everything about a fused A2A request that is checkable before the heuristic runs.
inline rocblaslt_status validate_fused_a2a(const _rocblaslt_handle*           handle,
                                           const RocblasltContractionProblem& problem)
{
    RocblasltFusedEpilogueInfo info;
    const bool                 fused = rocblaslt_resolve_fused_a2a(problem, info);
    if(info.misattached)
        return rocblaslt_status_invalid_value;
    if(!fused)
        return rocblaslt_status_success;

    if(handle == nullptr || handle->device_comm_world == 0)
        return rocblaslt_status_invalid_value;
    if(problem.epilogue != ROCBLASLT_EPILOGUE_DEFAULT)
        return rocblaslt_status_invalid_value;
    if(problem.batch_count != 1)
        return rocblaslt_status_invalid_value;
    if(info.commChannel >= handle->device_comm_channels)
        return rocblaslt_status_invalid_value;

    if(info.mode == RocblasltFusedEpilogueInfo::Mode::A2AGemm)
        return validate_a2a_gemm(handle, problem, info);

    if(info.a2aExtent > int64_t(problem.m)
       || info.a2aExtent % int64_t(handle->device_comm_world) != 0)
        return rocblaslt_status_invalid_value;
    if(info.a2aRecvPtrs == nullptr)
        return rocblaslt_status_invalid_value;
    return rocblaslt_status_success;
}

inline bool fused_a2a_lacks_sdma_queues(const RocblasltContractionProblem& problem)
{
    RocblasltFusedEpilogueInfo info;
    return rocblaslt_resolve_fused_a2a(problem, info) && info.a2aSdmaQueues == nullptr;
}

#endif
