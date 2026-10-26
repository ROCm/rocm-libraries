// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "handle.h"

#include <cstdint>

#if HIPBLASLT_HAS_GEMM_A2A_FUSION

inline size_t fused_a2a_element_bytes(hipDataType type)
{
    switch(type)
    {
    case HIP_R_64F:
    case HIP_C_32F:
        return 8;
    case HIP_R_32F:
    case HIP_R_32I:
        return 4;
    case HIP_R_16F:
    case HIP_R_16BF:
        return 2;
    default:
        return 1;
    }
}

inline bool fused_a2a_spans_overlap(const void* a, size_t aBytes, const void* b, size_t bBytes)
{
    if(a == nullptr || b == nullptr || aBytes == 0 || bBytes == 0)
        return false;
    const uintptr_t a0 = reinterpret_cast<uintptr_t>(a);
    const uintptr_t b0 = reinterpret_cast<uintptr_t>(b);
    return a0 < b0 + bBytes && b0 < a0 + aBytes;
}

// Byte span of a column-major matrix.
inline size_t fused_a2a_matrix_bytes(int64_t rows, int64_t cols, size_t ld, hipDataType type)
{
    if(rows <= 0 || cols <= 0)
        return 0;
    return ((size_t)(cols - 1) * ld + (size_t)rows) * fused_a2a_element_bytes(type);
}

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
    const size_t  bpe      = fused_a2a_element_bytes(problem.b_type);
    const int64_t rowBytes = (k / world) * (int64_t)bpe;
    if(rowBytes / 16 > (int64_t{1} << 14) || (n * rowBytes) % 128 != 0)
        return rocblaslt_status_invalid_value;

    if(problem.B == nullptr)
        return rocblaslt_status_success;

    if(info.a2aPeerBPtrs[handle->device_comm_rank] != problem.B)
        return rocblaslt_status_invalid_value;

    const size_t gatherBytes = (size_t)n * (size_t)k * bpe;
    const bool   transA      = problem.trans_a != HIPBLAS_OP_N;
    const size_t aBytes      = fused_a2a_matrix_bytes(
        transA ? k : problem.m, transA ? problem.m : k, problem.col_stride_a, problem.a_type);
    const size_t bBytes = fused_a2a_matrix_bytes(k, n, problem.col_stride_b, problem.b_type);
    const size_t cBytes
        = fused_a2a_matrix_bytes(problem.m, n, problem.col_stride_c, problem.c_type);
    const size_t dBytes
        = fused_a2a_matrix_bytes(problem.m, n, problem.col_stride_d, problem.d_type);
    if(fused_a2a_spans_overlap(info.a2aGatherBuffer, gatherBytes, problem.A, aBytes)
       || fused_a2a_spans_overlap(info.a2aGatherBuffer, gatherBytes, problem.B, bBytes)
       || fused_a2a_spans_overlap(info.a2aGatherBuffer, gatherBytes, problem.C, cBytes)
       || fused_a2a_spans_overlap(info.a2aGatherBuffer, gatherBytes, problem.D, dBytes))
        return rocblaslt_status_invalid_value;

    return rocblaslt_status_success;
}

// Everything about a fused A2A request that is checkable before the heuristic runs,
// plus, when the problem carries its pointers, the ones that depend on them.
inline rocblaslt_status validate_fused_a2a(const _rocblaslt_handle*           handle,
                                           const RocblasltContractionProblem& problem)
{
    RocblasltFusedEpilogueInfo info;
    if(!rocblaslt_resolve_fused_epilogue(problem.fused_epilogue, problem.fused_prologue, info))
        return rocblaslt_status_success;
    if(info.misattached)
        return rocblaslt_status_invalid_value;
    if(info.mode == RocblasltFusedEpilogueInfo::Mode::Disabled)
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
    return rocblaslt_resolve_fused_epilogue(problem.fused_epilogue, problem.fused_prologue, info)
           && info.mode != RocblasltFusedEpilogueInfo::Mode::Disabled
           && info.a2aSdmaQueues == nullptr;
}

#endif
