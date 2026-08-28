// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "handle.h"

// Everything about a fused A2A request that is checkable before the heuristic runs.
inline rocblaslt_status validate_fused_a2a(const _rocblaslt_handle*           handle,
                                           const RocblasltContractionProblem& problem)
{
    RocblasltFusedEpilogueInfo info;
    if(!rocblaslt_resolve_fused_epilogue(problem.fused_epilogue, info) || !info.hasA2APrefix)
        return rocblaslt_status_success;

    if(handle == nullptr || handle->device_comm_world == 0)
        return rocblaslt_status_invalid_value;
    if(problem.epilogue != ROCBLASLT_EPILOGUE_DEFAULT)
        return rocblaslt_status_invalid_value;
    if(problem.batch_count != 1)
        return rocblaslt_status_invalid_value;
    if(info.a2aExtent > int64_t(problem.m)
       || info.a2aExtent % int64_t(handle->device_comm_world) != 0)
        return rocblaslt_status_invalid_value;
    if(info.commChannel >= handle->device_comm_channels)
        return rocblaslt_status_invalid_value;
    if(info.a2aRecvPtrs == nullptr)
        return rocblaslt_status_invalid_value;
    return rocblaslt_status_success;
}

inline bool fused_a2a_lacks_sdma_queues(const RocblasltContractionProblem& problem)
{
    RocblasltFusedEpilogueInfo info;
    return rocblaslt_resolve_fused_epilogue(problem.fused_epilogue, info) && info.hasA2APrefix
           && info.a2aSdmaQueues == nullptr;
}
