// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * pipeline.cpp -- C++ port of SoftwarePipeline.run_ping_pong_dynamic
 * (rocke/helpers/pipeline.py). See helper_rocke.helpers.pipeline.h.
 */
#include "rocke/helper_rocke.helpers.pipeline.h"

#include <vector>

#include "rocke/ir_internal.h" /* rocke_i_set_err */

namespace
{

/* _ping_pong_phase: one prefetch+compute phase. Issue the next tile into the
 * buffer the current compute is *not* reading, wait for everything except
 * that just-issued load, then compute. */
void ping_pong_phase(const rocke_software_pipeline_t* pipe,
                     rocke_ir_builder_t* b,
                     rocke_value_t* k_cur,
                     rocke_value_t* k_next,
                     const rocke_buffer_pair_t* cur_buf,
                     const rocke_buffer_pair_t* nxt_buf,
                     std::vector<rocke_value_t*>& state,
                     rocke_pipeline_issue_load_fn issue_load_fn,
                     rocke_pipeline_compute_fn compute_fn,
                     void* user,
                     const rocke_schedule_policy_t* schedule)
{
    if(pipe->sync_before_issue)
    {
        /* Close the ABA window: every wave must be done reading nxt_buf (two
         * phases ago) before it is overwritten. */
        if(pipe->overlap_vmcnt)
            rocke_b_sync_lds_only(b);
        else
            rocke_b_sync(b);
    }
    issue_load_fn(b, k_next, nxt_buf, user);
    if(pipe->wait_vmcnt)
    {
        /* prefetch_depth == 1: leave the just-issued load in flight. */
        rocke_b_s_waitcnt(b, pipe->overlap_vmcnt ? 1 : 0, -1, -1);
    }
    if(pipe->sync_after_wait)
    {
        if(pipe->overlap_vmcnt)
            rocke_b_sync_lds_only(b);
        else
            rocke_b_sync(b);
    }
    if(schedule != NULL)
        rocke_schedule_policy_emit_compute_prologue(schedule, b);
    std::vector<rocke_value_t*> next(state.size());
    compute_fn(b, k_cur, cur_buf, state.data(), (int)state.size(), next.data(), user);
    state.swap(next);
    if(schedule != NULL)
        rocke_schedule_policy_emit_compute_epilogue(schedule, b);
}

} // namespace

extern "C" bool
    rocke_software_pipeline_run_ping_pong_dynamic(const rocke_software_pipeline_t* pipe,
                                                  rocke_ir_builder_t* b,
                                                  rocke_value_t* k_extent,
                                                  int block_k,
                                                  rocke_value_t* k_lo,
                                                  rocke_value_t* k_zero_fill,
                                                  const rocke_buffer_pair_t buffers[2],
                                                  const rocke_iter_arg_t* iter_args,
                                                  int num_iter_args,
                                                  rocke_pipeline_issue_load_fn issue_load_fn,
                                                  rocke_pipeline_compute_fn compute_fn,
                                                  void* user,
                                                  const rocke_schedule_policy_t* schedule,
                                                  rocke_value_t** results)
{
    if(pipe == NULL || buffers == NULL || issue_load_fn == NULL || compute_fn == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "run_ping_pong_dynamic: missing argument");
        return false;
    }
    if(block_k <= 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "block_k must be positive, got %d", block_k);
        return false;
    }

    const rocke_buffer_pair_t* buf0 = &buffers[0];
    const rocke_buffer_pair_t* buf1 = &buffers[1];
    rocke_value_t* c_bk = rocke_b_const_i32(b, block_k);
    rocke_value_t* c_2bk = rocke_b_const_i32(b, 2 * block_k);
    rocke_value_t* k_first = (k_lo == NULL) ? rocke_b_const_i32(b, 0) : k_lo;

    /* Prologue: stage the first tile into buf0 so the first phase's compute
     * has something to read. Every later tile is staged by the phase before
     * the one that consumes it. */
    issue_load_fn(b, k_first, buf0, user);

    rocke_for_t for_op = rocke_b_scf_for_iter(b,
                                              k_first,
                                              k_extent,
                                              c_2bk,
                                              iter_args,
                                              num_iter_args,
                                              "k_pipe",
                                              /*unroll=*/false,
                                              /*elide_trailing_barrier=*/true);
    rocke_b_region_enter(b, for_op.body);
    {
        std::vector<rocke_value_t*> state(for_op.iter_vars,
                                          for_op.iter_vars + for_op.num_iter_vars);
        rocke_value_t* k1 = rocke_b_add(b, for_op.iv, c_bk);
        rocke_value_t* k2 = rocke_b_add(b, for_op.iv, c_2bk);
        /* Whether tile k+1 exists: picks the zero-fill prefetch and gates Phase
         * B's state. Emitted once, only when one of them needs it. */
        rocke_value_t* k1_in = NULL;
        if(k_zero_fill != NULL || pipe->mask_tail_state)
            k1_in = rocke_b_cmp_lt(b, k1, k_extent);
        rocke_value_t* k1_load = k1;
        if(k_zero_fill != NULL)
            k1_load = rocke_b_select(b, k1_in, k1, k_zero_fill);
        /* Phase A: compute tile k out of buf0 while tile k+1 streams into buf1. */
        ping_pong_phase(pipe,
                        b,
                        for_op.iv,
                        k1_load,
                        buf0,
                        buf1,
                        state,
                        issue_load_fn,
                        compute_fn,
                        user,
                        schedule);
        /* Phase B: the buffers swap roles. */
        std::vector<rocke_value_t*> state_a(state);
        ping_pong_phase(
            pipe, b, k1, k2, buf1, buf0, state, issue_load_fn, compute_fn, user, schedule);
        if(pipe->mask_tail_state)
        {
            for(size_t i = 0; i < state.size(); ++i)
                state[i] = rocke_b_select(b, k1_in, state[i], state_a[i]);
        }
        rocke_b_scf_yield(b, state.data(), (int)state.size());
    }
    rocke_b_region_leave(b);

    /* The final phase left a prefetch in flight and (with overlap_vmcnt) only
     * an LDS-scoped barrier behind it. Drain both before the epilogue, which
     * stages its own data through the same LDS. */
    rocke_b_s_waitcnt(b, 0, -1, -1);
    rocke_b_sync(b);

    for(int i = 0; i < for_op.op->num_results; ++i)
        results[i] = for_op.op->results[i];
    return true;
}
