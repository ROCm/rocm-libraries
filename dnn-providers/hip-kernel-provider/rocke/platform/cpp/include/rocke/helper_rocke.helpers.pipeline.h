/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_rocke.helpers.pipeline.h -- C++ port of the runtime-extent ping-pong
 * of rocke/helpers/pipeline.py:
 *
 *   SoftwarePipeline.run_ping_pong_dynamic  -> rocke_software_pipeline_run_ping_pong_dynamic
 *   SoftwarePipeline._ping_pong_phase       -> (internal)
 *
 * Only the dynamic (runtime reduction extent) driver is ported: the
 * compile-time variants unroll in Python and have no C++ caller. The builder
 * call sequence is the Python one, so a kernel built through either emits the
 * same IR -- which the byte-identity gate checks for every kernel that uses it.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_PIPELINE_H
#define ROCKE_HELPER_ROCKE_HELPERS_PIPELINE_H

#include <stdbool.h>

#include "rocke/helper_rocke.helpers.schedule.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The SoftwarePipeline fields the dynamic ping-pong reads. */
typedef struct rocke_software_pipeline
{
    bool wait_vmcnt;
    bool sync_after_wait;
    bool sync_before_issue;
    bool overlap_vmcnt;
} rocke_software_pipeline_t;

/* One LDS buffer set (A and B tiles). */
typedef struct rocke_buffer_pair
{
    rocke_value_t* a;
    rocke_value_t* b;
} rocke_buffer_pair_t;

/* issue_load_fn(k_offset, buf_pair) */
typedef void (*rocke_pipeline_issue_load_fn)(rocke_ir_builder_t* b,
                                             rocke_value_t* k_offset,
                                             const rocke_buffer_pair_t* buf,
                                             void* user);

/* compute_fn(k_offset, buf_pair, state) -> new state (num_state values). */
typedef void (*rocke_pipeline_compute_fn)(rocke_ir_builder_t* b,
                                          rocke_value_t* k_offset,
                                          const rocke_buffer_pair_t* buf,
                                          rocke_value_t* const* state_in,
                                          int num_state,
                                          rocke_value_t** state_out,
                                          void* user);

/* Double-buffered ping-pong over a runtime reduction extent.
 *
 * k_extent: i32 value, the reduction extent in elements (a slice end under
 *   split-K). block_k: compile-time tile width.
 * k_lo: first tile offset; NULL means const 0.
 * k_zero_fill: offset whose tile reads as zero (the global extent); NULL means
 *   none. The body computes two tiles per step, so with an odd tile count the
 *   phase-A prefetch lands at or past k_extent. That reads zero only at the
 *   real end of the tensor; under split-K it would be the next slice's first
 *   tile, so it is redirected here.
 * buffers: exactly two buffer pairs. iter_args: the loop-carried state.
 * schedule: nullable; its compute prologue/epilogue wrap each compute.
 * results: receives num_iter_args loop results.
 *
 * Returns false with the builder error set on invalid arguments. */
bool rocke_software_pipeline_run_ping_pong_dynamic(const rocke_software_pipeline_t* pipe,
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
                                                   rocke_value_t** results);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_PIPELINE_H */
