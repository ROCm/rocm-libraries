/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C ABI mirror of instances/common/peer_gemm.py.
 * PeerGemmSpec -> rocke_peer_gemm_spec_t; build_peer_gemm -> rocke_build_peer_gemm_new.
 * Reuses existing MFMA helpers; HIP owns addressability and input publication.
 * Device A/C pointer tables select contiguous row-major panels by grid.z.
 * Scatter storage is panel-packed C_owner[compute_rank, rows, columns], not
 * ordinary full-width row-major C. Caller owns tables/data until all users drain.
 */
#ifndef ROCKE_INSTANCE_PEER_GEMM_H
#define ROCKE_INSTANCE_PEER_GEMM_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct rocke_peer_gemm_spec
    {
        int rows_per_source;
        int cols_per_rank;
        int k;
        int sources;
        const char* dtype;
        const char* name;
    } rocke_peer_gemm_spec_t;

    /* Required panel dimensions are zero until filled; other fields match Python defaults. */
    rocke_peer_gemm_spec_t rocke_peer_gemm_spec_default(void);
    bool rocke_peer_gemm_is_valid_spec(
        const rocke_peer_gemm_spec_t* spec, const char* arch, char* reason, size_t reason_cap);
    rocke_status_t rocke_peer_gemm_kernel_name(
        const rocke_peer_gemm_spec_t* spec, char* out, size_t out_cap);
    /* Initializes b; returned nodes belong to b and are freed with rocke_ir_builder_free. */
    rocke_kernel_def_t* rocke_build_peer_gemm_new(
        rocke_ir_builder_t* b, const rocke_peer_gemm_spec_t* spec, const char* arch);
    rocke_status_t rocke_peer_gemm_grid(const rocke_peer_gemm_spec_t* spec, int out[3]);
    /* On success out_ll is malloc'd and caller frees it. */
    rocke_status_t rocke_peer_gemm_lower_to_llvm(const rocke_peer_gemm_spec_t* spec,
        const char* arch, rocke_llvm_flavor_t flavor, char** out_ll, char* err, size_t err_cap);

    /* Additive opt-in schedule API; rocke_peer_gemm_spec_t layout is unchanged.
     * PeerGemmScheduleSpec -> rocke_peer_gemm_schedule_spec_t.
     * A CTA cap bounds submitted workgroups, not residency or physical peer traffic.
     */
    typedef struct rocke_peer_gemm_schedule_spec
    {
        rocke_peer_gemm_spec_t gemm;
        const char* work_order;
        int max_ctas;
    } rocke_peer_gemm_schedule_spec_t;

    rocke_peer_gemm_schedule_spec_t rocke_peer_gemm_schedule_spec_default(void);
    bool rocke_peer_gemm_schedule_is_valid_spec(const rocke_peer_gemm_schedule_spec_t* spec,
        const char* arch, char* reason, size_t reason_cap);
    rocke_status_t rocke_peer_gemm_schedule_kernel_name(
        const rocke_peer_gemm_schedule_spec_t* spec, char* out, size_t out_cap);
    rocke_status_t rocke_peer_gemm_schedule_grid(
        const rocke_peer_gemm_schedule_spec_t* spec, int out[3]);
    rocke_kernel_def_t* rocke_build_scheduled_peer_gemm_new(
        rocke_ir_builder_t* b, const rocke_peer_gemm_schedule_spec_t* spec, const char* arch);
    rocke_status_t rocke_peer_gemm_schedule_lower_to_llvm(
        const rocke_peer_gemm_schedule_spec_t* spec, const char* arch, rocke_llvm_flavor_t flavor,
        char** out_ll, char* err, size_t err_cap);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_INSTANCE_PEER_GEMM_H */
