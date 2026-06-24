/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_ck_dsl.helpers.grid.h -- C99 port of ck_dsl.helpers.grid.
 *
 * Grid / workgroup-id remapping helpers for chiplet locality.
 *
 * This phase ports ONLY:
 *   chiplet_aware_super_tile_dynamic
 *
 * which is the runtime composition of:
 *   chiplet_transform_chunked_dynamic  (helper, ported here as a file-local
 *                                       static so the public entry is exact)
 *   super_tile_swizzle_dynamic         (likewise)
 *
 * The op sequence emitted into the IRBuilder is byte-identical to the Python.
 */
#ifndef CKC_HELPER_CK_DSL_HELPERS_GRID_H
#define CKC_HELPER_CK_DSL_HELPERS_GRID_H

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default chiplet counts per AMDGPU family (Python module constants). */
#define CKC_NUM_XCDS_MI300X 8 /* CDNA3 */
#define CKC_NUM_XCDS_MI325X 8 /* CDNA3 */
#define CKC_NUM_XCDS_MI350X 8 /* CDNA4 */

/* Python: @dataclass(frozen=True) class SuperTileSwizzleResult
 *   row: Value  # i32 SSA, M-tile index
 *   col: Value  # i32 SSA, N-tile index
 */
typedef struct ckc_super_tile_swizzle_result
{
    ckc_value_t* row; /* i32 SSA, M-tile index */
    ckc_value_t* col; /* i32 SSA, N-tile index */
} ckc_super_tile_swizzle_result_t;

/* Runtime composition of chiplet remap + super-tile swizzle.
 *
 * Python signature:
 *   chiplet_aware_super_tile_dynamic(
 *       b, wgid, *, num_pid_m, num_pid_n, wgm=8,
 *       num_xcds=NUM_XCDS_MI300X, chunk_size=64)
 *
 * wgid / num_pid_m / num_pid_n are i32 SSA Values; wgm / num_xcds /
 * chunk_size are compile-time ints. On invalid args (matching the Python
 * ValueError paths) the builder's sticky error is set and result fields
 * are NULL.
 */
ckc_super_tile_swizzle_result_t ckc_chiplet_aware_super_tile_dynamic(ckc_ir_builder_t* b,
                                                                     ckc_value_t* wgid,
                                                                     ckc_value_t* num_pid_m,
                                                                     ckc_value_t* num_pid_n,
                                                                     int wgm,
                                                                     int num_xcds,
                                                                     int chunk_size);

/* Compile-time composition of chiplet remap + super-tile swizzle.
 *
 * Python signature:
 *   chiplet_aware_super_tile(
 *       b, wgid, *, num_pid_m, num_pid_n, wgm=8,
 *       num_xcds=NUM_XCDS_MI300X, chunk_size=64)
 *
 * Unlike the *_dynamic peer, num_pid_m / num_pid_n are compile-time ints, so
 * the tile-count derived quantities (limit, num_wgid_in_group, c_num_pid_m) are
 * emitted as folded i32 constants rather than as IR div/mul ops. wgid is an i32
 * SSA Value; wgm / num_xcds / chunk_size / num_pid_m / num_pid_n are
 * compile-time ints. The emitted op stream is byte-identical to the Python
 * compile-time helper. On invalid args the builder's sticky error is set and
 * result fields are NULL.
 */
ckc_super_tile_swizzle_result_t ckc_chiplet_aware_super_tile(ckc_ir_builder_t* b,
                                                             ckc_value_t* wgid,
                                                             int num_pid_m,
                                                             int num_pid_n,
                                                             int wgm,
                                                             int num_xcds,
                                                             int chunk_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_HELPERS_GRID_H */
