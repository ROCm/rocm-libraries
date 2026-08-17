/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._matmul_nbits_large_n_opt.h -- C99 port of
 * the single public symbol of
 * rocke/instances/common/_matmul_nbits_large_n_opt.py:
 *
 *   Python (_matmul_nbits_large_n_opt.py)        C99 (this header)
 *   -------------------------------------------  ---------------------------------
 *   build_large_n_opt_matmul_nbits(spec, arch)   rocke_build_large_n_opt_matmul_nbits()
 *
 * This is the combined-optimization large-N WMMA body for MatMulNBits
 * (gfx1201). It folds three optimizations into one kernel so their combination
 * can be measured on hardware against the baseline body:
 *
 *   (1) LDS-staged A: the tile_m x tile_k activation tile is staged once per
 *       K-group into LDS, then every wave reads its A fragments from LDS.
 *   (2) tile_k == group_size: the per-group scale is constant across the tile;
 *       B fragments are dequantised UNSCALED (raw int4 -> f16), contracted into
 *       a group-local f32 accumulator, then scaled once per output column with a
 *       single vector_fma into the main accumulator.
 *   (3) LDS epilogue transpose: the column-distributed WMMA accumulator is
 *       written to LDS, then stored to global cooperatively as coalesced b128.
 *
 * Preconditions (asserted by the builder / caller): gfx1201, tile_k ==
 * group_size, and M a multiple of tile_m. The per-element store M-guard is
 * dropped (full-tile precondition).
 *
 * This function EMITS IR. It reproduces the Python build's builder-call sequence
 * op-for-op so the produced rocke_kernel_def_t is byte-faithful to the Python
 * IRBuilder output.
 *
 * Peer dependencies (declared in sibling helper_*.h, resolved at link time):
 *   - rocke_matmul_nbits_spec_t / rocke_matmul_nbits_kernel_name /
 *     rocke_matmul_nbits_scale_wire_dtype  (_matmul_nbits_common.h)
 *   - rocke_wmma_params / rocke_wmma_params_t (_matmul_nbits_large_n.h)
 *   - rocke_unpack_i4_byte_to_pair_f16      (i4_dequant.h)
 *
 * Lifetime: every node is arena-owned (rocke_ir_builder_t.arena); nothing is freed
 * individually, mirroring the Python GC lifetime.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_MATMUL_NBITS_LARGE_N_OPT_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_MATMUL_NBITS_LARGE_N_OPT_H

#include "rocke/helper_rocke.instances.common._matmul_nbits_common.h" /* spec */
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* build_large_n_opt_matmul_nbits(spec, arch="gfx1201").
 *
 * Builds the combined-optimization large-N WMMA IR into the supplied (already
 * rocke_ir_builder_init'd) builder `b`, exactly as the Python build does, and
 * returns the kernel (b->kernel) on success, or NULL with b's sticky error set.
 *
 * `arch` NULL => "gfx1201" (the Python default). Like the Python, this expects
 * the builder to have been created with spec.kernel_name(); it does NOT re-init
 * the builder (the caller owns its lifetime). Use
 * rocke_build_large_n_opt_matmul_nbits_new() for the init-from-spec convenience.
 *
 * Validation failures (tile_k != group_size; tile_m*tile_k or tile_m*tile_n not
 * divisible by block_size*8; bad scale_dtype) set the builder's sticky error
 * (ROCKE_ERR_VALUE) and return NULL, matching the Python ValueErrors. */
rocke_kernel_def_t* rocke_build_large_n_opt_matmul_nbits(rocke_ir_builder_t* b,
                                                         const rocke_matmul_nbits_spec_t* spec,
                                                         const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_large_n_opt_matmul_nbits_new(rocke_ir_builder_t* b,
                                                             const rocke_matmul_nbits_spec_t* spec,
                                                             const char* arch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_MATMUL_NBITS_LARGE_N_OPT_H */
