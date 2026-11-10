/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_fwd_wmma_gate_internal.h -- PRIVATE capability gate
 * and shared facts of the C++ ports of the wave32 WMMA forward convolutions.
 * Declares no public API. Its users, as in Python:
 *   - rocke_wmma_fwd_gate: conv_direct_fwd_wmma.cpp, conv_fwd_small_wmma.cpp,
 *     conv_direct_fwd_gstream_wmma.cpp, conv_direct_fwd_tapouter_wmma.cpp;
 *   - rocke_wmma_fwd_limits_ok (the validator tail): conv_direct_fwd_wmma.cpp,
 *     conv_direct_fwd_gstream_wmma.cpp, conv_direct_fwd_tapouter_wmma.cpp;
 *   - rocke_wmma_fwd_tensor_bytes_ok (the tail's i32 byte-offset check): the
 *     tail and conv_fwd_small_wmma.cpp;
 *   - the halo helpers: conv_direct_fwd_wmma.cpp,
 *     conv_direct_fwd_gstream_wmma.cpp;
 *   - ROCKE_FWD_WMMA_CH_PAD: the four TDM-staging families above;
 *   - the WMMA atom (ROCKE_FWD_WMMA_*, rocke_wmma_ab_dtype,
 *     rocke_wmma_fwd_op_id): all of the above and conv_direct_dw_fwd_wmma.cpp,
 *     which uses neither the gate nor the validator tail.
 *
 *   Python (conv_fwd_wmma_gate.py)       C++ (this header)
 *   ----------------------------------   ----------------------------------
 *   WMMA_M / WMMA_N / WMMA_K             ROCKE_FWD_WMMA_M / _N / _K
 *   CH_PAD                               ROCKE_FWD_WMMA_CH_PAD
 *   wmma_ab_dtype(dtype)                 rocke_wmma_ab_dtype
 *   wmma_fwd_op(dtype, arch)             rocke_wmma_fwd_op_id
 *   wmma_fwd_gate(p, arch, name, wave,   rocke_wmma_fwd_gate
 *                 per_axis_pad=...)
 *   wmma_fwd_limits_reason(p, sched,     rocke_wmma_fwd_limits_ok
 *                          n_blocks)
 *   wmma_fwd_tensor_bytes_reason(p)      rocke_wmma_fwd_tensor_bytes_ok
 *   wmma_fwd_halo_extent(...)            rocke_wmma_fwd_halo_extent
 *   wmma_fwd_halo_cols(...)              rocke_wmma_fwd_halo_cols
 *   wmma_fwd_lead_cols(...)              rocke_wmma_fwd_lead_cols
 *
 * The checks run in the Python order and write the same rejection messages.
 */
#ifndef ROCKE_INSTANCE_CONV_FWD_WMMA_GATE_INTERNAL_H
#define ROCKE_INSTANCE_CONV_FWD_WMMA_GATE_INTERNAL_H

#include <stddef.h>

#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_t */
#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h" /* rocke_ir_builder_t */

/* The fp32-accumulating WMMA atom every forward family issues: M x N x K. */
#define ROCKE_FWD_WMMA_M 16
#define ROCKE_FWD_WMMA_N 16
#define ROCKE_FWD_WMMA_K 32

/* Channels of row skew past each LDS row of a staged X / W tile (16 bytes). */
#define ROCKE_FWD_WMMA_CH_PAD 8

/* wmma_ab_dtype: the WMMA operand dtype of a conv dtype (NULL reads as fp16). */
const char* rocke_wmma_ab_dtype(const char* dtype);

/* wmma_fwd_op: the op id of the M x N x K fp32-accumulating WMMA of conv
 * `dtype` on `arch`. Emits no IR; on a missing atom sets the builder error
 * "<who>: no WMMA atom" and returns NULL. */
const char* rocke_wmma_fwd_op_id(rocke_ir_builder_t* b,
                                 const char* dtype,
                                 const char* arch,
                                 const char* who);

/* wmma_fwd_gate: whether kernel `name` can run `p` on `arch` at the spec's
 * `wave_size`, with every extent at least 1 (rocke_direct_problem_dims_reason)
 * and one padding for both axes (any column padding when `per_axis_pad`: the
 * kernel offsets columns by PAD_W). On success stores the resolved target in
 * `*target_out` and returns true, so callers need not resolve it again;
 * otherwise writes the reason and returns false. */
bool rocke_wmma_fwd_gate(const rocke_direct_conv_problem_t* p,
                         const char* arch,
                         const char* name,
                         int wave_size,
                         bool per_axis_pad,
                         const rocke_archtarget_t** target_out,
                         char* reason,
                         size_t reason_cap);

/* wmma_fwd_limits_reason: the checks a forward validator ends with -- a known
 * scheduler strategy (`sched` NULL: none), i32 byte offsets into X, W and Y of
 * `p`, and an i32 flat block id for `n_blocks` workgroups. Returns true when
 * all hold; otherwise writes the reason and returns false. */
bool rocke_wmma_fwd_limits_ok(const rocke_direct_conv_problem_t* p,
                              const char* sched,
                              long long n_blocks,
                              char* reason,
                              size_t reason_cap);

/* wmma_fwd_tensor_bytes_reason: true when i32 byte offsets address X, W and
 * Y of `p`; otherwise writes the reason and returns false. */
bool rocke_wmma_fwd_tensor_bytes_ok(const rocke_direct_conv_problem_t* p,
                                    char* reason,
                                    size_t reason_cap);

/* wmma_fwd_halo_extent: input rows (columns) a tile of `tile` output rows
 * (columns) reads through a `k`-tap filter axis at `stride` and `dil`. */
int rocke_wmma_fwd_halo_extent(int stride, int tile, int k, int dil);

/* wmma_fwd_halo_cols: columns of one (phase) sub-halo -- the whole halo row,
 * or with `phase_split` one of the `p->stride` column phases. */
int rocke_wmma_fwd_halo_cols(const rocke_direct_conv_problem_t* p, int tile_w, bool phase_split);

/* wmma_fwd_lead_cols: most leading zero columns a (phase) sub-halo gets at
 * the left edge. */
int rocke_wmma_fwd_lead_cols(int pad_w, int stride, bool phase_split);

#endif /* ROCKE_INSTANCE_CONV_FWD_WMMA_GATE_INTERNAL_H */
