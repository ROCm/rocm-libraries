/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_dense_bwd_shell.h -- C port
 * of library/kernels/common/_attention_dense_bwd_shell.py, the shell of the
 * dense SDPA backward main kernel: its addressing, its compile-time band
 * classification and emit_dense_bwd_main (the kernel's entry point is
 * rocke_build_attn_dense_bwd_main_new, rocke/instance_attention_dense_bwd_spec.h).
 *
 *   Python (_attention_dense_bwd_shell.py)  C (this header)
 *   --------------------------------------  ----------------------------------
 *   BufferRowSlab (+ load_vec, load)        rocke_attn_bwd_row_slab_t with rsrc
 *                                           set (rocke_attn_bwd_buffer_row_slab,
 *                                           + _load_vec, _load; the RowSlab
 *                                           entry points dispatch to them)
 *   DenseFixedAddr (+ slab)                 rocke_attn_bwd_dense_fixed_addr_t
 *                                           (+ _init, _slab)
 *   dense_static_classes                    rocke_attn_bwd_dense_static_classes
 *   DiagonalBand / DenseNoMask              rocke_attn_bwd_diagonal_band_init /
 *     (_StaticClassMixin)                     rocke_attn_bwd_dense_no_mask_init
 *                                             (the mask policy plus any_edge /
 *                                             any_interior; static_runs and the
 *                                             interior_q_range shortcuts are
 *                                             read by the shell's kv-tile driver)
 *   emit_dense_bwd_main                     rocke_attn_bwd::emit_dense_bwd_main
 *                                           (C++ section)
 *
 * Errors: a Python ValueError records the status and message on the builder
 * (bool entries return false); the host-only dense_static_classes returns a
 * status and fills err.
 *
 * Not exposed: emit_dense_bwd_main's sentinel=False (negative-control tests
 * only); the builder always emits the dead-row term.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_DENSE_BWD_SHELL_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_DENSE_BWD_SHELL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_band.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* BufferRowSlab(b, rsrc, dtype, t_stride): a row slab (rsrc set). */
rocke_attn_bwd_row_slab_t rocke_attn_bwd_buffer_row_slab(rocke_ir_builder_t* b,
                                                         rocke_value_t* rsrc,
                                                         const rocke_type_t* dtype,
                                                         long long t_stride);

/* BufferRowSlab.load_vec(row, col0, n): n a multiple of 8 (else a
 * ValueError on the builder and NULL). */
rocke_value_t* rocke_attn_bwd_buffer_row_slab_load_vec(const rocke_attn_bwd_row_slab_t* s,
                                                       rocke_value_t* row,
                                                       rocke_attn_bwd_iv_t col0,
                                                       int n);

/* BufferRowSlab.load(row, col) */
rocke_value_t* rocke_attn_bwd_buffer_row_slab_load(const rocke_attn_bwd_row_slab_t* s,
                                                   rocke_value_t* row,
                                                   rocke_attn_bwd_iv_t col);

/* DenseFixedAddr. has_strided: the StridedTensorAddr view (oob = "clamp" or
 * the global view); else the buffer view over batch_ptr. */
typedef struct rocke_attn_bwd_dense_fixed_addr
{
    rocke_ir_builder_t* b;
    const rocke_type_t* dtype;
    long long strides[3];
    bool clamp; /* _clamp: oob == "clamp" or ragged */
    bool has_strided;
    rocke_attn_bwd_strided_tensor_addr_t strided;
    rocke_value_t* batch_ptr;
    long long num_records;
} rocke_attn_bwd_dense_fixed_addr_t;

/* DenseFixedAddr(b, ptr, dtype=, strides=, seq=, seqlen=, head_size=, oob=,
 * global_view=, ragged=). */
bool rocke_attn_bwd_dense_fixed_addr_init(rocke_attn_bwd_dense_fixed_addr_t* t,
                                          rocke_ir_builder_t* b,
                                          rocke_value_t* ptr,
                                          const rocke_type_t* dtype,
                                          const long long strides[3],
                                          const rocke_attn_bwd_seq_info_t* seq,
                                          long long seqlen,
                                          int head_size,
                                          const char* oob,
                                          bool global_view,
                                          bool ragged);

/* DenseFixedAddr.slab(h, row0, clamp=) */
rocke_attn_bwd_row_slab_t
    rocke_attn_bwd_dense_fixed_addr_slab(const rocke_attn_bwd_dense_fixed_addr_t* t,
                                         rocke_attn_bwd_iv_t h,
                                         rocke_attn_bwd_iv_t row0,
                                         bool clamp);

/* dense_static_classes(s_q=, s_kv=, k_m0=, k_n0=, left=, right=,
 * bottom_right=, masked=) -> (any_edge, any_interior). */
rocke_status_t rocke_attn_bwd_dense_static_classes(long long s_q,
                                                   long long s_kv,
                                                   int k_m0,
                                                   int k_n0,
                                                   int left,
                                                   int right,
                                                   bool bottom_right,
                                                   bool masked,
                                                   bool* any_edge,
                                                   bool* any_interior,
                                                   char* err,
                                                   size_t err_cap);

/* DiagonalBand(b, s_q=, s_kv=, left=, right=, bottom_right=, k_m0=, k_n0=):
 * the RuntimeBand policy over build-time ints, plus its static classes. */
bool rocke_attn_bwd_diagonal_band_init(rocke_attn_bwd_mask_policy_t* self,
                                       bool* any_edge,
                                       bool* any_interior,
                                       rocke_ir_builder_t* b,
                                       long long s_q,
                                       long long s_kv,
                                       int left,
                                       int right,
                                       bool bottom_right,
                                       int k_m0,
                                       int k_n0);

/* DenseNoMask(b, s_q=, s_kv=, k_m0=, k_n0=) */
bool rocke_attn_bwd_dense_no_mask_init(rocke_attn_bwd_mask_policy_t* self,
                                       bool* any_edge,
                                       bool* any_interior,
                                       rocke_ir_builder_t* b,
                                       long long s_q,
                                       long long s_kv,
                                       int k_m0,
                                       int k_n0);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>

#include "rocke/instance_attention_bwd.h"

struct rocke_attn_bwd_dense_spec;

namespace rocke_attn_bwd
{
/* emit_dense_bwd_main(spec, arch) after validation: body is the resolved,
 * validated body spec, b initialised with the spec's kernel name. Throws
 * ckc::Error (callers run it under ckc::guard_builder). */
rocke_kernel_def_t* emit_dense_bwd_main(rocke_ir_builder_t* b,
                                        const struct rocke_attn_bwd_dense_spec& spec,
                                        const std::string& arch,
                                        const rocke_attn_bwd_spec_t& body);
} // namespace rocke_attn_bwd
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_DENSE_BWD_SHELL_H */
