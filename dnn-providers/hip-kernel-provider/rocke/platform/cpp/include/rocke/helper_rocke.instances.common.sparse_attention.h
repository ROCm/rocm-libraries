/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common.sparse_attention.h -- C99 port of the
 * LDS bitmap primitives from rocke/instances/common/sparse_attention.py
 * (CK Tile ``50_sparse_attn`` parity).
 *
 * Only the shared LDS-bitmap building blocks of the Jenga / VSA sparse
 * attention kernels are ported here (the symbols the porting task names):
 *
 *   Python (sparse_attention.py)            C99 (this header)
 *   -------------------------------------   ------------------------------------
 *   _BLOCK_SIZE = 64                        ROCKE_SPARSE_ATTN_BLOCK_SIZE
 *   _const_i8(b, value)                     rocke_sparse_attn_const_i8
 *   _cooperative_iter(b, tid, total, body)  rocke_sparse_attn_cooperative_iter
 *   _stage_jenga_mask_to_lds(...)           rocke_sparse_attn_stage_jenga_mask_to_lds
 *   _stage_vsa_bitmap_to_lds(...)           rocke_sparse_attn_stage_vsa_bitmap_to_lds
 *   _lds_bitmap_predicate(b, lds, idx)      rocke_sparse_attn_lds_bitmap_predicate
 *
 * Each routine reproduces its Python counterpart's rocke_b_* builder-call
 * sequence byte-faithfully: same ops, same order, same operands, same
 * result-name hints. The host-side control structure (the chunk loop, the
 * static range checks, the scf_if scoping) is reproduced exactly so the
 * emitted op stream is identical to the Python.
 *
 * The Python ``_cooperative_iter`` takes a host-side ``body(slot)`` closure;
 * the C analogue threads a function pointer + an opaque ``user`` context
 * (the standard closure-emulation idiom used across this port).
 *
 * Error model mirrors the rest of the C port: everything routes through the
 * sticky-error IRBuilder (rocke_b_*); a first failure latches builder->status
 * and turns subsequent calls into no-ops.
 *
 * Lifetime: every node returned is arena-owned (rocke_ir_builder_t.arena).
 * Nothing is freed individually.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_SPARSE_ATTENTION_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_SPARSE_ATTENTION_H

#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_value_t, rocke_type_t */

#ifdef __cplusplus
extern "C" {
#endif

/* _BLOCK_SIZE = 64 -- one wave64 per CTA (matches the mfma_attention helper). */
#define ROCKE_SPARSE_ATTN_BLOCK_SIZE 64

/* _const_i8(b, value): build an i8 constant via ``arith.constant`` (there is no
 * scalar i8 const factory, so the generic op layer is used directly with the
 * "ci8" result-name hint -- byte-identical to the Python). Returns the i8 SSA
 * value, or NULL if the builder is already in an error state. */
rocke_value_t* rocke_sparse_attn_const_i8(rocke_ir_builder_t* b, int value);

/* The host-side ``body(slot)`` closure passed to _cooperative_iter and the
 * VSA scatter pass. ``slot`` is the per-lane i32 index; ``user`` is the
 * caller's opaque context. */
typedef void (*rocke_sparse_attn_slot_body_fn)(rocke_ir_builder_t* b,
                                               rocke_value_t* slot,
                                               void* user);

/* _cooperative_iter(b, tid, total, body): run ``body(slot)`` for each
 * ``slot in [0, total)`` distributed across the wave64. For ``total > 64`` a
 * chained ``chunk * 64`` walk is issued; the static in-range check is elided
 * when a chunk is fully covered, otherwise a per-lane ``scf_if(slot < total)``
 * guards the body. ``total <= 0`` is a no-op. */
void rocke_sparse_attn_cooperative_iter(rocke_ir_builder_t* b,
                                        rocke_value_t* tid,
                                        int total,
                                        rocke_sparse_attn_slot_body_fn body,
                                        void* user);

/* _stage_jenga_mask_to_lds(...): cooperatively copy one Q-block's mask row
 * (``Mask[q_block, :]``, ``num_k_blocks`` i8 bytes at ``mask_row_base``) into a
 * freshly allocated LDS i8 array. Returns the LDS allocation handle (so the
 * per-K-tile predicate can read it in O(1)); the caller issues the trailing
 * ``rocke_b_sync``. NULL on a builder error. */
rocke_value_t* rocke_sparse_attn_stage_jenga_mask_to_lds(rocke_ir_builder_t* b,
                                                         rocke_value_t* mask_global,
                                                         rocke_value_t* mask_row_base,
                                                         int num_k_blocks,
                                                         rocke_value_t* tid);

/* _stage_vsa_bitmap_to_lds(...): build the per-(q_block) K-attend bitmap in
 * LDS via three cooperative passes: allocate ``num_k_blocks`` i8 slots, zero
 * them (sync), then for each lane ``l in [0, max_blocks_per_q)`` with
 * ``l < block_count[q_block]`` scatter ``bitmap[block_lut[q_block, l]] = 1``.
 * Concurrent idempotent stores of ``1`` are safe. Returns the LDS bitmap
 * handle; the caller issues the trailing ``rocke_b_sync``. NULL on a builder
 * error. */
rocke_value_t* rocke_sparse_attn_stage_vsa_bitmap_to_lds(rocke_ir_builder_t* b,
                                                         rocke_value_t* block_lut,
                                                         rocke_value_t* block_count,
                                                         rocke_value_t* q_block_idx,
                                                         rocke_value_t* lut_row_base,
                                                         int num_k_blocks,
                                                         int max_blocks_per_q,
                                                         rocke_value_t* tid);

/* _lds_bitmap_predicate(b, bitmap_lds, k_block_idx): ``bitmap_lds[k_block_idx]
 * != 0`` -- one LDS byte read + i8 compare. Returns the i1 predicate, or NULL
 * on a builder error. */
rocke_value_t* rocke_sparse_attn_lds_bitmap_predicate(rocke_ir_builder_t* b,
                                                      rocke_value_t* bitmap_lds,
                                                      rocke_value_t* k_block_idx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_SPARSE_ATTENTION_H */
