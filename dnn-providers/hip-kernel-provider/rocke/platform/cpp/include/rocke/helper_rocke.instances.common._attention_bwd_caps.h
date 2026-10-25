/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_caps.h -- C port of the
 * per-arch capability facts of the attention backward kernels
 * (library/kernels/common/_attention_bwd_caps.py).
 *
 *   Python (_attention_bwd_caps.py)     C (this header)
 *   ---------------------------------   ---------------------------------------
 *   BWD_BARRIER_FENCE                   ROCKE_ATTN_BWD_BARRIER_FENCE
 *
 * Only the facts the aux kernels (attention_bwd_aux) need are ported so far.
 * Not ported yet (later units): BWD_ARCHES, BwdArchCaps / bwd_arch_caps,
 * bwd_atoms, select_bwd_atom, wide_k_atom_available, check_waves,
 * check_lds_bytes, check_lds_dma_width, check_transpose_read, the asm-MFMA
 * tables (mfma_asm_mnemonic, mfma_result_wait_states, asm_tail_nop_min,
 * check_asm_tail_nop, AGPR_ASM_TAIL_NOP, ASM_MFMA_VALU_LEAD_WAIT_STATES),
 * check_agpr_alloc and emit_lds_dma.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_CAPS_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_CAPS_H

/* Fence of every barrier of the backward kernels: the Python builders set it
 * as IRBuilder.fence_barriers, so every b.sync() carries it. The C builder has
 * no such field; every barrier of a backward kernel is emitted with
 * rocke_b_sync_fence(b, ROCKE_ATTN_BWD_BARRIER_FENCE) instead. */
#define ROCKE_ATTN_BWD_BARRIER_FENCE "compiler"

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_CAPS_H */
