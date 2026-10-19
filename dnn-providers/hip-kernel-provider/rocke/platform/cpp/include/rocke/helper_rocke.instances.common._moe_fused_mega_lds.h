/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._moe_fused_mega_lds.h -- C99 port of
 * rocke/instances/common/_moe_fused_mega_lds.py, the whole-kernel LDS
 * accounting shared by the fused-MoE mega-kernel family.
 *
 *   Python (_moe_fused_mega_lds.py)      C99 (this header / .cpp)
 *   ----------------------------------   -------------------------------------
 *   class LdsAlloc (NamedTuple)          rocke_mega_lds_alloc_t
 *     LdsAlloc.nbytes  (@property)       rocke_mega_lds_alloc_nbytes()
 *   mega_lds_pool_bytes(allocs)          rocke_mega_lds_pool_bytes()
 *   validate_mega_lds_budget(allocs,arch)  rocke_validate_mega_lds_budget()
 *
 * WHY THIS EXISTS. Each mega (moe_fused_mega f16/bf16, moe_fused_mega_fp8, and
 * the gfx1250 fused_moe_mega_wmma -- Python-only, no C mirror) validates its
 * gate/up and its down GEMM as two INDEPENDENT UniversalGemmSpec s. Neither
 * sub-validation sees Hidden_smem -- the persistent LDS bridge that is the whole
 * point of the fusion -- nor the fact that both GEMMs' operand buffers are
 * allocated in the builder prologue and therefore all coexist. A mega whose two
 * halves each fit the per-WG budget can still blow it as a whole; without this
 * module that only surfaces as a kernel-load failure on the device instead of a
 * spec rejection.
 *
 * WHY A PLAIN SUM IS A SAFE BOUND HERE. core/lower_llvm packs every
 * smem_alloc into one @smem_pool global with a liveness-driven linear scan, so
 * an allocation *may* be placed on top of a dead one (the cshuffle epilogue in
 * gemm_universal relies on exactly that). It cannot happen for the megas: a live
 * interval opens at the tile.smem_alloc op, and every mega buffer is allocated
 * in the prologue, before any of them is first used. All pairs therefore
 * interfere by construction and the packer gives each a disjoint range, so the
 * down GEMM's Bd_smem does NOT alias the by-then-dead Bg_smem / Bu_smem.
 *
 * The packer aligns each segment to the larger of its element rule and the
 * widest alignment its accesses claim, never more than the pool alignment
 * (ROCKE_SMEM_POOL_ALIGN, rocke/lower_llvm.h). The accesses are not known here,
 * so every segment is aligned to the pool alignment: the sum is then never
 * below rocke_ll_compute_smem_layout's L->smem_pool_size, whatever the accesses
 * claim. For the shipped megas every segment is a multiple of it, so the bound
 * is exact (the mega tests assert that equality against the lowered IR).
 *
 * Allocations the emitter declares but never references are dead-stripped by the
 * packer and must be left out of the sequence passed here (the fp8 mega's
 * BStage_smem under use_dtla=false is the live example).
 *
 * NOTHING HERE EMITS IR. The builder argument is used only for arena ownership
 * of the formatted reject reason, exactly as rocke_validate_arch_and_block_size
 * uses it.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__MOE_FUSED_MEGA_LDS_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__MOE_FUSED_MEGA_LDS_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_type_t */

#ifdef __cplusplus
extern "C" {
#endif

/* One smem_alloc as the LDS budget sees it (Python LdsAlloc).
 *
 * `name` is the emitter's name_hint (it names the buffer in the rejection
 * message); `elem_bytes` (the element type's rocke_i_type_bytes) and
 * `elem_count` give the segment size. Plain value
 * type: the caller owns the array, the strings are static literals. */
typedef struct rocke_mega_lds_alloc
{
    const char* name;
    int elem_bytes;
    int elem_count;
} rocke_mega_lds_alloc_t;

/* LdsAlloc.nbytes: elem_bytes * elem_count. */
long rocke_mega_lds_alloc_nbytes(const rocke_mega_lds_alloc_t* alloc);

/* mega_lds_pool_bytes(allocs): an upper bound on the bytes the @smem_pool
 * global occupies for `allocs`. Replays the packer's placement for the
 * all-interfering mega case with every segment aligned to ROCKE_SMEM_POOL_ALIGN
 * (the most the packer ever aligns one): each segment starts at the next
 * multiple of it past the previous one, and the pool is rounded up to it. A
 * segment whose size is not a multiple of ROCKE_SMEM_POOL_ALIGN can be counted
 * up to ROCKE_SMEM_POOL_ALIGN - 1 bytes above what the packer places. `allocs`
 * must be in emitter declaration order. */
long rocke_mega_lds_pool_bytes(const rocke_mega_lds_alloc_t* allocs, size_t num_allocs);

/* validate_mega_lds_budget(allocs, arch) -> (ok, reason).
 *
 * Returns true with *out_reason == "ok" when the total fits `arch`'s per-WG
 * budget; false with the Python reject text otherwise. `out_reason` may be NULL
 * to skip. The reason string is either a static literal or a builder-arena-owned
 * formatted message, so `b` must be a live builder; it is never read as IR. */
bool rocke_validate_mega_lds_budget(rocke_ir_builder_t* b,
                                    const rocke_mega_lds_alloc_t* allocs,
                                    size_t num_allocs,
                                    const char* arch,
                                    const char** out_reason);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__MOE_FUSED_MEGA_LDS_H */
