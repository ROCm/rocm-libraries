/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gemm_universal.h -- C99 port of the universal GEMM kernel
 * instance builder rocke/instances/common/gemm_universal.py.
 *
 * This is the DSL-side counterpart of CK's
 * dispatcher/codegen/unified_gemm_codegen.py: given the exact same config
 * schema CK's dispatcher enumerates, build a CK DSL IR (rocke_kernel_def_t)
 * that lowers to AMDGPU LLVM IR.
 *
 *   Python (gemm_universal.py)            C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class TileSpec                        rocke_gemm_tile_spec_t
 *   class TraitSpec                       rocke_gemm_trait_spec_t
 *   class DataSpec                        rocke_gemm_data_spec_t
 *   class UniversalGemmSpec               rocke_gemm_universal_spec_t
 *   is_valid_spec(spec, arch)             rocke_gemm_universal_is_valid_spec(...)
 *   build_universal_gemm(spec, arch)      rocke_build_universal_gemm(...)
 *   (+ convenience: build -> lower .ll)   rocke_gemm_universal_lower_to_llvm(...)
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python dataclasses are frozen value types
 * with defaults + a __post_init__ that derives block_size. In C the caller
 * fills a rocke_gemm_universal_spec_t. rocke_gemm_universal_spec_default() returns
 * a struct with every field set to the Python dataclass default; the caller
 * then overrides the fields it cares about (name + tile geometry are required)
 * and calls rocke_gemm_universal_spec_finalize() which runs the
 * WarpTileBlockSizeMixin._init_block_size() derivation (block_size==0 =>
 * warp_m*warp_n*warp_k*wave_size).
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_GEMM_UNIVERSAL_H
#define ROCKE_INSTANCE_GEMM_UNIVERSAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ TileSpec *
 *
 * Mirror of Python TileSpec. The three computed @property values
 * (mfmas_per_warp_m, mfmas_per_warp_n, k_atoms_per_tile_k) are NOT stored;
 * they are recomputed by the accessor helpers below (matching the Python
 * properties, including the divisibility ValueError -> -1 / builder error). */
typedef struct rocke_gemm_tile_spec
{
    int tile_m;
    int tile_n;
    int tile_k;
    int warp_m;
    int warp_n;
    int warp_k; /* default 1  */
    int warp_tile_m; /* default 32 */
    int warp_tile_n; /* default 32 */
    int warp_tile_k; /* default 16 */
} rocke_gemm_tile_spec_t;

/* TileSpec.mfmas_per_warp_m / _n / k_atoms_per_tile_k. Return -1 on the Python
 * divisibility ValueError (caller treats as invalid). */
int rocke_gemm_tile_mfmas_per_warp_m(const rocke_gemm_tile_spec_t* t);
int rocke_gemm_tile_mfmas_per_warp_n(const rocke_gemm_tile_spec_t* t);
int rocke_gemm_tile_k_atoms_per_tile_k(const rocke_gemm_tile_spec_t* t);

/* ----------------------------------------------------------------- TraitSpec *
 *
 * pipeline / scheduler / epilogue are the Python Literal strings:
 *   pipeline  : "mem" | "compv3" | "compv4" | "wsp3"
 *   scheduler : "intrawave" | "interwave"
 *   epilogue  : "default" | "cshuffle"
 * Stored as const char* (string literals); compared by strcmp like Python. */
typedef struct rocke_gemm_trait_spec
{
    const char* pipeline; /* default "compv4"    */
    const char* scheduler; /* default "intrawave" */
    const char* epilogue; /* default "cshuffle"  */
    bool pad_m;
    bool pad_n;
    bool pad_k;
    bool persistent;
    bool chiplet_swizzle;
    int chiplet_wgm; /* default 8  */
    int chiplet_num_xcds; /* default 8  */
    int chiplet_chunk_size; /* default 64 */
    /* Python waves_per_eu: Optional[int]. waves_per_eu_set==false => None. */
    bool waves_per_eu_set;
    int waves_per_eu;
    bool preshuffle_b;
    bool direct_to_lds;
    int dtl_cache_a; /* default 0 (CACHE_ALL)    */
    int dtl_cache_b; /* default 2 (CACHE_STREAM) */
    bool dtl_prefetch;
    bool active_tile_skip;
    int lds_k_pad; /* default 0 */
    bool lds_swizzle;
    /* compv4/compv3 schedule-hint policy. Python: Optional[bool] = None.
     * emit_sched_hints_set==false mirrors Python None (arch-resolved: hints
     * OFF on gfx950, ON elsewhere). When set, emit_sched_hints forces the
     * choice. Gates the per-cluster s_setprio/sched_barrier fences AND the
     * two-stage sched_group_barrier HotLoop interleave. */
    bool emit_sched_hints_set; /* false => Python None (arch-resolved) */
    bool emit_sched_hints;
    /* Split-K over the production universal body. When > 1, the kernel takes a
     * third grid dim block_id_z in [0, split_k) selecting a K-slice
     * [z*ks, (z+1)*ks) (ks = K // split_k) for the CTA's K-loop, replaces the
     * C output param with an f32 workspace Cf32[M, N], and the epilogue
     * atomic-adds each warp's f32 accumulator into it. split_k == 1 (default)
     * keeps the canonical single-K-pass body byte-identical. */
    int split_k; /* default 1 */
    /* cshuffle epilogue LDS aliasing. False (default): the smem-pool packer
     * aliases the cshuffle C tile onto the A/B staging bytes (pool = max(ab,c))
     * with a step-0 reuse barrier. True: C gets its own LDS bytes (pool = ab+c)
     * and the barrier is elided -> lower small-tile latency, more LDS. Only
     * affects the cshuffle epilogue; False keeps byte-identical output. */
    bool cshuffle_no_alias; /* default false */
    /* compv4 (VGPR-staged, non-DTL) AB LDS double buffer. Python:
     * Optional[bool] = None. ab_double_buffer_set==false mirrors None (the
     * historical automatic rule: direct epilogue only, and only when the
     * doubled AB fits 2 WG/CU). When set, ab_double_buffer forces the choice;
     * is_valid_spec rejects it off the compv4 non-DTL body. */
    bool ab_double_buffer_set; /* false => Python None (automatic) */
    bool ab_double_buffer;
    /* Opt-in MFMA-cluster priority bookends (Python interwave_setprio, default
     * false): sched_barrier(0); s_setprio(1); <K-atom MFMA cluster>;
     * s_setprio(0); sched_barrier(0). scheduler="interwave" alone keeps the
     * historical (intrawave) emission. MFMA only. */
    bool interwave_setprio;
    /* ---- DTL ring levers (all default-off; the defaults keep every
     * existing kernel byte-identical). Python TraitSpec fields of the same
     * names; see gemm_universal.py for the full rationale. ---- */
    /* Alias-scoped DTL LDS ring: 0 = historical dtl_prefetch body; N >= 2 = one
     * A and one B allocation per stage + lds_alias_scopes + a K-loop unrolled
     * by N with static stages and a partial vmcnt((N-2)*loads) wait. */
    int dtl_stages; /* default 0 */
    /* dtl_stages >= 2: issue the next tile's DTL loads between the MFMAs. */
    bool dtl_issue_spread;
    /* dtl_stages >= 2: one-K-atom-ahead fragment register prefetch + the
     * per-tile wait/barrier/DTL issue in front of the last K atom's MFMAs. */
    bool dtl_frag_prefetch;
    /* dtl_stages >= 2: the in-stream barrier without the fragment prefetch. */
    bool dtl_stream_barrier;
    /* MFMA fed (B, A): the accumulator holds C^T -> 8-byte vector stores in
     * the direct epilogue (MFMA, default epilogue, no pad_n / split-K). */
    bool mfma_swap_ab;
    /* Python iglp_opt: Optional[int]. iglp_opt_set==false => None. */
    bool iglp_opt_set;
    int iglp_opt;
    /* Persistent tile loop with the next tile's DTL prologue issued before the
     * current tile's epilogue stores (0 = off). */
    int persistent_ctas;
    /* MFMA accumulators in AGPRs (agpr_alloc = (n, n), n =
     * per-lane accumulator registers; opaque "=a,0" asm zero seed). */
    bool acc_agpr;
    /* (dtl_frag_prefetch body): every next-K-atom fragment ds_read and
     * every DTL issue between two MFMAs, each MFMA + its slot fenced by
     * sched_barrier(0) (Tensile-style interleave). */
    bool mfma_interleave;
    /* (acc_agpr): pin every MFMA result to the AGPR file
     * (asm "", "=a,0" after each MFMA). */
    bool acc_agpr_pin;
    /* (mfma_interleave): 0 = spread the DTL issues evenly over the
     * cluster; k >= 1 = one every k MFMAs from the first MFMA on. */
    int ilv_dtl_every;
    /* (mfma_interleave): the first k MFMAs of the K tile's last cluster
     * go in front of its wait + barrier (0 = barrier first). */
    int ilv_barrier_after;
    /* (mfma_interleave): ds_reads and DTL issues on one merged schedule. */
    bool ilv_merge;
    /* row shift R of the lds_swizzle XOR key (-1 = geometry default). */
    int lds_swizzle_r;
    /* (dtl_stages >= 2): one buffer resource + soffset per operand and
     * K tile for the DTL loads. */
    bool dtl_hoist;
    /* (persistent_ctas = P): stream-K tail balancing in the
     * persistent loop (in-kernel DP + 2-tile SK partition, linear fixup with
     * self-cleaning ready flags; params SkFlags / SkPart). */
    bool streamk;
    /* (dtl_stages >= 2): DTL buffer resources based at the tile origin
     * with num_records = the operand bytes left (rows past M / N read 0). */
    bool dtl_tight_rsrc;
    /* (pad_m / pad_n, default epilogue): uniform full-tile branch to the
     * unguarded epilogue. */
    bool pad_fast_tile;
    /* Early-release interleave levers, all default off. ilv_early_release (0..3):
     * split the per-tile sync into release + acquire (1: inside the last atom;
     * 2: release in the atom before it; 3: as 2 with the DTL issues spread over
     * the atom boundary). ilv_release_at / ilv_acquire_at: their MFMA
     * positions (0 = auto). */
    int ilv_early_release;
    int ilv_release_at;
    int ilv_acquire_at;
    /* epilogue AGPR hand-out asm "", "=v,0" instead of "=a,0" */
    bool acc_out_vgpr;
    /* front-load a cluster's ds_reads / DTL issues into the first p % of its
     * MFMAs (0 = off) */
    int ilv_read_span;
    int ilv_dtl_span;
} rocke_gemm_trait_spec_t;

/* ------------------------------------------------------------------ DataSpec */
typedef struct rocke_gemm_data_spec
{
    const char* dtype_a; /* default "fp16" */
    const char* dtype_b; /* default "fp16" */
    const char* dtype_c; /* default "fp16" */
    const char* dtype_acc; /* default "fp32" */
    const char* layout; /* default "RCR"  */
} rocke_gemm_data_spec_t;

/* -------------------------------------------------------- UniversalGemmSpec */
typedef struct rocke_gemm_universal_spec
{
    const char* name;
    rocke_gemm_tile_spec_t tile;
    rocke_gemm_trait_spec_t trait;
    rocke_gemm_data_spec_t data;
    int wave_size; /* default 64 */
    int block_size; /* default 0 => derived at finalize() */
    bool batched;
    /* object.__setattr__(spec, "_fused_epilogue", ep) side-channel. NULL =>
     * matmul-only (Python getattr(spec, "_fused_epilogue", None) -> None). The
     * pointed-to object is a rocke_fused_epilogue_t (stock) or, when
     * _fused_epilogue_is_mde is set, a rocke_multi_d_epilogue_t whose first member
     * is that base (so the base methods view it unchanged; only apply_vec
     * dispatches to the _MultiDEpilogue override). Set via
     * rocke_gemm_universal_spec_set_fused_epilogue. */
    void* _fused_epilogue;
    bool _fused_epilogue_is_mde;
} rocke_gemm_universal_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set `name` and the required `tile` geometry. */
rocke_gemm_universal_spec_t rocke_gemm_universal_spec_default(void);

/* WarpTileBlockSizeMixin._init_block_size(): when block_size==0, derive it as
 * warp_m*warp_n*warp_k*wave_size. Idempotent. Call after filling the spec. */
void rocke_gemm_universal_spec_finalize(rocke_gemm_universal_spec_t* spec);

/* Python acc_agpr_regs(spec): per-lane accumulator registers of one wave. */
int rocke_gemm_acc_agpr_regs(const rocke_gemm_universal_spec_t* spec);

/* UniversalGemmSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 * Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t rocke_gemm_universal_kernel_name(const rocke_gemm_universal_spec_t* spec,
                                                char* out,
                                                size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message; returns false. On accept returns true and writes "ok". */
bool rocke_gemm_universal_is_valid_spec(const rocke_gemm_universal_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap);

/* build_universal_gemm(spec, arch). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd) builder `b`, exactly as the Python build does, and
 * returns the kernel (b->kernel) on success or NULL with b's sticky error set.
 * `arch` NULL => "gfx950". The kernel name is set by the builder init; this
 * routine does NOT re-init the builder (so the caller controls its lifetime).
 *
 * NOTE: like the Python, this expects the builder to have been created with the
 * spec's kernel_name(). Use rocke_build_universal_gemm_new() for the
 * init-from-spec convenience. */
rocke_kernel_def_t* rocke_build_universal_gemm(rocke_ir_builder_t* b,
                                               const rocke_gemm_universal_spec_t* spec,
                                               const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_universal_gemm_new(rocke_ir_builder_t* b,
                                                   const rocke_gemm_universal_spec_t* spec,
                                                   const char* arch);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
rocke_status_t rocke_gemm_universal_lower_to_llvm(const rocke_gemm_universal_spec_t* spec,
                                                  const char* arch,
                                                  rocke_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GEMM_UNIVERSAL_H */
