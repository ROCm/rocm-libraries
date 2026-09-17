/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_moe_fused_mega_fp8.h -- C99 port of the single-launch fused-MoE
 * MEGA-kernel (FP8 e4m3 block-scale) instance builder
 * rocke/instances/common/moe_fused_mega_fp8.py.
 *
 *   Python (moe_fused_mega_fp8.py)             C99 (this header)
 *   ----------------------------------------   -----------------------------------
 *   @dataclass(frozen=True)                    rocke_fused_mega_kernel_spec_fp8_t
 *     FusedMegaKernelSpecFp8
 *   spec.* @property / methods                 rocke_fused_mega_fp8_spec_*(...)
 *     gate_up_atom / down_atom                 rocke_fused_mega_fp8_spec_{gate_up,down}_atom
 *     mfmas_m / mfmas_n /                       rocke_fused_mega_fp8_spec_mfmas_{m,n,m_down,n_down}
 *       mfmas_m_down / mfmas_n_down
 *     kernel_name()                            rocke_fused_mega_fp8_spec_kernel_name(...)
 *   moe_fused_mega_fp8_grid(...)               rocke_moe_fused_mega_fp8_grid(...)
 *   moe_fused_mega_fp8_persistent_grid(...)    rocke_moe_fused_mega_fp8_persistent_grid(...)
 *   moe_fused_mega_fp8_signature(spec, persi)  rocke_moe_fused_mega_fp8_signature(...)
 *   build_moe_fused_mega_gemm_fp8(spec, arch,  rocke_build_moe_fused_mega_gemm_fp8(...)
 *     persistent)
 *   (+ convenience: build -> lower .ll)        rocke_moe_fused_mega_fp8_lower_to_llvm(...)
 *
 * WHAT THIS KERNEL IS. A SINGLE fused kernel computes, per (inter-slice,
 * sorted-m-block) threadgroup, the full per-expert MoE path with fp8 e4m3
 * operands and per-128-block f32 scales:
 *
 *   STAGE 1a  gate(GEMM0) + up(GEMM1): fp8 X . fp8 W -> f32 acc, per-128-group
 *             dequant fold (group-accumulator pattern, scale applied POST-MFMA);
 *   STAGE 1b  SiLU(gate_dq) * up_dq in f32 -> f32 LDS scratch + per-lane amax;
 *             butterfly amax reduce -> per-128-inter-block dynamic scale;
 *             packed quantize the f32 Hidden to fp8 into the persistent LDS
 *             Hidden_smem (+ stash per-block scales in HiddenScale_smem);
 *   STAGE 2   down(GEMM2): fp8 Hidden(LDS-A) . fp8 W_down -> f32 acc, per-128-
 *             group dequant by (hidden_dyn_scale * down_scale) -> weighted,
 *             token-validity-masked atomic-add into Y.
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python FusedMegaKernelSpecFp8 is a frozen
 * dataclass with defaults plus a stack of derived @property values + the
 * optimization-lever flags (gate_up_k / down_k / use_dtla / sched_cadence) whose
 * defaults all equal the final-best config, so a default-built kernel is golden-
 * digest byte-identical to the on-disk kernel. In C the caller fills a
 * rocke_fused_mega_kernel_spec_fp8_t; rocke_fused_mega_kernel_spec_fp8_default()
 * returns a struct with every field set to the Python dataclass default
 * (including __post_init__'s block_size = warp_m*warp_n*wave_size). The derived
 * @property values are NOT stored; they are recomputed by the accessor helpers
 * below (matching the Python properties exactly, including the divisibility math).
 *
 * The hand-authored body in build_moe_fused_mega_gemm_fp8 is a ~2130-line stack
 * of module-level emitters + nested closures (_b_base / _scale_base /
 * _select_item / _emit_body). Its private shared state + per-phase function
 * contract lives in the sibling PRIVATE header
 * rocke/instance_moe_fused_mega_fp8_internal.h; public callers only touch THIS
 * header.
 *
 * OPTIONAL OPTIMIZATION LEVERS (Python module-level env-derived flags). The
 * default-off inline-asm/DTLA-X paths (_USE_ASM_AGPR_MFMA, _USE_ASM_AGPR_MFMA_DOWN,
 * _USE_X_DTLA, _USE_MFMA_CLUSTER, _ASM_MFMA_HAZARD_NOP, _SCHED_CADENCE) are
 * carried on a rocke_fused_mega_fp8_levers_t bundle: pass NULL to get the Python
 * import-time defaults (golden-safe). The two NEW inline-asm helpers
 * (mfma_f8f6f4_agpr / mfma_f8f6f4_agpr_cluster) are already ported in
 * rocke/helper_helpers.asm.h and selected only by the asm/cluster lever paths.
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity is enforced inside build
 * (Python raises ValueError; here the builder sticky error is set and NULL is
 * returned); the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_MOE_FUSED_MEGA_FP8_H
#define ROCKE_INSTANCE_MOE_FUSED_MEGA_FP8_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/arena.h" /* rocke_arena_t (signature entry storage) */
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom_t */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t / SignatureBuilder */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * Module constants (Python module-level)
 * ============================================================ */

/* GROUP_K: 128-wide contraction scale group (= 4 fp8_16x16x32 atoms). */
#define ROCKE_MOE_FP8_GROUP_K 128
/* FP8_MAX: quant_max_abs("fp8e4m3") == 448.0. */
#define ROCKE_MOE_FP8_FP8_MAX 448.0
/* AMAX_FLOOR: pyisa dynamic-quant amax floor. */
#define ROCKE_MOE_FP8_AMAX_FLOOR 1e-6
/* DTLA_CHUNK: 16-byte direct-to-LDS payload cap (global_load_lds_dwordx4). */
#define ROCKE_MOE_FP8_DTLA_CHUNK 16
/* PERSISTENT_P_CAP: default persistent-grid block cap. */
#define ROCKE_MOE_FP8_PERSISTENT_P_CAP 512

/* ============================================================ *
 * Optimization-lever bundle (Python module-level _USE_* / env flags)
 * ============================================================ *
 *
 * Mirrors the import-time module globals that select OPTIONAL code paths. All
 * defaults equal the Python import-time defaults so a build with this bundle (or
 * NULL) is golden-safe / byte-identical. The Python derives these from os.environ
 * at import time; the C caller sets them explicitly (or passes NULL ->
 * rocke_fused_mega_fp8_levers_default()).
 *
 * sched_cadence note: this bundle's `sched_cadence` is the *env default*
 * (_SCHED_CADENCE, default "iglp1"); the SPEC's sched_cadence field (Python
 * level-9 flag) OVERRIDES it per-build when non-NULL. */
typedef struct rocke_fused_mega_fp8_levers
{
    bool use_asm_agpr_mfma; /* _USE_ASM_AGPR_MFMA      (default false) */
    bool use_asm_agpr_mfma_down; /* _USE_ASM_AGPR_MFMA_DOWN (default false;
                                    ROCKE_FP8_AGPR_MFMA_DOWN) */
    bool use_x_dtla; /* _USE_X_DTLA             (default false; ROCKE_FP8_X_DTLA) */
    bool use_mfma_cluster; /* _USE_MFMA_CLUSTER       (default false; ROCKE_FP8_MFMA_CLUSTER) */
    int asm_mfma_hazard_nop; /* _ASM_MFMA_HAZARD_NOP    (default 8;     ROCKE_FP8_MFMA_NOP) */
    const char* sched_cadence; /* _SCHED_CADENCE env def   (default "iglp1"; ROCKE_FP8_SCHED) */
} rocke_fused_mega_fp8_levers_t;

/* Returns the Python import-time defaults (all _USE_* false, hazard_nop 8,
 * sched_cadence "iglp1"). Passing NULL to build/signature is equivalent. */
rocke_fused_mega_fp8_levers_t rocke_fused_mega_fp8_levers_default(void);

/* ============================================================ *
 * FusedMegaKernelSpecFp8   (Python lines 298-403)
 * ============================================================ *
 *
 * One concrete fp8 fused-MoE mega-kernel configuration. Field order follows the
 * Python dataclass declaration order. `dtype` is fixed "fp8e4m3". sched_cadence
 * is an Optional[str] (Python level-9 flag): has_sched_cadence==false encodes
 * Python None (defer to the levers/env default); when true, sched_cadence pins
 * the per-loop cadence on the spec ("iglp1" | "none" | "sgb"). */
typedef struct rocke_fused_mega_kernel_spec_fp8
{
    const char* name; /* required (no default)                        */

    int tile_m; /* default 16  (sorted tokens per m-block)      */
    int tile_n_inter; /* default 256 (inter cols this TG owns)        */
    int tile_k_gu; /* default 32                                   */
    int warp_m; /* default 1                                    */
    int warp_n; /* default 4                                    */
    int warp_tile_m; /* default 16                                   */
    int warp_tile_n; /* default 16                                   */
    int warp_tile_k; /* default 32                                   */
    int tile_n_down; /* default 256                                  */
    int tile_k_down; /* default 64                                   */
    int wave_size; /* default 64                                   */
    int block_size; /* default 0 -> __post_init__ sets warp_m*warp_n*wave_size */
    const char* dtype; /* fixed "fp8e4m3"                              */

    /* -- optimization-lever flags (defaults = final best) -- */
    int gate_up_k; /* default 128 (L7 hero atom); 32 = legacy baseline */
    int down_k; /* default 128 (L7 hero atom); 32 = legacy baseline */
    bool use_dtla; /* default true (L8 direct-to-LDS); false = global->VGPR */

    bool has_sched_cadence; /* false => Python None (defer to env)  */
    const char* sched_cadence; /* "iglp1" | "none" | "sgb" when has_*  */
} rocke_fused_mega_kernel_spec_fp8_t;

/* Default-constructed spec (every Python dataclass default, block_size resolved
 * via __post_init__). `name` is set to NULL: the caller MUST set it before use
 * (Python `name` has no default). */
rocke_fused_mega_kernel_spec_fp8_t rocke_fused_mega_kernel_spec_fp8_default(void);

/* __post_init__: if block_size==0, set it to warp_m*warp_n*wave_size. Idempotent;
 * call after mutating warp_m/warp_n/wave_size with block_size left 0. */
void rocke_fused_mega_kernel_spec_fp8_post_init(rocke_fused_mega_kernel_spec_fp8_t* spec);

/* ---- derived geometry @property/method ports ---- */

/* gate_up_atom(): gate_up_k==32 -> fp8_16x16x32, else fp8_16x16x128 hero atom.
 * down_atom():    down_k==32    -> fp8_16x16x32, else fp8_16x16x128 hero atom.
 * Return the catalog MfmaAtom (borrowed, never freed). */
const rocke_mfma_atom_t*
    rocke_fused_mega_fp8_spec_gate_up_atom(const rocke_fused_mega_kernel_spec_fp8_t* spec);
const rocke_mfma_atom_t*
    rocke_fused_mega_fp8_spec_down_atom(const rocke_fused_mega_kernel_spec_fp8_t* spec);

/* mfmas_m   = (tile_m       // warp_m) // warp_tile_m
 * mfmas_n   = (tile_n_inter // warp_n) // warp_tile_n
 * mfmas_m_down = (tile_m     // warp_m) // warp_tile_m
 * mfmas_n_down = (tile_n_down // warp_n) // warp_tile_n   */
int rocke_fused_mega_fp8_spec_mfmas_m(const rocke_fused_mega_kernel_spec_fp8_t* spec);
int rocke_fused_mega_fp8_spec_mfmas_n(const rocke_fused_mega_kernel_spec_fp8_t* spec);
int rocke_fused_mega_fp8_spec_mfmas_m_down(const rocke_fused_mega_kernel_spec_fp8_t* spec);
int rocke_fused_mega_fp8_spec_mfmas_n_down(const rocke_fused_mega_kernel_spec_fp8_t* spec);

/* kernel_name() -> "{name}_moe_fused_mega_fp8_m{tile_m}n{tile_n_inter}k{tile_k_gu}".
 * Writes the NUL-terminated string into out (capacity out_cap). Returns ROCKE_OK
 * or ROCKE_ERR_VALUE on NULL args / too-small buffer. */
rocke_status_t rocke_fused_mega_fp8_spec_kernel_name(const rocke_fused_mega_kernel_spec_fp8_t* spec,
                                                     char* out,
                                                     size_t out_cap);

/* ============================================================ *
 * Grid + signature   (Python lines 411-496)
 * ============================================================ */

/* moe_fused_mega_fp8_grid(num_m_blocks, inter, spec) ->
 *   grid = (ceil(inter / tile_n_inter), num_m_blocks, 1). Writes the 3 dims into
 * out_grid[3]. Returns ROCKE_OK or ROCKE_ERR_VALUE on NULL. */
rocke_status_t rocke_moe_fused_mega_fp8_grid(int num_m_blocks,
                                             int inter,
                                             const rocke_fused_mega_kernel_spec_fp8_t* spec,
                                             int out_grid[3]);

/* moe_fused_mega_fp8_persistent_grid(num_m_blocks, inter, spec, p_cap) ->
 *   grid_x     = ceil(inter / tile_n_inter)
 *   total_work = grid_x * num_m_blocks
 *   P          = min(total_work, p_cap)  (1 if total_work==0)
 *   launch grid = (P, 1, 1).
 * Pass p_cap<=0 to use ROCKE_MOE_FP8_PERSISTENT_P_CAP. Writes out_grid[3] = (P,1,1)
 * and the three scalars through the out params (any may be NULL). */
rocke_status_t
    rocke_moe_fused_mega_fp8_persistent_grid(int num_m_blocks,
                                             int inter,
                                             const rocke_fused_mega_kernel_spec_fp8_t* spec,
                                             int p_cap,
                                             int out_grid[3],
                                             int* out_grid_x,
                                             int* out_total_work,
                                             int* out_P);

/* moe_fused_mega_fp8_signature(spec, persistent): build the kernel ABI signature
 * (the SignatureBuilder.build() result). `persistent` appends the
 * (grid_x, total_work, P) trailing scalars. The entries are allocated in the
 * caller-supplied `arena` (its lifetime owns the strings + array). On ROCKE_OK
 * *out_items / *out_count receive the arena-owned signature array. */
rocke_status_t rocke_moe_fused_mega_fp8_signature(const rocke_fused_mega_kernel_spec_fp8_t* spec,
                                                  bool persistent,
                                                  rocke_arena_t* arena,
                                                  const rocke_sig_entry_t** out_items,
                                                  size_t* out_count);

/* ============================================================ *
 * Build entry + lower convenience   (Python lines 1554-2129)
 * ============================================================ */

/* build_moe_fused_mega_gemm_fp8(spec, arch="gfx950", persistent=False).
 *
 * Builds the full fused fp8 MoE mega-kernel on the supplied builder `b` (created
 * with the spec's kernel_name()), populating the private build-context and
 * driving the phase functions in Python execution order (see
 * instance_moe_fused_mega_fp8_internal.h). Returns the kernel (b->kernel) on
 * success or NULL with b's sticky error set.
 *
 *   arch       NULL => "gfx950".
 *   persistent => emit the grid-stride persistent ABI variant (3 extra params,
 *                 work-id loop, inter-item LDS barrier); false => byte-identical
 *                 one-shot path.
 *   levers     NULL => rocke_fused_mega_fp8_levers_default() (golden-safe). */
rocke_kernel_def_t*
    rocke_build_moe_fused_mega_gemm_fp8(rocke_ir_builder_t* b,
                                        const rocke_fused_mega_kernel_spec_fp8_t* spec,
                                        const char* arch,
                                        bool persistent,
                                        const rocke_fused_mega_fp8_levers_t* levers);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns `b`
 * and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t*
    rocke_build_moe_fused_mega_gemm_fp8_new(rocke_ir_builder_t* b,
                                            const rocke_fused_mega_kernel_spec_fp8_t* spec,
                                            const char* arch,
                                            bool persistent,
                                            const rocke_fused_mega_fp8_levers_t* levers);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". `levers` NULL => defaults. On ROCKE_OK *out_ll receives
 * a malloc'd NUL-terminated string the caller frees with free(); on failure it is
 * left NULL and (if err!=NULL, capacity err_cap) a diagnostic is written.
 * Internally owns and frees its IRBuilder. */
rocke_status_t
    rocke_moe_fused_mega_fp8_lower_to_llvm(const rocke_fused_mega_kernel_spec_fp8_t* spec,
                                           const char* arch,
                                           bool persistent,
                                           const rocke_fused_mega_fp8_levers_t* levers,
                                           rocke_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_MOE_FUSED_MEGA_FP8_H */
