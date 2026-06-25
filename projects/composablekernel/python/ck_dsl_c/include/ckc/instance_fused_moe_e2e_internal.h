/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_fused_moe_e2e_internal.h -- PRIVATE shared state + phase-function
 * contract for the C99 port of the end-to-end fused-MoE forward orchestrator
 * (ck_dsl/instances/common/fused_moe_e2e.py, 2678 LOC).
 *
 * WHY THIS HEADER EXISTS.
 *   FusedMoeForward is a host runtime driver, not a single closure stack like
 *   the GEMM/conv bodies -- but the structure that makes a C port tractable is
 *   identical: the long methods (__init__, _forward_static, _forward_dynamic,
 *   _dispatch_grouped_gemm) share a LARGE set of instance state + per-call
 *   locals (the resolved arch + adjusted tile, ~20 lazily-compiled launcher
 *   slots, the workspace pool, cached/packed weights, the resolved workspace
 *   tensor handles, the per-call sizing scalars slot_size/total_padded/
 *   max_padded_m, and the path-selection booleans). In Python those are `self.`
 *   attributes + method locals captured implicitly by the helper methods.
 *
 *   In C there is no implicit capture. The faithful port turns each Python
 *   method / launcher-ensure / host helper into a free function that takes a
 *   POINTER to one shared context struct, ckc_fmoe_build_ctx_t, which holds
 *   EXACTLY the state these methods share. The driver
 *   (ckc_build_fused_moe_forward in ckc/instance_fused_moe_e2e.h) populates the
 *   ctx in __init__ order, binds forward_fn, and the forward entries call the
 *   phase functions in the Python pipeline order.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing agent binds
 *   to. It is DESIGNED TO BE COMPLETE: every instance attribute (self._*) and
 *   every cross-phase local the Python body shares is a field here. A body agent
 *   implementing a phase .c file MUST be able to read/write only ctx fields and
 *   call the prototypes below WITHOUT editing this header. If a phase genuinely
 *   needs a value not present, that is a design bug in this header to be fixed
 *   once, deliberately, not patched per-phase.
 *
 *   Naming: ctx fields mirror the Python attribute / local names 1:1 with the
 *   leading underscore dropped (Python `self._batched_gemm_launcher` ->
 *   `ctx->batched_gemm_launcher`; local `max_padded_m` -> `ctx->max_padded_m`).
 *   Phase functions mirror the Python method names with a `ckc_fmoe_` prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by
 * the instance_fused_moe_e2e_*.c translation units. Public callers use
 * ckc/instance_fused_moe_e2e.h.
 */
#ifndef CKC_INSTANCE_FUSED_MOE_E2E_INTERNAL_H
#define CKC_INSTANCE_FUSED_MOE_E2E_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ckc/ir.h"
#include "ckc/instance_fused_moe_e2e.h"

/* The spec converters + tile policy (already ported). */
#include "ckc/helper_ck_dsl.helpers.fused_moe_e2e_spec.h"

/* Sub-kernel spec / build / grid / signature surfaces the orchestrator drives.
 * Every launcher slot below caches an artifact built by one of these. */
#include "ckc/instance_topk_softmax.h"                         /* ckc_topk_softmax_spec_t */
#include "ckc/instance_batched_gemm.h"                         /* ckc_batched_gemm_spec_t */
#include "ckc/instance_grouped_gemm.h"                         /* ckc_grouped_gemm_* */
#include "ckc/instance_gemm_universal.h"                       /* tile / trait spec */
#include "ckc/helper_ck_dsl.instances.common.moe_gemm_fused.h" /* gate_up/down/interleaved specs */

#ifdef __cplusplus
extern "C" {
#endif

/* Number of workspace specs _workspace_specs returns (constant: 15). */
#define CKC_FMOE_NUM_WORKSPACE_SPECS 15

/* Upper bound on experts for the per-expert host loops (counts/offsets walks,
 * BlockExpertIds, per-expert padded copies). Datacenter shapes use E<=32; 1024
 * is generous headroom. */
#define CKC_FMOE_MAX_EXPERTS 1024

/* ============================================================ *
 * Opaque runtime peers (host-side; not IR types)
 * ============================================================ *
 *
 * These mirror Python runtime objects with no C IR analogue. They are forward-
 * declared as opaque so the ctx can hold them by pointer without pulling a HIP
 * runtime into the codegen library. A future HIP backend defines them.
 *
 *   ckc_kernel_launcher_t  : runtime.launcher.KernelLauncher (cached HSACO +
 *                            module + function + signature).
 *   ckc_grouped_gemm_launcher_t : instances GroupedGemmLauncher (the deprecated
 *                            per-expert E-launch loop launcher).
 *   ckc_moe_sorting_launcher_t  : MoeSortingLauncher (3-phase sort).
 *   ckc_fused_moe_launcher_t    : FusedMoeLauncher (gather/silu_mul/topk_reduce).
 *   ckc_workspace_pool_t   : runtime.launcher.WorkspacePool.
 *   ckc_tensor_t           : an opaque torch-tensor-like device buffer handle
 *                            (data_ptr + shape + dtype) the host helpers pass
 *                            around; in a HIP backend this wraps a device ptr.
 */
typedef struct ckc_kernel_launcher ckc_kernel_launcher_t;
typedef struct ckc_grouped_gemm_launcher ckc_grouped_gemm_launcher_t;
typedef struct ckc_moe_sorting_launcher ckc_moe_sorting_launcher_t;
typedef struct ckc_fused_moe_launcher ckc_fused_moe_launcher_t;
typedef struct ckc_workspace_pool ckc_workspace_pool_t;
typedef struct ckc_tensor ckc_tensor_t;

/* ------------------------------------------------------------ moe_gemm_launcher_cache *
 *
 * Python `self._moe_gemm_launcher_cache: dict` keyed on
 *   ("batched", preshuffle_b, active_tile_skip)
 *   ("interleaved_gate_up_silu", preshuffle_b, active_tile_skip)
 *   ("grouped_gate_up_silu",)
 *   ("grouped_down_reduce",)
 * Each value is a cached KernelLauncher for that unique trait combo. In C the
 * dict is a small fixed table: `kind` is one of the enum below, plus the two
 * trait bools (ignored for the grouped kinds). Linear scan on lookup (N<=10). */
typedef enum ckc_fmoe_gemm_kind
{
    CKC_FMOE_GEMM_BATCHED = 0,         /* "batched"                  */
    CKC_FMOE_GEMM_INTERLEAVED_GATE_UP, /* "interleaved_gate_up_silu" */
    CKC_FMOE_GEMM_GROUPED_GATE_UP,     /* "grouped_gate_up_silu"     */
    CKC_FMOE_GEMM_GROUPED_DOWN_REDUCE  /* "grouped_down_reduce"      */
} ckc_fmoe_gemm_kind_t;

typedef struct ckc_fmoe_gemm_cache_entry
{
    bool used;
    ckc_fmoe_gemm_kind_t kind;
    bool preshuffle_b;               /* part of key for batched / interleaved kinds */
    bool active_tile_skip;           /* part of key for batched / interleaved kinds */
    ckc_kernel_launcher_t* launcher; /* cached value */
} ckc_fmoe_gemm_cache_entry_t;

#define CKC_FMOE_GEMM_CACHE_CAP 16

/* ============================================================ *
 * ckc_fmoe_workspace_handles_t
 * ============================================================ *
 *
 * The resolved workspace tensor handles for one forward call. _forward_dynamic
 * / _forward_static call self._pool.prepare(self._workspace_specs(device)) and
 * then unpack the dict by name; this struct names every entry both paths read,
 * plus the per-call padded/packed scratch buffers each path allocates via
 * self._pool.get_spec(WorkspaceSpec(...)). NULL == not allocated on this path. */
typedef struct ckc_fmoe_workspace_handles
{
    /* ---- the 15-entry _workspace_specs table (both paths) ---- */
    ckc_tensor_t* TopkIds;        /* (T, K) i32             */
    ckc_tensor_t* TopkWeights;    /* (T, K) f32             */
    ckc_tensor_t* Hist;           /* (E,)  i32              */
    ckc_tensor_t* Counter;        /* (E,)  i32              */
    ckc_tensor_t* Offsets;        /* (E,)  i32              */
    ckc_tensor_t* Counts;         /* (E,)  i32              */
    ckc_tensor_t* SortedTokenIds; /* (T*K,) i32             */
    ckc_tensor_t* SortedTopkIds;  /* (T*K,) i32             */
    ckc_tensor_t* SortedWeights;  /* (T*K,) f32             */
    ckc_tensor_t* GroupedInput;   /* (T*K, H) act           */
    ckc_tensor_t* GateOut;        /* (T*K, I) act           */
    ckc_tensor_t* UpOut;          /* (T*K, I) act           */
    ckc_tensor_t* Hidden;         /* (T*K, I) act           */
    ckc_tensor_t* DownOut;        /* (T*K, H) act           */
    ckc_tensor_t* Y_f32;          /* (T, H) f32 accumulator */

    /* ---- dynamic-path uniform-padded scratch (get_spec) ---- */
    ckc_tensor_t* GroupedInputPaddedUniform;   /* (E*MAX, H) act  */
    ckc_tensor_t* GateOutPaddedUniform;        /* (E*MAX, I) act  */
    ckc_tensor_t* UpOutPaddedUniform;          /* (E*MAX, I) act  */
    ckc_tensor_t* HiddenPaddedUniform;         /* (E*MAX, I) act  */
    ckc_tensor_t* DownOutPaddedUniform;        /* (E*MAX, H) act  */
    ckc_tensor_t* SortedTokenIdsPaddedUniform; /* (E*MAX,) i32  */
    ckc_tensor_t* SortedWeightsPaddedUniform;  /* (E*MAX,) f32  */

    /* ---- grouped (de-padded) dynamic-path scratch (_dispatch_grouped_gemm) ---- */
    ckc_tensor_t* GroupedInputPacked;   /* (total_packed, H) act */
    ckc_tensor_t* HiddenPacked;         /* (total_packed, I) act */
    ckc_tensor_t* SortedTokenIdsPacked; /* (total_packed,) i32   */
    ckc_tensor_t* SortedWeightsPacked;  /* (total_packed,) f32   */
    ckc_tensor_t* BlockExpertIds;       /* (num_m_blocks,) i32   */

    /* ---- static-path padded scratch (get_spec) ---- */
    ckc_tensor_t* StaticSortedTokenIdsPadded; /* (E*S,) i32      */
    ckc_tensor_t* StaticSortedTopkIdsPadded;  /* (E*S,) i32      */
    ckc_tensor_t* StaticSortedWeightsPadded;  /* (E*S,) f32      */
    ckc_tensor_t* StaticGroupedInputPadded;   /* (E*S, H) act    */
    ckc_tensor_t* StaticGateUpPacked;         /* (E*S, 2I) act   */
    ckc_tensor_t* StaticHiddenPadded;         /* (E*S, I) act    */
    ckc_tensor_t* StaticDownOutPadded;        /* (E*S, H) act    */
} ckc_fmoe_workspace_handles_t;

/* ===================================================================== *
 *  ckc_fmoe_build_ctx_t
 *
 *  The single shared state object. Holds every FusedMoeForward instance
 *  attribute (self._*) plus the cross-phase per-call locals the two forward
 *  paths share. Field order follows __init__'s computation order, then the
 *  per-call working set; the populate routine reads top-to-bottom against the
 *  source (lines 692-831 for the attributes, 1629-2678 for the per-call set).
 * ===================================================================== */
typedef struct ckc_fmoe_build_ctx
{
    /* ---- resolved environment (FusedMoeForward.__init__, lines 692-749) ---- */
    ckc_fmoe_forward_spec_t spec; /* self.spec (AFTER the tile-swap policy) */
    const char* arch;             /* self.arch = _resolve_launch_arch(...)  */
    bool is_gfx942;               /* self.arch == "gfx942"                  */
    bool is_bf16;                 /* spec.dtype == "bf16"                   */

    /* ---- sub-kernel launchers cached as instance attrs (lines 750-801) ---- *
     * Each is lazily compiled by its _ensure_* phase; NULL == not yet built. */
    ckc_moe_sorting_launcher_t* sort_launcher;    /* self._sort_launcher       */
    ckc_fused_moe_launcher_t* fused_moe_launcher; /* self._fused_moe_launcher  */
    ckc_kernel_launcher_t* topk_launcher;         /* self._topk_launcher       */
    ckc_grouped_gemm_launcher_t* gemm_launcher;   /* self._gemm_launcher       */
    ckc_kernel_launcher_t* batched_gemm_launcher; /* self._batched_gemm_launcher */
    ckc_kernel_launcher_t* batched_gemm_preshuffle_b_launcher;
    /* self._batched_gemm_preshuffle_b_launcher */
    ckc_kernel_launcher_t* interleaved_gate_up_silu_preshuffle_launcher;
    /* self._interleaved_gate_up_silu_preshuffle_launcher */
    ckc_kernel_launcher_t* gate_up_silu_launcher; /* self._gate_up_silu_launcher */
    ckc_kernel_launcher_t* interleaved_gate_up_silu_launcher;
    /* self._interleaved_gate_up_silu_launcher */
    ckc_kernel_launcher_t* down_reduce_launcher; /* self._down_reduce_launcher */
    ckc_kernel_launcher_t* static_scatter_gather_launcher;
    /* self._static_scatter_gather_launcher */
    ckc_kernel_launcher_t* silu_mul_packed_launcher; /* self._silu_mul_packed_launcher */

    /* parameterized active-tile / preshuffle launcher cache (the Python dict). */
    ckc_fmoe_gemm_cache_entry_t moe_gemm_launcher_cache[CKC_FMOE_GEMM_CACHE_CAP];
    int moe_gemm_launcher_cache_len;

    /* ---- workspace pool (lines 754) ---- */
    ckc_workspace_pool_t* pool; /* self._pool = WorkspacePool()           */

    /* ---- cached / preshuffled weights (lines 759-800) ---- *
     * Each cached tensor + its data_ptr key tuple. The key arrays mirror the
     * Python Optional[Tuple[...]] (presence via the *_key_valid flag). */
    ckc_tensor_t* w_down_preshuffled; /* self._w_down_preshuffled */
    bool w_down_preshuffled_key_valid;
    uint64_t w_down_preshuffled_key[3]; /* (data_ptr, block_n, block_k) */

    ckc_tensor_t* gu_concat_preshuffled; /* self._gu_concat_preshuffled */
    bool gu_concat_preshuffled_key_valid;
    uint64_t gu_concat_preshuffled_key[3]; /* (Wg_ptr, Wu_ptr, block_n) */

    ckc_tensor_t* gu_interleaved_preshuffled; /* self._gu_interleaved_preshuffled */
    bool gu_interleaved_preshuffled_key_valid;
    uint64_t gu_interleaved_preshuffled_key[3]; /* (Wg_ptr, Wu_ptr, block_n) */

    ckc_tensor_t* gu_concat; /* self._gu_concat */
    bool gu_concat_key_valid;
    uint64_t gu_concat_key[2]; /* (Wg_ptr, Wu_ptr) */

    ckc_tensor_t* gu_interleaved; /* self._gu_interleaved */
    bool gu_interleaved_key_valid;
    uint64_t gu_interleaved_key[2]; /* (Wg_ptr, Wu_ptr) */

    /* ---- HIP graph capture (lines 780-781) ---- */
    void* captured_graph;        /* self._captured_graph (opaque CUDAGraph) */
    void* captured_graph_stream; /* self._captured_graph_stream             */

    /* ---- static-offset mode state (lines 810-830) ---- */
    ckc_tensor_t* static_offsets; /* self._static_offsets (E,) i32 arange*S, lazy */
    bool use_static_offsets;      /* T*K*E <= 512                             */
    int static_slot_size;         /* ceil(T*K / tile_m) * tile_m (>= tile_m)  */

    /* ===== per-call working set (shared by the forward phase functions) ===== *
     * Populated freshly at the head of each _forward_static / _forward_dynamic
     * call; not persistent instance state. */

    /* resolved call inputs (device ptrs, mirror the forward() kwargs) */
    uint64_t routing_logits;
    uint64_t X;
    uint64_t W_gate;
    uint64_t W_up;
    uint64_t W_down;
    uint64_t Y;
    uint64_t stream;
    ckc_tensor_t* device; /* X.device (torch device handle)          */

    /* resolved workspace handles for this call (self._pool.prepare + get_spec). */
    ckc_fmoe_workspace_handles_t ws;

    /* per-call sizing scalars (the locals both paths compute). */
    int tile_m;          /* spec.gemm_tile.tile_m                    */
    int tile_n;          /* spec.gemm_tile.tile_n                    */
    int slot_size;       /* static path: == static_slot_size        */
    int total_padded;    /* static: E*slot_size; dyn-uniform: E*MAX  */
    int max_padded_m;    /* dynamic uniform path                     */
    int num_m_blocks;    /* grouped path: sum ceil(count[e]/tile_m)  */
    int total_packed;    /* grouped path: num_m_blocks * tile_m      */
    int gate_up_n;       /* packed gate-up N = 2 * intermediate      */
    int gemm_block_size; /* to_batched_gemm_spec().block_size        */

    /* host-resident per-expert count/offset arrays (D->H copy of Counts/Offsets
     * in the dynamic path). Length == spec.experts. */
    int32_t counts_cpu[CKC_FMOE_MAX_EXPERTS];
    int32_t offsets_cpu[CKC_FMOE_MAX_EXPERTS];
    int blocks_per_expert[CKC_FMOE_MAX_EXPERTS];        /* grouped path */
    int padded_counts_per_expert[CKC_FMOE_MAX_EXPERTS]; /* uniform path */

    /* ---- static-path resolved per-call path-selection booleans (lines 2303-2357) */
    bool use_experimental_fused;       /* spec.use_experimental_fused_gate_up_silu */
    bool use_experimental_interleaved; /* interleaved && !fused                     */
    bool use_experimental_down_reduce; /* spec.use_experimental_fused_down_reduce   */
    bool use_experimental_static_sg;   /* spec.use_experimental_static_scatter_gather */

    /* ---- static-path resolved launcher / weight selections for this call ---- *
     * These mirror the locals _forward_static binds after the if/elif chains;
     * holding them in ctx lets the gate / down / route phase functions read the
     * exact selection the prologue made. NULL when the path does not use it. */
    ckc_kernel_launcher_t* sel_gate_up_silu_launcher;
    ckc_kernel_launcher_t* sel_interleaved_gate_up_silu_launcher;
    ckc_kernel_launcher_t* sel_down_reduce_launcher;
    ckc_kernel_launcher_t* sel_silu_mul_packed_launcher;
    ckc_kernel_launcher_t* sel_static_scatter_gather_launcher;
    ckc_moe_sorting_launcher_t* sel_sort_launchers; /* None when static_sg */
    ckc_fused_moe_launcher_t* sel_fmoe_launchers;
    ckc_tensor_t* sel_gu_concat;      /* gu_concat or None for this call      */
    ckc_tensor_t* sel_gu_interleaved; /* gu_interleaved or None for this call */

    /* dynamic-path resolved gate/down launcher + B-tensor selections. */
    ckc_kernel_launcher_t* sel_gate_up_dyn_launcher;
    ckc_kernel_launcher_t* sel_down_b_launcher;
    ckc_tensor_t* sel_down_b_tensor;
    ckc_kernel_launcher_t* sel_gate_up_b_launcher;
    ckc_tensor_t* sel_gate_up_b_tensor;
} ckc_fmoe_build_ctx_t;

/* ===================================================================== *
 *  PHASE FUNCTIONS -- one per Python method / closure / module helper.
 *
 *  Each reads/writes only ctx (and the resources it carries). The launcher-
 *  ensure functions mirror FusedMoeForward._ensure_* (lazy compile + cache);
 *  the value-producing host helpers reproduce the Python arithmetic exactly;
 *  the forward driver phases reproduce the host-side dispatch decisions + the
 *  launch-chain ordering (the actual HIP launch is TODO(port)).
 * ===================================================================== */

/* ----- module-level pure helpers (file-scope Python functions) ----- *
 * Faithfully ported in the spec helper header; re-declared here as ctx-free for
 * the body TUs' convenience. (See helper_ck_dsl.helpers.fused_moe_e2e_spec.h.)
 *   _resolve_launch_arch / _gemm_dtype_to_universal / _ensure_2byte_dtype
 *   + the seven tile factories. */

/* ----- ctx lifecycle ----- */

/* FusedMoeForward.__init__ (lines 692-831): resolve arch, run the shape-aware
 * tile-swap policy, compute the static gate + slot size, zero all launcher /
 * cache / weight slots. Returns CKC_OK or CKC_ERR_VALUE on a dtype / tile
 * ValueError. */
ckc_status_t ckc_fmoe_build_ctx_init(ckc_fmoe_build_ctx_t* ctx,
                                     const ckc_fmoe_forward_spec_t* spec,
                                     const char* arch);

/* Release pool / launchers / cached weights. NULL-safe. */
void ckc_fmoe_build_ctx_destroy(ckc_fmoe_build_ctx_t* ctx);

/* ----- lazy launcher ensures (FusedMoeForward._ensure_* / _grouped_* / _moe_*)
 * Each compiles its sub-kernel on first call and caches the launcher on ctx,
 * returning the cached launcher (or NULL with the builder error set). ----- */
ckc_kernel_launcher_t* ckc_fmoe_ensure_topk_launcher(ckc_fmoe_build_ctx_t* ctx);
ckc_grouped_gemm_launcher_t* ckc_fmoe_ensure_gemm_launcher(ckc_fmoe_build_ctx_t* ctx);
ckc_kernel_launcher_t* ckc_fmoe_ensure_batched_gemm_launcher(ckc_fmoe_build_ctx_t* ctx);
ckc_kernel_launcher_t*
ckc_fmoe_ensure_batched_gemm_preshuffle_b_launcher(ckc_fmoe_build_ctx_t* ctx);
ckc_kernel_launcher_t* ckc_fmoe_ensure_silu_mul_packed_launcher(ckc_fmoe_build_ctx_t* ctx);
ckc_kernel_launcher_t* ckc_fmoe_ensure_gate_up_silu_launcher(ckc_fmoe_build_ctx_t* ctx);
ckc_kernel_launcher_t* ckc_fmoe_ensure_interleaved_gate_up_silu_launcher(ckc_fmoe_build_ctx_t* ctx);
ckc_kernel_launcher_t*
ckc_fmoe_ensure_interleaved_gate_up_silu_preshuffle_launcher(ckc_fmoe_build_ctx_t* ctx);
ckc_kernel_launcher_t* ckc_fmoe_ensure_down_reduce_launcher(ckc_fmoe_build_ctx_t* ctx);
ckc_kernel_launcher_t* ckc_fmoe_ensure_static_scatter_gather_launcher(ckc_fmoe_build_ctx_t* ctx);

/* Parameterized cache lookups (the self._moe_gemm_launcher_cache dict). */
ckc_kernel_launcher_t* ckc_fmoe_moe_batched_gemm_launcher(ckc_fmoe_build_ctx_t* ctx,
                                                          bool preshuffle_b,
                                                          bool active_tile_skip);
ckc_kernel_launcher_t* ckc_fmoe_moe_interleaved_gate_up_silu_launcher(ckc_fmoe_build_ctx_t* ctx,
                                                                      bool preshuffle_b,
                                                                      bool active_tile_skip);
ckc_kernel_launcher_t* ckc_fmoe_grouped_gate_up_silu_launcher(ckc_fmoe_build_ctx_t* ctx);
ckc_kernel_launcher_t* ckc_fmoe_grouped_down_reduce_launcher(ckc_fmoe_build_ctx_t* ctx);

/* _grouped_gate_up_spec / _grouped_down_spec: the per-call FusedGateUpSilu /
 * FusedDownReduce specs used by the grouped path. Returned by value. */
ckc_moe_gate_up_silu_gemm_spec_t ckc_fmoe_grouped_gate_up_spec(const ckc_fmoe_build_ctx_t* ctx);
ckc_moe_down_reduce_gemm_spec_t ckc_fmoe_grouped_down_spec(const ckc_fmoe_build_ctx_t* ctx);

/* _ensure_compiled (lines 1334-1353): compile every component up front. */
ckc_status_t ckc_fmoe_ensure_compiled(ckc_fmoe_build_ctx_t* ctx);

/* ----- value-producing host helpers (faithfully ported) ----- */

/* _workspace_specs (lines 1359-1382): fill the 15-entry table in declaration
 * order. Writes min(cap, 15) entries + sets *n_written. The dtype kinds /
 * element sizes mirror the orchestrator helper header. */
typedef enum ckc_fmoe_ws_dtype
{
    CKC_FMOE_WS_I32 = 0,
    CKC_FMOE_WS_F32,
    CKC_FMOE_WS_ACT
} ckc_fmoe_ws_dtype_t;

typedef struct ckc_fmoe_ws_spec
{
    const char* name; /* static string */
    int shape[2];
    int rank;
    ckc_fmoe_ws_dtype_t dtype;
    int elem_bytes;
} ckc_fmoe_ws_spec_t;

ckc_status_t ckc_fmoe_workspace_specs(const ckc_fmoe_build_ctx_t* ctx,
                                      ckc_fmoe_ws_spec_t* out,
                                      size_t cap,
                                      size_t* n_written);

/* _host_preshuffle_b (lines 1053-1075): (E,N,K) -> (E,k_tiles,n_tiles,block_n,
 * block_k) shape contract + the N%block_n / K%block_k ValueError precondition.
 * out_shape needs room for 5 ints. CKC_ERR_VALUE on the precondition failure. */
ckc_status_t
ckc_fmoe_host_preshuffle_b(int E, int N, int K, int block_n, int block_k, int out_shape[5]);

/* _ensure_w_down_preshuffled / _ensure_gu_concat(_preshuffled) /
 * _ensure_gu_interleaved(_preshuffled) / _ensure_gu_concat: the data_ptr-keyed
 * lazy weight-packing caches. Each returns the cached/built tensor (the actual
 * torch cat/stack/permute is TODO(port); the cache key arithmetic is ported). */
ckc_tensor_t* ckc_fmoe_ensure_w_down_preshuffled(ckc_fmoe_build_ctx_t* ctx, ckc_tensor_t* W_down);
ckc_tensor_t* ckc_fmoe_ensure_gu_concat_preshuffled(ckc_fmoe_build_ctx_t* ctx,
                                                    ckc_tensor_t* W_gate,
                                                    ckc_tensor_t* W_up);
ckc_tensor_t* ckc_fmoe_ensure_gu_interleaved_preshuffled(ckc_fmoe_build_ctx_t* ctx,
                                                         ckc_tensor_t* W_gate,
                                                         ckc_tensor_t* W_up);
ckc_tensor_t*
ckc_fmoe_ensure_gu_concat(ckc_fmoe_build_ctx_t* ctx, ckc_tensor_t* W_gate, ckc_tensor_t* W_up);
ckc_tensor_t*
ckc_fmoe_ensure_gu_interleaved(ckc_fmoe_build_ctx_t* ctx, ckc_tensor_t* W_gate, ckc_tensor_t* W_up);

/* _build_per_expert_problems (lines 1384-1420): walk counts/offsets, emit one
 * GroupedGemmProblem per active expert with the exact pointer arithmetic
 *   A_ptr = a_base + offset * a_inner_dim * elem_bytes
 *   B_ptr = b_base[e]
 *   C_ptr = c_base + offset * c_inner_dim * elem_bytes
 *   M=count, N=c_inner_dim, K=b_inner_dim.
 * Fills out (capacity cap), sets *n_written to the active-expert count. */
ckc_status_t ckc_fmoe_build_per_expert_problems(ckc_fmoe_build_ctx_t* ctx,
                                                const int32_t* counts_cpu,
                                                const int32_t* offsets_cpu,
                                                uint64_t a_base,
                                                const uint64_t* b_base,
                                                uint64_t c_base,
                                                int a_inner_dim,
                                                int b_inner_dim,
                                                int c_inner_dim,
                                                int elem_bytes,
                                                ckc_grouped_gemm_problem_t* out,
                                                size_t cap,
                                                size_t* n_written);

/* ----- forward driver phases (the launch chain; HIP launch is TODO(port)) ----- */

/* forward (lines 1597-1627): dispatch to static or dynamic on
 * ctx->use_static_offsets. This is the function bound to KernelDef.forward_fn. */
ckc_status_t ckc_fmoe_forward_dispatch(ckc_fmoe_build_ctx_t* ctx,
                                       uint64_t routing_logits,
                                       uint64_t X,
                                       uint64_t W_gate,
                                       uint64_t W_up,
                                       uint64_t W_down,
                                       uint64_t Y,
                                       uint64_t stream);

/* _forward_dynamic (lines 1629-2058): the host-roundtrip path. Prologue
 * (workspaces + pre-zero), stage 1+2 router+sort, stage 2.5 gather, D->H copy of
 * Counts/Offsets, grouped vs uniform-padded dispatch, stages 3-6 GEMM chain,
 * stage 7 cast. Reads/writes ctx; returns CKC_ERR_NOTIMPL for the stubbed HIP
 * launch body. */
ckc_status_t ckc_fmoe_forward_dynamic(ckc_fmoe_build_ctx_t* ctx);

/* _dispatch_grouped_gemm (lines 1426-1582): the de-padded grouped path. Builds
 * blocks_per_expert / num_m_blocks / total_packed, packs the dense tile-aligned
 * layout + BlockExpertIds, dispatches the grouped gate-up + down-reduce kernels.
 * Returns true (handled) / false (fall back to uniform-padded) into *handled;
 * the host-side packing is ported, the HIP launch is TODO(port). */
ckc_status_t ckc_fmoe_dispatch_grouped_gemm(ckc_fmoe_build_ctx_t* ctx, bool* handled);

/* _forward_static (lines 2175-2678): the no-roundtrip static-offset path.
 * Prologue (workspaces + one-shot static_offsets + per-call resets), resolve the
 * path-selection booleans + launcher/weight selections, build the router /
 * route-stage / gate-stage / down-stage callables, single launch_kernel chain,
 * stage-7 cast. Returns CKC_ERR_NOTIMPL for the stubbed HIP launch body. */
ckc_status_t ckc_fmoe_forward_static(ckc_fmoe_build_ctx_t* ctx);

/* capture_graph / replay_graph (lines 2064-2169): HIP-graph capture of the
 * static-mode forward (opt-in benchmark path). TODO(port): require a HIP
 * runtime; declared for surface completeness. */
ckc_status_t ckc_fmoe_capture_graph(ckc_fmoe_build_ctx_t* ctx,
                                    uint64_t routing_logits,
                                    uint64_t X,
                                    uint64_t W_gate,
                                    uint64_t W_up,
                                    uint64_t W_down,
                                    uint64_t Y,
                                    int warmup_iters);
ckc_status_t ckc_fmoe_replay_graph(ckc_fmoe_build_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_FUSED_MOE_E2E_INTERNAL_H */
