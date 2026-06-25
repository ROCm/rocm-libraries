/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.helpers.fused_moe_e2e_orchestrator.h -- C99 port of the
 * end-to-end fused-MoE forward orchestrator.
 *
 * The orchestrator's real Python source resolves to
 *   ck_dsl/instances/common/fused_moe_e2e.py
 * (there is no ck_dsl/helpers/fused_moe_e2e_orchestrator.py; the C file name
 * follows the requested target name). The ported symbols and their Python
 * peers:
 *
 *   Python (FusedMoeForward / module)        C99 (this header)
 *   --------------------------------------   ----------------------------------
 *   class FusedMoeForward                     ckc_fused_moe_forward_t
 *   FusedMoeForward.forward (kernel entry)    ckc_fused_moe_forward_forward()
 *   FusedMoeForward._forward_static           ckc_fused_moe_forward_static()
 *   FusedMoeForward._forward_dynamic          ckc_fused_moe_forward_dynamic()
 *   FusedMoeForward._workspace_specs          ckc_fused_moe_forward_workspace_specs()
 *   FusedMoeForward._host_preshuffle_b        ckc_fused_moe_host_preshuffle_b()
 *   FusedMoeForward._build_per_expert_problems
 *                                             ckc_fused_moe_build_per_expert_problems()
 *
 * Scope note (what is and is not byte-faithful)
 * ---------------------------------------------
 * Unlike the IR-emitting helpers in this port (preshuffle, sweep, ...),
 * FusedMoeForward is a *host runtime driver*: it allocates torch workspaces,
 * issues chained HIP kernel launches, copies (experts,) i32 arrays D->H, and
 * casts an f32 accumulator into the caller's output. There is no IR builder
 * sequence to reproduce. Accordingly:
 *
 *   - The three *value-producing* host helpers are ported faithfully (the
 *     arithmetic / layout is reproduced exactly):
 *       * ckc_fused_moe_forward_workspace_specs  -- the 15-entry workspace
 *         shape/dtype table, in declaration order.
 *       * ckc_fused_moe_host_preshuffle_b        -- the (E,N,K) -> (E,k_tiles,
 *         n_tiles,block_n,block_k) shape/stride + divisibility precondition.
 *       * ckc_fused_moe_build_per_expert_problems -- the per-expert pointer
 *         arithmetic that fills GroupedGemmProblem entries.
 *   - The three *launch driver* entries (forward / static / dynamic) are kept
 *     bounded: they reproduce the host-side dispatch *decision* (static vs
 *     dynamic, the slot_size / total_padded sizing) but the actual HIP launch
 *     chain, torch buffer zeroing, and D->H copies are out of scope for a
 *     Python-free C IR port and are marked TODO(port). They return
 *     CKC_ERR_NOTIMPL so a caller cannot mistake the stub for a working launch.
 *
 * Error model mirrors the rest of the port: ckc_status_t return + an out-param
 * for value producers; the Python 'raise ValueError' maps to CKC_ERR_VALUE.
 */
#ifndef CKC_HELPER_CK_DSL_HELPERS_FUSED_MOE_E2E_ORCHESTRATOR_H
#define CKC_HELPER_CK_DSL_HELPERS_FUSED_MOE_E2E_ORCHESTRATOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ckc/instance_gemm_universal.h" /* ckc_gemm_tile_spec_t */
#include "ckc/ir.h" /* ckc_status_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------ dtype labels *
 *
 * Mirror of the string dtype labels the Python orchestrator accepts. Only the
 * 2-byte activation dtypes are supported (Python _ensure_2byte_dtype /
 * _gemm_dtype_to_universal raise ValueError on anything else).
 */
typedef enum ckc_fmoe_dtype
{
    CKC_FMOE_DTYPE_F16 = 0, /* "f16" / "fp16" */
    CKC_FMOE_DTYPE_BF16 /* "bf16"         */
} ckc_fmoe_dtype_t;

/* element_size in bytes (Python _ensure_2byte_dtype): always 2 for the
 * supported activation dtypes. Returns 0 + CKC_ERR_VALUE-style 0 for an
 * unrecognized enum value. */
int ckc_fmoe_dtype_elem_bytes(ckc_fmoe_dtype_t dtype);

/* "fp16"/"bf16" canonical string (Python _gemm_dtype_to_universal). Returns a
 * static string, or NULL for an unsupported value. */
const char* ckc_fmoe_dtype_to_universal(ckc_fmoe_dtype_t dtype);

/* ------------------------------------------------------------ FusedMoeForwardSpec *
 *
 * Value type mirroring the subset of FusedMoeForwardSpec the ported symbols
 * read. Fields are 1:1 with the Python dataclass declaration order for the
 * shape/dtype block; the many boolean tuning knobs that only steer the launch
 * chain (which is stubbed) are kept as a single flags set so the struct stays
 * a faithful but bounded mirror.
 */
typedef struct ckc_fmoe_forward_spec
{
    int tokens; /* T */
    int experts; /* E */
    int topk; /* K */
    int hidden; /* H */
    int intermediate; /* I */
    ckc_fmoe_dtype_t dtype;

    int streaming_block_size; /* default 256 */
    int streaming_vec; /* default 8   */
    int sort_block_size; /* default 64  */
    int router_block_size; /* default 64  */

    ckc_gemm_tile_spec_t gemm_tile;

    /* Tuning knobs that select which path / launcher runs. The faithfully
     * ported value helpers do not depend on these; they are carried so the
     * forward() dispatch decision (static vs dynamic) can be reproduced. */
    bool use_grouped_gemm; /* default true  */
    bool active_tile_skip_gemms; /* default true  */
    bool preshuffle_w_down; /* default false */
} ckc_fmoe_forward_spec_t;

/* FusedMoeForwardSpec.total_pairs == tokens * topk. */
int ckc_fmoe_forward_spec_total_pairs(const ckc_fmoe_forward_spec_t* spec);

/* ------------------------------------------------------------ workspace specs *
 *
 * Mirror of the WorkspaceSpec tuple produced by _workspace_specs. Each entry
 * carries the name, rank-<=2 shape, dtype kind, and element size so a caller
 * can size scratch without a torch dependency. The dtype kind is one of:
 */
typedef enum ckc_ws_dtype
{
    CKC_WS_I32 = 0,
    CKC_WS_F32,
    CKC_WS_ACT /* activation dtype (f16 / bf16) */
} ckc_ws_dtype_t;

typedef struct ckc_ws_spec
{
    const char* name; /* static string, e.g. "TopkIds" */
    int shape[2]; /* up to 2 dims; rank gives the valid count */
    int rank; /* 1 or 2 */
    ckc_ws_dtype_t dtype;
    int elem_bytes; /* 4 for i32/f32; 2 for act */
} ckc_ws_spec_t;

/* Number of workspace entries _workspace_specs returns (constant: 15). */
#define CKC_FMOE_NUM_WORKSPACE_SPECS 15

/* Fill `out` (capacity `cap` entries) with the workspace table for `spec`,
 * in the exact Python declaration order. Writes min(cap, 15) entries and sets
 * *n_written. Returns CKC_OK, or CKC_ERR_VALUE on a NULL arg / unsupported
 * dtype. The strings point at static storage. */
ckc_status_t ckc_fused_moe_forward_workspace_specs(const ckc_fmoe_forward_spec_t* spec,
                                                   ckc_ws_spec_t* out,
                                                   size_t cap,
                                                   size_t* n_written);

/* ------------------------------------------------------------ host preshuffle B *
 *
 * Mirror of _host_preshuffle_b. Pure shape/stride producer: given the input
 * (E, N, K) row-major B and (block_n, block_k), it returns the output shape
 * (E, k_tiles, n_tiles, block_n, block_k) and the precondition check. The
 * actual element permutation (torch .view/.permute/.contiguous) is a runtime
 * concern out of scope here; this reproduces the shape contract + the Python
 * 'raise ValueError' precondition exactly.
 *
 * out_shape must have room for 5 ints. Returns CKC_ERR_VALUE when
 * N % block_n != 0 or K % block_k != 0 (mirroring the Python ValueError).
 */
ckc_status_t ckc_fused_moe_host_preshuffle_b(
    int E, int N, int K, int block_n, int block_k, int out_shape[5]);

/* ------------------------------------------------------------ per-expert problems *
 *
 * Mirror of GroupedGemmProblem (instances/common/grouped_gemm.py): one entry
 * of a grouped GEMM workload. Pointers are device pointers (uint64).
 */
typedef struct ckc_grouped_gemm_problem
{
    int M;
    int N;
    int K;
    uint64_t A_ptr;
    uint64_t B_ptr;
    uint64_t C_ptr;
} ckc_grouped_gemm_problem_t;

/* Mirror of _build_per_expert_problems. Walks the per-expert counts/offsets
 * (host i32 arrays of length `experts`) and emits one problem per active
 * expert (count > 0) with the exact pointer arithmetic of the Python:
 *
 *   A_ptr = a_base + offset * a_inner_dim * elem_bytes
 *   B_ptr = b_base[e]                    (per-expert weight base)
 *   C_ptr = c_base + offset * c_inner_dim * elem_bytes
 *   M=count, N=c_inner_dim, K=b_inner_dim
 *
 * `b_base` is an array of `experts` device pointers (b_weights[e].data_ptr()).
 * Fills `out` (capacity `cap`) and sets *n_written to the number of active
 * experts. Returns CKC_OK, or CKC_ERR_VALUE on a NULL arg.
 */
ckc_status_t ckc_fused_moe_build_per_expert_problems(int experts,
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

/* ------------------------------------------------------------ forward driver *
 *
 * Mirror of FusedMoeForward: the host runtime instance plus its three launch
 * entries. The launch chain itself (HIP launches, torch buffers, D->H copies)
 * is a runtime concern and is a TODO(port) stub here; these entries reproduce
 * the host-side dispatch *decision* and sizing only, returning CKC_ERR_NOTIMPL
 * to flag the stub.
 */
typedef struct ckc_fused_moe_forward
{
    ckc_fmoe_forward_spec_t spec;
    const char* arch; /* resolved launch arch ("gfx942"/"gfx950") */
    bool use_static_offsets; /* T*K*E <= 256 gate (set by init)          */
    int static_slot_size; /* ceil(T*K / tile_m) * tile_m (static path) */
} ckc_fused_moe_forward_t;

/* Construct the driver from a spec (mirrors FusedMoeForward.__init__: resolve
 * arch, compute the static-vs-dynamic gate and static slot size). `arch` may
 * be NULL to default to "gfx950" (the Python no-HIP fallback). */
ckc_status_t ckc_fused_moe_forward_init(ckc_fused_moe_forward_t* self,
                                        const ckc_fmoe_forward_spec_t* spec,
                                        const char* arch);

/* forward(): dispatch to static or dynamic. Reproduces the
 * `if self._use_static_offsets` decision; the launch body is TODO(port). */
ckc_status_t ckc_fused_moe_forward_forward(ckc_fused_moe_forward_t* self);

/* _forward_static / _forward_dynamic: bounded launch-driver stubs. */
ckc_status_t ckc_fused_moe_forward_static(ckc_fused_moe_forward_t* self);
ckc_status_t ckc_fused_moe_forward_dynamic(ckc_fused_moe_forward_t* self);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_HELPERS_FUSED_MOE_E2E_ORCHESTRATOR_H */
