/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_helper_ck_dsl.instances.gfx942.attention_tiled_2d.h -- C99 port of
 * the symbols named by the porting task from
 * ck_dsl/instances/gfx942/attention_tiled_2d.py (the gfx942/CDNA3 narrow-atom
 * tiled-2D unified-attention kernel).
 *
 * Ported symbols (task list):
 *
 *   Python (attention_tiled_2d.py)              C99 (this header)
 *   -----------------------------------------   ----------------------------------------------
 *   class UnifiedAttention2DTiledSpec           ckc_attention_tiled_2d_spec_t
 *     (the spec dataclass + __post_init__)        + ckc_attention_tiled_2d_spec_default()
 *                                                 + ckc_attention_tiled_2d_spec_validate()
 *   (config constants derived in build_*)       ckc_unified_attention_2d_tiled_config_from_spec
 *   build_unified_attention_2d_tiled            ckc_build_unified_attention_2d_tiled_scalar
 *   _mfma_32x32_c_row(b, lane, elem_idx)        ckc__mfma_32x32_c_row
 *   _mfma_32x32_c_col(b, lane, n_tile32)        ckc__mfma_32x32_c_col
 *
 * The two ``_mfma_32x32_c_*`` helpers reproduce the Python builder-call
 * sequence byte-faithfully (same ops, same order, same operands), routing the
 * 32x32x16 C-warp ``TileDistribution.calculate_x`` through
 * ckc_tile_distribution_calculate_x exactly as the Python drives
 * ``_C32_DIST.calculate_x``.
 *
 * ``ckc_attention_tiled_2d_spec_t`` is a structural mirror of the Python
 * dataclass: every field carries its Python default (materialised by
 * ckc_attention_tiled_2d_spec_default). Python ``Optional`` fields are encoded
 * as a ``has_<field>`` flag + value pair (the absence flag == Python None).
 * ``__post_init__`` maps to ckc_attention_tiled_2d_spec_validate (a builder
 * variant that latches the first Python ValueError onto the sticky-error
 * IRBuilder). The derived ``@property`` accessors (num_queries_per_kv, block_m,
 * tile_size_eff, ...) are folded into
 * ckc_unified_attention_2d_tiled_config_from_spec, which is the faithful port of
 * the "config constants derived from spec" block at the head of
 * build_unified_attention_2d_tiled.
 *
 * ``ckc_build_unified_attention_2d_tiled_scalar`` is the kernel emitter. Its
 * Python body is ~4000 lines of IR emission; per the porting task it is a
 * stub-to-link entry point here (it validates the spec + arch and computes the
 * config, then records CKC_ERR_UNIMPLEMENTED on the builder and returns NULL).
 * The faithful kernel-body port is a follow-on; this surface lets a C driver
 * link, construct/validate the spec, and read the derived config today.
 *
 * Error model mirrors the rest of the C port: pure helpers return a sentinel
 * (NULL / false); the ckc_b_* / builder variants route the same failure through
 * the sticky-error IRBuilder. A dead builder is a NULL/false no-op.
 *
 * Lifetime: every IR node returned is arena-owned (ckc_ir_builder_t.arena).
 * The spec/config structs are plain by-value PODs the caller owns.
 */
#ifndef CKC_HELPER_HELPER_CK_DSL_INSTANCES_GFX942_ATTENTION_TILED_2D_H
#define CKC_HELPER_HELPER_CK_DSL_INSTANCES_GFX942_ATTENTION_TILED_2D_H

#include <stdbool.h>

#include "ckc/ir.h"                          /* ckc_ir_builder_t, ckc_value_t, ckc_kernel_def_t */
#include "ckc/helper_ck_dsl.helpers.atoms.h" /* ckc_mfma_atom_t, ckc_make_c_warp_dstr_encoding  */
#include "ckc/helper_ck_dsl.helpers.distribution.h" /* ckc_tile_distribution_t, calculate_x            */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------- module constants *
 * MFMA_M / MFMA_N (module-level in the Python). */
#define CKC_ATTN_TILED_2D_MFMA_M 16
#define CKC_ATTN_TILED_2D_MFMA_N 16

/* --------------------------------------------------------------- spec struct *
 *
 * Faithful mirror of UnifiedAttention2DTiledSpec. The first block holds the
 * required (no-default) fields; the rest carry the Python field defaults
 * (materialised by ckc_attention_tiled_2d_spec_default).
 *
 * Optional[int] fields are (has_<x>, <x>): has_<x>==false encodes Python None.
 * Optional[str] fields (kv_storage_dtype) are a ``const char*`` that is NULL
 * when Python is None. ``kv_cache_policy`` is a non-optional string ("stream"
 * default). All strings are caller-owned static/long-lived literals; this
 * struct does not copy them. */
typedef struct ckc_attention_tiled_2d_spec
{
    /* ---- required ---- */
    int head_size;
    int block_size;
    int num_query_heads;
    int num_kv_heads;
    const char* dtype; /* "fp16" or "bf16" */
    bool use_sinks;
    int sliding_window;
    bool has_softcap;

    /* ---- defaulted ---- */
    bool use_alibi;   /* False */
    bool use_qq_bias; /* False */
    int num_seqs;     /* 0 */
    int num_warps;    /* 1 */

    bool has_waves_per_eu; /* Optional[int] None */
    int waves_per_eu;

    const char* kv_storage_dtype; /* Optional[str] None ("fp8e4m3") */

    bool use_fp8_mfma_qk; /* False */
    bool use_fp8_mfma_pv; /* False */
    bool use_register_pv; /* False */

    bool has_tile_size; /* Optional[int] None */
    int tile_size;

    int block_m_per_warp;                 /* 16 */
    bool use_mfma_32x32;                  /* False */
    bool use_mfma_32x32x8;                /* False */
    bool use_transposed_qk_32x32;         /* False */
    bool use_transposed_scalar_state;     /* False */
    bool use_transposed_invariant_hoist;  /* False */
    bool use_transposed_mask_once;        /* False */
    bool use_transposed_half_local_pv;    /* False */
    bool use_mfma32_skip_legacy_qreg;     /* False */
    bool use_transposed_mask_limit;       /* False */
    bool use_grouped_kv2_softmax;         /* False */
    bool use_fast_paged_kv_desc;          /* False */
    bool use_i64_kv_addr;                 /* False */
    bool use_early_v_schedule;            /* False */
    bool use_agpr_alloc_zero;             /* False */
    bool use_conflict_free_v;             /* False */
    bool use_conflict_free_v_store;       /* False */
    bool use_conflict_free_v_store_split; /* True  */
    bool use_conflict_free_v_ck_vlds;     /* True  */
    bool use_k_single_buffer;             /* False */
    bool use_k_sliced_ring;               /* False */
    bool use_k_sliced_ldsseq;             /* False */
    bool use_iglp_opt;                    /* False */
    bool use_q_direct_global;             /* False */
    const char* kv_cache_policy;          /* "stream" */
    bool use_global_load_lds_k;           /* False */
    bool use_q_major_grid;                /* False */
} ckc_attention_tiled_2d_spec_t;

/* Materialise every defaulted field (the dataclass defaults). The required
 * fields are zero/NULL-initialised; the caller must set them before use. */
ckc_attention_tiled_2d_spec_t ckc_attention_tiled_2d_spec_default(void);

/* ------------------------------------------------ derived @property accessors *
 *
 * Pure, IR-free; faithful ports of the dataclass @property bodies. */

/* num_queries_per_kv = num_query_heads // num_kv_heads */
int ckc_attention_tiled_2d_spec_num_queries_per_kv(const ckc_attention_tiled_2d_spec_t* s);
/* block_m = block_m_per_warp * num_warps */
int ckc_attention_tiled_2d_spec_block_m(const ckc_attention_tiled_2d_spec_t* s);
/* regs_per_lane: 16 for the 32x32 paths, else block_m_per_warp // 4 */
int ckc_attention_tiled_2d_spec_regs_per_lane(const ckc_attention_tiled_2d_spec_t* s);
/* block_q = block_m // num_queries_per_kv */
int ckc_attention_tiled_2d_spec_block_q(const ckc_attention_tiled_2d_spec_t* s);
/* tile_size_eff = tile_size if has_tile_size else block_size */
int ckc_attention_tiled_2d_spec_tile_size_eff(const ckc_attention_tiled_2d_spec_t* s);
/* n_blocks_per_tile = tile_size_eff // block_size */
int ckc_attention_tiled_2d_spec_n_blocks_per_tile(const ckc_attention_tiled_2d_spec_t* s);
/* dtype_ir: F16 for "fp16", BF16 otherwise (returns a ckc_type_t*). */
const ckc_type_t* ckc_attention_tiled_2d_spec_dtype_ir(const ckc_attention_tiled_2d_spec_t* s);
/* binary_search_iters: 32 if num_seqs<=0, else max(1, ceil(log2(num_seqs+1))) */
int ckc_attention_tiled_2d_spec_binary_search_iters(const ckc_attention_tiled_2d_spec_t* s);

/* __post_init__ validator (builder variant). Reproduces the Python validation
 * order and messages; on the first failing check it records the Python
 * ValueError text on ``b`` (CKC_ERR_VALUE) and returns false. Returns true when
 * the spec is accepted. A dead/NULL builder is a false no-op. */
bool ckc_attention_tiled_2d_spec_validate(ckc_ir_builder_t* b,
                                          const ckc_attention_tiled_2d_spec_t* s);

/* ----------------------------------------------- config-from-spec (build head) *
 *
 * The pile of ALL-CAPS config constants the build function derives from the
 * spec before any IR is emitted (HD/T/BS/BLOCK_M/... plus the umbrella
 * predicates and byte-stride helpers). Faithful port of that block. */
typedef struct ckc_unified_attention_2d_tiled_config
{
    int HD;                /* spec.head_size */
    int T;                 /* spec.tile_size_eff */
    int BS;                /* spec.block_size */
    int N_BLOCKS_PER_TILE; /* spec.n_blocks_per_tile */
    int BLOCK_M;           /* spec.block_m */
    int BLOCK_Q;           /* spec.block_q */
    int NQK;               /* spec.num_queries_per_kv */
    int NUM_KV;            /* spec.num_kv_heads */
    int NUM_QH;            /* spec.num_query_heads */
    int SLIDING_WINDOW;    /* spec.sliding_window */
    bool USE_SOFTCAP;      /* spec.has_softcap */
    bool USE_SINKS;        /* spec.use_sinks */
    bool USE_ALIBI;        /* spec.use_alibi */
    bool USE_QQ_BIAS;      /* spec.use_qq_bias */

    /* fp8 K/V cache predicates */
    bool KV_FP8;        /* kv_storage_dtype == "fp8e4m3" */
    bool FP8_MFMA_QK;   /* KV_FP8 && use_fp8_mfma_qk */
    bool FP8_MFMA_PV;   /* KV_FP8 && use_fp8_mfma_pv */
    bool FP8_NATIVE_QK; /* always False (documented dead path) */
    int KV_BYTES;       /* 1 if KV_FP8 else 2 */

    /* 32x32 umbrella predicates */
    bool USE_MFMA_32X32X8; /* spec.use_mfma_32x32x8 */
    bool USE_MFMA_32X32;   /* spec.use_mfma_32x32 || USE_MFMA_32X32X8 */

    bool REGISTER_PV; /* spec.use_register_pv */
    bool TRANSPOSED_QK_32X32;
    bool CONFLICT_FREE_V;
    bool CONFLICT_FREE_V_STORE;
    bool K_SINGLE_BUF;

    /* the selected element dtype (F16/BF16) and the KV io dtype */
    const ckc_type_t* dtype;       /* F16 / BF16 */
    const ckc_type_t* kv_io_dtype; /* FP8E4M3 if KV_FP8 else dtype */
} ckc_unified_attention_2d_tiled_config_t;

/* Faithful port of the build_unified_attention_2d_tiled config-derivation head.
 *
 * Validates the spec (must pass ckc_attention_tiled_2d_spec_validate),
 * dtype-gates ("fp16"/"bf16" only -> Python NotImplementedError on others),
 * and fills *out with the derived constants. Returns true on success; on the
 * Python ValueError/NotImplementedError paths it records the message on ``b``
 * and returns false (leaving *out unspecified). */
bool ckc_unified_attention_2d_tiled_config_from_spec(ckc_ir_builder_t* b,
                                                     const ckc_attention_tiled_2d_spec_t* spec,
                                                     ckc_unified_attention_2d_tiled_config_t* out);

/* ---------------------------------------------------- kernel build (stub) *
 *
 * build_unified_attention_2d_tiled(spec, arch="gfx942") analogue.
 *
 * STUB-TO-LINK: the Python kernel body is ~4000 lines of IR emission and is not
 * yet ported. This entry validates the arch and spec, derives the config, then
 * records CKC_ERR_UNIMPLEMENTED on ``b`` and returns NULL. ``arch`` NULL == the
 * Python default "gfx942". Defined so a C driver can link against the gfx942
 * tiled-2D surface today; the faithful body port is a follow-on. */
ckc_kernel_def_t* ckc_build_unified_attention_2d_tiled_scalar(
    ckc_ir_builder_t* b, const ckc_attention_tiled_2d_spec_t* spec, const char* arch);

/* ----------------------------------------------------- 32x32 C helpers *
 *
 * _mfma_32x32_c_row(b, lane, elem_idx): MFMA-local output row for a 32x32x16 C
 * element ``elem_idx`` (0..15). Drives CK Tile's C-warp distribution
 * (_C32_DIST.calculate_x) with ys=[elem_idx//4, elem_idx%4], ps=[[lane//32,
 * lane%32]] and returns the row X-coord. ``elem_idx`` outside 0..15 is a Python
 * ValueError: the builder records CKC_ERR_VALUE and the call returns NULL. */
ckc_value_t* ckc__mfma_32x32_c_row(ckc_ir_builder_t* b, ckc_value_t* lane, int elem_idx);

/* _mfma_32x32_c_col(b, lane, n_tile32=0): MFMA-local output col for 32x32x16 C
 * elements in N-tile ``n_tile32``. Drives _C32_DIST.calculate_x with ys=[0,0],
 * ps=[[lane//32, lane%32]] and returns the col X-coord; when n_tile32 != 0,
 * ``n_tile32*32`` is added on top. Returns NULL on a builder error. */
ckc_value_t* ckc__mfma_32x32_c_col(ckc_ir_builder_t* b, ckc_value_t* lane, int n_tile32);

/* Re-entrancy reset for the lazily-built _C32_DIST cache. The cache holds a
 * tile distribution arena-allocated off a build's IRBuilder; that builder is
 * freed at the end of the build, so the cached pointer dangles. Call this at the
 * start of every tiled-attention build entry (gfx942 + gfx950) so a fresh build
 * rebuilds the distribution against its own arena instead of reading freed
 * memory. Safe to call when nothing is cached (no-op). */
void ckc_attn2d_c32_dist_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* CKC_HELPER_HELPER_CK_DSL_INSTANCES_GFX942_ATTENTION_TILED_2D_H */
