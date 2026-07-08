// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_3d_gfx942_attn_tiled_3d_public_glue.c --
 * PUBLIC entry + glue for the C99 chunked port of
 * rocke/instances/gfx942/attention_tiled_3d.py (arch gfx942).
 *
 * SCOPE (this TU only):
 *   - The two by-value spec structs' _default() / _validate() (__post_init__)
 *     and every @property accessor (num_queries_per_kv, block_m=16, block_q,
 *     tile_size, dtype_ir, binary_search_iters, kernel_name via
 *     kernel_name_join).  (Python lines 91-172, 977-999.)
 *   - supports_tiled_3d coverage gate (pure host predicate, static reasons).
 *     (Python lines 175-231.)
 *   - rocke_gfx942_attention_tiled_3d_ctx_init + the two config_from_spec
 *     derivations (validate + arch gate + dtype gate + every ALL-CAPS const +
 *     load-feed asserts + C16_DIST build).  (Python lines 122-127, 175-231,
 *     242-273, 438-667, 1008-1057.)
 *   - The two PUBLIC build entries that drive ctx_init then call the phase fns
 *     in Python execution order.  (Python build_unified_attention_3d_tiled
 *     234-969 ; build_unified_attention_reduce_tiled 1002-1133.)
 *   - The two build -> lower-to-.ll convenience fns (own/free an internal
 *     IRBuilder, malloc *out_ll).
 *
 * The phase functions (declare_params / emit_prologue / emit_early_zero_fill /
 * emit_q_to_lds / emit_loop_init / emit_softmax_loop / emit_epilogue and the
 * reduce declare_and_prologue / max_pass / combine_pass / normalize_pass) and
 * the inner IR helpers (mfma_16x16_c_row / strided_v_b_operand / the KV load
 * issuers) are PEERS implemented in sibling TUs (the segment + reduce body
 * buckets) and declared in rocke/instance_gfx942_attention_tiled_3d_internal.h.
 * This TU calls them through that header but does not implement them.
 *
 * ARCH GATE NOTE (validate_tiled_attention_arch). The Python validator
 * (instances/common/attention_arch.py) admits two families: the gfx950 wide-K +
 * ds_read_tr path and the gfx942 narrow 16x16x16 strided-V path. The C arch
 * surface (rocke/helper_rocke.core.arch.h) exposes op_for_shape but not the
 * memory.has_ds_read_tr predicate, so the wide-K admission is reproduced via the
 * catalogued 16x16x32 f16 atom only (the canonical wide-K marker). On gfx942 the
 * relevant branch is the narrow path (gfx942 in {_NARROW_TILED_2D_ARCHES} AND
 * the 16x16x16 f16+bf16 atoms present), which IS fully reproducible here. The
 * fallback reason strings mirror the Python text byte-for-byte.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h" /* rocke_arena_strdup */
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx, op_for_shape */
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom, make_c_warp_dstr_encoding */
#include "rocke/helper_rocke.helpers.distribution.h" /* make_static_tile_distribution */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/instance_gfx942_attention_tiled_3d.h"
#include "rocke/instance_gfx942_attention_tiled_3d_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */
#include "rocke/lower_llvm.h"

/* Module consts (Python module-level MFMA_M / MFMA_N). */
#define ROCKE_ATTN3D_MFMA_M 16
#define ROCKE_ATTN3D_MFMA_N 16

/* ===================================================================== *
 *  small helpers
 * ===================================================================== */

static bool rocke_attn3d_streq(const char* a, const char* b)
{
    if(a == NULL || b == NULL)
    {
        return a == b;
    }
    return strcmp(a, b) == 0;
}

/* Copy `msg` into (err, err_cap), NUL-terminated and truncated. No-op if err is
 * NULL / err_cap is 0. */
static void rocke_attn3d_set_err_buf(char* err, size_t err_cap, const char* msg)
{
    size_t n;
    if(err == NULL || err_cap == 0)
    {
        return;
    }
    if(msg == NULL)
    {
        msg = "";
    }
    n = strlen(msg);
    if(n >= err_cap)
    {
        n = err_cap - 1;
    }
    memcpy(err, msg, n);
    err[n] = '\0';
}

/* True iff the target catalog has the narrow 16x16x16 f16 AND bf16 (a,a,fp32)
 * atom (_narrow_k_mfma_available). */
static bool rocke_attn3d_narrow_k_available(const rocke_archtarget_t* t)
{
    const rocke_mmaop_t* f16
        = rocke_archtarget_op_for_shape(t, "mma", "f16", "f16", "fp32", 16, 16, 16);
    const rocke_mmaop_t* bf16
        = rocke_archtarget_op_for_shape(t, "mma", "bf16", "bf16", "fp32", 16, 16, 16);
    return f16 != NULL && bf16 != NULL;
}

/* True iff the target catalog has the wide-K 16x16x32 f16 atom
 * (_wide_k_mfma_available, the queryable half; the ds_read_tr cross-check is not
 * exposed by the C arch surface -- see the ARCH GATE NOTE above). */
static bool rocke_attn3d_wide_k_available(const rocke_archtarget_t* t)
{
    return rocke_archtarget_op_for_shape(t, "mma", "f16", "f16", "fp32", 16, 16, 32) != NULL;
}

/* ===================================================================== *
 *  UnifiedAttention3DTiledSpec -- _default / _validate / @property
 * ===================================================================== */

rocke_unified_attention_3d_tiled_spec_t rocke_unified_attention_3d_tiled_spec_default(void)
{
    rocke_unified_attention_3d_tiled_spec_t s;
    memset(&s, 0, sizeof(s));
    /* required fields stay zero/NULL (caller must set). Defaulted fields: */
    s.use_alibi = false;
    s.use_qq_bias = false;
    s.num_seqs = 0;
    s.has_waves_per_eu = false;
    s.waves_per_eu = 0;
    s.kv_storage_dtype = NULL;
    s.has_tile_size_override = false;
    s.tile_size_override = 0;
    s.use_invariant_hoist = false;
    s.use_wide_kv_load = false;
    return s;
}

/* __post_init__: kv_storage_dtype must be None or 'fp8e4m3'. */
bool rocke_unified_attention_3d_tiled_spec_validate(
    rocke_ir_builder_t* b, const rocke_unified_attention_3d_tiled_spec_t* s)
{
    if(b == NULL || !rocke_ir_builder_ok(b))
    {
        return false;
    }
    if(s == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "attention_tiled_3d spec is NULL");
        return false;
    }
    if(s->kv_storage_dtype != NULL && !rocke_attn3d_streq(s->kv_storage_dtype, "fp8e4m3"))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "kv_storage_dtype must be None or 'fp8e4m3' (got '%s')",
                        s->kv_storage_dtype);
        return false;
    }
    return true;
}

int rocke_unified_attention_3d_tiled_spec_num_queries_per_kv(
    const rocke_unified_attention_3d_tiled_spec_t* s)
{
    if(s == NULL || s->num_kv_heads == 0)
    {
        return -1;
    }
    return s->num_query_heads / s->num_kv_heads;
}

int rocke_unified_attention_3d_tiled_spec_block_m(const rocke_unified_attention_3d_tiled_spec_t* s)
{
    (void)s;
    return 16;
}

int rocke_unified_attention_3d_tiled_spec_block_q(const rocke_unified_attention_3d_tiled_spec_t* s)
{
    int nqk = rocke_unified_attention_3d_tiled_spec_num_queries_per_kv(s);
    if(nqk <= 0)
    {
        return -1;
    }
    return 16 / nqk;
}

int rocke_unified_attention_3d_tiled_spec_tile_size(
    const rocke_unified_attention_3d_tiled_spec_t* s)
{
    if(s == NULL)
    {
        return -1;
    }
    return s->has_tile_size_override ? s->tile_size_override : s->block_size;
}

const rocke_type_t*
    rocke_unified_attention_3d_tiled_spec_dtype_ir(const rocke_unified_attention_3d_tiled_spec_t* s)
{
    if(s != NULL && rocke_attn3d_streq(s->dtype, "fp16"))
    {
        return rocke_f16();
    }
    return rocke_bf16();
}

int rocke_unified_attention_3d_tiled_spec_binary_search_iters(
    const rocke_unified_attention_3d_tiled_spec_t* s)
{
    double v;
    int r;
    if(s == NULL)
    {
        return -1;
    }
    if(s->num_seqs <= 0)
    {
        return 32;
    }
    /* max(1, ceil(log2(num_seqs + 1))) */
    v = ceil(log2((double)(s->num_seqs + 1)));
    r = (int)v;
    return r < 1 ? 1 : r;
}

int rocke_unified_attention_3d_tiled_spec_kernel_name(
    const rocke_unified_attention_3d_tiled_spec_t* s, char* buf, size_t cap)
{
    /* kernel_name_join("rocke_uattn3d_tiled_gfx942", d.., b.., h..kv.., seg..,
     *   dtype, kv<...> if kv_storage_dtype else "", "sinks" if use_sinks else "",
     *   "sw<sw>" if sw>0 else "", "softcap" if has_softcap else "",
     *   "alibi" if use_alibi else "", "qqb" if use_qq_bias else "",
     *   "hoist" if use_invariant_hoist else "", "wkv" if use_wide_kv_load else "")
     *
     * The whole thing is a positional parts list (NOT flags=...): empty parts
     * are skipped by kernel_name_join. We materialise each part string and feed
     * the non-empty ones in source order. */
    char d_part[32];
    char b_part[32];
    char h_part[64];
    char seg_part[32];
    char kv_part[32];
    char sw_part[32];
    const char* parts[16];
    size_t np = 0;
    size_t out_len = 0;
    rocke_status_t st;

    if(s == NULL || buf == NULL || cap == 0)
    {
        return -1;
    }

    snprintf(d_part, sizeof(d_part), "d%d", s->head_size);
    snprintf(b_part, sizeof(b_part), "b%d", s->block_size);
    snprintf(h_part, sizeof(h_part), "h%dkv%d", s->num_query_heads, s->num_kv_heads);
    snprintf(seg_part, sizeof(seg_part), "seg%d", s->num_segments);

    parts[np++] = "rocke_uattn3d_tiled_gfx942";
    parts[np++] = d_part;
    parts[np++] = b_part;
    parts[np++] = h_part;
    parts[np++] = seg_part;
    parts[np++] = (s->dtype != NULL) ? s->dtype : "";
    if(s->kv_storage_dtype != NULL && s->kv_storage_dtype[0] != '\0')
    {
        snprintf(kv_part, sizeof(kv_part), "kv%s", s->kv_storage_dtype);
        parts[np++] = kv_part;
    }
    if(s->use_sinks)
    {
        parts[np++] = "sinks";
    }
    if(s->sliding_window > 0)
    {
        snprintf(sw_part, sizeof(sw_part), "sw%d", s->sliding_window);
        parts[np++] = sw_part;
    }
    if(s->has_softcap)
    {
        parts[np++] = "softcap";
    }
    if(s->use_alibi)
    {
        parts[np++] = "alibi";
    }
    if(s->use_qq_bias)
    {
        parts[np++] = "qqb";
    }
    if(s->use_invariant_hoist)
    {
        parts[np++] = "hoist";
    }
    if(s->use_wide_kv_load)
    {
        parts[np++] = "wkv";
    }

    /* kernel_name_join takes prefix + parts; we pass everything as parts with a
     * "" prefix so the join is identical (empty prefix is skipped). */
    st = rocke_kernel_name_join("", parts, np, NULL, NULL, 0, buf, cap, &out_len);
    if(st != ROCKE_OK)
    {
        return -1;
    }
    return (int)out_len;
}

/* ===================================================================== *
 *  UnifiedAttentionReduceTiledSpec -- _default / dtype_ir / kernel_name
 * ===================================================================== */

rocke_unified_attention_reduce_tiled_spec_t rocke_unified_attention_reduce_tiled_spec_default(void)
{
    rocke_unified_attention_reduce_tiled_spec_t s;
    memset(&s, 0, sizeof(s));
    s.has_waves_per_eu = false;
    s.waves_per_eu = 0;
    return s;
}

const rocke_type_t* rocke_unified_attention_reduce_tiled_spec_dtype_ir(
    const rocke_unified_attention_reduce_tiled_spec_t* s)
{
    if(s != NULL && rocke_attn3d_streq(s->dtype, "fp16"))
    {
        return rocke_f16();
    }
    return rocke_bf16();
}

int rocke_unified_attention_reduce_tiled_spec_kernel_name(
    const rocke_unified_attention_reduce_tiled_spec_t* s, char* buf, size_t cap)
{
    /* kernel_name_join("rocke_uattn_reduce_tiled_gfx942", d<HD>, h<NUM_QH>,
     *   seg<NUM_SEG>, dtype) */
    char d_part[32];
    char h_part[32];
    char seg_part[32];
    const char* parts[8];
    size_t np = 0;
    size_t out_len = 0;
    rocke_status_t st;

    if(s == NULL || buf == NULL || cap == 0)
    {
        return -1;
    }

    snprintf(d_part, sizeof(d_part), "d%d", s->head_size);
    snprintf(h_part, sizeof(h_part), "h%d", s->num_query_heads);
    snprintf(seg_part, sizeof(seg_part), "seg%d", s->num_segments);

    parts[np++] = "rocke_uattn_reduce_tiled_gfx942";
    parts[np++] = d_part;
    parts[np++] = h_part;
    parts[np++] = seg_part;
    parts[np++] = (s->dtype != NULL) ? s->dtype : "";

    st = rocke_kernel_name_join("", parts, np, NULL, NULL, 0, buf, cap, &out_len);
    if(st != ROCKE_OK)
    {
        return -1;
    }
    return (int)out_len;
}

/* ===================================================================== *
 *  supports_tiled_3d coverage gate (pure host predicate, static reasons)
 *
 *  Reasons that interpolate values are rendered into a static thread-local
 *  buffer; the fixed reasons are string literals. Faithful port of Python
 *  lines 175-231 (the value-interpolated cases share the exact format text).
 * ===================================================================== */
bool rocke_gfx942_attention_tiled_3d_supports(int head_size,
                                              int block_size,
                                              const char* dtype,
                                              int num_queries_per_kv,
                                              bool use_alibi,
                                              bool use_qq_bias,
                                              bool use_fp8,
                                              const char* q_dtype,
                                              const char* kv_storage_dtype,
                                              const char* arch,
                                              const char** out_reason)
{
    static char reason_buf[ROCKE_ERR_MSG_CAP];
    const char* eff_arch = (arch != NULL) ? arch : "gfx942";
    const rocke_archtarget_t* target;
    bool arch_ok;

    (void)use_alibi;
    (void)use_qq_bias;

#define ATTN3D_REASON(s)       \
    do                         \
    {                          \
        if(out_reason != NULL) \
        {                      \
            *out_reason = (s); \
        }                      \
    } while(0)

    /* arch_ok, arch_reason = validate_tiled_attention_arch(arch) */
    target = rocke_archtarget_from_gfx(eff_arch);
    if(target == NULL)
    {
        /* KeyError str() wrapping (matches _build_target / ArchTarget.from_gfx). */
        snprintf(reason_buf, sizeof(reason_buf), "\"unknown gfx target '%s'\"", eff_arch);
        ATTN3D_REASON(reason_buf);
        return false;
    }
    arch_ok = false;
    if(rocke_attn3d_wide_k_available(target))
    {
        /* gfx950 wide-K + ds_read_tr path (ds_read_tr not queryable in C; the
         * wide-K marker is taken as admission, see ARCH GATE NOTE). */
        arch_ok = true;
    }
    else if(rocke_attn3d_streq(eff_arch, "gfx942") && rocke_attn3d_narrow_k_available(target))
    {
        arch_ok = true;
    }
    if(!arch_ok)
    {
        snprintf(reason_buf,
                 sizeof(reason_buf),
                 "tiled attention requires either the wide-K MFMA atoms "
                 "(mfma_f32_16x16x32 / mfma_f32_32x32x16, gfx950) or, on a narrow "
                 "variant arch (gfx942), the 16x16x16 f16/bf16 atom; neither path "
                 "is available on %s",
                 eff_arch);
        ATTN3D_REASON(reason_buf);
        return false;
    }

    if(!rocke_attn3d_streq(dtype, "fp16") && !rocke_attn3d_streq(dtype, "bf16"))
    {
        snprintf(reason_buf,
                 sizeof(reason_buf),
                 "tiled 3D kernel currently supports fp16/bf16 (got '%s')",
                 dtype != NULL ? dtype : "(null)");
        ATTN3D_REASON(reason_buf);
        return false;
    }
    if(head_size != 64 && head_size != 128 && head_size != 256)
    {
        snprintf(reason_buf,
                 sizeof(reason_buf),
                 "tiled 3D kernel only supports head_size in {64,128,256} (got %d)",
                 head_size);
        ATTN3D_REASON(reason_buf);
        return false;
    }
    if(head_size % 32 != 0)
    {
        snprintf(reason_buf,
                 sizeof(reason_buf),
                 "tiled 3D kernel requires head_size divisible by 32 (got %d)",
                 head_size);
        ATTN3D_REASON(reason_buf);
        return false;
    }
    if(block_size != 16 && block_size != 32 && block_size != 64)
    {
        snprintf(reason_buf,
                 sizeof(reason_buf),
                 "tiled 3D kernel only supports block_size in {16,32,64} (got %d)",
                 block_size);
        ATTN3D_REASON(reason_buf);
        return false;
    }
    if(kv_storage_dtype != NULL && !rocke_attn3d_streq(kv_storage_dtype, "fp8e4m3"))
    {
        snprintf(reason_buf,
                 sizeof(reason_buf),
                 "tiled 3D kernel: unsupported kv_storage_dtype '%s'",
                 kv_storage_dtype);
        ATTN3D_REASON(reason_buf);
        return false;
    }
    if(use_fp8 && kv_storage_dtype == NULL)
    {
        ATTN3D_REASON("tiled 3D kernel: use_fp8=True requires kv_storage_dtype='fp8e4m3'");
        return false;
    }
    if(q_dtype != NULL && !rocke_attn3d_streq(q_dtype, "fp16")
       && !rocke_attn3d_streq(q_dtype, "bf16"))
    {
        snprintf(
            reason_buf, sizeof(reason_buf), "tiled 3D kernel: unsupported q_dtype '%s'", q_dtype);
        ATTN3D_REASON(reason_buf);
        return false;
    }
    if(num_queries_per_kv > 16 || num_queries_per_kv < 1)
    {
        snprintf(reason_buf,
                 sizeof(reason_buf),
                 "tiled 3D kernel needs 1<=num_queries_per_kv<=16 (got %d)",
                 num_queries_per_kv);
        ATTN3D_REASON(reason_buf);
        return false;
    }
    if(16 % num_queries_per_kv != 0)
    {
        ATTN3D_REASON("tiled 3D kernel needs num_queries_per_kv to divide BLOCK_M=16");
        return false;
    }
    ATTN3D_REASON("supported");
    return true;
#undef ATTN3D_REASON
}

/* ===================================================================== *
 *  config_from_spec -- segment kernel (Python lines 242-273, 438-667)
 * ===================================================================== */
bool rocke_gfx942_attn_tiled_3d_config_from_spec(
    rocke_ir_builder_t* b,
    const rocke_unified_attention_3d_tiled_spec_t* spec,
    const char* arch,
    rocke_gfx942_attn_tiled_3d_config_t* out)
{
    const char* eff_arch = (arch != NULL) ? arch : "gfx942";
    const rocke_archtarget_t* target;
    bool arch_ok;
    int nqk;

    if(b == NULL || !rocke_ir_builder_ok(b))
    {
        return false;
    }
    if(spec == NULL || out == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "attn_tiled_3d config_from_spec: NULL spec/out");
        return false;
    }

    /* spec.__post_init__ validation first (validate()). */
    if(!rocke_unified_attention_3d_tiled_spec_validate(b, spec))
    {
        return false;
    }

    /* require_tiled_attention_arch(arch): NotImplementedError on reject. */
    target = rocke_archtarget_from_gfx(eff_arch);
    arch_ok = false;
    if(target != NULL)
    {
        if(rocke_attn3d_wide_k_available(target))
        {
            arch_ok = true;
        }
        else if(rocke_attn3d_streq(eff_arch, "gfx942") && rocke_attn3d_narrow_k_available(target))
        {
            arch_ok = true;
        }
    }
    if(!arch_ok)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_NOTIMPL,
                        "tiled attention requires either the wide-K MFMA atoms "
                        "(gfx950) or, on a narrow variant arch (gfx942), the "
                        "16x16x16 f16/bf16 atom; neither path is available on %s",
                        eff_arch);
        return false;
    }

    /* dtype gate fp16/bf16 (NotImplementedError). */
    if(!rocke_attn3d_streq(spec->dtype, "fp16") && !rocke_attn3d_streq(spec->dtype, "bf16"))
    {
        rocke_i_set_err(b, ROCKE_ERR_NOTIMPL, "tiled 3D kernel supports fp16/bf16");
        return false;
    }

    memset(out, 0, sizeof(*out));

    out->HD = spec->head_size;
    out->T = rocke_unified_attention_3d_tiled_spec_tile_size(spec);
    out->BS = spec->block_size;
    out->BLOCK_M = rocke_unified_attention_3d_tiled_spec_block_m(spec);
    out->BLOCK_Q = rocke_unified_attention_3d_tiled_spec_block_q(spec);
    out->NQK = rocke_unified_attention_3d_tiled_spec_num_queries_per_kv(spec);
    out->NUM_KV = spec->num_kv_heads;
    out->NUM_QH = spec->num_query_heads;
    out->NUM_SEG = spec->num_segments;
    out->SLIDING_WINDOW = spec->sliding_window;
    out->USE_SOFTCAP = spec->has_softcap;
    out->USE_SINKS = spec->use_sinks;
    out->USE_ALIBI = spec->use_alibi;
    out->USE_QQ_BIAS = spec->use_qq_bias;
    out->USE_INVARIANT_HOIST = spec->use_invariant_hoist;
    out->KV_FP8
        = (spec->kv_storage_dtype != NULL && rocke_attn3d_streq(spec->kv_storage_dtype, "fp8e4m3"));
    out->KV_BYTES = out->KV_FP8 ? 1 : 2;

    nqk = out->NQK;

    out->dtype = rocke_unified_attention_3d_tiled_spec_dtype_ir(spec);
    out->kv_io_dtype = out->KV_FP8 ? rocke_fp8e4m3() : out->dtype;

    /* narrow-atom loop trip counts (lines 266-271). */
    out->QK_K_STEP = 16;
    out->PV_K_STEP = 16;
    out->QK_K_ITERS = out->HD / out->QK_K_STEP;
    out->QK_N_TILES = out->T / ROCKE_ATTN3D_MFMA_N;
    out->PV_K_ITERS = out->T / out->PV_K_STEP;
    out->PV_N_TILES = out->HD / ROCKE_ATTN3D_MFMA_N;

    out->THREADS = 64;
    out->binary_search_iters = rocke_unified_attention_3d_tiled_spec_binary_search_iters(spec);

    if(out->PV_N_TILES < 0 || out->PV_N_TILES > 16)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "attn_tiled_3d: PV_N_TILES=%d outside [0,16]", out->PV_N_TILES);
        return false;
    }

    /* ---- async / wide / fp8 KV feed geometry (lines 549-667) ---- */
    out->ASYNC_LDS_DWORDS = 1;
    out->HALVES_PER_LANE = out->ASYNC_LDS_DWORDS * 2;
    out->KV_HALVES_PER_CALL = out->THREADS * out->HALVES_PER_LANE;

    /* assert (T * HD) % KV_HALVES_PER_CALL == 0 (line 556). */
    if(out->KV_HALVES_PER_CALL == 0 || (out->T * out->HD) % out->KV_HALVES_PER_CALL != 0)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "attn_tiled_3d: (T*HD)=%d not divisible by KV_HALVES_PER_CALL=%d",
                        out->T * out->HD,
                        out->KV_HALVES_PER_CALL);
        return false;
    }
    out->kv_calls_per_tile = (out->T * out->HD) / out->KV_HALVES_PER_CALL;
    out->bytes_per_call = out->KV_HALVES_PER_CALL * 2;
    out->kv_stride_blk_b = out->BS * out->NUM_KV * out->HD * out->KV_BYTES;
    out->kv_stride_tok_b = out->NUM_KV * out->HD * out->KV_BYTES;
    out->kv_stride_h_b = out->HD * out->KV_BYTES;
    out->bytes_per_buf = out->T * out->HD * 2;

    out->WIDE_ELEMS = 8;
    out->WIDE_OK = ((out->T * out->HD) % (out->THREADS * out->WIDE_ELEMS)) == 0;
    out->wide_chunks_per_thread
        = out->WIDE_OK ? (out->T * out->HD) / (out->THREADS * out->WIDE_ELEMS) : 0;

    out->fp8_elems_per_chunk = 8;
    out->fp8_total_chunks = (out->T * out->HD) / out->fp8_elems_per_chunk;
    if(out->KV_FP8)
    {
        /* assert fp8_total_chunks % THREADS == 0 (lines 663-666). */
        if(out->fp8_total_chunks % out->THREADS != 0)
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "fp8 loader: total chunks %d must be divisible by "
                            "THREADS=%d (T=%d, HD=%d)",
                            out->fp8_total_chunks,
                            out->THREADS,
                            out->T,
                            out->HD);
            return false;
        }
    }
    out->fp8_chunks_per_thread = out->fp8_total_chunks / out->THREADS;

    /* WIDE_KV = use_wide_kv_load && !KV_FP8 && WIDE_OK (line 703). */
    out->WIDE_KV = spec->use_wide_kv_load && !out->KV_FP8 && out->WIDE_OK;

    /* Q -> LDS feed (lines 438-439). */
    out->Q_VECS_PER_ROW = out->HD / 8;
    out->Q_VECS_PER_THREAD = (out->BLOCK_M * out->Q_VECS_PER_ROW) / out->THREADS;
    out->bm1_div_nqk = nqk > 0 ? (out->BLOCK_M - 1) / nqk : -1;

    /* T != BS path requires BS % T == 0 (line 582). */
    if(out->T != out->BS)
    {
        if(out->T == 0 || out->BS % out->T != 0)
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "3D tile_size_override must divide block_size (T=%d, BS=%d)",
                            out->T,
                            out->BS);
            return false;
        }
    }

    /* reduce-kernel-only fields stay 0 for the segment kind. */
    out->HALFS_PER_THREAD = 0;
    out->SEG_PER_LANE = 0;

    if(!rocke_ir_builder_ok(b))
    {
        return false;
    }
    return true;
}

/* ===================================================================== *
 *  config_from_spec -- reduce kernel (Python lines 1008-1057)
 * ===================================================================== */
bool rocke_gfx942_attn_tiled_3d_reduce_config_from_spec(
    rocke_ir_builder_t* b,
    const rocke_unified_attention_reduce_tiled_spec_t* spec,
    rocke_gfx942_attn_tiled_3d_config_t* out)
{
    if(b == NULL || !rocke_ir_builder_ok(b))
    {
        return false;
    }
    if(spec == NULL || out == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "attn_tiled_3d reduce config_from_spec: NULL spec/out");
        return false;
    }

    memset(out, 0, sizeof(*out));

    out->HD = spec->head_size;
    out->NUM_SEG = spec->num_segments;
    out->NUM_QH = spec->num_query_heads;
    out->NUM_KV = spec->num_kv_heads;
    out->dtype = rocke_unified_attention_reduce_tiled_spec_dtype_ir(spec);
    out->THREADS = 64;

    out->HALFS_PER_THREAD = out->HD / out->THREADS;
    /* assert HALFS_PER_THREAD * THREADS == HD (line 1015). */
    if(out->HALFS_PER_THREAD * out->THREADS != out->HD)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "attn_tiled_3d reduce: HALFS_PER_THREAD*THREADS != HD (HD=%d)",
                        out->HD);
        return false;
    }
    out->SEG_PER_LANE = (out->NUM_SEG + out->THREADS - 1) / out->THREADS;

    if(!rocke_ir_builder_ok(b))
    {
        return false;
    }
    return true;
}

/* ===================================================================== *
 *  rocke_gfx942_attention_tiled_3d_ctx_init
 *
 *  Zero-init the ctx, copy the spec slice + derive cfg + build C16_DIST.
 * ===================================================================== */
bool rocke_gfx942_attention_tiled_3d_ctx_init(
    rocke_gfx942_attention_tiled_3d_build_ctx_t* ctx,
    rocke_ir_builder_t* b,
    rocke_gfx942_attn_tiled_3d_kind_t kind,
    const rocke_unified_attention_3d_tiled_spec_t* spec,
    const rocke_unified_attention_reduce_tiled_spec_t* reduce_spec,
    const char* arch)
{
    const rocke_tile_distribution_encoding_t* enc;
    const rocke_mfma_atom_t* c_atom;

    if(ctx == NULL || b == NULL)
    {
        if(b != NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "attn_tiled_3d ctx_init: NULL ctx");
        }
        return false;
    }
    if(!rocke_ir_builder_ok(b))
    {
        return false;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->b = b;
    ctx->kind = kind;
    ctx->spec = NULL;
    ctx->reduce_spec = NULL;
    ctx->kernel = b->kernel;

    if(kind == ROCKE_GFX942_ATTN_TILED_3D_SEGMENT)
    {
        if(spec == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "attn_tiled_3d ctx_init: NULL segment spec");
            return false;
        }
        ctx->spec = spec;
        if(!rocke_gfx942_attn_tiled_3d_config_from_spec(b, spec, arch, &ctx->cfg))
        {
            return false;
        }
    }
    else /* ROCKE_GFX942_ATTN_TILED_3D_REDUCE */
    {
        if(reduce_spec == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "attn_tiled_3d ctx_init: NULL reduce spec");
            return false;
        }
        ctx->reduce_spec = reduce_spec;
        if(!rocke_gfx942_attn_tiled_3d_reduce_config_from_spec(b, reduce_spec, &ctx->cfg))
        {
            return false;
        }
    }

    /* _C16_DIST = make_static_tile_distribution(
     *                 make_c_warp_dstr_encoding(MfmaAtom.f16_16x16x16()))
     * (Python lines 74-76). The C accumulator layout is dtype/arch independent,
     * so the f16 atom drives it for both segment and reduce ctx. */
    c_atom = rocke_mfma_atom("f16", 16, 16, 16);
    if(c_atom == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "attn_tiled_3d: no f16 16x16x16 MFMA atom");
        return false;
    }
    enc = rocke_make_c_warp_dstr_encoding(b, c_atom);
    if(enc == NULL)
    {
        return false;
    }
    ctx->C16_DIST = rocke_make_static_tile_distribution(b, enc);
    if(ctx->C16_DIST == NULL)
    {
        return false;
    }

    if(!rocke_ir_builder_ok(b))
    {
        return false;
    }
    return true;
}

/* ===================================================================== *
 *  PUBLIC build entry -- segment kernel
 *
 *  Drives ctx_init then calls the phase fns in Python execution order
 *  (build_unified_attention_3d_tiled, lines 234-969). max_workgroup_size /
 *  waves_per_eu kernel attrs are set here (Python lines 276-278) before the
 *  param declarations.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_unified_attention_3d_tiled_gfx942(
    rocke_ir_builder_t* b, const rocke_unified_attention_3d_tiled_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        rocke_gfx942_attention_tiled_3d_build_ctx_t ctx;

        if(b == NULL)
        {
            return NULL;
        }
        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        if(spec == NULL)
        {
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "build_unified_attention_3d_tiled_gfx942: NULL spec");
            return NULL;
        }

        /* Name the kernel from spec.kernel_name() (Python b = IRBuilder(
         * spec.kernel_name())). The C entry reuses a caller-supplied builder. */
        if(b->kernel != NULL)
        {
            char name[256];
            if(rocke_unified_attention_3d_tiled_spec_kernel_name(spec, name, sizeof(name)) < 0)
            {
                rocke_i_set_err(
                    b,
                    ROCKE_ERR_VALUE,
                    "build_unified_attention_3d_tiled_gfx942: kernel_name encode failed");
                return NULL;
            }
            b->kernel->name = rocke_arena_strdup(&b->arena, name);
            if(b->kernel->name == NULL)
            {
                rocke_i_set_err(b, ROCKE_ERR_OOM, "attn_tiled_3d: OOM kernel name");
                return NULL;
            }
        }

        if(!rocke_gfx942_attention_tiled_3d_ctx_init(
               &ctx, b, ROCKE_GFX942_ATTN_TILED_3D_SEGMENT, spec, NULL, arch))
        {
            return NULL;
        }

        /* b.kernel.attrs["max_workgroup_size"] = THREADS (line 276). */
        if(b->kernel != NULL)
        {
            rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx.cfg.THREADS);
            /* waves_per_eu when set (lines 277-278). */
            if(spec->has_waves_per_eu)
            {
                rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
            }
        }

        /* ---- phase functions in Python execution order ----
         * declare params (281-329) -> prologue (331-483, which emits the early
         * seg_start_tile_pos>=seq_len zero-fill via emit_early_zero_fill in order)
         * -> Q->LDS (437-467) -> loop init + first K load (508-536, 721-724) ->
         * online-softmax scf.for (726-912) -> segment-workspace epilogue (914-967).
         */
        rocke_gfx942_attention_tiled_3d_declare_params(&ctx);
        /* emit_prologue already emits the early zero-fill AND the Q->LDS feed inline
         * (matching the single Python build function); calling emit_q_to_lds again
         * here would duplicate the Q->LDS loop. */
        rocke_gfx942_attention_tiled_3d_emit_prologue(&ctx);
        rocke_gfx942_attention_tiled_3d_emit_loop_init(&ctx);
        rocke_gfx942_attention_tiled_3d_emit_softmax_loop(&ctx);
        rocke_gfx942_attention_tiled_3d_emit_epilogue(&ctx);

        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel; /* return b.kernel (line 969) */
    });
}

/* ===================================================================== *
 *  PUBLIC build entry -- reduce kernel (lines 1002-1133)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_unified_attention_reduce_tiled_gfx942(
    rocke_ir_builder_t* b,
    const rocke_unified_attention_reduce_tiled_spec_t* spec,
    const char* arch)
{
    rocke_gfx942_attention_tiled_3d_build_ctx_t ctx;

    if(b == NULL)
    {
        return NULL;
    }
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    if(spec == NULL)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "build_unified_attention_reduce_tiled_gfx942: NULL spec");
        return NULL;
    }

    if(b->kernel != NULL)
    {
        char name[256];
        if(rocke_unified_attention_reduce_tiled_spec_kernel_name(spec, name, sizeof(name)) < 0)
        {
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "attn_reduce_tiled_gfx942: kernel_name encode failed");
            return NULL;
        }
        b->kernel->name = rocke_arena_strdup(&b->arena, name);
        if(b->kernel->name == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "attn_reduce_tiled_gfx942: OOM kernel name");
            return NULL;
        }
    }

    if(!rocke_gfx942_attention_tiled_3d_ctx_init(
           &ctx, b, ROCKE_GFX942_ATTN_TILED_3D_REDUCE, NULL, spec, arch))
    {
        return NULL;
    }

    /* b.kernel.attrs["max_workgroup_size"] = THREADS (line 1018). */
    if(b->kernel != NULL)
    {
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx.cfg.THREADS);
        if(spec->has_waves_per_eu)
        {
            rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
        }
    }

    /* ---- reduce phase functions in Python execution order ----
     * declare + prologue (1022-1058) -> pass1 max (1060-1083) -> pass2 combine
     * (1085-1104) -> pass3 normalize (1106-1131). */
    rocke_gfx942_attention_tiled_3d_reduce_declare_and_prologue(&ctx);
    rocke_gfx942_attention_tiled_3d_reduce_max_pass(&ctx);
    rocke_gfx942_attention_tiled_3d_reduce_combine_pass(&ctx);
    rocke_gfx942_attention_tiled_3d_reduce_normalize_pass(&ctx);

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel; /* return b.kernel (line 1133) */
}

/* ===================================================================== *
 *  build -> lower-to-.ll convenience (own/free internal IRBuilder)
 * ===================================================================== */
rocke_status_t rocke_build_unified_attention_3d_tiled_gfx942_lower_to_llvm(
    const rocke_unified_attention_3d_tiled_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;
    char name[256];

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        rocke_attn3d_set_err_buf(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }

    if(rocke_unified_attention_3d_tiled_spec_kernel_name(spec, name, sizeof(name)) < 0)
    {
        rocke_attn3d_set_err_buf(err, err_cap, "lower_to_llvm: kernel_name encode failed");
        return ROCKE_ERR_VALUE;
    }
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        rocke_attn3d_set_err_buf(err, err_cap, "lower_to_llvm: IRBuilder init failed");
        return ROCKE_ERR_OOM;
    }

    kernel = rocke_build_unified_attention_3d_tiled_gfx942(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_attn3d_set_err_buf(
            err,
            err_cap,
            (m != NULL && m[0] != '\0') ? m : "build_unified_attention_3d_tiled_gfx942 failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(
        kernel, flavor, (arch != NULL) ? arch : "gfx942", out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

rocke_status_t rocke_build_unified_attention_reduce_tiled_gfx942_lower_to_llvm(
    const rocke_unified_attention_reduce_tiled_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;
    char name[256];

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        rocke_attn3d_set_err_buf(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }

    if(rocke_unified_attention_reduce_tiled_spec_kernel_name(spec, name, sizeof(name)) < 0)
    {
        rocke_attn3d_set_err_buf(err, err_cap, "lower_to_llvm: kernel_name encode failed");
        return ROCKE_ERR_VALUE;
    }
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        rocke_attn3d_set_err_buf(err, err_cap, "lower_to_llvm: IRBuilder init failed");
        return ROCKE_ERR_OOM;
    }

    kernel = rocke_build_unified_attention_reduce_tiled_gfx942(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_attn3d_set_err_buf(
            err,
            err_cap,
            (m != NULL && m[0] != '\0') ? m : "build_unified_attention_reduce_tiled_gfx942 failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(
        kernel, flavor, (arch != NULL) ? arch : "gfx942", out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
