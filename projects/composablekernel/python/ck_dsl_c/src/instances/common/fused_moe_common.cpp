// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_fused_moe_fused_moe_common.c -- shared substrate for the chunked C99
 * port of ck_dsl/instances/common/fused_moe.py.
 *
 * SCOPE OF THIS TU (the shared substrate every MoE body TU links against):
 *   * FusedMoeSpec value-type glue:
 *       ckc_fused_moe_spec_default
 *       ckc_fused_moe_spec_total_pairs
 *       ckc_fused_moe_spec_elems_per_thread_hidden
 *       ckc_fused_moe_spec_elems_per_thread_inter
 *       ckc_fused_moe_spec_kernel_name
 *   * ckc_fused_moe_is_valid_spec (the full ordered Python is_valid_spec gate).
 *   * The three ported module-level helpers (instance_fused_moe_internal.h):
 *       ckc_moe_effective_vec     (_effective_vec)
 *       ckc_moe_chunk_distribution(_chunk_distribution)
 *       ckc_moe_silu_mul_f32      (_silu_mul_f32)
 *   * The five grids (ckc_moe_*_grid -> (total_pairs, 1, 1)).
 *   * The five SignatureBuilder manifests (ckc_moe_*_signature).
 *   * ckc_moe_fused_workspace_bytes.
 *
 * The per-phase prologue/body functions and the public build entries live in the
 * sibling body TUs; they are declared in instance_fused_moe_internal.h and
 * instance_fused_moe.h respectively, and call into the helpers below.
 *
 * Byte-identical builder-call sequence vs the Python: same op order / attrs.
 */
#include "ckc/instance_fused_moe.h"
#include "ckc/instance_fused_moe_internal.h"

#include <stdio.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/helper_ck_dsl.helpers.distribution.h"
#include "ckc/helper_ck_dsl.helpers.io.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/ir.h"

/* ===================================================================== *
 *  FusedMoeSpec value-type glue.
 * ===================================================================== */

/* FusedMoeSpec with the @dataclass defaults: dtype "f16", block_size 256,
 * vec 4, name "ck_dsl_fused_moe", bf16_accumulator false. The five shape fields
 * are zeroed (the caller fills tokens/experts/topk/hidden/intermediate). */
ckc_fused_moe_spec_t ckc_fused_moe_spec_default(void)
{
    ckc_fused_moe_spec_t spec;
    spec.tokens           = 0;
    spec.experts          = 0;
    spec.topk             = 0;
    spec.hidden           = 0;
    spec.intermediate     = 0;
    spec.dtype            = "f16";
    spec.block_size       = 256;
    spec.vec              = 4;
    spec.name             = "ck_dsl_fused_moe";
    spec.bf16_accumulator = false;
    return spec;
}

/* @property total_pairs -> tokens * topk. */
int ckc_fused_moe_spec_total_pairs(const ckc_fused_moe_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return spec->tokens * spec->topk;
}

/* @property elems_per_thread_hidden -> hidden // block_size. */
int ckc_fused_moe_spec_elems_per_thread_hidden(const ckc_fused_moe_spec_t* spec)
{
    if(spec == NULL || spec->block_size == 0)
    {
        return 0;
    }
    return spec->hidden / spec->block_size;
}

/* @property elems_per_thread_inter -> intermediate // block_size. */
int ckc_fused_moe_spec_elems_per_thread_inter(const ckc_fused_moe_spec_t* spec)
{
    if(spec == NULL || spec->block_size == 0)
    {
        return 0;
    }
    return spec->intermediate / spec->block_size;
}

/* kernel_name(phase):
 *   kernel_name_join(name, phase, f"T{tokens}", f"E{experts}", f"K{topk}",
 *                    f"H{hidden}", f"I{intermediate}", dtype,
 *                    f"b{block_size}", f"v{vec}") */
ckc_status_t ckc_fused_moe_spec_kernel_name(const ckc_fused_moe_spec_t* spec,
                                            const char* phase,
                                            char* out,
                                            size_t out_cap)
{
    char t_buf[32];
    char e_buf[32];
    char k_buf[32];
    char h_buf[32];
    char i_buf[32];
    char b_buf[32];
    char v_buf[32];
    const char* parts[9];

    if(spec == NULL || phase == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }

    snprintf(t_buf, sizeof(t_buf), "T%d", spec->tokens);
    snprintf(e_buf, sizeof(e_buf), "E%d", spec->experts);
    snprintf(k_buf, sizeof(k_buf), "K%d", spec->topk);
    snprintf(h_buf, sizeof(h_buf), "H%d", spec->hidden);
    snprintf(i_buf, sizeof(i_buf), "I%d", spec->intermediate);
    snprintf(b_buf, sizeof(b_buf), "b%d", spec->block_size);
    snprintf(v_buf, sizeof(v_buf), "v%d", spec->vec);

    parts[0] = phase;
    parts[1] = t_buf;
    parts[2] = e_buf;
    parts[3] = k_buf;
    parts[4] = h_buf;
    parts[5] = i_buf;
    parts[6] = spec->dtype;
    parts[7] = b_buf;
    parts[8] = v_buf;

    return ckc_kernel_name_join(spec->name, parts, 9, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec) -> (ok, why). Mirrors the Python gate exactly, in
 *  the same order; writes the Python-matching message into `reason` on
 *  reject, "ok" on accept.
 * ===================================================================== */
bool ckc_fused_moe_is_valid_spec(const ckc_fused_moe_spec_t* spec, char* reason, size_t reason_cap)
{
    const char* dtype;

    if(spec == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "null spec");
        }
        return false;
    }

    if(spec->tokens <= 0 || spec->experts <= 0 || spec->topk <= 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "tokens / experts / topk must be > 0 (got %d, %d, %d)",
                     spec->tokens,
                     spec->experts,
                     spec->topk);
        }
        return false;
    }
    if(spec->hidden <= 0 || spec->intermediate <= 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "hidden / intermediate must be > 0 (got %d, %d)",
                     spec->hidden,
                     spec->intermediate);
        }
        return false;
    }
    if(spec->topk > spec->experts)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "topk (%d) > experts (%d)", spec->topk, spec->experts);
        }
        return false;
    }
    if(spec->block_size != 64 && spec->block_size != 128 && spec->block_size != 256 &&
       spec->block_size != 512 && spec->block_size != 1024)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "block_size %d not in {64..1024}", spec->block_size);
        }
        return false;
    }
    if(spec->vec != 2 && spec->vec != 4 && spec->vec != 8)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "vec %d not in {2, 4, 8}", spec->vec);
        }
        return false;
    }
    dtype = spec->dtype;
    if(dtype == NULL ||
       (strcmp(dtype, "f16") != 0 && strcmp(dtype, "fp16") != 0 && strcmp(dtype, "bf16") != 0))
    {
        if(reason != NULL && reason_cap > 0)
        {
            /* Python: f"unsupported dtype {spec.dtype!r}" -- repr-quoted. */
            snprintf(reason, reason_cap, "unsupported dtype '%s'", dtype != NULL ? dtype : "None");
        }
        return false;
    }
    if(spec->hidden % spec->vec != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(
                reason, reason_cap, "hidden %d not divisible by vec %d", spec->hidden, spec->vec);
        }
        return false;
    }
    if(spec->intermediate % spec->vec != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "intermediate %d not divisible by vec %d",
                     spec->intermediate,
                     spec->vec);
        }
        return false;
    }
    if(spec->hidden % spec->block_size != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "hidden %d not divisible by block_size %d; v1 requires one CTA "
                     "per bucket row to cover the full hidden vector",
                     spec->hidden,
                     spec->block_size);
        }
        return false;
    }
    if(spec->intermediate % spec->block_size != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "intermediate %d not divisible by block_size %d; same "
                     "one-CTA-per-row constraint as hidden",
                     spec->intermediate,
                     spec->block_size);
        }
        return false;
    }

    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "ok");
    }
    return true;
}

/* ===================================================================== *
 *  SHARED MODULE-LEVEL HELPERS.
 * ===================================================================== */

/* _effective_vec(spec_vec, block_size, n):
 *   ev = min(spec_vec, 8)
 *   while ev > 1 and (n % (block_size * ev) != 0): ev //= 2
 *   return ev */
int ckc_moe_effective_vec(int spec_vec, int block_size, int n)
{
    int ev = spec_vec < 8 ? spec_vec : 8;
    while(ev > 1 && (n % (block_size * ev) != 0))
    {
        ev /= 2;
    }
    return ev;
}

/* _chunk_distribution(block_size, vec):
 *   TileDistributionEncoding(Hs=((block_size, vec),),
 *                            Ps2RHs_major=((1,),), Ps2RHs_minor=((0,),),
 *                            Ys2RHs_major=(1,), Ys2RHs_minor=(1,))
 *   -> make_static_tile_distribution(encoding) */
const ckc_tile_distribution_t*
ckc_moe_chunk_distribution(ckc_ir_builder_t* b, int block_size, int vec)
{
    int h_levels[2];
    ckc_h_row_t hs[1];
    int p_major[1];
    int p_minor[1];
    ckc_p_seq_t ps[1];
    int ys_major[1];
    int ys_minor[1];
    ckc_tile_distribution_encoding_t* encoding;

    if(b == NULL)
    {
        return NULL;
    }

    /* Hs = ((block_size, vec),) */
    h_levels[0]  = block_size;
    h_levels[1]  = vec;
    hs[0].levels = h_levels;
    hs[0].count  = 2;

    /* Ps2RHs_major = ((1,),), Ps2RHs_minor = ((0,),) */
    p_major[0]  = 1;
    p_minor[0]  = 0;
    ps[0].major = p_major;
    ps[0].minor = p_minor;
    ps[0].count = 1;

    /* Ys2RHs_major = (1,), Ys2RHs_minor = (1,) */
    ys_major[0] = 1;
    ys_minor[0] = 1;

    encoding = ckc_make_tile_distribution_encoding(b,
                                                   /*Rs*/ NULL,
                                                   /*num_R*/ 0,
                                                   hs,
                                                   /*num_X*/ 1,
                                                   ps,
                                                   /*num_P*/ 1,
                                                   ys_major,
                                                   ys_minor,
                                                   /*num_Y*/ 1);
    if(encoding == NULL)
    {
        return NULL;
    }
    return ckc_make_static_tile_distribution(b, encoding);
}

/* _silu_mul_f32(b, g, u, one_f32, c_neg_log2e):
 *   sig  = rcp(fadd(one_f32, exp2(fmul(c_neg_log2e, g))))
 *   silu = fmul(g, sig)
 *   return fmul(silu, u) */
ckc_value_t* ckc_moe_silu_mul_f32(ckc_ir_builder_t* b,
                                  ckc_value_t* g,
                                  ckc_value_t* u,
                                  ckc_value_t* one_f32,
                                  ckc_value_t* c_neg_log2e)
{
    ckc_value_t* sig;
    ckc_value_t* silu;
    if(b == NULL)
    {
        return NULL;
    }
    sig  = ckc_b_rcp(b, ckc_b_fadd(b, one_f32, ckc_b_exp2(b, ckc_b_fmul(b, c_neg_log2e, g))));
    silu = ckc_b_fmul(b, g, sig);
    return ckc_b_fmul(b, silu, u);
}

/* ===================================================================== *
 *  GRIDS -- every phase returns (total_pairs, 1, 1).
 * ===================================================================== */

static void ckc_moe_fill_grid(const ckc_fused_moe_spec_t* spec, int out[3])
{
    if(out == NULL)
    {
        return;
    }
    out[0] = spec != NULL ? (spec->tokens * spec->topk) : 0;
    out[1] = 1;
    out[2] = 1;
}

void ckc_moe_gather_grid(const ckc_fused_moe_spec_t* spec, int out[3])
{
    ckc_moe_fill_grid(spec, out);
}

void ckc_moe_silu_mul_grid(const ckc_fused_moe_spec_t* spec, int out[3])
{
    ckc_moe_fill_grid(spec, out);
}

void ckc_moe_silu_mul_packed_grid(const ckc_fused_moe_spec_t* spec, int out[3])
{
    ckc_moe_fill_grid(spec, out);
}

void ckc_moe_static_scatter_gather_grid(const ckc_fused_moe_spec_t* spec, int out[3])
{
    ckc_moe_fill_grid(spec, out);
}

void ckc_moe_topk_weighted_reduce_grid(const ckc_fused_moe_spec_t* spec, int out[3])
{
    ckc_moe_fill_grid(spec, out);
}

/* ===================================================================== *
 *  SIGNATURES (manifests). Each builds the SignatureBuilder chain, then
 *  copies the resulting entries into the caller's out[] (cap out_cap) and
 *  sets *out_count.
 * ===================================================================== */

/* Copy the built entries into out[]/out_count, mirroring SignatureBuilder
 * .build() followed by the caller materialising the list into a fixed array. */
static ckc_status_t ckc_moe_sig_emit(const ckc_signature_builder_t* sb,
                                     ckc_sig_entry_t* out,
                                     size_t out_cap,
                                     size_t* out_count)
{
    const ckc_sig_entry_t* items;
    size_t count;
    size_t i;
    ckc_status_t st;

    st = ckc_signature_builder_build(sb, &items, &count);
    if(st != CKC_OK)
    {
        return st;
    }
    if(count > out_cap)
    {
        return CKC_ERR_VALUE;
    }
    for(i = 0; i < count; ++i)
    {
        out[i] = items[i];
    }
    *out_count = count;
    return CKC_OK;
}

ckc_status_t ckc_moe_gather_signature(struct ckc_arena* arena,
                                      const ckc_fused_moe_spec_t* spec,
                                      struct ckc_sig_entry* out,
                                      size_t out_cap,
                                      size_t* out_count)
{
    ckc_signature_builder_t sb;
    ckc_status_t st;

    if(arena == NULL || spec == NULL || out == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }
    st = ckc_signature_builder_init(&sb, arena);
    if(st != CKC_OK)
    {
        return st;
    }
    ckc_signature_builder_ptr(&sb, "X", spec->dtype, NULL);
    ckc_signature_builder_ptr(&sb, "SortedTokenIds", "i32", NULL);
    ckc_signature_builder_ptr(&sb, "GroupedInput", spec->dtype, NULL);
    ckc_signature_builder_scalar(&sb, "tokens", "i32");
    ckc_signature_builder_scalar(&sb, "hidden", "i32");
    return ckc_moe_sig_emit(&sb, out, out_cap, out_count);
}

ckc_status_t ckc_moe_silu_mul_signature(struct ckc_arena* arena,
                                        const ckc_fused_moe_spec_t* spec,
                                        struct ckc_sig_entry* out,
                                        size_t out_cap,
                                        size_t* out_count)
{
    ckc_signature_builder_t sb;
    ckc_status_t st;

    if(arena == NULL || spec == NULL || out == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }
    st = ckc_signature_builder_init(&sb, arena);
    if(st != CKC_OK)
    {
        return st;
    }
    ckc_signature_builder_ptr(&sb, "GateOut", spec->dtype, NULL);
    ckc_signature_builder_ptr(&sb, "UpOut", spec->dtype, NULL);
    ckc_signature_builder_ptr(&sb, "Hidden", spec->dtype, NULL);
    ckc_signature_builder_scalar(&sb, "total_pairs", "i32");
    ckc_signature_builder_scalar(&sb, "intermediate", "i32");
    return ckc_moe_sig_emit(&sb, out, out_cap, out_count);
}

ckc_status_t ckc_moe_silu_mul_packed_signature(struct ckc_arena* arena,
                                               const ckc_fused_moe_spec_t* spec,
                                               struct ckc_sig_entry* out,
                                               size_t out_cap,
                                               size_t* out_count)
{
    ckc_signature_builder_t sb;
    ckc_status_t st;

    if(arena == NULL || spec == NULL || out == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }
    st = ckc_signature_builder_init(&sb, arena);
    if(st != CKC_OK)
    {
        return st;
    }
    ckc_signature_builder_ptr(&sb, "GateUp", spec->dtype, NULL);
    ckc_signature_builder_ptr(&sb, "Hidden", spec->dtype, NULL);
    ckc_signature_builder_scalar(&sb, "total_pairs", "i32");
    ckc_signature_builder_scalar(&sb, "intermediate", "i32");
    return ckc_moe_sig_emit(&sb, out, out_cap, out_count);
}

ckc_status_t ckc_moe_static_scatter_gather_signature(struct ckc_arena* arena,
                                                     const ckc_fused_moe_spec_t* spec,
                                                     struct ckc_sig_entry* out,
                                                     size_t out_cap,
                                                     size_t* out_count)
{
    ckc_signature_builder_t sb;
    ckc_status_t st;

    if(arena == NULL || spec == NULL || out == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }
    st = ckc_signature_builder_init(&sb, arena);
    if(st != CKC_OK)
    {
        return st;
    }
    ckc_signature_builder_ptr(&sb, "TopkIds", "i32", NULL);
    ckc_signature_builder_ptr(&sb, "TopkWeights", "f32", NULL);
    ckc_signature_builder_ptr(&sb, "Counter", "i32", NULL);
    ckc_signature_builder_ptr(&sb, "X", spec->dtype, NULL);
    ckc_signature_builder_ptr(&sb, "SortedTokenIds", "i32", NULL);
    ckc_signature_builder_ptr(&sb, "SortedWeights", "f32", NULL);
    ckc_signature_builder_ptr(&sb, "GroupedInput", spec->dtype, NULL);
    ckc_signature_builder_scalar(&sb, "tokens", "i32");
    ckc_signature_builder_scalar(&sb, "topk", "i32");
    ckc_signature_builder_scalar(&sb, "num_experts", "i32");
    ckc_signature_builder_scalar(&sb, "hidden", "i32");
    ckc_signature_builder_scalar(&sb, "slot_size", "i32");
    return ckc_moe_sig_emit(&sb, out, out_cap, out_count);
}

ckc_status_t ckc_moe_topk_weighted_reduce_signature(struct ckc_arena* arena,
                                                    const ckc_fused_moe_spec_t* spec,
                                                    struct ckc_sig_entry* out,
                                                    size_t out_cap,
                                                    size_t* out_count)
{
    ckc_signature_builder_t sb;
    ckc_status_t st;

    if(arena == NULL || spec == NULL || out == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }
    st = ckc_signature_builder_init(&sb, arena);
    if(st != CKC_OK)
    {
        return st;
    }
    ckc_signature_builder_ptr(&sb, "DownOut", spec->dtype, NULL);
    ckc_signature_builder_ptr(&sb, "SortedTokenIds", "i32", NULL);
    ckc_signature_builder_ptr(&sb, "SortedWeights", "f32", NULL);
    ckc_signature_builder_ptr(&sb, "Y", "f32", NULL);
    ckc_signature_builder_scalar(&sb, "total_pairs", "i32");
    ckc_signature_builder_scalar(&sb, "hidden", "i32");
    ckc_signature_builder_scalar(&sb, "tokens", "i32");
    return ckc_moe_sig_emit(&sb, out, out_cap, out_count);
}

/* ===================================================================== *
 *  WORKSPACE
 *
 *  moe_fused_workspace_bytes(spec):
 *    elem_bytes = 2; grouped + gate + up + hidden_buf + down where
 *    grouped/down = total_pairs*hidden*2 and
 *    gate/up/hidden_buf = total_pairs*intermediate*2. Returned as int64.
 * ===================================================================== */
long long ckc_moe_fused_workspace_bytes(const ckc_fused_moe_spec_t* spec)
{
    long long elem_bytes = 2; /* f16 / bf16 */
    long long total_pairs;
    long long grouped;
    long long gate;
    long long up;
    long long hidden_buf;
    long long down;

    if(spec == NULL)
    {
        return 0;
    }
    total_pairs = (long long)spec->tokens * (long long)spec->topk;
    grouped     = total_pairs * (long long)spec->hidden * elem_bytes;
    gate        = total_pairs * (long long)spec->intermediate * elem_bytes;
    up          = total_pairs * (long long)spec->intermediate * elem_bytes;
    hidden_buf  = total_pairs * (long long)spec->intermediate * elem_bytes;
    down        = total_pairs * (long long)spec->hidden * elem_bytes;
    return grouped + gate + up + hidden_buf + down;
}
