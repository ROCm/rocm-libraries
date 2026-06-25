// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of the append-KV kernel instance builder
 * ck_dsl/instances/common/fmha_appendkv.py.
 *
 * See ckc/instance_fmha_appendkv.h for the public symbol map.
 *
 * appendkv is a pure vectorised KV-cache scatter (optionally fused with rotary
 * on K): no MFMA atoms, no LDS. The build reproduces the Python builder-call
 * sequence byte-for-byte:
 *   - kernel attr max_workgroup_size = block_size,
 *   - the param() declaration order (K_new, V_new, K_cache, V_cache, seqlen_kv,
 *     cu_seqlens_new, [cos_table, sin_table], total_new_q, batch, four strides),
 *   - the grid decode (tid, bid, kv_head_idx, new_token),
 *   - the in_bounds scf.if guard wrapping the per-thread body,
 *   - the per-batch sequence scan, address math, and the K / V copy paths
 *     (no-rotary vector copy, rotary-half two-chunk, rotary-interleaved chunk,
 *     scalar-fallback pair loop).
 *
 * Nested Python expressions are sequenced explicitly because C
 * argument-evaluation order is unspecified, so the emitted SSA numbering stays
 * identical to the Python.
 */
#include "ckc/instance_fmha_appendkv.h"

#include <stdio.h> /* snprintf */
#include <string.h> /* memset, memcpy, strcmp, strlen */

#include "ckc/arch_target.h" /* arch lookup        */
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "ckc/helper_ck_dsl.helpers.io.h" /* io_ir_type, copies */
#include "ckc/helper_ck_dsl.helpers.rotary.h" /* rotary helpers     */
#include "ckc/helper_ck_dsl.helpers.spec.h" /* ceil_div_grid, sig */
#include "ckc/ir_internal.h" /* ckc_i_set_err      */

/* CK Tile's appendkv default policy reads 16 bytes at a time => _VEC = 8 for
 * f16 / bf16 (the same shape buffer_load_dwordx4 lowers to). */
#define CKC_FMHA_APPENDKV_VEC 8

/* ===================================================================== *
 *  small helpers
 * ===================================================================== */

static void ckc_appendkv_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    ckc_spec_set_reason(reason, reason_cap, msg);
}

/* ===================================================================== *
 *  FmhaAppendKvSpec helpers
 * ===================================================================== */

ckc_fmha_appendkv_spec_t ckc_fmha_appendkv_spec_default(ckc_fmha_common_spec_t common, int batch)
{
    ckc_fmha_appendkv_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.common = common;
    spec.batch = batch;
    spec.has_rotary = false;
    spec.block_size = 256; /* dataclass default */
    spec.name = NULL; /* => "ck_dsl_fmha_appendkv" */
    return spec;
}

const char* ckc_fmha_appendkv_spec_name(const ckc_fmha_appendkv_spec_t* spec)
{
    if(spec == NULL || spec->name == NULL)
    {
        return "ck_dsl_fmha_appendkv";
    }
    return spec->name;
}

/* kernel_name():
 *   kernel_name_join(name, f"H{H}", f"HK{HK}", dtype, f"B{batch}",
 *                    "rope"|"norope", f"b{block_size}") */
ckc_status_t
    ckc_fmha_appendkv_kernel_name(const ckc_fmha_appendkv_spec_t* spec, char* out, size_t out_cap)
{
    const char* parts[6];
    char h_buf[32];
    char hk_buf[32];
    char b_buf[32];
    char bs_buf[32];
    const ckc_fmha_shape_t* s;

    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    s = &spec->common.shape;

    snprintf(h_buf, sizeof(h_buf), "H%d", s->head_size);
    snprintf(hk_buf, sizeof(hk_buf), "HK%d", s->num_kv_heads);
    snprintf(b_buf, sizeof(b_buf), "B%d", spec->batch);
    snprintf(bs_buf, sizeof(bs_buf), "b%d", spec->block_size);

    parts[0] = h_buf;
    parts[1] = hk_buf;
    parts[2] = spec->common.dtype;
    parts[3] = b_buf;
    parts[4] = spec->has_rotary ? "rope" : "norope";
    parts[5] = bs_buf;

    return ckc_kernel_name_join(
        ckc_fmha_appendkv_spec_name(spec), parts, 6, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch) -> (ok, reason)
 * ===================================================================== */

bool ckc_fmha_appendkv_is_valid_spec(const ckc_fmha_appendkv_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap)
{
    const ckc_arch_target_t* target;
    int max_tpb;
    char buf[200];
    ckc_arena_t arena;
    const char* why = NULL;
    bool ok;

    if(spec == NULL)
    {
        ckc_appendkv_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* try: target = ArchTarget.from_gfx(arch) except KeyError as e:
     *     return False, str(e) */
    target = ckc_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf, sizeof(buf), "'%s'", arch);
        ckc_appendkv_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* ok, why = validate_common_spec(spec.common); if not ok: return False, why */
    if(ckc_arena_init(&arena, 4096) != 0)
    {
        ckc_appendkv_set_reason(reason, reason_cap, "arena init failed");
        return false;
    }
    ok = ckc_fmha_validate_common_spec(&arena, &spec->common, &why);
    if(!ok)
    {
        ckc_appendkv_set_reason(reason, reason_cap, why != NULL ? why : "invalid common spec");
        ckc_arena_destroy(&arena);
        return false;
    }
    ckc_arena_destroy(&arena);

    /* if spec.batch <= 0: return False, f"batch must be > 0 (got {batch})" */
    if(spec->batch <= 0)
    {
        snprintf(buf, sizeof(buf), "batch must be > 0 (got %d)", spec->batch);
        ckc_appendkv_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if rotary is not None and rotary.head_size != common.shape.head_size: ... */
    if(spec->has_rotary && spec->rotary.head_size != spec->common.shape.head_size)
    {
        snprintf(buf,
                 sizeof(buf),
                 "rotary head_size (%d) != common head_size (%d)",
                 spec->rotary.head_size,
                 spec->common.shape.head_size);
        ckc_appendkv_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if block_size > target.max_threads_per_block: ... */
    max_tpb = ckc_arch_max_threads_per_block(target);
    if(spec->block_size > max_tpb)
    {
        snprintf(buf,
                 sizeof(buf),
                 "block_size %d > max_threads_per_block %d on %s",
                 spec->block_size,
                 max_tpb,
                 arch);
        ckc_appendkv_set_reason(reason, reason_cap, buf);
        return false;
    }

    ckc_appendkv_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  _copy_row_vec -- vector_row_copy with vec_bytes = _VEC*2 (the f16/bf16
 *  16-byte default). Mirrors fmha_appendkv._copy_row_vec.
 * ===================================================================== */

static void ckc_appendkv_copy_row_vec(ckc_ir_builder_t* b,
                                      int H,
                                      const char* dtype,
                                      ckc_value_t* src_ptr,
                                      ckc_value_t* dst_ptr,
                                      ckc_value_t* src_row_base,
                                      ckc_value_t* dst_row_base)
{
    ckc_b_vector_row_copy(
        b, src_ptr, dst_ptr, src_row_base, dst_row_base, H, dtype, CKC_FMHA_APPENDKV_VEC * 2);
}

/* ===================================================================== *
 *  _appendkv_copy_k -- K-cache copy, optionally fused with rotary.
 * ===================================================================== */

static ckc_status_t ckc_appendkv_copy_k(ckc_ir_builder_t* b,
                                        const ckc_fmha_appendkv_spec_t* spec,
                                        int H,
                                        const char* dtype,
                                        ckc_value_t* K_new,
                                        ckc_value_t* K_cache,
                                        ckc_value_t* cos_table,
                                        ckc_value_t* sin_table,
                                        ckc_value_t* dst_pos,
                                        ckc_value_t* in_row_base,
                                        ckc_value_t* cache_row_base)
{
    const ckc_type_t* ty;
    int elem_bytes = 2;
    ckc_rotary_layout_t layout;
    int pair_count;
    const int VEC = CKC_FMHA_APPENDKV_VEC;

    /* if spec.rotary is None: straight vector copy; return */
    if(!spec->has_rotary)
    {
        ckc_appendkv_copy_row_vec(b, H, dtype, K_new, K_cache, in_row_base, cache_row_base);
        return CKC_OK;
    }

    ty = ckc_io_ir_type(dtype);
    layout = spec->rotary.layout;
    pair_count = ckc_rotary_spec_pair_count(&spec->rotary);

    /* if layout == "half" and (H // 2) % _VEC == 0: */
    if(layout == CKC_ROTARY_HALF && (H / 2) % VEC == 0)
    {
        int half = H / 2;
        int n_chunks = half / VEC;
        int c;
        for(c = 0; c < n_chunks; ++c)
        {
            int d_lo = c * VEC;
            int d_hi = half + c * VEC;
            ckc_value_t* lo_vec;
            ckc_value_t* hi_vec;
            ckc_value_t* out_lo_f32[CKC_FMHA_APPENDKV_VEC];
            ckc_value_t* out_hi_f32[CKC_FMHA_APPENDKV_VEC];
            ckc_value_t* lo_pack;
            ckc_value_t* hi_pack;
            int j;

            lo_vec = ckc_b_global_load_vN(b,
                                          K_new,
                                          ckc_b_add(b, in_row_base, ckc_b_const_i32(b, d_lo)),
                                          ty,
                                          VEC,
                                          VEC * elem_bytes);
            hi_vec = ckc_b_global_load_vN(b,
                                          K_new,
                                          ckc_b_add(b, in_row_base, ckc_b_const_i32(b, d_hi)),
                                          ty,
                                          VEC,
                                          VEC * elem_bytes);
            for(j = 0; j < VEC; ++j)
            {
                int pair = c * VEC + j;
                ckc_value_t* cos_v = NULL;
                ckc_value_t* sin_v = NULL;
                ckc_value_t* lo_f32;
                ckc_value_t* hi_f32;
                ckc_value_t* new_lo = NULL;
                ckc_value_t* new_hi = NULL;

                ckc_rotary_load_cos_sin(b,
                                        cos_table,
                                        sin_table,
                                        dst_pos,
                                        ckc_b_const_i32(b, pair),
                                        &spec->rotary,
                                        &cos_v,
                                        &sin_v);
                lo_f32 = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, lo_vec, j));
                hi_f32 = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, hi_vec, j));
                ckc_rotary_apply_pair_f32(b, lo_f32, hi_f32, cos_v, sin_v, &new_lo, &new_hi);
                out_lo_f32[j] = new_lo;
                out_hi_f32[j] = new_hi;
            }
            lo_pack = ckc_b_pack_f32_to(b, out_lo_f32, VEC, dtype);
            hi_pack = ckc_b_pack_f32_to(b, out_hi_f32, VEC, dtype);
            ckc_b_global_store_vN(b,
                                  K_cache,
                                  ckc_b_add(b, cache_row_base, ckc_b_const_i32(b, d_lo)),
                                  lo_pack,
                                  VEC,
                                  VEC * elem_bytes);
            ckc_b_global_store_vN(b,
                                  K_cache,
                                  ckc_b_add(b, cache_row_base, ckc_b_const_i32(b, d_hi)),
                                  hi_pack,
                                  VEC,
                                  VEC * elem_bytes);
        }
        return CKC_OK;
    }

    /* if layout == "interleaved" and H % _VEC == 0 and _VEC % 2 == 0: */
    if(layout == CKC_ROTARY_INTERLEAVED && H % VEC == 0 && VEC % 2 == 0)
    {
        int pairs_per_chunk = VEC / 2;
        int n_chunks = H / VEC;
        int c;
        for(c = 0; c < n_chunks; ++c)
        {
            int d = c * VEC;
            ckc_value_t* vec;
            ckc_value_t* out_f32[CKC_FMHA_APPENDKV_VEC];
            ckc_value_t* packed;
            int j;

            vec = ckc_b_global_load_vN(b,
                                       K_new,
                                       ckc_b_add(b, in_row_base, ckc_b_const_i32(b, d)),
                                       ty,
                                       VEC,
                                       VEC * elem_bytes);
            for(j = 0; j < pairs_per_chunk; ++j)
            {
                int pair = c * pairs_per_chunk + j;
                ckc_value_t* cos_v = NULL;
                ckc_value_t* sin_v = NULL;
                ckc_value_t* lo_f32;
                ckc_value_t* hi_f32;
                ckc_value_t* new_lo = NULL;
                ckc_value_t* new_hi = NULL;

                ckc_rotary_load_cos_sin(b,
                                        cos_table,
                                        sin_table,
                                        dst_pos,
                                        ckc_b_const_i32(b, pair),
                                        &spec->rotary,
                                        &cos_v,
                                        &sin_v);
                lo_f32 = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, vec, 2 * j));
                hi_f32 = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, vec, 2 * j + 1));
                ckc_rotary_apply_pair_f32(b, lo_f32, hi_f32, cos_v, sin_v, &new_lo, &new_hi);
                out_f32[2 * j] = new_lo;
                out_f32[2 * j + 1] = new_hi;
            }
            packed = ckc_b_pack_f32_to(b, out_f32, VEC, dtype);
            ckc_b_global_store_vN(b,
                                  K_cache,
                                  ckc_b_add(b, cache_row_base, ckc_b_const_i32(b, d)),
                                  packed,
                                  VEC,
                                  VEC * elem_bytes);
        }
        return CKC_OK;
    }

    /* Catch-all fallback: scalar pair loop. */
    {
        int pair;
        for(pair = 0; pair < pair_count; ++pair)
        {
            int lo_d = 0;
            int hi_d = 0;
            ckc_value_t* k_lo;
            ckc_value_t* k_hi;
            ckc_value_t* cos_v = NULL;
            ckc_value_t* sin_v = NULL;
            ckc_value_t* new_lo = NULL;
            ckc_value_t* new_hi = NULL;

            ckc_rotary_pair_indices(&spec->rotary, pair, &lo_d, &hi_d);
            k_lo = ckc_b_load_scalar_as_f32(
                b, K_new, ckc_b_add(b, in_row_base, ckc_b_const_i32(b, lo_d)), dtype);
            k_hi = ckc_b_load_scalar_as_f32(
                b, K_new, ckc_b_add(b, in_row_base, ckc_b_const_i32(b, hi_d)), dtype);
            ckc_rotary_load_cos_sin(b,
                                    cos_table,
                                    sin_table,
                                    dst_pos,
                                    ckc_b_const_i32(b, pair),
                                    &spec->rotary,
                                    &cos_v,
                                    &sin_v);
            ckc_rotary_apply_pair_f32(b, k_lo, k_hi, cos_v, sin_v, &new_lo, &new_hi);
            ckc_b_store_scalar_from_f32(
                b, K_cache, ckc_b_add(b, cache_row_base, ckc_b_const_i32(b, lo_d)), new_lo, dtype);
            ckc_b_store_scalar_from_f32(
                b, K_cache, ckc_b_add(b, cache_row_base, ckc_b_const_i32(b, hi_d)), new_hi, dtype);
        }
    }
    return CKC_OK;
}

/* ===================================================================== *
 *  _appendkv_copy_v -- always a plain row copy (no rotary on V).
 * ===================================================================== */

static void ckc_appendkv_copy_v(ckc_ir_builder_t* b,
                                int H,
                                const char* dtype,
                                ckc_value_t* V_new,
                                ckc_value_t* V_cache,
                                ckc_value_t* in_row_base,
                                ckc_value_t* cache_row_base)
{
    ckc_appendkv_copy_row_vec(b, H, dtype, V_new, V_cache, in_row_base, cache_row_base);
}

/* ===================================================================== *
 *  _appendkv_body -- per-thread sequence lookup + address math + K/V copy.
 * ===================================================================== */

static void ckc_appendkv_body(ckc_ir_builder_t* b,
                              const ckc_fmha_appendkv_spec_t* spec,
                              int H,
                              const char* dtype,
                              ckc_value_t* new_token,
                              ckc_value_t* kv_head_idx,
                              ckc_value_t* K_new,
                              ckc_value_t* V_new,
                              ckc_value_t* K_cache,
                              ckc_value_t* V_cache,
                              ckc_value_t* seqlen_kv,
                              ckc_value_t* cu_seqlens_new,
                              ckc_value_t* cos_table,
                              ckc_value_t* sin_table,
                              ckc_value_t* stride_in_token,
                              ckc_value_t* stride_in_head,
                              ckc_value_t* stride_cache_token,
                              ckc_value_t* stride_cache_head)
{
    ckc_value_t* seq;
    ckc_value_t* seq_idx;
    ckc_value_t* cu_base;
    ckc_value_t* local_new;
    ckc_value_t* seqlen_cur;
    ckc_value_t* dst_pos;
    ckc_value_t* in_row_base;
    ckc_value_t* cache_row_base;
    int i;

    /* seq = b.const_i32(0)
     * for i in range(batch):
     *     cuq_next = b.global_load_i32(cu_seqlens_new, const_i32(i+1))
     *     is_in_seq = b.cmp_lt(new_token, cuq_next)
     *     seq = b.select(is_in_seq, seq, b.add(seq, const_i32(1))) */
    seq = ckc_b_const_i32(b, 0);
    for(i = 0; i < spec->batch; ++i)
    {
        ckc_value_t* cuq_next
            = ckc_b_global_load_i32(b, cu_seqlens_new, ckc_b_const_i32(b, i + 1), 0);
        ckc_value_t* is_in_seq = ckc_b_cmp_lt(b, new_token, cuq_next);
        ckc_value_t* seq_plus = ckc_b_add(b, seq, ckc_b_const_i32(b, 1));
        seq = ckc_b_select(b, is_in_seq, seq, seq_plus);
    }
    seq_idx = seq;

    /* cu_base = b.global_load_i32(cu_seqlens_new, seq_idx)
     * local_new = b.sub(new_token, cu_base)
     * seqlen_cur = b.global_load_i32(seqlen_kv, seq_idx)
     * dst_pos = b.add(seqlen_cur, local_new) */
    cu_base = ckc_b_global_load_i32(b, cu_seqlens_new, seq_idx, 0);
    local_new = ckc_b_sub(b, new_token, cu_base);
    seqlen_cur = ckc_b_global_load_i32(b, seqlen_kv, seq_idx, 0);
    dst_pos = ckc_b_add(b, seqlen_cur, local_new);

    /* in_row_base = b.add(b.mul(new_token, stride_in_token),
     *                     b.mul(kv_head_idx, stride_in_head)) */
    {
        ckc_value_t* a0 = ckc_b_mul(b, new_token, stride_in_token);
        ckc_value_t* a1 = ckc_b_mul(b, kv_head_idx, stride_in_head);
        in_row_base = ckc_b_add(b, a0, a1);
    }
    /* cache_row_base = b.add(b.mul(dst_pos, stride_cache_token),
     *                        b.mul(kv_head_idx, stride_cache_head)) */
    {
        ckc_value_t* c0 = ckc_b_mul(b, dst_pos, stride_cache_token);
        ckc_value_t* c1 = ckc_b_mul(b, kv_head_idx, stride_cache_head);
        cache_row_base = ckc_b_add(b, c0, c1);
    }

    ckc_appendkv_copy_k(b,
                        spec,
                        H,
                        dtype,
                        K_new,
                        K_cache,
                        cos_table,
                        sin_table,
                        dst_pos,
                        in_row_base,
                        cache_row_base);
    ckc_appendkv_copy_v(b, H, dtype, V_new, V_cache, in_row_base, cache_row_base);
}

/* ===================================================================== *
 *  build_fmha_fwd_appendkv(spec, arch) -- THE DRIVER
 * ===================================================================== */

ckc_kernel_def_t* ckc_build_fmha_fwd_appendkv(ckc_ir_builder_t* b,
                                              const ckc_fmha_appendkv_spec_t* spec,
                                              const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        int H;
        int BS;
        const char* dtype;
        const ckc_type_t* ty;
        char reason[200];

        ckc_value_t* K_new;
        ckc_value_t* V_new;
        ckc_value_t* K_cache;
        ckc_value_t* V_cache;
        ckc_value_t* seqlen_kv;
        ckc_value_t* cu_seqlens_new;
        ckc_value_t* cos_table = NULL;
        ckc_value_t* sin_table = NULL;
        ckc_value_t* total_new_q;
        ckc_value_t* batch_param;
        ckc_value_t* stride_in_token;
        ckc_value_t* stride_in_head;
        ckc_value_t* stride_cache_token;
        ckc_value_t* stride_cache_head;

        ckc_value_t* tid;
        ckc_value_t* bid;
        ckc_value_t* kv_head_idx;
        ckc_value_t* new_token;
        ckc_value_t* in_bounds;

        if(b == NULL || spec == NULL)
        {
            if(b != NULL)
            {
                ckc_i_set_err(b, CKC_ERR_VALUE, "build_fmha_fwd_appendkv: null spec");
            }
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx950";
        }

        /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError */
        if(!ckc_fmha_appendkv_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "invalid fmha_appendkv spec: %s", reason);
            return NULL;
        }

        H = spec->common.shape.head_size;
        BS = spec->block_size;
        dtype = spec->common.dtype;
        ty = ckc_io_ir_type(dtype);

        /* b = IRBuilder(spec.kernel_name())  -- caller already did this.
         * b.kernel.attrs["max_workgroup_size"] = BS */
        ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

        /* K_new/V_new: ptr<ty,global>, noalias, readonly, align=16. */
        {
            ckc_param_opts_t opts;
            const ckc_type_t* ptr_ty = ckc_ptr_type(b, ty, "global");
            const ckc_type_t* ptr_i32 = ckc_ptr_type(b, ckc_i32(), "global");
            const ckc_type_t* ptr_f32 = ckc_ptr_type(b, ckc_f32(), "global");

            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.readonly = true;
            opts.readonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            K_new = ckc_b_param(b, "K_new", ptr_ty, &opts);
            V_new = ckc_b_param(b, "V_new", ptr_ty, &opts);

            /* K_cache/V_cache: ptr<ty,global>, noalias, align=16 (no readonly). */
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.align = 16;
            opts.align_set = true;
            K_cache = ckc_b_param(b, "K_cache", ptr_ty, &opts);
            V_cache = ckc_b_param(b, "V_cache", ptr_ty, &opts);

            /* seqlen_kv/cu_seqlens_new: ptr<i32,global>, noalias, readonly, align=4. */
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.readonly = true;
            opts.readonly_set = true;
            opts.align = 4;
            opts.align_set = true;
            seqlen_kv = ckc_b_param(b, "seqlen_kv", ptr_i32, &opts);
            cu_seqlens_new = ckc_b_param(b, "cu_seqlens_new", ptr_i32, &opts);

            /* cos_table/sin_table (rotary only): ptr<f32,global>, readonly, align=4. */
            if(spec->has_rotary)
            {
                memset(&opts, 0, sizeof(opts));
                opts.readonly = true;
                opts.readonly_set = true;
                opts.align = 4;
                opts.align_set = true;
                cos_table = ckc_b_param(b, "cos_table", ptr_f32, &opts);
                sin_table = ckc_b_param(b, "sin_table", ptr_f32, &opts);
            }
        }

        /* total_new_q = b.param("total_new_q", I32)
         * _batch = b.param("batch", I32)
         * stride_in_token/head, stride_cache_token/head = b.param(..., I32) */
        total_new_q = ckc_b_param(b, "total_new_q", ckc_i32(), NULL);
        batch_param = ckc_b_param(b, "batch", ckc_i32(), NULL);
        (void)batch_param; /* ABI-only, mirrors Python `_batch` */
        stride_in_token = ckc_b_param(b, "stride_in_token", ckc_i32(), NULL);
        stride_in_head = ckc_b_param(b, "stride_in_head", ckc_i32(), NULL);
        stride_cache_token = ckc_b_param(b, "stride_cache_token", ckc_i32(), NULL);
        stride_cache_head = ckc_b_param(b, "stride_cache_head", ckc_i32(), NULL);

        /* tid = b.thread_id_x(); bid = b.block_id_x(); kv_head_idx = b.block_id_y()
         * new_token = b.add(b.mul(bid, b.const_i32(BS)), tid) */
        tid = ckc_b_thread_id_x(b);
        bid = ckc_b_block_id_x(b);
        kv_head_idx = ckc_b_block_id_y(b);
        new_token = ckc_b_add(b, ckc_b_mul(b, bid, ckc_b_const_i32(b, BS)), tid);

        /* in_bounds = b.cmp_lt(new_token, total_new_q)
         * with b.scf_if(in_bounds): _appendkv_body(...) */
        in_bounds = ckc_b_cmp_lt(b, new_token, total_new_q);
        {
            ckc_if_t gate = ckc_b_scf_if(b, in_bounds);
            ckc_b_region_enter(b, gate.then_region);
            ckc_appendkv_body(b,
                              spec,
                              H,
                              dtype,
                              new_token,
                              kv_head_idx,
                              K_new,
                              V_new,
                              K_cache,
                              V_cache,
                              seqlen_kv,
                              cu_seqlens_new,
                              cos_table,
                              sin_table,
                              stride_in_token,
                              stride_in_head,
                              stride_cache_token,
                              stride_cache_head);
            ckc_b_region_leave(b);
        }

        /* b.ret(); return b.kernel */
        ckc_b_ret(b);

        if(!ckc_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

/* ===================================================================== *
 *  build_fmha_fwd_appendkv_new -- init the builder, then build.
 * ===================================================================== */

ckc_kernel_def_t* ckc_build_fmha_fwd_appendkv_new(ckc_ir_builder_t* b,
                                                  const ckc_fmha_appendkv_spec_t* spec,
                                                  const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_fmha_appendkv_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_fmha_fwd_appendkv(b, spec, arch);
    });
}

/* ===================================================================== *
 *  fmha_appendkv_grid(spec, total_new_q)
 * ===================================================================== */

ckc_status_t
    ckc_fmha_appendkv_grid(const ckc_fmha_appendkv_spec_t* spec, int total_new_q, int out[3])
{
    int totals[1];
    int tiles[1];
    int tmp[3];
    ckc_status_t st;

    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    /* gx, _, _ = ceil_div_grid((total_new_q, block_size)) */
    totals[0] = total_new_q;
    tiles[0] = spec->block_size;
    st = ckc_ceil_div_grid(totals, tiles, 1, tmp);
    if(st != CKC_OK)
    {
        return st;
    }
    /* return (gx, num_kv_heads, 1) */
    out[0] = tmp[0];
    out[1] = spec->common.shape.num_kv_heads;
    out[2] = 1;
    return CKC_OK;
}

/* ===================================================================== *
 *  fmha_appendkv_signature(spec)
 * ===================================================================== */

ckc_status_t ckc_fmha_appendkv_signature(ckc_arena_t* arena,
                                         const ckc_fmha_appendkv_spec_t* spec,
                                         ckc_sig_entry_t* out_items,
                                         size_t out_cap,
                                         size_t* out_count)
{
    ckc_signature_builder_t sb;
    const ckc_sig_entry_t* items;
    size_t count;
    size_t need;
    ckc_status_t st;

    if(arena == NULL || spec == NULL || out_items == NULL)
    {
        return CKC_ERR_VALUE;
    }
    need = spec->has_rotary ? 14u : 12u;
    if(out_cap < need)
    {
        return CKC_ERR_VALUE;
    }

    st = ckc_signature_builder_init(&sb, arena);
    if(st != CKC_OK)
    {
        return st;
    }

    ckc_signature_builder_ptr(&sb, "K_new", spec->common.dtype, NULL);
    ckc_signature_builder_ptr(&sb, "V_new", spec->common.dtype, NULL);
    ckc_signature_builder_ptr(&sb, "K_cache", spec->common.dtype, NULL);
    ckc_signature_builder_ptr(&sb, "V_cache", spec->common.dtype, NULL);
    ckc_signature_builder_ptr(&sb, "seqlen_kv", "i32", NULL);
    ckc_signature_builder_ptr(&sb, "cu_seqlens_new", "i32", NULL);
    if(spec->has_rotary)
    {
        ckc_signature_builder_ptr(&sb, "cos_table", "f32", NULL);
        ckc_signature_builder_ptr(&sb, "sin_table", "f32", NULL);
    }
    ckc_signature_builder_scalar(&sb, "total_new_q", "i32");
    ckc_signature_builder_scalar(&sb, "batch", "i32");
    ckc_signature_builder_scalar(&sb, "stride_in_token", "i32");
    ckc_signature_builder_scalar(&sb, "stride_in_head", "i32");
    ckc_signature_builder_scalar(&sb, "stride_cache_token", "i32");
    ckc_signature_builder_scalar(&sb, "stride_cache_head", "i32");

    st = ckc_signature_builder_build(&sb, &items, &count);
    if(st != CKC_OK)
    {
        return st;
    }
    if(count > out_cap)
    {
        return CKC_ERR_VALUE;
    }
    memcpy(out_items, items, count * sizeof(*out_items));
    if(out_count != NULL)
    {
        *out_count = count;
    }
    return CKC_OK;
}

/* ===================================================================== *
 *  fmha_appendkv_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */

static void ckc_appendkv_copy_err(char* err, size_t err_cap, const char* m)
{
    size_t n;
    if(err == NULL || err_cap == 0)
    {
        return;
    }
    if(m == NULL)
    {
        m = "fmha_appendkv lower failed";
    }
    n = strlen(m);
    if(n >= err_cap)
    {
        n = err_cap - 1;
    }
    memcpy(err, m, n);
    err[n] = '\0';
}

ckc_status_t ckc_fmha_appendkv_lower_to_llvm(const ckc_fmha_appendkv_spec_t* spec,
                                             const char* arch,
                                             ckc_llvm_flavor_t flavor,
                                             char** out_ll,
                                             char* err,
                                             size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        ckc_appendkv_copy_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_fmha_fwd_appendkv_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        ckc_appendkv_copy_err(err, err_cap, ckc_ir_builder_error(&b));
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
