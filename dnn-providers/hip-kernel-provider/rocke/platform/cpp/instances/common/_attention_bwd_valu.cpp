// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_valu.py: the softmax-gradient
 * VALU levers of the attention backward main kernel.
 *
 * See rocke/helper_rocke.instances.common._attention_bwd_valu.h for the mapping
 * and the byte-identity rules. Every op is created in the Python evaluation
 * order; each comment `py:` quotes the Python expression a block reproduces.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_valu.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/helper_rocke.helpers.attention_band.h" /* ROCKE_ATTN_I32_FAR */
#include "rocke/ir_internal.h" /* rocke_i_set_err */

const char* const ROCKE_ATTN_BWD_BF16_CASTS[ROCKE_ATTN_BWD_BF16_CASTS_N]
    = {"generic", "rne_noguard"};
const char* const ROCKE_ATTN_BWD_BWD_EXP2S[ROCKE_ATTN_BWD_BWD_EXP2S_N] = {"exact", "fast"};
const int ROCKE_ATTN_BWD_STATS_VECS[ROCKE_ATTN_BWD_STATS_VECS_N] = {1, 2, 4};
const char* const ROCKE_ATTN_BWD_NATIVE_BF16_CVT_ARCHS[ROCKE_ATTN_BWD_NATIVE_BF16_CVT_ARCHS_N]
    = {"gfx950"};

/* v_perm_b32 selector taking bytes 2, 3 of the low source and bytes 2, 3 of
 * the high source: the upper halves of two words, low element first. */
static const int64_t HI_HALVES = 0x07060302;

/* py: b.vector_splat(b.const_i32(v), n) */
static rocke_value_t* splat(rocke_ir_builder_t* b, int64_t v, int n)
{
    rocke_value_t* c = rocke_b_const_i32(b, v);
    return rocke_b_vector_splat(b, c, n);
}

rocke_value_t* rocke_attn_bwd_rne_noguard_bf16(rocke_ir_builder_t* b, rocke_value_t* frag)
{
    if(!rocke_ir_builder_ok(b) || frag == NULL)
        return NULL;
    if(frag->type == NULL || frag->type->kind != ROCKE_TYPE_VECTOR || frag->type->count < 1)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "rne_noguard_bf16 needs a vector fragment");
        return NULL;
    }
    const int n = frag->type->count;
    if(n % 2)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "rne_noguard_bf16 packs element pairs: n must be even");
        return NULL;
    }
    const rocke_type_t* vi = rocke_vector_type(b, rocke_i32(), n);
    rocke_value_t* u = rocke_b_vec_bitcast(b, frag, vi);
    /* py: lsb = b.vector_and(b.vector_lshr(u, splat(16)), splat(1)) */
    rocke_value_t* s16 = splat(b, 16, n);
    rocke_value_t* sh = rocke_b_vector_lshr(b, u, s16);
    rocke_value_t* s1 = splat(b, 1, n);
    rocke_value_t* lsb = rocke_b_vector_and(b, sh, s1);
    /* py: r = b.vector_add(b.vector_add(u, lsb), splat(0x7FFF)) */
    rocke_value_t* ul = rocke_b_vector_add(b, u, lsb);
    rocke_value_t* sr = splat(b, 0x7FFF, n);
    rocke_value_t* r = rocke_b_vector_add(b, ul, sr);
    std::vector<rocke_value_t*> words((size_t)(n / 2));
    for(int j = 0; j < n / 2; ++j)
    {
        rocke_value_t* hi = rocke_b_vec_extract(b, r, 2 * j + 1);
        rocke_value_t* lo = rocke_b_vec_extract(b, r, 2 * j);
        words[(size_t)j] = rocke_b_byte_perm(b, hi, lo, HI_HALVES);
    }
    rocke_value_t* packed = rocke_b_vec_pack(b, words.data(), n / 2, rocke_i32());
    const rocke_type_t* vb = rocke_vector_type(b, rocke_bf16(), n);
    return rocke_b_vec_bitcast(b, packed, vb);
}

rocke_value_t* rocke_attn_bwd_cast_c_frag(rocke_ir_builder_t* b,
                                          rocke_value_t* frag,
                                          const rocke_type_t* elem,
                                          const char* bf16_cast)
{
    if(bf16_cast == NULL)
        bf16_cast = "generic";
    if(strcmp(bf16_cast, "generic") == 0)
        return rocke_b_vec_cast_f32_to(b, frag, elem);
    if(strcmp(bf16_cast, "rne_noguard") != 0)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "bf16_cast='%s' not in ('generic', 'rne_noguard')", bf16_cast);
        return NULL;
    }
    if(elem == NULL || !rocke_type_eq(elem, rocke_bf16()))
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "bf16_cast='rne_noguard' applies to bf16 operands only");
        return NULL;
    }
    return rocke_attn_bwd_rne_noguard_bf16(b, frag);
}

rocke_value_t* rocke_attn_bwd_softmax_p(rocke_ir_builder_t* b,
                                        rocke_value_t* sv,
                                        rocke_value_t* scale_log2,
                                        rocke_value_t* lse2,
                                        rocke_value_t* neg_lse2,
                                        const char* bwd_exp2)
{
    if(bwd_exp2 == NULL)
        bwd_exp2 = "exact";
    rocke_value_t* arg;
    if(neg_lse2 == NULL)
    {
        rocke_value_t* m = rocke_b_fmul(b, sv, scale_log2);
        arg = rocke_b_fsub(b, m, lse2);
    }
    else
    {
        arg = rocke_b_fma(b, sv, scale_log2, neg_lse2);
    }
    if(strcmp(bwd_exp2, "exact") == 0)
        return rocke_b_exp2(b, arg);
    if(strcmp(bwd_exp2, "fast") == 0)
        return rocke_b_exp2_fast(b, arg);
    rocke_i_set_err(b, ROCKE_ERR_VALUE, "bwd_exp2='%s' not in ('exact', 'fast')", bwd_exp2);
    return NULL;
}

/* ====================================================================== *
 * Host-integer evaluation of a lane map (Python eval_layout / IntLaneBuilder)
 * ====================================================================== */

/* py: a // b and a % b (floor semantics) */
static int64_t floor_div(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        q -= 1;
    return q;
}

/* Interpret the integer-arithmetic ops a catalog lane map emits on a scratch
 * builder (const_i32, add, sub, mul, div, mod, xor, shl, lshr, land). false on anything else. */
static bool eval_int(const rocke_value_t* v, int64_t* out)
{
    if(v == NULL || v->op == NULL)
        return false;
    const rocke_op_t* op = v->op;
    if(op->opcode == ROCKE_OP_ARITH_CONSTANT)
        return rocke_attr_get_int(&op->attrs, "value", out);
    if(op->num_operands != 2)
        return false;
    int64_t x = 0;
    int64_t y = 0;
    if(!eval_int(op->operands[0], &x) || !eval_int(op->operands[1], &y))
        return false;
    switch(op->opcode)
    {
    case ROCKE_OP_ARITH_ADD:
        *out = x + y;
        return true;
    case ROCKE_OP_ARITH_SUB:
        *out = x - y;
        return true;
    case ROCKE_OP_ARITH_MUL:
        *out = x * y;
        return true;
    case ROCKE_OP_ARITH_DIV:
        if(y == 0)
            return false;
        *out = floor_div(x, y);
        return true;
    case ROCKE_OP_ARITH_MOD:
        if(y == 0)
            return false;
        *out = x - floor_div(x, y) * y;
        return true;
    case ROCKE_OP_ARITH_XOR:
        *out = x ^ y;
        return true;
    case ROCKE_OP_ARITH_SHL:
        if(y < 0 || y > 62)
            return false;
        *out = (int64_t)((uint64_t)x << y);
        return true;
    case ROCKE_OP_ARITH_LSHR:
        if(y < 0 || y > 62)
            return false;
        *out = x >> y; /* py: a >> b is an arithmetic shift on ints */
        return true;
    case ROCKE_OP_ARITH_AND:
        *out = x & y;
        return true;
    default:
        return false;
    }
}

bool rocke_attn_bwd_check_stats_vec_layout(rocke_ir_builder_t* b,
                                           const rocke_mma_op_t* op,
                                           int width)
{
    bool known = false;
    for(int i = 0; i < ROCKE_ATTN_BWD_STATS_VECS_N; ++i)
        known = known || ROCKE_ATTN_BWD_STATS_VECS[i] == width;
    if(!known)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "stats_vec=%d not in (1, 2, 4)", width);
        return false;
    }
    const rocke_layout_map_t* m = op ? op->c_layout : NULL;
    if(m == NULL || m->fn == NULL)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_NOTIMPL,
                        "no verified C lane map for %s",
                        op && op->op_id ? op->op_id : "(null)");
        return false;
    }
    const int n = m->frag_len;
    const int lanes = m->wave_size;
    /* the C fragment as host ints: [lane][slot] -> (row, col) */
    rocke_ir_builder_t scratch;
    if(rocke_ir_builder_init(&scratch, "stats_vec_scratch") != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "stats_vec: scratch builder failed");
        return false;
    }
    std::vector<int64_t> coord((size_t)lanes * (size_t)(n > 0 ? n : 0) * 2);
    bool ok = true;
    for(int lane = 0; lane < lanes && ok; ++lane)
    {
        for(int s = 0; s < n && ok; ++s)
        {
            rocke_value_t* lv = rocke_b_const_i32(&scratch, lane);
            rocke_value_t* o0 = NULL;
            rocke_value_t* o1 = NULL;
            m->fn(&scratch, lv, s, &o0, &o1);
            int64_t c0 = 0;
            int64_t c1 = 0;
            ok = rocke_ir_builder_ok(&scratch) && eval_int(o0, &c0) && eval_int(o1, &c1);
            coord[((size_t)lane * (size_t)n + (size_t)s) * 2] = c0;
            coord[((size_t)lane * (size_t)n + (size_t)s) * 2 + 1] = c1;
        }
    }
    rocke_ir_builder_free(&scratch);
    if(!ok)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "stats_vec: cannot evaluate the C lane map of %s", op->op_id);
        return false;
    }
    if(n % width)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "stats_vec=%d: the C fragment of %s has %d slots",
                        width,
                        op->op_id,
                        n);
        return false;
    }
    for(int lane = 0; lane < lanes; ++lane)
    {
        const int64_t* c = &coord[(size_t)lane * (size_t)n * 2];
        for(int g = 0; g < n; g += width)
        {
            const int64_t r0 = c[2 * g];
            const int64_t c0 = c[2 * g + 1];
            for(int j = 0; j < width; ++j)
            {
                if(c[2 * (g + j)] != r0 + j || c[2 * (g + j) + 1] != c0)
                {
                    /* Python's repr of the lane's coords tuple */
                    std::string held = "(";
                    for(int s = 0; s < n; ++s)
                    {
                        if(s)
                            held += ", ";
                        held += "(" + std::to_string((long long)c[2 * s]) + ", "
                                + std::to_string((long long)c[2 * s + 1]) + ")";
                    }
                    held += n == 1 ? ",)" : ")";
                    rocke_i_set_err(b,
                                    ROCKE_ERR_VALUE,
                                    "stats_vec=%d: the C rows of %s are not runs of %d consecutive "
                                    "rows (a lane holds %s)",
                                    width,
                                    op->op_id,
                                    width,
                                    held.c_str());
                    return false;
                }
            }
        }
    }
    return true;
}

bool rocke_attn_bwd_batched_stats_rows(rocke_ir_builder_t* b,
                                       const rocke_attn_bwd_workspace_addr_t* lse2,
                                       const rocke_attn_bwd_workspace_addr_t* dsum,
                                       rocke_value_t* h,
                                       rocke_attn_bwd_iv_t q0,
                                       rocke_value_t* const* rows,
                                       int n_rows,
                                       int width,
                                       rocke_attn_bwd_iv_t n_heads,
                                       rocke_value_t** out_lse2,
                                       rocke_value_t** out_dsum)
{
    const rocke_attn_bwd_workspace_addr_t* was[2] = {lse2, dsum};
    for(int i = 0; i < 2; ++i)
    {
        if(was[i]->layout != ROCKE_ATTN_BWD_WS_HEAD_MAJOR || was[i]->width != 1)
        {
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "stats_vec > 1 needs head-major one-float stats rows");
            return false;
        }
    }
    if(lse2->ws_rows != dsum->ws_rows || lse2->row0 != dsum->row0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "lse2 / Dsum workspaces must share their row model");
        return false;
    }
    if(width < 1)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "batched_stats_rows: width must be >= 1");
        return false;
    }
    if(n_rows % width)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%d rows do not split into runs of %d", n_rows, width);
        return false;
    }
    /* Uniform part (once per step). */
    rocke_value_t* base = lse2->row0;
    if(!rocke_attn_bwd_iv_is_int(q0, 0))
    {
        rocke_value_t* q64 = rocke_attn_bwd_as_i64(b, q0, true);
        base = rocke_b_add(b, base, q64);
    }
    rocke_value_t* rows64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(lse2->ws_rows), false);
    rocke_value_t* one64 = rocke_b_const_i64(b, 1);
    /* py: _max64(b, _min64(b, base, b.sub(rows64, one64)), b.const_i64(0)) */
    rocke_value_t* last64 = rocke_b_sub(b, rows64, one64);
    rocke_value_t* mn = rocke_attn_bwd_min64(b, base, last64);
    rocke_value_t* c64 = rocke_b_const_i64(b, 0);
    rocke_value_t* base_ws = rocke_attn_bwd_max64(b, mn, c64);
    /* py: _clamp64_to_i32(b, b.sub(rows64, base), b.const_i32(I32_FAR)) */
    rocke_value_t* left64 = rocke_b_sub(b, rows64, base);
    rocke_value_t* far = rocke_b_const_i32(b, ROCKE_ATTN_I32_FAR);
    rocke_value_t* rows_left = rocke_attn_bwd_clamp64_to_i32(b, left64, far);
    /* py: elem = b.add(b.mul(as_i64(b, h), rows64), base_ws) */
    rocke_value_t* h64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(h), false);
    rocke_value_t* hr = rocke_b_mul(b, h64, rows64);
    rocke_value_t* elem = rocke_b_add(b, hr, base_ws);
    /* py: total = b.mul(as_i64(b, n_heads), rows64) */
    rocke_value_t* nh64 = rocke_attn_bwd_as_i64(b, n_heads, false);
    rocke_value_t* total = rocke_b_mul(b, nh64, rows64);
    /* py: back = _min64(b, elem, b.const_i64(width - 1))  # in [0, width - 1] */
    rocke_value_t* cw = rocke_b_const_i64(b, width - 1);
    rocke_value_t* back = rocke_attn_bwd_min64(b, elem, cw);
    /* py: lo = b.sub(b.const_i32(0), b.trunc(back, I32)) */
    rocke_value_t* z32 = rocke_b_const_i32(b, 0);
    rocke_value_t* bt = rocke_b_trunc(b, back, rocke_i32());
    rocke_value_t* lo = rocke_b_sub(b, z32, bt);
    /* py: hi = _clamp64_to_i32(b, b.sub(b.sub(total, one64), elem), b.const_i32(I32_FAR)) */
    rocke_value_t* t1 = rocke_b_sub(b, total, one64);
    rocke_value_t* t2 = rocke_b_sub(b, t1, elem);
    rocke_value_t* far2 = rocke_b_const_i32(b, ROCKE_ATTN_I32_FAR);
    rocke_value_t* hi = rocke_attn_bwd_clamp64_to_i32(b, t2, far2);
    /* py: top = b.sub(hi, b.const_i32(width - 1)) */
    rocke_value_t* cw32 = rocke_b_const_i32(b, width - 1);
    rocke_value_t* top = rocke_b_sub(b, hi, cw32);
    /* py: off = b.mul(b.sub(elem, back), b.const_i64(4)) */
    rocke_value_t* eb = rocke_b_sub(b, elem, back);
    rocke_value_t* c4 = rocke_b_const_i64(b, 4);
    rocke_value_t* off = rocke_b_mul(b, eb, c4);
    rocke_value_t* p_l = rocke_b_global_ptr_add(b, lse2->ptr, off);
    rocke_value_t* p_d = rocke_b_global_ptr_add(b, dsum->ptr, off);
    rocke_value_t* inf = rocke_b_const_f32(b, (double)INFINITY);
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    std::vector<rocke_value_t*> shift((size_t)width, (rocke_value_t*)NULL);
    int o = 0;
    for(int g = 0; g < n_rows; g += width)
    {
        rocke_value_t* r0 = rows[g];
        rocke_value_t* mi = rocke_b_smin(b, r0, top);
        rocke_value_t* a = rocke_b_smax(b, mi, lo);
        rocke_value_t* idx = rocke_b_sub(b, a, lo);
        rocke_value_t* vl = rocke_b_global_load_vN(b, p_l, idx, rocke_f32(), width, 4);
        rocke_value_t* vd = rocke_b_global_load_vN(b, p_d, idx, rocke_f32(), width, 4);
        rocke_value_t* d = rocke_b_sub(b, r0, a);
        for(int k = 1; k < width; ++k)
        {
            rocke_value_t* ck = rocke_b_const_i32(b, k);
            shift[(size_t)k] = rocke_b_cmp_eq(b, d, ck);
        }
        for(int i = 0; i < width; ++i)
        {
            rocke_value_t* r = rows[g + i];
            rocke_value_t* ok = rocke_b_cmp_lt(b, r, rows_left);
            rocke_value_t* pair[2];
            rocke_value_t* vs[2] = {vl, vd};
            rocke_value_t* neutral[2] = {inf, zero};
            for(int p = 0; p < 2; ++p)
            {
                rocke_value_t* val = rocke_b_vec_extract(b, vs[p], i);
                for(int k = 1; k < width - i; ++k)
                {
                    rocke_value_t* ex = rocke_b_vec_extract(b, vs[p], i + k);
                    val = rocke_b_select(b, shift[(size_t)k], ex, val);
                }
                pair[p] = rocke_b_select(b, ok, val, neutral[p]);
            }
            out_lse2[o] = pair[0];
            out_dsum[o] = pair[1];
            ++o;
        }
    }
    return rocke_ir_builder_ok(b);
}

void rocke_attn_bwd_stats_rows_neg(rocke_ir_builder_t* b,
                                   rocke_value_t* const* lse2,
                                   int n,
                                   rocke_value_t** out)
{
    for(int i = 0; i < n; ++i)
        out[i] = rocke_b_fneg(b, lse2[i]);
}
