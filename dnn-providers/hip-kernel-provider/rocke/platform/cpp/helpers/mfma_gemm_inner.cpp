// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.mfma_gemm_inner.c -- C99 port of nine symbols from
 * rocke/helpers/mfma_gemm_inner.py. See the header for the symbol map and the
 * binding / error-model notes.
 *
 * Fidelity contract: each rocke_* function reproduces its Python counterpart's
 * builder-call sequence op-for-op, in the same order, with the same operands and
 * the same compile-time constants -- so the emitted IR is byte-identical to what
 * the Python helper produces. Where the Python calls a MfmaAtom method that the
 * atoms.h port does not expose (emit / zero_acc / lane_to_output), the tiny
 * builder sequence is inlined here verbatim.
 */

#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arena.h" /* rocke_arena_printf                       */
#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live              */

/* ============================================================== _ir_type_for_dtype *
 *
 * Python:
 *
 *     def _ir_type_for_dtype(dtype_in):
 *         if dtype_in in ("f16", "fp16"): return F16
 *         if dtype_in == "bf16":          return BF16
 *         if dtype_in == "fp8e4m3":       return FP8E4M3
 *         if dtype_in == "bf8e5m2":       return BF8E5M2
 *         raise ValueError(f"unsupported dtype {dtype_in!r}")
 *
 * Private helper (not a requested symbol). Returns the interned scalar singleton,
 * or NULL on the Python ValueError path (no builder here). */
static const rocke_type_t* rocke_i_ir_type_for_dtype(const char* dtype_in)
{
    if(dtype_in == NULL)
    {
        return NULL;
    }
    if(strcmp(dtype_in, "f16") == 0 || strcmp(dtype_in, "fp16") == 0)
    {
        return rocke_f16();
    }
    if(strcmp(dtype_in, "bf16") == 0)
    {
        return rocke_bf16();
    }
    if(strcmp(dtype_in, "fp8e4m3") == 0)
    {
        return rocke_fp8e4m3();
    }
    if(strcmp(dtype_in, "bf8e5m2") == 0)
    {
        return rocke_bf8e5m2();
    }
    return NULL;
}

/* dtype_in in ("f16", "fp16", "bf16") -- the vec-load fast path predicate used by
 * load_a_row_major_contiguous. */
static bool rocke_i_dtype_is_f16_bf16(const char* dtype_in)
{
    return dtype_in != NULL
           && (strcmp(dtype_in, "f16") == 0 || strcmp(dtype_in, "fp16") == 0
               || strcmp(dtype_in, "bf16") == 0);
}

/* dtype_in in ("f16", "bf16") -- the align==2 predicate used by
 * load_b_col_strided_scalars (NOTE: Python uses "f16"/"bf16" only, NOT "fp16"). */
static bool rocke_i_dtype_align2(const char* dtype_in)
{
    return dtype_in != NULL && (strcmp(dtype_in, "f16") == 0 || strcmp(dtype_in, "bf16") == 0);
}

/* ====================================================================== LaneDecode *
 *
 * decode_mfma_lanes -- byte-for-byte:
 *     c_m = b.const_i32(atom.m)
 *     c_n = b.const_i32(atom.n)
 *     m_in_atom = b.mod(lane, c_m)
 *     n_in_atom = b.mod(lane, c_n)
 *     k_blk     = b.div(lane, c_m)
 */
rocke_lane_decode_t rocke_decode_mfma_lanes(rocke_ir_builder_t* b,
                                            const rocke_mfma_atom_t* atom,
                                            rocke_value_t* lane)
{
    rocke_lane_decode_t out;
    rocke_value_t* c_m;
    rocke_value_t* c_n;

    out.lane = lane;
    out.m_in_atom = NULL;
    out.n_in_atom = NULL;
    out.k_blk = NULL;

    if(!rocke_i_live(b) || atom == NULL)
    {
        return out;
    }

    c_m = rocke_b_const_i32(b, atom->m);
    c_n = rocke_b_const_i32(b, atom->n);
    out.m_in_atom = rocke_b_mod(b, lane, c_m);
    out.n_in_atom = rocke_b_mod(b, lane, c_n);
    out.k_blk = rocke_b_div(b, lane, c_m);
    return out;
}

/* ============================================================== mfma_atom_for_dtype *
 *
 * Faithful port of the dtype/(m,n)/prefer_packed_k decision tree. Each branch
 * resolves the same atom the Python MfmaAtom.<factory>() would, via the catalog
 * lookup rocke_mfma_atom(dtype, m, n, k). The (dtype, m, n, k) tuples come straight
 * from the factory bodies in atoms.py:
 *
 *   f16  (16,16): packed -> f16_16x16x32 (k=32) ; else f16_16x16x16 (k=16)
 *   f16  (32,32): packed -> f16_32x32x16 (k=16) ; else f16_32x32x8  (k=8)
 *   bf16 (16,16): packed -> bf16_16x16x32(k=32) ; else bf16_16x16x16(k=16)
 *   bf16 (32,32):           bf16_32x32x16(k=16)   (no prefer_packed_k branch)
 *   fp8e4m3 (16,16): fp8_16x16x32(k=32) ; (32,32): fp8_32x32x16(k=16)
 *   bf8e5m2 (16,16): bf8_16x16x32(k=32) ; (32,32): bf8_32x32x16(k=16)
 *   else: ValueError
 */
const rocke_mfma_atom_t*
    rocke_mfma_atom_for_dtype(const char* dtype_in, int m, int n, bool prefer_packed_k)
{
    if(dtype_in == NULL)
    {
        return NULL;
    }

    /* if dtype_in in ("f16", "fp16"): */
    if(strcmp(dtype_in, "f16") == 0 || strcmp(dtype_in, "fp16") == 0)
    {
        if(m == 16 && n == 16)
        {
            return prefer_packed_k ? rocke_mfma_atom("f16", 16, 16, 32)
                                   : rocke_mfma_atom("f16", 16, 16, 16);
        }
        if(m == 32 && n == 32)
        {
            return prefer_packed_k ? rocke_mfma_atom("f16", 32, 32, 16)
                                   : rocke_mfma_atom("f16", 32, 32, 8);
        }
    }
    /* if dtype_in == "bf16": */
    if(strcmp(dtype_in, "bf16") == 0)
    {
        if(m == 16 && n == 16)
        {
            return prefer_packed_k ? rocke_mfma_atom("bf16", 16, 16, 32)
                                   : rocke_mfma_atom("bf16", 16, 16, 16);
        }
        if(m == 32 && n == 32)
        {
            return rocke_mfma_atom("bf16", 32, 32, 16);
        }
    }
    /* if dtype_in == "fp8e4m3": */
    if(strcmp(dtype_in, "fp8e4m3") == 0)
    {
        if(m == 16 && n == 16)
        {
            return rocke_mfma_atom("fp8e4m3", 16, 16, 32);
        }
        if(m == 32 && n == 32)
        {
            return rocke_mfma_atom("fp8e4m3", 32, 32, 16);
        }
    }
    /* if dtype_in == "bf8e5m2": */
    if(strcmp(dtype_in, "bf8e5m2") == 0)
    {
        if(m == 16 && n == 16)
        {
            return rocke_mfma_atom("bf8e5m2", 16, 16, 32);
        }
        if(m == 32 && n == 32)
        {
            return rocke_mfma_atom("bf8e5m2", 32, 32, 16);
        }
    }
    /* raise ValueError(...) */
    return NULL;
}

const rocke_mfma_atom_t* rocke_b_mfma_atom_for_dtype(
    rocke_ir_builder_t* b, const char* dtype_in, int m, int n, bool prefer_packed_k)
{
    const rocke_mfma_atom_t* atom;

    if(!rocke_i_live(b))
    {
        return NULL;
    }
    atom = rocke_mfma_atom_for_dtype(dtype_in, m, n, prefer_packed_k);
    if(atom == NULL)
    {
        /* Mirror: raise ValueError(f"no MFMA atom for dtype_in={dtype_in!r} "
         *                          f"shape {m}x{n}") */
        return (const rocke_mfma_atom_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "no MFMA atom for dtype_in=%s%s%s shape %dx%d",
            dtype_in ? "'" : "",
            dtype_in ? dtype_in : "None",
            dtype_in ? "'" : "",
            m,
            n);
    }
    return atom;
}

/* ======================================================= load_a_row_major_contiguous *
 *
 * Python:
 *     dtype_ir = _ir_type_for_dtype(atom.dtype_in)
 *     m_row = b.add(m_tile_base, lane_decode.m_in_atom)
 *     k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.a_per_lane))
 *     k_base = b.add(k_tile_base, k_lane_start)
 *     a_addr = b.add(b.mul(m_row, b.const_i32(K)), k_base)
 *     if atom.dtype_in in ("f16","fp16","bf16"):
 *         return b.global_load_vN(A, a_addr, dtype_ir, atom.a_per_lane,
 *                                 align=atom.a_per_lane * 2)
 *     out = b.zero_vec(dtype_ir, atom.a_per_lane)
 *     for j in range(atom.a_per_lane):
 *         addr = b.add(a_addr, b.const_i32(j))
 *         s = b.global_load(A, addr, dtype_ir, align=1)
 *         out = b.vec_insert(out, s, j)
 *     return out
 */
rocke_value_t* rocke_load_a_row_major_contiguous(rocke_ir_builder_t* b,
                                                 rocke_value_t* A,
                                                 const rocke_mfma_atom_t* atom,
                                                 const rocke_lane_decode_t* lane_decode,
                                                 rocke_value_t* m_tile_base,
                                                 rocke_value_t* k_tile_base,
                                                 int K)
{
    const rocke_type_t* dtype_ir;
    rocke_value_t* m_row;
    rocke_value_t* k_lane_start;
    rocke_value_t* k_base;
    rocke_value_t* a_addr;
    rocke_value_t* out;
    int j;

    if(!rocke_i_live(b) || atom == NULL || lane_decode == NULL)
    {
        return NULL;
    }

    dtype_ir = rocke_i_ir_type_for_dtype(atom->dtype_in);
    if(dtype_ir == NULL)
    {
        return (rocke_value_t*)rocke_i_set_err(b,
                                               ROCKE_ERR_VALUE,
                                               "unsupported dtype %s%s%s",
                                               atom->dtype_in ? "'" : "",
                                               atom->dtype_in ? atom->dtype_in : "None",
                                               atom->dtype_in ? "'" : "");
    }

    m_row = rocke_b_add(b, m_tile_base, lane_decode->m_in_atom);
    k_lane_start = rocke_b_mul(b, lane_decode->k_blk, rocke_b_const_i32(b, atom->a_per_lane));
    k_base = rocke_b_add(b, k_tile_base, k_lane_start);
    a_addr = rocke_b_add(b, rocke_b_mul(b, m_row, rocke_b_const_i32(b, K)), k_base);

    if(rocke_i_dtype_is_f16_bf16(atom->dtype_in))
    {
        return rocke_b_global_load_vN(
            b, A, a_addr, dtype_ir, atom->a_per_lane, atom->a_per_lane * 2);
    }

    /* fp8 / bf8: no vec load helper -- fall back to scalar loads. */
    out = rocke_b_zero_vec(b, dtype_ir, atom->a_per_lane);
    for(j = 0; j < atom->a_per_lane; ++j)
    {
        rocke_value_t* addr = rocke_b_add(b, a_addr, rocke_b_const_i32(b, j));
        rocke_value_t* s = rocke_b_global_load(b, A, addr, dtype_ir, 1);
        out = rocke_b_vec_insert(b, out, s, j);
    }
    return out;
}

/* ====================================================== load_b_col_strided_scalars *
 *
 * Python:
 *     dtype_ir = _ir_type_for_dtype(atom.dtype_in)
 *     n_col = b.add(n_tile_base, lane_decode.n_in_atom)
 *     k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.b_per_lane))
 *     k_base = b.add(k_tile_base, k_lane_start)
 *     out = b.zero_vec(dtype_ir, atom.b_per_lane)
 *     for j in range(atom.b_per_lane):
 *         addr = b.add(b.mul(b.add(k_base, b.const_i32(j)), b.const_i32(N)), n_col)
 *         s = b.global_load(B, addr, dtype_ir,
 *                           align=2 if atom.dtype_in in ("f16","bf16") else 1)
 *         out = b.vec_insert(out, s, j)
 *     return out
 */
rocke_value_t* rocke_load_b_col_strided_scalars(rocke_ir_builder_t* b,
                                                rocke_value_t* B,
                                                const rocke_mfma_atom_t* atom,
                                                const rocke_lane_decode_t* lane_decode,
                                                rocke_value_t* n_tile_base,
                                                rocke_value_t* k_tile_base,
                                                int N)
{
    const rocke_type_t* dtype_ir;
    rocke_value_t* n_col;
    rocke_value_t* k_lane_start;
    rocke_value_t* k_base;
    rocke_value_t* out;
    int align;
    int j;

    if(!rocke_i_live(b) || atom == NULL || lane_decode == NULL)
    {
        return NULL;
    }

    dtype_ir = rocke_i_ir_type_for_dtype(atom->dtype_in);
    if(dtype_ir == NULL)
    {
        return (rocke_value_t*)rocke_i_set_err(b,
                                               ROCKE_ERR_VALUE,
                                               "unsupported dtype %s%s%s",
                                               atom->dtype_in ? "'" : "",
                                               atom->dtype_in ? atom->dtype_in : "None",
                                               atom->dtype_in ? "'" : "");
    }

    n_col = rocke_b_add(b, n_tile_base, lane_decode->n_in_atom);
    k_lane_start = rocke_b_mul(b, lane_decode->k_blk, rocke_b_const_i32(b, atom->b_per_lane));
    k_base = rocke_b_add(b, k_tile_base, k_lane_start);

    align = rocke_i_dtype_align2(atom->dtype_in) ? 2 : 1;

    out = rocke_b_zero_vec(b, dtype_ir, atom->b_per_lane);
    for(j = 0; j < atom->b_per_lane; ++j)
    {
        /* Match Python's strict left-to-right evaluation order so the IR
         * value counter advances identically. Python evaluates
         *   addr = b.add(b.mul(b.add(k_base, b.const_i32(j)), b.const_i32(N)),
         *                n_col)
         * inside-out, left arg first: const_i32(j) -> add -> const_i32(N) ->
         * mul -> add. C function-argument evaluation order is unspecified, so
         * sequence each builder call into its own statement. */
        rocke_value_t* c_j = rocke_b_const_i32(b, j);
        rocke_value_t* k_off = rocke_b_add(b, k_base, c_j);
        rocke_value_t* c_n = rocke_b_const_i32(b, N);
        rocke_value_t* row_off = rocke_b_mul(b, k_off, c_n);
        rocke_value_t* addr = rocke_b_add(b, row_off, n_col);
        rocke_value_t* s = rocke_b_global_load(b, B, addr, dtype_ir, align);
        out = rocke_b_vec_insert(b, out, s, j);
    }
    return out;
}

/* ===================================================================== mfma_k_loop *
 *
 * Python:
 *     if K % atom.k != 0: raise ValueError(...)
 *     n_tiles = K // atom.k
 *     acc0 = initial_acc if initial_acc is not None else atom.zero_acc(b)
 *     kloop = b.scf_for_iter(b.const_i32(0), b.const_i32(n_tiles), b.const_i32(1),
 *                            [(acc_name, acc0)], iv_name=iv_name)
 *     with kloop as (kt, (acc_v,)):
 *         a_vec = load_a(b, kt)
 *         b_vec = load_b(b, kt)
 *         new_acc = atom.emit(b, a_vec, b_vec, acc_v)
 *         if per_tile_post_mfma is not None:
 *             new_acc = per_tile_post_mfma(b, new_acc, kt)
 *         b.scf_yield(new_acc)
 *     return kloop.results[0]
 *
 * Inlined atom methods:
 *     atom.zero_acc(b) -> b.zero_vec_f32(atom.c_per_lane)
 *     atom.emit(b,a,b,c) -> b.mma(atom.name, a, b, c)
 */
rocke_value_t* rocke_mfma_k_loop(rocke_ir_builder_t* b,
                                 int K,
                                 const rocke_mfma_atom_t* atom,
                                 rocke_mfma_load_fn load_a,
                                 rocke_mfma_load_fn load_b,
                                 rocke_mfma_post_fn per_tile_post_mfma,
                                 rocke_value_t* initial_acc,
                                 const char* iv_name,
                                 const char* acc_name,
                                 void* user)
{
    int n_tiles;
    rocke_value_t* acc0;
    rocke_value_t* lb;
    rocke_value_t* ub;
    rocke_value_t* step;
    rocke_iter_arg_t loop_args[1];
    rocke_for_t kloop;
    rocke_value_t* kt;
    rocke_value_t* acc_v;
    rocke_value_t* a_vec;
    rocke_value_t* b_vec;
    rocke_value_t* new_acc;
    rocke_value_t* yield_vals[1];

    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(atom == NULL || load_a == NULL || load_b == NULL)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "mfma_k_loop: atom/load_a/load_b must be non-NULL");
    }

    /* if K % atom.k != 0: raise ValueError */
    if(atom->k == 0 || (K % atom->k) != 0)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "mfma_k_loop: K=%d must be divisible by atom.k=%d", K, atom->k);
    }
    n_tiles = K / atom->k;

    /* acc0 = initial_acc if initial_acc is not None else atom.zero_acc(b) */
    acc0 = (initial_acc != NULL) ? initial_acc : rocke_b_zero_vec_f32(b, atom->c_per_lane);

    /* loop_args = [(acc_name, acc0)] ; Python defaults iv_name/acc_name. */
    loop_args[0].name = (acc_name != NULL) ? acc_name : "acc";
    loop_args[0].init = acc0;

    /* Python emits the loop bounds in positional (lb, ub, step) order AFTER the
     * iter-arg init (acc0 above). Hoist the bound constants into explicit
     * statements so the value-counter order matches Python -- C argument
     * evaluation order is unspecified (GCC is right-to-left). */
    lb = rocke_b_const_i32(b, 0);
    ub = rocke_b_const_i32(b, n_tiles);
    step = rocke_b_const_i32(b, 1);
    kloop = rocke_b_scf_for_iter(b,
                                 lb,
                                 ub,
                                 step,
                                 loop_args,
                                 1,
                                 (iv_name != NULL) ? iv_name : "kt",
                                 /*unroll=*/false,
                                 /*elide_trailing_barrier=*/true);

    rocke_b_region_enter(b, kloop.body);
    {
        kt = kloop.iv;
        acc_v = kloop.iter_vars[0];

        a_vec = load_a(b, kt, user);
        b_vec = load_b(b, kt, user);
        /* atom.emit(b, a_vec, b_vec, acc_v) -> b.mma(atom.name, ...) */
        new_acc = rocke_b_mma(b, atom->name, a_vec, b_vec, acc_v, NULL, 0);
        if(per_tile_post_mfma != NULL)
        {
            new_acc = per_tile_post_mfma(b, new_acc, kt, user);
        }
        yield_vals[0] = new_acc;
        rocke_b_scf_yield(b, yield_vals, 1);
    }
    rocke_b_region_leave(b);

    if(!rocke_i_live(b) || kloop.op == NULL || kloop.op->num_results < 1)
    {
        return NULL;
    }
    return kloop.op->results[0];
}

/* ============================================================== store_acc_to_global *
 *
 * Python:
 *     if epilogue is not None:
 *         epilogue(b, atom, lane_decode, C, m_tile_base, n_tile_base, acc, N,
 *                  out_dtype)
 *         return
 *     out_dtype_ir = F32 if out_dtype == "f32" else _ir_type_for_dtype(out_dtype)
 *     for i in range(atom.c_per_lane):
 *         row_in, col_in = atom.lane_to_output(b, lane_decode.lane, i)
 *         row = b.add(m_tile_base, row_in)
 *         col = b.add(n_tile_base, col_in)
 *         addr = b.add(b.mul(row, b.const_i32(N)), col)
 *         c_f32 = b.vec_extract(acc, i)
 *         if out_dtype == "f32": val = c_f32
 *         else: val = b.cast_f32_to(c_f32, out_dtype_ir)
 *         if atomic_add:
 *             if out_dtype != "f32": raise ValueError(...)
 *             b.global_atomic_add(C, addr, val)
 *         else:
 *             b.global_store(C, addr, val, align=4 if out_dtype=="f32" else 2)
 *
 * lane_to_output inlined (atoms.py:536-591).
 */

/* Inline of MfmaAtom.lane_to_output(b, lane, i): writes (row_in, col_in) for
 * accumulator slot i. Returns true on a supported (m,n); false sets the sticky
 * NotImplementedError (the Python `raise NotImplementedError` path). */
static bool rocke_i_lane_to_output(rocke_ir_builder_t* b,
                                   const rocke_mfma_atom_t* atom,
                                   rocke_value_t* lane,
                                   int i,
                                   rocke_value_t** out_row,
                                   rocke_value_t** out_col)
{
    if((atom->m == 16 && atom->n == 16))
    {
        rocke_value_t* c_atom_n = rocke_b_const_i32(b, atom->n);
        rocke_value_t* n_in_atom = rocke_b_mod(b, lane, c_atom_n);
        rocke_value_t* m_blk = rocke_b_div(b, lane, c_atom_n);
        /* Python: row = b.add(b.mul(m_blk, b.const_i32(c_per_lane)),
         *                     b.const_i32(i)). Sequence each builder call so the
         * value counter matches Python's left-to-right argument evaluation
         * (const c_per_lane -> mul -> const i -> add); C arg order is
         * unspecified. */
        rocke_value_t* c_cpl = rocke_b_const_i32(b, atom->c_per_lane);
        rocke_value_t* row_mul = rocke_b_mul(b, m_blk, c_cpl);
        rocke_value_t* c_i = rocke_b_const_i32(b, i);
        rocke_value_t* row = rocke_b_add(b, row_mul, c_i);
        *out_row = row;
        *out_col = n_in_atom;
        return true;
    }
    if((atom->m == 32 && atom->n == 32))
    {
        rocke_value_t* c_atom_n = rocke_b_const_i32(b, atom->n);
        rocke_value_t* n_in_atom = rocke_b_mod(b, lane, c_atom_n);
        rocke_value_t* m_blk = rocke_b_div(b, lane, c_atom_n);
        int rb = i / 4;
        int ri = i % 4;
        /* Python: row = b.add(b.add(b.const_i32(rb*8),
         *                           b.mul(m_blk, b.const_i32(4))),
         *                     b.const_i32(ri)). Left-to-right order:
         * const(rb*8) -> const(4) -> mul -> add(inner) -> const(ri) ->
         * add(outer). */
        rocke_value_t* c_rb = rocke_b_const_i32(b, rb * 8);
        rocke_value_t* c_4 = rocke_b_const_i32(b, 4);
        rocke_value_t* mblk4 = rocke_b_mul(b, m_blk, c_4);
        rocke_value_t* row_inner = rocke_b_add(b, c_rb, mblk4);
        rocke_value_t* c_ri = rocke_b_const_i32(b, ri);
        rocke_value_t* row = rocke_b_add(b, row_inner, c_ri);
        *out_row = row;
        *out_col = n_in_atom;
        return true;
    }
    if((atom->m == 4 && atom->n == 4))
    {
        rocke_value_t* c4 = rocke_b_const_i32(b, 4);
        rocke_value_t* lane_in_batch = rocke_b_mod(b, lane, c4);
        *out_row = rocke_b_const_i32(b, i);
        *out_col = lane_in_batch;
        return true;
    }
    rocke_i_set_err(
        b, ROCKE_ERR_NOTIMPL, "no lane_to_output dispatch for atom %dx%d", atom->m, atom->n);
    return false;
}

rocke_status_t rocke_store_acc_to_global(rocke_ir_builder_t* b,
                                         rocke_value_t* C,
                                         const rocke_mfma_atom_t* atom,
                                         const rocke_lane_decode_t* lane_decode,
                                         rocke_value_t* m_tile_base,
                                         rocke_value_t* n_tile_base,
                                         rocke_value_t* acc,
                                         int N,
                                         const char* out_dtype,
                                         bool atomic_add,
                                         rocke_mfma_epilogue_fn epilogue,
                                         void* epilogue_user)
{
    bool is_f32;
    const rocke_type_t* out_dtype_ir;
    int i;

    if(!rocke_i_live(b))
    {
        return b ? b->status : ROCKE_ERR_VALUE;
    }
    if(atom == NULL || lane_decode == NULL)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "store_acc_to_global: atom/lane_decode must be non-NULL");
        return b->status;
    }

    /* out_dtype default "f16" (Python keyword default). */
    if(out_dtype == NULL)
    {
        out_dtype = "f16";
    }

    /* if epilogue is not None: epilogue(...); return */
    if(epilogue != NULL)
    {
        epilogue(
            b, atom, lane_decode, C, m_tile_base, n_tile_base, acc, N, out_dtype, epilogue_user);
        return b->status;
    }

    is_f32 = (strcmp(out_dtype, "f32") == 0);

    /* out_dtype_ir = F32 if out_dtype == "f32" else _ir_type_for_dtype(out_dtype) */
    if(is_f32)
    {
        out_dtype_ir = rocke_f32();
    }
    else
    {
        out_dtype_ir = rocke_i_ir_type_for_dtype(out_dtype);
        if(out_dtype_ir == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "unsupported dtype '%s'", out_dtype);
            return b->status;
        }
    }

    for(i = 0; i < atom->c_per_lane; ++i)
    {
        rocke_value_t* row_in;
        rocke_value_t* col_in;
        rocke_value_t* row;
        rocke_value_t* col;
        rocke_value_t* addr;
        rocke_value_t* c_f32;
        rocke_value_t* val;

        if(!rocke_i_lane_to_output(b, atom, lane_decode->lane, i, &row_in, &col_in))
        {
            return b->status;
        }
        row = rocke_b_add(b, m_tile_base, row_in);
        col = rocke_b_add(b, n_tile_base, col_in);
        addr = rocke_b_add(b, rocke_b_mul(b, row, rocke_b_const_i32(b, N)), col);
        c_f32 = rocke_b_vec_extract(b, acc, i);

        if(is_f32)
        {
            val = c_f32;
        }
        else
        {
            val = rocke_b_cast_f32_to(b, c_f32, out_dtype_ir);
        }

        if(atomic_add)
        {
            /* if out_dtype != "f32": raise ValueError */
            if(!is_f32)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "atomic_add output requires out_dtype='f32'");
                return b->status;
            }
            rocke_b_global_atomic_add(b, C, addr, val, NULL);
        }
        else
        {
            rocke_b_global_store(b, C, addr, val, is_f32 ? 4 : 2);
        }
    }
    return b->status;
}

/* ====================================================== validate_arch_and_block_size *
 *
 * Python:
 *     try: target = ArchTarget.from_gfx(arch)
 *     except KeyError as e: return False, str(e), None
 *     if block_size > target.max_threads_per_block:
 *         return (False, f"block_size {block_size} > "
 *                        f"{target.max_threads_per_block} (hardware cap) on {arch}",
 *                 target)
 *     return True, "ok", target
 *
 * The reason strings never enter IR; they surface only through ValueError
 * messages. The from_gfx KeyError str(e) is reconstructed (the Python KeyError
 * repr wraps the message in double quotes). The known-arches list + data file
 * name come from the canonical arch port. */
bool rocke_validate_arch_and_block_size(rocke_ir_builder_t* b,
                                        const char* arch,
                                        int block_size,
                                        const char** out_reason,
                                        const rocke_archtarget_t** out_target)
{
    const rocke_archtarget_t* target;

    if(out_target != NULL)
    {
        *out_target = NULL;
    }

    /* try: target = ArchTarget.from_gfx(arch) except KeyError: */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        if(out_reason != NULL)
        {
            /* Reconstruct str(KeyError(...)): the KeyError repr wraps the
             * message in double-quotes, and the message is
             *   unknown gfx target {arch!r}; known: {sorted}. Add a row to
             *   arch_specs.json.
             * Built into the builder arena (never reaches IR). */
            char known[512];
            const char* const* arches;
            int count = 0;
            int k;
            size_t pos = 0;

            arches = rocke_known_arches(&count);
            known[pos++] = '[';
            for(k = 0; k < count && arches != NULL && pos + 8 < sizeof(known); ++k)
            {
                int wrote;
                wrote = snprintf(
                    known + pos, sizeof(known) - pos, "%s'%s'", (k == 0) ? "" : ", ", arches[k]);
                if(wrote < 0)
                {
                    break;
                }
                pos += (size_t)wrote;
            }
            if(pos + 1 < sizeof(known))
            {
                known[pos++] = ']';
            }
            known[pos] = '\0';

            if(b != NULL)
            {
                *out_reason
                    = rocke_arena_printf(&b->arena,
                                         "\"unknown gfx target '%s'; known: %s. Add a row to "
                                         "arch_specs.json.\"",
                                         arch ? arch : "None",
                                         known);
            }
            else
            {
                *out_reason = "unknown gfx target";
            }
        }
        return false;
    }

    /* if block_size > target.max_threads_per_block: */
    if(block_size > rocke_archtarget_max_threads_per_block(target))
    {
        if(out_target != NULL)
        {
            *out_target = target;
        }
        if(out_reason != NULL)
        {
            if(b != NULL)
            {
                *out_reason = rocke_arena_printf(&b->arena,
                                                 "block_size %d > %d (hardware cap) on %s",
                                                 block_size,
                                                 rocke_archtarget_max_threads_per_block(target),
                                                 arch ? arch : "None");
            }
            else
            {
                *out_reason = "block_size over hardware cap";
            }
        }
        return false;
    }

    /* return True, "ok", target */
    if(out_target != NULL)
    {
        *out_target = target;
    }
    if(out_reason != NULL)
    {
        *out_reason = "ok";
    }
    return true;
}

/* ==================================================== validate_mfma_atom_in_catalog *
 *
 * Python:
 *     target = ArchTarget.from_gfx(arch)
 *     if not target.mma.has_shape(a_dtype=atom.dtype_in, b_dtype=atom.dtype_in,
 *             c_dtype=atom.dtype_out, m=atom.m, n=atom.n, k=atom.k):
 *         raise NotImplementedError(
 *             f"{where} MFMA atom {atom.name!r} "
 *             f"({atom.dtype_in} {atom.m}x{atom.n}x{atom.k}) is not in the "
 *             f"{arch} MMA catalog; this configuration requires a different "
 *             f"target.")
 */
rocke_status_t rocke_validate_mfma_atom_in_catalog(rocke_ir_builder_t* b,
                                                   const rocke_mfma_atom_t* atom,
                                                   const char* arch,
                                                   const char* where)
{
    const rocke_archtarget_t* target;
    const rocke_arch_mma_catalog_t* mma;

    if(!rocke_i_live(b))
    {
        return b ? b->status : ROCKE_ERR_VALUE;
    }
    if(atom == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "validate_mfma_atom_in_catalog: atom must be non-NULL");
        return b->status;
    }

    /* target = ArchTarget.from_gfx(arch) -- Python would raise KeyError on an
     * unknown gfx; here a NULL target is reported as a value error (the
     * uncatchable-name path). */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "unknown gfx target '%s'", arch ? arch : "None");
        return b->status;
    }
    mma = rocke_archtarget_mma(target);

    /* if not target.mma.has_shape(...): raise NotImplementedError(...) */
    if(!rocke_mma_catalog_has_shape(mma,
                                    /*family=*/NULL,
                                    atom->dtype_in,
                                    atom->dtype_in,
                                    atom->dtype_out,
                                    atom->m,
                                    atom->n,
                                    atom->k))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_NOTIMPL,
                        "%s MFMA atom '%s' (%s %dx%dx%d) is not in the %s MMA catalog; "
                        "this configuration requires a different target.",
                        where ? where : "None",
                        atom->name ? atom->name : "None",
                        atom->dtype_in ? atom->dtype_in : "None",
                        atom->m,
                        atom->n,
                        atom->k,
                        arch ? arch : "None");
        return b->status;
    }
    return ROCKE_OK;
}
