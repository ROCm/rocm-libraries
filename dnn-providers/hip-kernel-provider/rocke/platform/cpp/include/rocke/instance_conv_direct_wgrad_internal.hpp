/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_wgrad_internal.hpp -- PRIVATE helpers shared by the
 * C++ ports of the direct wgrad kernels (conv_direct_wgrad_tiled.cpp,
 * conv_direct_dw_wgrad.cpp, conv_direct_wgrad_gather.cpp). Emits no IR of its
 * own and declares no public API.
 *
 * BYTE-IDENTITY CONTRACT. The Python builders emit one op per builder call, in
 * Python's left-to-right argument order; C++ leaves argument evaluation order
 * unspecified. The `Em` helpers below therefore take at most ONE argument that
 * emits: `e.addi(x, 3)` is Python `b.add(x, ci(3))` (the constant, then the
 * add). Every call site in the ports keeps to the same rule -- a call may nest
 * one emitting argument, never two.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_WGRAD_INTERNAL_HPP
#define ROCKE_INSTANCE_CONV_DIRECT_WGRAD_INTERNAL_HPP

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err_msg */
#include "rocke/lower_llvm.h"

namespace rocke_dwg
{

using V = rocke_value_t*;

constexpr int kWave = 64;

inline int cdiv(int a, int b)
{
    return (a + b - 1) / b;
}

/* Python `a % m` for m > 0 (never negative). */
inline int pymod(int a, int m)
{
    int r = a % m;
    return r < 0 ? r + m : r;
}

/* Python `a // m` for m > 0 (floors). */
inline int pyfloordiv(int a, int m)
{
    int q = a / m;
    return (a % m != 0 && (a < 0) != (m < 0)) ? q - 1 : q;
}

/* A NULL dtype is not defaulted here: the spec-default constructors set
 * "fp16", and the validators reject NULL like Python rejects dtype=None. */
inline bool problem_is_bf16(const rocke_direct_conv_problem_t* p)
{
    return p->dtype != NULL && strcmp(p->dtype, "bf16") == 0;
}

inline bool problem_dtype_ok(const rocke_direct_conv_problem_t* p)
{
    const char* d = p->dtype;
    return d != NULL && (strcmp(d, "fp16") == 0 || strcmp(d, "bf16") == 0);
}

/* DirectConvProblem.Ho / .Wo. */
inline int problem_ho(const rocke_direct_conv_problem_t* p)
{
    return pyfloordiv(p->H + 2 * p->PAD - (p->KH - 1) * p->dil_h - 1, p->stride) + 1;
}

inline int problem_wo(const rocke_direct_conv_problem_t* p)
{
    return pyfloordiv(p->W + 2 * p->PAD - (p->KW - 1) * p->dil_w - 1, p->stride) + 1;
}

inline void set_reason(char* out, size_t cap, const char* msg)
{
    if(out != NULL && cap > 0)
    {
        snprintf(out, cap, "%s", msg);
    }
}

/* conv_direct_grouped.dilation_reason. */
inline bool dilation_reason(const rocke_direct_conv_problem_t* p, char* why, size_t cap)
{
    if(p->dil_h != 1 || p->dil_w != 1)
    {
        ROCKE_ERR_SNPRINTF(why,
                           cap,
                           "this direct conv kernel has no dilation (got %dx%d); only the "
                           "non-grouped forward kernel computes a dilated convolution",
                           p->dil_h,
                           p->dil_w);
        return true;
    }
    return false;
}

/* `raise ValueError(f"{prefix}: {reason}")` on the builder. */
inline void fail_invalid(rocke_ir_builder_t* b, const char* prefix, const char* reason)
{
    if(b->status == ROCKE_OK)
    {
        std::string msg = std::string(prefix) + ": " + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
    }
}

/* Emission helpers (see the contract at the top of this header). */
struct Em
{
    rocke_ir_builder_t* b;

    V ci(int64_t v) const
    {
        return rocke_b_const_i32(b, v);
    }

    V add(V x, V y) const
    {
        return rocke_b_add(b, x, y);
    }
    V sub(V x, V y) const
    {
        return rocke_b_sub(b, x, y);
    }
    V mul(V x, V y) const
    {
        return rocke_b_mul(b, x, y);
    }
    V div(V x, V y) const
    {
        return rocke_b_div(b, x, y);
    }
    V mod(V x, V y) const
    {
        return rocke_b_mod(b, x, y);
    }
    V smin(V x, V y) const
    {
        return rocke_b_smin(b, x, y);
    }
    V smax(V x, V y) const
    {
        return rocke_b_smax(b, x, y);
    }
    V land(V x, V y) const
    {
        return rocke_b_land(b, x, y);
    }
    V lor(V x, V y) const
    {
        return rocke_b_lor(b, x, y);
    }
    V lnot(V x) const
    {
        return rocke_b_lnot(b, x);
    }
    V bxor(V x, V y) const
    {
        return rocke_b_xor(b, x, y);
    }
    V lt(V x, V y) const
    {
        return rocke_b_cmp_lt(b, x, y);
    }
    V ge(V x, V y) const
    {
        return rocke_b_cmp_ge(b, x, y);
    }
    V eq(V x, V y) const
    {
        return rocke_b_cmp_eq(b, x, y);
    }
    V select(V c, V x, V y) const
    {
        return rocke_b_select(b, c, x, y);
    }
    V zext64(V x) const
    {
        return rocke_b_zext(b, x, rocke_i64());
    }

    /* Python b.op(x, ci(v)): the constant first, then the op. */
    V addi(V x, int64_t v) const
    {
        V k = ci(v);
        return add(x, k);
    }
    V subi(V x, int64_t v) const
    {
        V k = ci(v);
        return sub(x, k);
    }
    V muli(V x, int64_t v) const
    {
        V k = ci(v);
        return mul(x, k);
    }
    V divi(V x, int64_t v) const
    {
        V k = ci(v);
        return div(x, k);
    }
    V modi(V x, int64_t v) const
    {
        V k = ci(v);
        return mod(x, k);
    }
    V smini(V x, int64_t v) const
    {
        V k = ci(v);
        return smin(x, k);
    }
    V lti(V x, int64_t v) const
    {
        V k = ci(v);
        return lt(x, k);
    }
    V gei(V x, int64_t v) const
    {
        V k = ci(v);
        return ge(x, k);
    }
    V eqi(V x, int64_t v) const
    {
        V k = ci(v);
        return eq(x, k);
    }
    V lshri(V x, int64_t v) const
    {
        V k = ci(v);
        return rocke_b_lshr(b, x, k);
    }
    V shli(V x, int64_t v) const
    {
        V k = ci(v);
        return rocke_b_shl(b, x, k);
    }

    /* Python b.div(b.add(x, ci(d - 1)), ci(d)): ceil-divide by a constant. */
    V cdivi(V x, int64_t d) const
    {
        V t = addi(x, d - 1);
        return divi(t, d);
    }

    /* (2-D smem) Python b.smem_load_vN(smem, c0, idx, dtype=..., n=...). */
    V smem_load(V smem, V c0, V idx, const rocke_type_t* dtype, int n) const
    {
        V ix[2] = {c0, idx};
        return rocke_b_smem_load_vN(b, smem, ix, 2, dtype, n);
    }
    V smem_load_f32(V smem, V c0, V idx, int n) const
    {
        V ix[2] = {c0, idx};
        return rocke_b_smem_load_vN_f32(b, smem, ix, 2, n);
    }
    void smem_store(V smem, V c0, V idx, V value, int n) const
    {
        V ix[2] = {c0, idx};
        rocke_b_smem_store_vN(b, smem, ix, 2, value, n);
    }
    void smem_store_f32(V smem, V c0, V idx, V value, int n) const
    {
        V ix[2] = {c0, idx};
        rocke_b_smem_store_vN_f32(b, smem, ix, 2, value, n);
    }
    V ds_read_tr(V smem, V c0, V idx, const rocke_type_t* dtype) const
    {
        V ix[2] = {c0, idx};
        return rocke_b_ds_read_tr16_b64(b, smem, ix, 2, dtype);
    }
    V smem_alloc_1d(const rocke_type_t* elem, int n, const char* hint) const
    {
        int shape[2] = {1, n};
        return rocke_b_smem_alloc(b, elem, shape, 2, hint);
    }

    /* Python swz_val(row, bits): XOR term 2 * sum(((row >> bits[j]) & 1) << j),
     * or NULL for no swizzle. */
    V swz_val(V row, const std::vector<int>& bits, V c1) const
    {
        V v = NULL;
        for(size_t j = 0; j < bits.size(); ++j)
        {
            V t = lshri(row, bits[j]);
            t = land(t, c1);
            t = shli(t, (int64_t)j + 1);
            v = (v == NULL) ? t : lor(v, t);
        }
        return v;
    }

    /* _mfma(b, dtype, "16x16x32", a, bb, acc). */
    V mfma16x16x32(bool bf16, V a, V bb, V acc) const
    {
        return bf16 ? rocke_b_mfma_f32_16x16x32_bf16(b, a, bb, acc)
                    : rocke_b_mfma_f32_16x16x32_f16(b, a, bb, acc);
    }

    /* `with b.scf_if(cond): body()`. */
    template <class Fn>
    void if_(V cond, Fn&& body) const
    {
        rocke_if_t f = rocke_b_scf_if(b, cond);
        if(f.then_region == NULL)
        {
            return;
        }
        rocke_b_region_enter(b, f.then_region);
        body();
        rocke_b_region_leave(b);
    }

    /* `loop = b.scf_for_iter(lo, hi, step, iter_args, iv_name=...,
     * elide_trailing_barrier=False)`; `with loop as (iv, carried): body`.
     * body(iv, carried) returns the yielded values; returns loop.results. */
    template <class Fn>
    std::vector<V> for_iter(V lo,
                            V hi,
                            V step,
                            const std::vector<std::string>& names,
                            const std::vector<V>& inits,
                            const char* iv_name,
                            Fn&& body) const
    {
        std::vector<rocke_iter_arg_t> args(names.size());
        for(size_t i = 0; i < names.size(); ++i)
        {
            args[i].name = names[i].c_str();
            args[i].init = inits[i];
        }
        rocke_for_t f = rocke_b_scf_for_iter(
            b, lo, hi, step, args.data(), (int)args.size(), iv_name, false, false);
        if(f.op == NULL || f.body == NULL)
        {
            return std::vector<V>();
        }
        std::vector<V> carried(f.iter_vars, f.iter_vars + f.num_iter_vars);
        rocke_b_region_enter(b, f.body);
        std::vector<V> out = body(f.iv, carried);
        rocke_b_scf_yield(b, out.data(), (int)out.size());
        rocke_b_region_leave(b);
        return std::vector<V>(f.op->results, f.op->results + f.op->num_results);
    }
};

/* The `_new` entry: zero the builder, initialise it with the spec's kernel
 * name, then build -- the Python IRBuilder(spec.kernel_name()) + build pair.
 * name_fn(out, cap) -> rocke_status_t; build_fn() -> rocke_kernel_def_t*. */
template <class NameFn, class BuildFn>
rocke_kernel_def_t* build_new(
    rocke_ir_builder_t* b, bool have_spec, const char* who, NameFn&& name_fn, BuildFn&& build_fn)
{
    if(b != NULL)
    {
        memset(b, 0, sizeof(*b));
    }
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[512];
        if(b == NULL)
        {
            return NULL;
        }
        if(!have_spec)
        {
            std::string msg = std::string(who) + ": null spec";
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
            return NULL;
        }
        if(name_fn(name, sizeof(name)) != ROCKE_OK)
        {
            std::string msg = std::string(who) + ": kernel name does not fit";
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return build_fn();
    });
}

/* The `lower_to_llvm` convenience entry: new_fn(&b) builds into a fresh
 * builder; the kernel is lowered at `flavor` for `arch`. */
template <class NewFn>
rocke_status_t lower_to_llvm(bool have_spec,
                             const char* arch,
                             rocke_llvm_flavor_t flavor,
                             char** out_ll,
                             char* err,
                             size_t err_cap,
                             const char* who,
                             NewFn&& new_fn)
{
    rocke_ir_builder_t b = {};
    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(!have_spec || out_ll == NULL)
    {
        set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    rocke_kernel_def_t* kernel = new_fn(&b);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        rocke_status_t st = rocke_ir_builder_status(&b);
        set_reason(err, err_cap, (m != NULL && m[0] != '\0') ? m : who);
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

/* Python [(f"{prefix}{i}", v) for i, v in enumerate(values)] names. */
inline std::vector<std::string> iter_names(const char* prefix, size_t n)
{
    std::vector<std::string> out;
    out.reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        out.push_back(std::string(prefix) + std::to_string(i));
    }
    return out;
}

} // namespace rocke_dwg

#endif /* ROCKE_INSTANCE_CONV_DIRECT_WGRAD_INTERNAL_HPP */
