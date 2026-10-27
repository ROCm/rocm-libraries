// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke_py_adapters.hpp -- adapters between Python helper shapes and their C
 * twins, for C++ translated from the attention backward kernel builders
 * (tr/adapters.py lists them).
 *
 * PRIVATE: support for translated sources in this directory; not installed,
 * not API; EXPERIMENTAL. Package data of tools/translator (attention_bwd):
 * tr.regen copies it next to the sources that use it and never over an
 * existing copy.
 */
#ifndef ROCKE_ROCKE_PY_ADAPTERS_HPP
#define ROCKE_ROCKE_PY_ADAPTERS_HPP

#include <array>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"

#include "rocke_py.hpp"
#include <string.h>

namespace py_adapt
{

inline const char* c_str(const char* s)
{
    return s;
}
inline const char* c_str(const std::string& s)
{
    return s.c_str();
}

/* IRBuilder(name) on a builder the C caller owns. kernel_entry() initialises it
 * with a placeholder when a check raises before this runs, so the caller can
 * always read the error and free the builder. */
inline thread_local bool g_builder_ready = false;

inline void init_builder(rocke_ir_builder_t* b, const char* name)
{
    if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        throw ckc::OOMError("builder init failed");
    g_builder_ready = true;
}

template <class Fn>
rocke_kernel_def_t* kernel_entry(rocke_ir_builder_t* b, Fn&& fn)
{
    g_builder_ready = false;
    try
    {
        return fn();
    }
    catch(const ckc::Error& e)
    {
        if(!g_builder_ready)
            init_builder(b, "rocke_invalid_spec");
        rocke_i_set_err_msg(b, e.code(), e.what());
        return nullptr;
    }
}

/* ArchTarget.from_gfx(gfx): KeyError for an unknown target. */
inline const rocke_arch_target_t* arch_target(const char* gfx)
{
    const rocke_arch_target_t* t = rocke_arch_target_from_gfx(gfx);
    if(t == nullptr)
        throw ckc::KeyError(py::repr(gfx));
    return t;
}

/* [(name, init), ...] of scf_for_iter; names stay owned by the caller's list. */
template <class L>
std::vector<rocke_iter_arg_t> iter_args(const L& l)
{
    std::vector<rocke_iter_arg_t> out;
    for(const auto& t : l)
    {
        rocke_iter_arg_t a;
        a.name = c_str(std::get<0>(t));
        a.init = std::get<1>(t);
        out.push_back(a);
    }
    return out;
}

/* **attrs of IRBuilder.param: NULL when empty, as the C API expects. */
inline const rocke_param_opts_t* param_opts(const py::attrs& a, rocke_param_opts_t* o)
{
    if(a.kv.empty())
        return nullptr;
    for(const auto& kv : a.kv)
    {
        if(kv.first == "readonly")
        {
            o->readonly = kv.second != 0;
            o->readonly_set = true;
        }
        else if(kv.first == "noalias")
        {
            o->noalias = kv.second != 0;
            o->noalias_set = true;
        }
        else if(kv.first == "writeonly")
        {
            o->writeonly = kv.second != 0;
            o->writeonly_set = true;
        }
        else if(kv.first == "align")
            o->align = (int)kv.second;
        else
            throw ckc::ValueError("unknown param attribute " + kv.first);
    }
    return o;
}

/* Host entries that report a Python raise as a return value plus a reason text. */
template <class R, class Fn>
R with_reason(char* reason, size_t cap, R fail, Fn&& fn)
{
    try
    {
        return fn();
    }
    catch(const ckc::Error& e)
    {
        if(reason != nullptr && cap > 0)
        {
            strncpy(reason, e.what(), cap - 1);
            reason[cap - 1] = '\0';
        }
        return fail;
    }
}

/* Entries that take the builder: a Python raise is recorded on the builder
 * (its sticky error) and the entry returns fail. */
template <class R, class Fn>
R on_builder(rocke_ir_builder_t* b, R fail, Fn&& fn)
{
    try
    {
        return fn();
    }
    catch(const ckc::Error& e)
    {
        rocke_i_set_err_msg(b, e.code(), e.what());
        return fail;
    }
}

/* A returned list the caller owns: *out = malloc'd copy of get(x) for each x,
 * *n_out = its length; free(*out) when done. */
template <class T, class S, class Get>
void own_out(const S& r, T** out, Get get)
{
    *out = (T*)malloc(sizeof(T) * (r.size() ? r.size() : 1));
    if(*out == nullptr)
        throw ckc::OOMError("out of memory");
    size_t i = 0;
    for(const auto& x : r)
        (*out)[i++] = (T)get(x);
}

/* A C helper that reports a Python raise through an err record {code, msg}:
 * throw it here, after the call, as the same exception. */
template <class E>
void check_err_void(const E& e)
{
    if(e.code != ROCKE_OK)
        ckc::raise_status(e.code, e.msg);
}

template <class R, class E>
R check_err(R r, const E& e)
{
    check_err_void(e);
    return r;
}

/* Host entries that return a Python str through (out, out_cap). */
template <class Fn>
rocke_status_t str_out(char* out, size_t cap, Fn&& fn)
{
    try
    {
        const std::string s = fn();
        if(out == nullptr || s.size() + 1 > cap)
            return ROCKE_ERR_VALUE;
        memcpy(out, s.c_str(), s.size() + 1);
        return ROCKE_OK;
    }
    catch(const ckc::Error& e)
    {
        return e.code();
    }
}

/* MmaOp.c_layout(): NotImplementedError when the op has no verified C lane map. */
inline const rocke_layout_map_t* c_layout(const rocke_mma_op_t* op)
{
    if(op == nullptr || op->c_layout == nullptr || op->c_layout->fn == nullptr)
        throw ckc::NotImplError(std::string("no verified C lane map for ")
                                + (op && op->op_id ? op->op_id : "(null)"));
    return op->c_layout;
}

/* MmaOp.a_layout() / b_layout(): NotImplementedError when the op has no verified C lane map. */
inline const rocke_layout_map_t* layout_of(const rocke_mma_op_t* op, const rocke_layout_map_t* m)
{
    if(m == nullptr || m->fn == nullptr)
        throw ckc::NotImplError(std::string("no verified C lane map for ")
                                + (op && op->op_id ? op->op_id : "(null)"));
    return m;
}

/* _attention_bwd_frag.eval_layout(layout): lanes[lane][slot] = (row, col). */
inline py::tuple<py::tuple<std::array<int64_t, 2>>> eval_layout(rocke_ir_builder_t* b,
                                                                const rocke_layout_map_t* m)
{
    const int lanes = m->wave_size;
    const int n = m->frag_len;
    std::vector<int> flat((size_t)lanes * (size_t)n * 2);
    if(!rocke_attn_bwd_eval_layout(b, m, flat.data(), (int)flat.size()))
        throw ckc::ValueError("eval_layout failed");
    py::tuple<py::tuple<std::array<int64_t, 2>>> out;
    for(int l = 0; l < lanes; ++l)
    {
        py::tuple<std::array<int64_t, 2>> lane;
        for(int s = 0; s < n; ++s)
        {
            const size_t k = ((size_t)l * (size_t)n + (size_t)s) * 2;
            lane.push_back({(int64_t)flat[k], (int64_t)flat[k + 1]});
        }
        out.push_back(lane);
    }
    return out;
}

/* ArchTarget.has_mfma / has_wmma: an atom of that family in the catalog. */
inline bool arch_has_family(const rocke_arch_target_t* t, const char* family)
{
    for(int i = 0; i < t->mma.num_ops; ++i)
    {
        const char* f = rocke_arch_mma_op_id_family(t->mma.ops[i].op_id);
        if(f != nullptr && strcmp(f, family) == 0)
            return true;
    }
    return false;
}

/* eval_layout with no builder in scope: on a scratch builder of its own
 * (the lane map's integer ops are interpreted; nothing reaches a kernel). */
inline py::tuple<py::tuple<std::array<int64_t, 2>>> eval_layout_scratch(const rocke_layout_map_t* m)
{
    rocke_ir_builder_t s;
    if(rocke_ir_builder_init(&s, "rocke_eval_layout_scratch") != ROCKE_OK)
        throw ckc::OOMError("scratch builder init failed");
    try
    {
        auto r = eval_layout(&s, m);
        rocke_ir_builder_free(&s);
        return r;
    }
    catch(...)
    {
        rocke_ir_builder_free(&s);
        throw;
    }
}

/* U2 host twins of platform emitters (designs/host_clones.md): the IR twin runs
 * on a scratch builder, and the values it returns are folded with the host
 * builder's own methods, ops(name, a, c) (name: the IRBuilder method). */
struct host_scratch
{
    rocke_ir_builder_t b;
    host_scratch()
    {
        if(rocke_ir_builder_init(&b, "rocke_host_fold") != ROCKE_OK)
            throw ckc::OOMError("scratch builder init failed");
    }
    ~host_scratch()
    {
        rocke_ir_builder_free(&b);
    }
    host_scratch(const host_scratch&) = delete;
    host_scratch& operator=(const host_scratch&) = delete;
};

template <class Ops>
int64_t host_fold(const rocke_value_t* v, Ops&& ops)
{
    if(v == nullptr || v->op == nullptr)
        throw ckc::ValueError("host fold: not an integer program of the inputs");
    const rocke_op_t* op = v->op;
    if(op->opcode == ROCKE_OP_ARITH_CONSTANT)
    {
        int64_t x = 0;
        if(!rocke_attr_get_int(&op->attrs, "value", &x))
            throw ckc::ValueError("host fold: a constant with no integer value");
        return ops("const_i32", x, 0);
    }
    const char* name = nullptr;
    switch(op->opcode)
    {
    case ROCKE_OP_ARITH_ADD:
        name = "add";
        break;
    case ROCKE_OP_ARITH_SUB:
        name = "sub";
        break;
    case ROCKE_OP_ARITH_MUL:
        name = "mul";
        break;
    case ROCKE_OP_ARITH_DIV:
        name = "div";
        break;
    case ROCKE_OP_ARITH_MOD:
        name = "mod";
        break;
    case ROCKE_OP_ARITH_XOR:
        name = "xor";
        break;
    case ROCKE_OP_ARITH_SHL:
        name = "shl";
        break;
    case ROCKE_OP_ARITH_LSHR:
        name = "lshr";
        break;
    case ROCKE_OP_ARITH_AND:
        name = "land";
        break;
    case ROCKE_OP_ARITH_OR:
        name = "lor";
        break;
    case ROCKE_OP_ARITH_SMAX:
        name = "smax";
        break;
    case ROCKE_OP_ARITH_SMIN:
        name = "smin";
        break;
    default:
        break;
    }
    if(name == nullptr || op->num_operands != 2)
        throw ckc::ValueError("host fold: an op the host builder does not fold");
    const int64_t a = host_fold(op->operands[0], ops);
    const int64_t c = host_fold(op->operands[1], ops);
    return ops(name, a, c);
}

/* == of two int | Value: both ints and equal, or the same Value */
/* the platform's IntOrValue (rocke_int_or_value_t) as the backward one:
 * IV is deduced from the caller, so this header needs neither header */
template <class IV = struct rocke_attn_bwd_iv, class P>
IV iv_of(const P& p)
{
    IV x{};
    x.v = p.is_value ? p.v : nullptr;
    x.i = p.is_value ? 0 : p.i;
    return x;
}

/* an int | Value used as an IR operand: Python raises on an int */
template <class IV>
rocke_value_t* iv_value(const IV& x)
{
    if(x.v == nullptr)
        throw ckc::TypeError("an int where an IR value is needed");
    return x.v;
}

/* an int | Value used as a host int: Python fails on a Value there */
template <class IV>
int64_t iv_int_of(const IV& x)
{
    if(x.v != nullptr)
        throw ckc::TypeError("an IR value where an int is needed");
    return x.i;
}

template <class IV> /* rocke_attn_bwd_iv_t: a template, so this header needs no addr header */
bool iv_eq(const IV& a, const IV& b)
{
    return a.v == b.v && (a.v != nullptr || a.i == b.i);
}

/* MmaCatalog.enumerate(family=, a_dtype=, b_dtype=, c_dtype=, m=, n=): the
 * matching atoms in catalog order; m / n < 0 is Python's None. */
inline std::vector<const rocke_mma_op_t*> mma_enumerate(const rocke_mma_catalog_t* cat,
                                                        const char* family,
                                                        const char* a_dtype,
                                                        const char* b_dtype,
                                                        const char* c_dtype,
                                                        int64_t m,
                                                        int64_t n)
{
    const int count = rocke_mma_catalog_enumerate(
        cat, family, a_dtype, b_dtype, c_dtype, (int)m, (int)n, nullptr, 0);
    std::vector<const rocke_mma_op_t*> out((size_t)(count > 0 ? count : 0));
    if(count > 0)
        rocke_mma_catalog_enumerate(
            cat, family, a_dtype, b_dtype, c_dtype, (int)m, (int)n, out.data(), count);
    return out;
}

/* A C entry that took the builder failed: its sticky error, as an exception */
[[noreturn]] inline void raise_builder(rocke_ir_builder_t* b)
{
    const char* m = rocke_ir_builder_error(b);
    ckc::raise_status(rocke_ir_builder_status(b) != ROCKE_OK ? rocke_ir_builder_status(b)
                                                             : ROCKE_ERR_VALUE,
                      m ? m : "builder error");
}

/* A string of an owned list: a malloc'd copy the caller frees. */
inline const char* own_str(const std::string& s)
{
    char* p = (char*)malloc(s.size() + 1);
    if(p == nullptr)
        throw ckc::OOMError("out of memory");
    memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

inline const char* own_str(const char* s)
{
    return own_str(std::string(s ? s : ""));
}

/* IRBuilder.inline_asm_multi(..., result_name_hint=): the C entry has no
 * name hint, so the tile.inline_asm op is built here with the same attrs. */
inline std::vector<rocke_value_t*>
    inline_asm_multi(rocke_ir_builder_t* b,
                     const char* tmpl,
                     const char* constraints,
                     const std::vector<rocke_value_t*>& operands,
                     const std::vector<const rocke_type_t*>& result_types,
                     bool sideeffect,
                     bool convergent,
                     const char* name_hint)
{
    rocke_attr_map_t attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "template", tmpl);
    rocke_attr_set_str(b, &attrs, "constraints", constraints);
    rocke_attr_set_bool(b, &attrs, "sideeffect", sideeffect);
    rocke_attr_set_bool(b, &attrs, "convergent", convergent);
    rocke_op_t* op = rocke_b_op(b,
                                ROCKE_OP_TILE_INLINE_ASM,
                                const_cast<rocke_value_t**>(operands.data()),
                                (int)operands.size(),
                                const_cast<const rocke_type_t**>(result_types.data()),
                                (int)result_types.size(),
                                &attrs,
                                NULL,
                                0,
                                name_hint,
                                NULL);
    if(op == NULL || !rocke_ir_builder_ok(b))
        raise_builder(b);
    return std::vector<rocke_value_t*>(op->results, op->results + op->num_results);
}

} // namespace py_adapt

#endif /* ROCKE_ROCKE_PY_ADAPTERS_HPP */
