/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_hiptensor_aot_internal.h -- C++ mirror of
 * hiptensor_aot/common.py (the Emit helpers, operator codes, dtype table) for
 * the hiptensor_aot builders. Internal to the engine (C++ only).
 *
 * Every helper makes the builder calls of its Python twin, in Python's order:
 * an argument expression that emits ops is computed in its own statement
 * before the call that consumes it.
 */
#ifndef ROCKE_INSTANCE_HIPTENSOR_AOT_INTERNAL_H
#define ROCKE_INSTANCE_HIPTENSOR_AOT_INTERNAL_H

#ifndef __cplusplus
#error "rocke/instance_hiptensor_aot_internal.h is C++ only"
#endif

#include "rocke/instance_hiptensor_aot.h"
#include "rocke/ir.h"

#include <string>
#include <utility>
#include <vector>

namespace rocke
{
namespace htaot
{

using V = rocke_value_t;

/* hiptensorOperator_t values (common.OP_*). */
enum : int
{
    OP_IDENTITY = 1,
    OP_SQRT = 2,
    OP_ADD = 3,
    OP_MUL = 5,
    OP_MAX = 6,
    OP_MIN = 7,
    OP_RELU = 8,
    OP_CONJ = 9,
    OP_RCP = 10,
    OP_SIGMOID = 11,
    OP_TANH = 12,
    OP_EXP = 22,
    OP_LOG = 23,
    OP_ABS = 24,
    OP_NEG = 25,
    OP_SIN = 26,
    OP_COS = 27,
    OP_TAN = 28,
    OP_SINH = 29,
    OP_COSH = 30,
    OP_ASIN = 31,
    OP_ACOS = 32,
    OP_ATAN = 33,
    OP_ASINH = 34,
    OP_ACOSH = 35,
    OP_ATANH = 36,
    OP_CEIL = 37,
    OP_FLOOR = 38,
};

constexpr int MAX_RANK = ROCKE_HTAOT_MAX_RANK;

/* common.DTYPES: IR type and element size; false for an unknown name. */
bool dtype_of(const char* dt, const rocke_type_t** ty, int* nbytes);
bool is_dtype(const char* dt);
bool streq(const char* a, const char* b);
bool present(const char* s); /* not NULL and not "" */

/* common.PACK_TARGETS / vector_bytes / max_vec / lds_bytes, from the target catalog
 * (rocke/arch_target.h): the widest global vector access in bytes (max_vector_load_dwords * 4;
 * the narrowest over the pack targets), the elements of nbytes bytes in it, and the LDS capacity
 * of `arch` (nullptr: the largest over the pack targets). Each returns 0 where the catalog gives
 * no positive value (the Python functions raise). */
int vector_bytes();
int max_vec(int nbytes);
int lds_bytes(const char* arch);

/* Python common.Emit. */
class Emit
{
public:
    Emit(rocke_ir_builder_t* b, bool idx64, const char* cdt, bool math);

    rocke_ir_builder_t* b;
    bool math;
    bool idx64;
    bool f64; /* cdt == "f64" */
    const rocke_type_t* cty;

    /* constants / params */
    V* c32(int64_t v);
    V* cf(double v);
    V* cf_f32(double v);
    V* zero_off();
    V* stride_param(const char* name);
    V* scalar_param(const char* name);

    /* integer helpers */
    V* udiv(V* x, V* m, V* s);
    std::pair<V*, V*> divmod(V* x, V* d, V* m, V* s);
    V* off(V* coord, V* stride);

    /* memory */
    std::pair<V*, V*> addr(V* ptr, V* off, int nbytes);
    std::vector<V*> load(V* ptr, V* off, const char* dt, bool cplx = false);
    void store(V* ptr, V* off, const std::vector<V*>& vals, const char* dt);

    /* conversions */
    V* to_c(V* v, const char* dt);
    V* from_c(V* v, const char* dt);
    V* narrow(V* v, const rocke_type_t* ty);
    V* f64_to_f32_odd(V* v);

    /* uniform branches: fn(vals) when cond holds, else vals */
    template <class Fn>
    std::vector<V*> when(V* cond, const std::vector<V*>& vals, Fn&& fn)
    {
        rocke_for_t loop = when_open(cond, vals);
        std::vector<V*> cur(loop.iter_vars, loop.iter_vars + loop.num_iter_vars);
        std::vector<V*> out = fn(cur);
        return when_close(loop, out);
    }

    /* operators */
    V* unary(V* v, V* code);
    std::vector<V*> unary_list(const std::vector<V*>& vs, V* code);
    std::vector<V*> unary_list(const std::vector<V*>& vs, V* code, bool math);
    std::vector<V*> unary_inputs(const std::vector<V*>& vs, const std::vector<V*>& codes);
    V* sigmoid(V* y);
    std::vector<V*> math_ops(const std::vector<V*>& xs, V* code);
    V* binary(V* a, V* c, V* code, bool nan_max = true);
    std::pair<V*, V*> cmul(V* xr, V* xi, V* yr, V* yi);

    int next_uid()
    {
        return uid_++;
    }

private:
    rocke_for_t when_open(V* cond, const std::vector<V*>& vals);
    std::vector<V*> when_close(const rocke_for_t& loop, const std::vector<V*>& out);
    std::vector<V*> math_loop(const char* fam,
                              const std::vector<V*>& xs,
                              const std::vector<std::pair<const char*, V*>>& preds);
    V* math_family(const char* fam, V* y, const std::vector<std::pair<const char*, V*>>& preds);

    int uid_ = 0;
};

/* common.ptr(b, name, dt, readonly) */
V* ptr(rocke_ir_builder_t* b, const char* name, const char* dt, bool readonly);

/* A kernel parameter of type `t` with no attributes. */
V* param(rocke_ir_builder_t* b, const std::string& name, const rocke_type_t* t);

/* Structured control flow (Python `with b.scf_if(c):` / `with b.scf_for(...)`). */
template <class Fn>
void scf_if(rocke_ir_builder_t* b, V* cond, Fn&& fn)
{
    rocke_if_t f = rocke_b_scf_if(b, cond);
    rocke_b_region_enter(b, f.then_region);
    fn();
    rocke_b_region_leave(b);
}

/* Python `rocke_i_set_err` with a ValueError message; returns NULL. */
rocke_kernel_def_t* fail(rocke_ir_builder_t* b, const std::string& msg);

} // namespace htaot
} // namespace rocke

#endif /* ROCKE_INSTANCE_HIPTENSOR_AOT_INTERNAL_H */
