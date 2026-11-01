/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/elementary.h -- C++ mirror of rocke/helpers/elementary.py: accurate
 * f64 elementary functions written in IR ops (sin, cos, tan, sinh, cosh, asin,
 * acos, atan, asinh, acosh, atanh, floor, ceil).
 *
 * Only the IR side (Python IRMath) is mirrored; the host model (HostMath) and
 * the coefficient generator stay Python-only. Every function emits the same
 * builder calls in the same order as the Python program, so the kernels that
 * use it (hipTensor's hiptensor_aot family, the GEMM prologue operand
 * operators) lower to the same LLVM IR in both engines.
 *
 * C entry points:
 *
 *   elementary.opaque_zero(b)          rocke_elementary_opaque_zero(b)
 *   elementary.emit(b, name, x, anch)  rocke_elementary_emit(b, name, x, anch)
 *
 * C++ callers that build a family program inside their own control flow (the
 * hiptensor_aot Emit.math_family) use rocke::elementary::Math and family().
 */
#ifndef ROCKE_ELEMENTARY_H
#define ROCKE_ELEMENTARY_H

#include <stdbool.h>
#include <stdint.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Python elementary.opaque_zero: an i64 zero the compiler cannot see through
 * (one `s_mov_b64` with side effects, result named "ozero"). */
rocke_value_t* rocke_elementary_opaque_zero(rocke_ir_builder_t* b);

/* Python elementary.emit(b, name, x, anchored): name(x) for an f32 or f64
 * value (f32: evaluated in f64, rounded once to f32). NULL (and the builder's
 * error set) for an unknown name or another type. */
rocke_value_t*
    rocke_elementary_emit(rocke_ir_builder_t* b, const char* name, rocke_value_t* x, bool anchored);

/* Python elementary.FUNCTIONS (13 names) and FAMILIES (6 groups). */
int rocke_elementary_num_functions(void);
const char* rocke_elementary_function(int i);
int rocke_elementary_num_families(void);
const char* rocke_elementary_family(int i);
/* Members of family `fam` in Python order; count returned, names in *out. */
int rocke_elementary_family_members(const char* fam, const char* const** out);
/* Python family_of(name); NULL for an unknown name. */
const char* rocke_elementary_family_of(const char* name);

#ifdef __cplusplus
} /* extern "C" */

#include <unordered_map>
#include <utility>
#include <vector>

namespace rocke
{
namespace elementary
{

/* A Python value of the math context: an IR value, a Python float, a Python
 * int (an i64 bit pattern) or a Python bool (a build-time predicate). */
struct MV
{
    enum Kind : uint8_t
    {
        VAL,
        FLT,
        INT,
        BOOL
    };
    Kind k = VAL;
    rocke_value_t* v = nullptr;
    double f = 0.0;
    int64_t i = 0;
    bool bv = false;

    MV() = default;
    MV(rocke_value_t* x)
        : k(VAL)
        , v(x)
    {
    }
    MV(double x)
        : k(FLT)
        , f(x)
    {
    }
    MV(int x)
        : k(INT)
        , i(x)
    {
    }
    MV(int64_t x)
        : k(INT)
        , i(x)
    {
    }
    MV(bool x)
        : k(BOOL)
        , bv(x)
    {
    }

    bool is_false() const
    {
        return k == BOOL && !bv;
    } /* Python `p is False` */
    bool is_true() const
    {
        return k == BOOL && bv;
    } /* Python `p is True`  */
    bool is_bool() const
    {
        return k == BOOL;
    }
};

/* Argument packs. A braced list evaluates its elements left to right, so
 * `m.add({m.mul({a, c}), d})` makes the builder calls in Python's order. */
struct A2
{
    MV a, c;
};
struct A3
{
    MV a, c, d;
};

/* Python IRMath (and _RecordingMath when `recording`). */
class Math
{
public:
    Math(rocke_ir_builder_t* b, rocke_value_t* anchor = nullptr)
        : b_(b)
        , anchor_(anchor)
    {
    }

    rocke_ir_builder_t* builder() const
    {
        return b_;
    }

    /* values */
    MV f(const MV& v);
    void materialize(const std::vector<double>& constants);
    rocke_value_t* k64(int64_t v);
    MV i(const MV& v);

    MV add(const A2& x);
    MV sub(const A2& x);
    MV mul(const A2& x);
    MV div(const A2& x);
    MV fma(const A3& x);
    MV sqrt(const MV& a);
    MV neg(const MV& a);
    MV abs(const MV& a);

    /* predicates */
    MV lt(const A2& x);
    MV le(const A2& x);
    MV gt(const A2& x);
    MV ge(const A2& x);
    MV eq(const A2& x);
    MV isnan(const MV& a);
    MV band(const A2& x);
    MV bor(const A2& x);
    MV bnot(const MV& p);
    MV sel(const A3& x);

    /* f64 encodings */
    MV bits(const MV& a);
    MV from_bits(const MV& n);
    MV iand(const A2& x);
    MV ior(const A2& x);
    MV ixor(const A2& x);
    MV ishl(const MV& n, int64_t s);
    MV ilshr(const MV& n, int64_t s);
    MV iadd(const A2& x);
    MV isub(const A2& x);
    MV ieq(const A2& x);

    /* hardware */
    MV trig_preop(const MV& x, int seg);

    /* control: fn(vals) for the lanes where cond holds, else vals */
    template <class Fn>
    std::vector<MV> when(const MV& cond, std::vector<MV> vals, Fn&& fn)
    {
        if(cond.is_bool())
            return cond.bv ? std::vector<MV>(fn(vals)) : vals;
        std::vector<rocke_value_t*> init = when_inputs(vals);
        rocke_for_t loop = when_open(cond, init);
        std::vector<MV> cur(loop.iter_vars, loop.iter_vars + loop.num_iter_vars);
        std::vector<MV> out = fn(cur);
        return when_close(loop, out);
    }

    /* _RecordingMath: f64 constants in first-use order and their use counts */
    bool recording = false;
    std::vector<double> seen;
    std::unordered_map<int64_t, int> uses;

private:
    std::vector<rocke_value_t*> when_inputs(const std::vector<MV>& vals);
    rocke_for_t when_open(const MV& cond, const std::vector<rocke_value_t*>& init);
    std::vector<MV> when_close(const rocke_for_t& loop, const std::vector<MV>& out);
    MV cmp(const char* pred, const A2& x);
    rocke_value_t* val(const MV& v); /* an MV that must already be an IR value */

    rocke_ir_builder_t* b_;
    rocke_value_t* anchor_;
    std::unordered_map<int64_t, rocke_value_t*> pool_;
};

/* A predicate of `family`: Python preds[name] (a bool or an i1 value). */
struct Pred
{
    const char* name;
    MV p;
};

/* Python family(m, fam, x, preds): the function of `fam` whose predicate
 * holds, applied to the f64 `x` (missing names are false). */
MV family(Math& m, const char* fam, const MV& x, const std::vector<Pred>& preds);

/* Python family_constants(fam, preds): the f64 constants the program uses
 * more than once, in first-use order (a recording build on a scratch builder;
 * memoized per (fam, predicate kinds)). */
std::vector<double> family_constants(const char* fam, const std::vector<Pred>& preds);

} // namespace elementary
} // namespace rocke

#endif /* __cplusplus */

#endif /* ROCKE_ELEMENTARY_H */
