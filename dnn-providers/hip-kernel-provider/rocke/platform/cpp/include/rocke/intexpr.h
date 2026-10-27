/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/intexpr.h -- the C++ evaluator of integer expressions (intexprs).
 *
 * An intexpr is a small JSON tree of integer arithmetic, read through the
 * json_dom DOM (jd_val_t); recipes use it for sizes, loop bounds and admission
 * guards. This evaluator mirrors rocke.core.intexpr.eval_intexpr in Python:
 * both must return the same value, or the same error text, for every tree, and
 * tests/core/test_intexpr.py pins both against one table of cases.
 *
 *   <intexpr> := integer
 *              | {"spec": NAME} | {"var": NAME} | {"param": NAME}
 *              | {"spec_str_eq": [NAME, literal]}
 *              | {"<OP>": [e, e]}  for OP in add sub mul div mod eq ne lt le gt ge
 *              | {"<FN>": e}       for FN in magic_multiplier magic_shift
 *
 * Evaluation is in int64_t. It refuses, with an error, where C and Python
 * could disagree: a number that is not an exact integer within +/-(2^53 - 1),
 * a div or mod by zero, a div or mod with a negative operand (C truncates,
 * Python floors), and an add, sub or mul whose result leaves int64.
 *
 * The evaluator owns no names. Leaves are resolved through a caller-supplied
 * lookup, so the recipe VM (spec values and loop indices) and the verifier
 * (kernel parameter names) share one evaluator.
 */
#ifndef ROCKE_INTEXPR_H
#define ROCKE_INTEXPR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/json_dom.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The kinds of leaf that name an integer. */
typedef enum
{
    ROCKE_INTEXPR_SPEC, /* {"spec": NAME}: a spec value bound at replay   */
    ROCKE_INTEXPR_VAR, /* {"var": NAME}: a compile-time loop index       */
    ROCKE_INTEXPR_PARAM /* {"param": NAME}: a kernel parameter's value     */
} rocke_intexpr_leaf_t;

/* Name lookup for the leaves. Either callback may be NULL, which binds
 * nothing of that kind. `lookup_int` returns false when `name` is not bound;
 * `lookup_str` returns NULL when the spec string `name` is not bound. */
typedef struct
{
    bool (*lookup_int)(void* ctx, rocke_intexpr_leaf_t kind, const char* name, int64_t* out);
    const char* (*lookup_str)(void* ctx, const char* name);
    void* ctx;
} rocke_intexpr_env_t;

/* Evaluate `expr` into *out. Returns false and writes the reason into
 * err/err_cap when the tree is malformed, names an unbound leaf, or is refused
 * (see above); *out is then left unchanged. */
bool rocke_intexpr_eval(
    const jd_val_t* expr, const rocke_intexpr_env_t* env, int64_t* out, char* err, size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INTEXPR_H */
