/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/recipe_vm.h -- execute a "builder recipe" (schema "ck.dsl.recipe/v1")
 * against a runtime spec to emit a ckc_kernel_def_t, with no embedded CPython.
 *
 * This is the "compact per-builder artifact + runtime shape flexibility" path:
 * instead of shipping one concrete portable-IR graph per shape, a recipe encodes
 * the *builder algorithm* once -- including compile-time control flow
 * (`static_for` over a spec value, spec-derived constants) -- and a tiny C VM
 * replays it at JIT time with concrete spec values, producing the
 * shape-specialized IR. One small recipe therefore covers a whole family
 * (e.g. all head dims) and the specialization happens in C at runtime.
 *
 * The emitted kernel is owned by `out_builder`'s arena (free with
 * ckc_ir_builder_free). On failure *out_kernel is NULL and a diagnostic is
 * written into err/err_cap.
 */
#ifndef CKC_RECIPE_VM_H
#define CKC_RECIPE_VM_H

#include <stddef.h>

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Spec inputs supplied at JIT time (the values the recipe specializes on). */
typedef struct
{
    const char* name;
    long value;
} ckc_recipe_spec_int_t;

typedef struct
{
    const char* name;
    const char* value;
} ckc_recipe_spec_str_t;

ckc_status_t ckc_recipe_run_from_json(const char* text,
                                      const ckc_recipe_spec_int_t* ints,
                                      int n_ints,
                                      const ckc_recipe_spec_str_t* strs,
                                      int n_strs,
                                      ckc_ir_builder_t* out_builder,
                                      ckc_kernel_def_t** out_kernel,
                                      char* err,
                                      size_t err_cap);

/* Same as ckc_recipe_run_from_json but the recipe is a CBOR blob (the compact
 * shipping form). Decodes into the same DOM and runs identically. */
ckc_status_t ckc_recipe_run_from_cbor(const unsigned char* data,
                                      size_t len,
                                      const ckc_recipe_spec_int_t* ints,
                                      int n_ints,
                                      const ckc_recipe_spec_str_t* strs,
                                      int n_strs,
                                      ckc_ir_builder_t* out_builder,
                                      ckc_kernel_def_t** out_kernel,
                                      char* err,
                                      size_t err_cap);

/* Run one recipe out of a CBOR BUNDLE (schema "ck.dsl.bundle/v1"), selected by
 * `key` and optionally `arch` (NULL matches any arch). The bundle packs many
 * concrete/rolled recipes into one blob the runtime can load once and serve by
 * key, with no per-recipe files. */
ckc_status_t ckc_recipe_run_from_bundle_cbor(const unsigned char* data,
                                             size_t len,
                                             const char* key,
                                             const char* arch,
                                             const ckc_recipe_spec_int_t* ints,
                                             int n_ints,
                                             const ckc_recipe_spec_str_t* strs,
                                             int n_strs,
                                             ckc_ir_builder_t* out_builder,
                                             ckc_kernel_def_t** out_kernel,
                                             char* err,
                                             size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_RECIPE_VM_H */
