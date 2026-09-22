/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/recipe_vm.h -- execute a "builder recipe" (schema "rocke.recipe/v1")
 * against a runtime spec to emit a rocke_kernel_def_t, with no embedded CPython.
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
 * rocke_ir_builder_free). On failure *out_kernel is NULL and a diagnostic is
 * written into err/err_cap.
 */
#ifndef ROCKE_RECIPE_VM_H
#define ROCKE_RECIPE_VM_H

#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Spec inputs supplied at JIT time (the values the recipe specializes on). */
typedef struct
{
    const char* name;
    long value;
} rocke_recipe_spec_int_t;

typedef struct
{
    const char* name;
    const char* value;
} rocke_recipe_spec_str_t;

rocke_status_t rocke_recipe_run_from_json(const char* text,
                                          const rocke_recipe_spec_int_t* ints,
                                          int n_ints,
                                          const rocke_recipe_spec_str_t* strs,
                                          int n_strs,
                                          rocke_ir_builder_t* out_builder,
                                          rocke_kernel_def_t** out_kernel,
                                          char* err,
                                          size_t err_cap);

/* Same as rocke_recipe_run_from_json but the recipe is a CBOR blob (the compact
 * shipping form). Decodes into the same DOM and runs identically. */
rocke_status_t rocke_recipe_run_from_cbor(const unsigned char* data,
                                          size_t len,
                                          const rocke_recipe_spec_int_t* ints,
                                          int n_ints,
                                          const rocke_recipe_spec_str_t* strs,
                                          int n_strs,
                                          rocke_ir_builder_t* out_builder,
                                          rocke_kernel_def_t** out_kernel,
                                          char* err,
                                          size_t err_cap);

/* Run one recipe out of a CBOR BUNDLE (schema "rocke.bundle/v1"), selected by
 * `key` and optionally `arch` (NULL matches any arch). The bundle packs many
 * concrete/rolled recipes into one blob the runtime can load once and serve by
 * key, with no per-recipe files. */
rocke_status_t rocke_recipe_run_from_bundle_cbor(const unsigned char* data,
                                                 size_t len,
                                                 const char* key,
                                                 const char* arch,
                                                 const rocke_recipe_spec_int_t* ints,
                                                 int n_ints,
                                                 const rocke_recipe_spec_str_t* strs,
                                                 int n_strs,
                                                 rocke_ir_builder_t* out_builder,
                                                 rocke_kernel_def_t** out_kernel,
                                                 char* err,
                                                 size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_RECIPE_VM_H */
