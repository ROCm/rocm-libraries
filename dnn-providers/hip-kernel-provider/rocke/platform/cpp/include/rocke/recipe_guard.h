/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/recipe_guard.h -- ask, without compiling anything, whether a recipe
 * actually serves a given shape.
 *
 * This is the enforcement surface for a JIT caller such as hipDNN.
 *
 * Why a rolled recipe needs to be asked at all
 * --------------------------------------------
 * A CONCRETE recipe is its own validity statement: it was recorded from a build
 * the kernel accepted, so being in the bundle means it is supported and a lookup
 * miss is the rejection. A ROLLED recipe (non-empty "spec") generalizes the
 * emission over one or more free axes, and generalizing emission is not the same
 * as generalizing legality. The VM will happily replay it at any value the
 * caller supplies -- including values whose kernel the family's own admission
 * gate would have refused to build. Rolling is what creates the need for this
 * check, not what removes it.
 *
 * The guard is a handful of predicates over the free axes, derived at bundle
 * generation time by measuring the family's Python gate with the recipe's baked
 * values already fixed, and verified against that gate out of sample before
 * shipping (see python/rocke/portable_ir/src/guard.py). It is carried inside the
 * recipe and evaluated here by the same intexpr evaluator that runs the recipe
 * itself -- no second implementation of the language, and nothing for a C++ port
 * of the kernel gates to drift away from.
 *
 * The guarantee is one-way, and callers should design around the asymmetry:
 *
 *     ADMITTED  ==>  the family's gate accepts this binding
 *     REFUSED   ==>  either the gate rejects it, or generation could not prove
 *                    that it does not
 *
 * So a refusal means "do not compile this here", not "this is impossible".
 * Treat it as a routing decision -- fall back to another provider -- rather than
 * as an error to report to the user.
 *
 * Typical use, before committing to a JIT compile:
 *
 *     rocke_guard_verdict_t v;
 *     char why[ROCKE_ERR_MSG_CAP];
 *     rocke_status_t st = rocke_bundle_check_guard_cbor(
 *         blob, blob_len, "attention_tiled_2d", "gfx950",
 *         ints, n_ints, strs, n_strs, 0, &v, why, sizeof why);
 *     if(st == ROCKE_ERR_KEY)      // no such recipe: not our shape
 *         return fallback();
 *     if(st != ROCKE_OK)           // malformed bundle: a build/packaging bug
 *         return hard_error(why);
 *     if(v == ROCKE_GUARD_REFUSED) // unsupported configuration
 *         return fallback();       //   `why` says which constraint failed
 *     ... rocke_recipe_run_from_bundle_cbor(...) ...
 *
 * Cost: the rules themselves are a handful of integer comparisons, but each call
 * parses the whole blob to reach them, and that parse dominates -- around 0.5ms
 * on a 100KiB bundle, against roughly 1.7ms to replay and lower the same recipe
 * and far more to finish a real compile. So the check is worth making before a
 * compile and is not worth making in a loop over thousands of shapes; if that
 * becomes a pattern, the fix is a parse-once handle rather than a faster
 * evaluator.
 *
 * A caller that will immediately replay on success can skip the separate call
 * entirely: the VM enforces the same guard internally before it emits its first
 * op, so a refused binding fails the run rather than producing a kernel.
 *
 * Version skew (see rocke/abi.h)
 * ------------------------------
 * A bundle built by a NEWER generator than this engine comes back as
 * ROCKE_ERR_VALUE with a reason naming the levels, NOT as REFUSED. The
 * distinction matters to the caller above: a refusal means route elsewhere and
 * carry on, while this means the deployed engine and the shipped artifacts do
 * not match, which no amount of falling back will fix. Reporting it as a
 * refusal would file a deployment fault under "unsupported shape", where it
 * would show up as a quiet loss of coverage that nobody goes looking for.
 *
 * The sample above already handles it correctly: it is caught by the
 * `st != ROCKE_OK` arm. A caller that lumps all non-OK statuses into fallback
 * gets the wrong behaviour here, which is the reason for spelling this out.
 *
 * Separately, and before any of this: check that the library you loaded matches
 * the header you compiled against, with `rocke_abi_version() ==
 * ROCKE_ABI_VERSION`. That one is not a data question -- a mismatch means the
 * struct layouts in this header disagree with the binary, so the call above is
 * undefined behaviour rather than a wrong answer.
 */
#ifndef ROCKE_RECIPE_GUARD_H
#define ROCKE_RECIPE_GUARD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/recipe_vm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The answer, kept separate from rocke_status_t on purpose: "this shape is not
 * supported" is a normal result of a working call, not a failure of the call.
 * Folding the two together would force a caller to tell a routing decision apart
 * from a corrupt bundle by parsing an error string. */
typedef enum rocke_guard_verdict
{
    ROCKE_GUARD_ADMITTED = 0, /* every rule passed; the gate accepts this      */
    ROCKE_GUARD_REFUSED, /* a rule failed; `reason` names which           */
    ROCKE_GUARD_ABSENT /* no guard on this recipe -- see note below     */
} rocke_guard_verdict_t;

/* ROCKE_GUARD_ABSENT is reported, not silently treated as admitted, because the
 * two mean different things to a caller. A concrete recipe legitimately has no
 * guard (its presence in the bundle is the validity statement). A ROLLED recipe
 * without one is an ungoverned bundle -- generated before guards existed, or by
 * a generator that skipped derivation -- and a caller that wants enforcement
 * should be able to notice that rather than infer safety from silence. Both
 * ABSENT and ADMITTED return ROCKE_OK: guards are additive and every recipe that
 * replayed before them still replays. */

/* Demand that the binding be one of the points the generator actually built and
 * compared, not merely one the rules accept.
 *
 * The default answers "the kernel's gate accepts this", which is what governs
 * whether a compile can succeed. This flag answers the stricter "this exact
 * point was verified byte-for-byte at generation time", giving up the rolled
 * interior -- the whole compression win -- for the strongest available evidence.
 * Reasonable while bringing a family up or for a conformance run; expensive as a
 * steady state. Refuses everything if the recipe carries no verified points. */
#define ROCKE_GUARD_REQUIRE_VERIFIED 0x1u

/* Check the guard on a standalone CBOR recipe.
 *
 * `ints`/`strs` bind the free axes, exactly as they would be passed to
 * rocke_recipe_run_from_cbor -- the point is to ask the identical question the
 * replay would, before paying for it. Extra bindings the recipe does not use are
 * ignored; a missing one is a refusal naming the axis.
 *
 * Returns ROCKE_OK when the check ran, and writes the answer to *out_verdict.
 * Passing NULL for out_verdict collapses a refusal into ROCKE_ERR_VALUE, for
 * callers that only want pass/fail. `reason` (optional) receives a
 * human-readable diagnostic on refusal or error. */
rocke_status_t rocke_recipe_check_guard_cbor(const unsigned char* data,
                                             size_t len,
                                             const rocke_recipe_spec_int_t* ints,
                                             int n_ints,
                                             const rocke_recipe_spec_str_t* strs,
                                             int n_strs,
                                             unsigned flags,
                                             rocke_guard_verdict_t* out_verdict,
                                             char* reason,
                                             size_t reason_cap);

/* Same, for one recipe inside a CBOR bundle, selected by `key` and optionally
 * `arch` (NULL matches any arch).
 *
 * Returns ROCKE_ERR_KEY when the bundle holds no such recipe. That is a distinct
 * and expected answer: for a pruned bundle -- one the generator filtered with
 * the kernel gates -- absence IS the rejection for concrete recipes, and the
 * guard covers the rolled ones. A caller can therefore use this single call as
 * the complete admission test for both kinds. */
rocke_status_t rocke_bundle_check_guard_cbor(const unsigned char* data,
                                             size_t len,
                                             const char* key,
                                             const char* arch,
                                             const rocke_recipe_spec_int_t* ints,
                                             int n_ints,
                                             const rocke_recipe_spec_str_t* strs,
                                             int n_strs,
                                             unsigned flags,
                                             rocke_guard_verdict_t* out_verdict,
                                             char* reason,
                                             size_t reason_cap);

/* Is `key` (for `arch`, or any arch when NULL) in this bundle at all?
 *
 * The cheap half of the question above, for a dispatcher deciding whether this
 * provider is a candidate before it has a shape in hand. */
bool rocke_bundle_contains(const unsigned char* data,
                           size_t len,
                           const char* key,
                           const char* arch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_RECIPE_GUARD_H */
