/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gemm_prologue.h -- C++ mirror of
 * rocke/instances/common/gemm_prologue.py: the fused operand prologue of the
 * universal GEMM body (Prologue, attach_prologue, PrologueEmitter).
 *
 * A prologue is attached to a rocke_gemm_universal_spec_t through its
 * `_prologue` side channel (Python spec._prologue). NULL keeps every kernel
 * byte-identical; with one, rocke_build_universal_gemm transforms each operand
 * fragment between its LDS read and the MFMA, as the Python builder does:
 *
 *   src "f32"   f32 operands read through the 16-bit pair view and split into
 *               `parts` compute-type terms (recipes cvt / x3 / x4 / x6 / x9);
 *   out_f32     the epilogue stores the f32 accumulators;
 *   cplx_b      complex-f32 B through its real embedding (conj_a / conj_b).
 *
 * Mirrored fields: src, recipe, unary_a, unary_b, unary_impl, out_f32, cap,
 * order, split_at, pin_v, fast_exp, psub, cplx_b, conj_a, conj_b, agpr_out.
 * Not mirrored yet (rocke_gemm_prologue_check refuses them, so the C++ engine
 * never builds a different kernel than Python): stage "vgpr", epi "alpha" /
 * "bilinear".
 */
#ifndef ROCKE_INSTANCE_GEMM_PROLOGUE_H
#define ROCKE_INSTANCE_GEMM_PROLOGUE_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Python Prologue (frozen dataclass); strings are the Python literals. */
typedef struct rocke_gemm_prologue
{
    const char* src; /* "same" | "f32"                      (default "same") */
    const char* recipe; /* "cvt" | "x3" | "x4" | "x6" | "x9"  (default "cvt")  */
    const char* unary_a; /* "" | an operand op (relu, exp, sigmoid, ...)      */
    const char* unary_b; /* ""                                                */
    const char* unary_impl; /* "f32" | "bits"                   (default "f32") */
    bool out_f32;
    bool cap;
    const char* order; /* "small_first" | "big_first"   (default small_first) */
    const char* split_at; /* "load" | "use"                    (default "load") */
    const char* stage; /* "lds" (only "lds" is mirrored)    (default "lds")  */
    bool pin_v;
    bool fast_exp; /* default true                                           */
    bool psub;
    bool cplx_b;
    bool conj_a;
    bool conj_b;
    const char* epi; /* "" (only "" is mirrored)                             */
    bool agpr_out;
} rocke_gemm_prologue_t;

/* Python Prologue() defaults. */
rocke_gemm_prologue_t rocke_gemm_prologue_default(void);

/* Python Prologue.check() plus the mirror's own scope (see the header
 * comment). False with `reason` (capacity reason_cap) on a reject. */
bool rocke_gemm_prologue_check(const rocke_gemm_prologue_t* pro, char* reason, size_t reason_cap);

/* Python Prologue.kmul / .parts. */
int rocke_gemm_prologue_kmul(const rocke_gemm_prologue_t* pro);
int rocke_gemm_prologue_parts(const rocke_gemm_prologue_t* pro);

/* Python Prologue.suffix(): appended to the kernel name. Returns 0, or -1 when
 * `out` (capacity out_cap) is too small. */
int rocke_gemm_prologue_suffix(const rocke_gemm_prologue_t* pro, char* out, size_t out_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GEMM_PROLOGUE_H */
