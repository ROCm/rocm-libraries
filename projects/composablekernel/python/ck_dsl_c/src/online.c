/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/online.c -- see ckc/online.h. Thin wrappers: (recipe VM | IR import) +
 * lower, with optional phase timing, returning a malloc'd .ll string.
 */
#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 199309L
#define _POSIX_C_SOURCE 199309L /* clock_gettime / CLOCK_MONOTONIC */
#endif

#include "ckc/online.h"

#include <stdlib.h>
#include <time.h>

#include "ckc/ir_import.h"
#include "ckc/lower_llvm.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* Lower an already-built kernel (owned by `b`) to .ll, time it, free the
 * builder, and emit the phase timings. Consumes `b` (frees it). */
static ckc_status_t online_finish(ckc_ir_builder_t* b,
                                  ckc_kernel_def_t* kernel,
                                  const char* arch,
                                  char** out_ll,
                                  double build_ms,
                                  double* out_build_ms,
                                  double* out_lower_ms,
                                  char* err,
                                  size_t err_cap)
{
    double t1 = now_ms();
    ckc_status_t st =
        ckc_lower_kernel_to_llvm_ex(kernel, CKC_LLVM_FLAVOR_AUTO, arch, out_ll, err, err_cap);
    double t2 = now_ms();
    ckc_ir_builder_free(b);
    if (out_build_ms)
        *out_build_ms = build_ms;
    if (out_lower_ms)
        *out_lower_ms = t2 - t1;
    return st;
}

ckc_status_t ckc_online_recipe_cbor_to_llvm(const unsigned char* data,
                                            size_t len,
                                            const ckc_recipe_spec_int_t* ints,
                                            int n_ints,
                                            const ckc_recipe_spec_str_t* strs,
                                            int n_strs,
                                            const char* arch,
                                            char** out_ll,
                                            double* out_build_ms,
                                            double* out_lower_ms,
                                            char* err,
                                            size_t err_cap)
{
    if (out_ll)
        *out_ll = NULL;
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel = NULL;
    double t0 = now_ms();
    ckc_status_t st = ckc_recipe_run_from_cbor(data, len, ints, n_ints, strs, n_strs, &b, &kernel,
                                               err, err_cap);
    double t1 = now_ms();
    if (st != CKC_OK || !kernel)
        return st; /* run_from_cbor already freed the builder on failure */
    return online_finish(&b, kernel, arch, out_ll, t1 - t0, out_build_ms, out_lower_ms, err,
                         err_cap);
}

ckc_status_t ckc_online_bundle_cbor_to_llvm(const unsigned char* data,
                                            size_t len,
                                            const char* key,
                                            const char* arch,
                                            const ckc_recipe_spec_int_t* ints,
                                            int n_ints,
                                            const ckc_recipe_spec_str_t* strs,
                                            int n_strs,
                                            char** out_ll,
                                            double* out_build_ms,
                                            double* out_lower_ms,
                                            char* err,
                                            size_t err_cap)
{
    if (out_ll)
        *out_ll = NULL;
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel = NULL;
    double t0 = now_ms();
    ckc_status_t st = ckc_recipe_run_from_bundle_cbor(data, len, key, arch, ints, n_ints, strs,
                                                      n_strs, &b, &kernel, err, err_cap);
    double t1 = now_ms();
    if (st != CKC_OK || !kernel)
        return st;
    return online_finish(&b, kernel, arch, out_ll, t1 - t0, out_build_ms, out_lower_ms, err,
                         err_cap);
}

ckc_status_t ckc_online_ir_json_to_llvm(const char* text,
                                        const char* arch,
                                        char** out_ll,
                                        double* out_build_ms,
                                        double* out_lower_ms,
                                        char* err,
                                        size_t err_cap)
{
    if (out_ll)
        *out_ll = NULL;
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel = NULL;
    double t0 = now_ms();
    ckc_status_t st = ckc_import_kernel_from_json(text, NULL, &b, &kernel, err, err_cap);
    double t1 = now_ms();
    if (st != CKC_OK || !kernel)
        return st;
    return online_finish(&b, kernel, arch, out_ll, t1 - t0, out_build_ms, out_lower_ms, err,
                         err_cap);
}

void ckc_online_free(char* p)
{
    free(p);
}
