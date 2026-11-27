/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_batchnorm2d_ro.h -- C99 port of
 * rocke/instances/common/batchnorm2d_ro.py: read-once (persistent grid)
 * NCHW BatchNorm training forward and backward. Each CTA keeps its tile in
 * registers across a cross-CTA exchange of per-channel partial statistics
 * (device-scope atomics plus a bounded spin on an arrival counter).
 *
 *   Python (batchnorm2d_ro.py)             C99 (this header)
 *   -----------------------------------    --------------------------------------
 *   RO_STRIDE / RO_MAX_SPLITS              ROCKE_BATCHNORM2D_RO_STRIDE / _MAX_SPLITS
 *   class BatchNorm2DRoSpec                rocke_batchnorm2d_ro_spec_t
 *   .cap_elems / .kernel_name(kind)        rocke_batchnorm2d_ro_cap_elems / _kernel_name
 *   class BatchNorm2DRoPlan                rocke_batchnorm2d_ro_plan_t
 *   is_valid_ro_spec(spec, arch)           rocke_batchnorm2d_is_valid_ro_spec(...)
 *   batchnorm2d_ro_plan(...)               rocke_batchnorm2d_ro_plan(...)
 *   batchnorm2d_ro_sync_ints(c)            rocke_batchnorm2d_ro_sync_ints(c)
 *   build_batchnorm2d_ro_train(spec)       rocke_build_batchnorm2d_ro_train(b, spec)
 *   build_batchnorm2d_ro_bwd(spec)         rocke_build_batchnorm2d_ro_bwd(b, spec)
 */
#ifndef ROCKE_INSTANCE_BATCHNORM2D_RO_H
#define ROCKE_INSTANCE_BATCHNORM2D_RO_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Ints per channel record of the sync buffer: [0] arrival counter, [1:3] and
 * [3:5] the two accumulator sets (selected by the launch epoch parity), the
 * rest padding. */
#define ROCKE_BATCHNORM2D_RO_STRIDE 32
/* Largest group of CTAs that share one channel (one arrival counter). */
#define ROCKE_BATCHNORM2D_RO_MAX_SPLITS 32

typedef struct rocke_batchnorm2d_ro_spec
{
    int block_size; /* default 256                        */
    int vec; /* default 8                          */
    int regs; /* default 16 (vectors held per thread) */
    const char* dtype; /* default "f16" ("f16"|"bf16"|"f32") */
    bool fast_reduce; /* default false                      */
    bool save_mean_invstd; /* default true                       */
    int wave_size; /* default 64                         */
    const char* name; /* default "rocke_batchnorm2d_ro"     */
} rocke_batchnorm2d_ro_spec_t;

typedef struct rocke_batchnorm2d_ro_plan
{
    int n_chunk;
    int hw_chunk;
    int sn;
    int sh;
    int grid;
} rocke_batchnorm2d_ro_plan_t;

rocke_batchnorm2d_ro_spec_t rocke_batchnorm2d_ro_spec_default(void);

/* .cap_elems = block_size * vec * regs. */
int rocke_batchnorm2d_ro_cap_elems(const rocke_batchnorm2d_ro_spec_t* spec);

/* .kernel_name(kind) ("train" | "bwd") into out (capacity out_cap). */
rocke_status_t rocke_batchnorm2d_ro_kernel_name(const rocke_batchnorm2d_ro_spec_t* spec,
                                                const char* kind,
                                                char* out,
                                                size_t out_cap);

/* is_valid_ro_spec(spec, arch) -> (ok, reason). */
bool rocke_batchnorm2d_is_valid_ro_spec(const rocke_batchnorm2d_ro_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap);

/* batchnorm2d_ro_plan(spec, n, c, h, w, sn_req=, sh_req=, max_resident=):
 * true and *out filled when admissible, false for Python's None. */
bool rocke_batchnorm2d_ro_plan(const rocke_batchnorm2d_ro_spec_t* spec,
                               int n,
                               int c,
                               int h,
                               int w,
                               int sn_req,
                               int sh_req,
                               int max_resident,
                               rocke_batchnorm2d_ro_plan_t* out);

/* batchnorm2d_ro_sync_ints(c) = c * RO_STRIDE. */
long long rocke_batchnorm2d_ro_sync_ints(int c);

/* Builders: `b` already initialised with kernel_name("train" / "bwd"). Return
 * the kernel or NULL with b's sticky error set. */
rocke_kernel_def_t* rocke_build_batchnorm2d_ro_train(rocke_ir_builder_t* b,
                                                     const rocke_batchnorm2d_ro_spec_t* spec);
rocke_kernel_def_t* rocke_build_batchnorm2d_ro_bwd(rocke_ir_builder_t* b,
                                                   const rocke_batchnorm2d_ro_spec_t* spec);
/* Convenience: init `b` with kernel_name(kind), then build train ("train")
 * or backward ("bwd"). */
rocke_kernel_def_t* rocke_build_batchnorm2d_ro_new(rocke_ir_builder_t* b,
                                                   const rocke_batchnorm2d_ro_spec_t* spec,
                                                   const char* kind);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_INSTANCE_BATCHNORM2D_RO_H */
