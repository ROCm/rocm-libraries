/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_batchnorm2d_mcw.h -- C99 port of the multi-channel-per-wave
 * BatchNorm2D kernels of rocke/instances/common/batchnorm2d.py (NCHW training
 * forward and backward, one launch, the tensor of a channel group held in
 * registers).
 *
 *   Python (batchnorm2d.py)                C99 (this header)
 *   -----------------------------------    --------------------------------------
 *   class BatchNorm2DMcwSpec               rocke_batchnorm2d_mcw_spec_t
 *   .block_size / .lpc / .regs             rocke_batchnorm2d_mcw_block_size/lpc/regs
 *   .kernel_name()                         rocke_batchnorm2d_mcw_kernel_name(...)
 *   is_valid_mcw_spec(spec, arch)          rocke_batchnorm2d_is_valid_mcw_spec(...)
 *   mcw_admits(spec, n, c, h, w)           rocke_batchnorm2d_mcw_admits(...)
 *   batchnorm2d_mcw_grid(c, spec)          rocke_batchnorm2d_mcw_grid(...)
 *   build_batchnorm2d_mcw_train(spec)      rocke_build_batchnorm2d_mcw_train(b, spec)
 *   build_batchnorm2d_mcw_bwd(spec)        rocke_build_batchnorm2d_mcw_bwd(b, spec)
 *   build_batchnorm2d_mcw(spec)            rocke_build_batchnorm2d_mcw(b, spec)
 */
#ifndef ROCKE_INSTANCE_BATCHNORM2D_MCW_H
#define ROCKE_INSTANCE_BATCHNORM2D_MCW_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* MCW_MAX_REGS: register-cached elements per lane. */
#define ROCKE_BATCHNORM2D_MCW_MAX_REGS 192

/* Mirror of Python BatchNorm2DMcwSpec (n / c / h / w are runtime). */
typedef struct rocke_batchnorm2d_mcw_spec
{
    const char* mode; /* default "train" ("train" | "bwd")   */
    int waves; /* default 4                           */
    int cpw; /* default 1 (channels per wave)       */
    int npw; /* default 32 (samples per wave)       */
    int slots; /* default 1                           */
    int vec; /* default 1                           */
    const char* dtype; /* default "f16" ("f16"|"bf16"|"f32") */
    int wave_size; /* default 64                          */
    bool dynamic; /* default true                        */
    const char* name; /* default "rocke_batchnorm2d_mcw"     */
} rocke_batchnorm2d_mcw_spec_t;

rocke_batchnorm2d_mcw_spec_t rocke_batchnorm2d_mcw_spec_default(void);

/* .block_size = waves * wave_size; .lpc = wave_size // cpw;
 * .regs = npw * slots * vec * (1 if mode == "train" else 2). */
int rocke_batchnorm2d_mcw_block_size(const rocke_batchnorm2d_mcw_spec_t* spec);
int rocke_batchnorm2d_mcw_lpc(const rocke_batchnorm2d_mcw_spec_t* spec);
int rocke_batchnorm2d_mcw_regs(const rocke_batchnorm2d_mcw_spec_t* spec);

/* BatchNorm2DMcwSpec.kernel_name() into out (capacity out_cap). */
rocke_status_t rocke_batchnorm2d_mcw_kernel_name(const rocke_batchnorm2d_mcw_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap);

/* is_valid_mcw_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". */
bool rocke_batchnorm2d_is_valid_mcw_spec(const rocke_batchnorm2d_mcw_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap);

/* mcw_admits(spec, n, c, h, w) -> (ok, reason): runtime-shape admission. */
bool rocke_batchnorm2d_mcw_admits(const rocke_batchnorm2d_mcw_spec_t* spec,
                                  int n,
                                  int c,
                                  int h,
                                  int w,
                                  char* reason,
                                  size_t reason_cap);

/* batchnorm2d_mcw_grid(c, spec) -> (c // cpw, 1, 1). */
rocke_status_t
    rocke_batchnorm2d_mcw_grid(int c, const rocke_batchnorm2d_mcw_spec_t* spec, int out[3]);

/* Builders: `b` already initialised with the spec's kernel name. Return the
 * kernel or NULL with b's sticky error set (invalid spec or wrong mode). */
rocke_kernel_def_t* rocke_build_batchnorm2d_mcw_train(rocke_ir_builder_t* b,
                                                      const rocke_batchnorm2d_mcw_spec_t* spec);
rocke_kernel_def_t* rocke_build_batchnorm2d_mcw_bwd(rocke_ir_builder_t* b,
                                                    const rocke_batchnorm2d_mcw_spec_t* spec);
/* build_batchnorm2d_mcw(spec): train or bwd by spec->mode. */
rocke_kernel_def_t* rocke_build_batchnorm2d_mcw(rocke_ir_builder_t* b,
                                                const rocke_batchnorm2d_mcw_spec_t* spec);
/* Convenience: init `b` with the kernel name, then rocke_build_batchnorm2d_mcw. */
rocke_kernel_def_t* rocke_build_batchnorm2d_mcw_new(rocke_ir_builder_t* b,
                                                    const rocke_batchnorm2d_mcw_spec_t* spec);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_INSTANCE_BATCHNORM2D_MCW_H */
