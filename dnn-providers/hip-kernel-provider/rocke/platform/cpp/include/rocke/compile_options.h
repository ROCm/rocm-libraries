/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/compile_options.h -- the COMGR codegen options of a kernel: the C++
 * twin of helpers/compile.py _comgr_options_for_kernel, a pure function of the
 * kernel attrs and the LLVM flavor the kernel is compiled for.
 *
 *   ["-O3"]
 *   + ["-mllvm", "-amdgpu-mfma-vgpr-form"]    attr mfma_vgpr_form, or agpr_alloc (0, 0)
 *   + ["-mllvm", "-amdgpu-mfma-vgpr-form=0"]  else attr mfma_agpr_form, flavor > llvm20
 *   + ["-mllvm", opt] per llvm_options entry (a list of strings)
 *
 * A COMGR client (hipTensor's pack builder and JIT) compiles the lowered .ll
 * of a kernel with these options to get the code object rocKE's own
 * build_hsaco_from_llvm_ir produces.
 */
#ifndef ROCKE_COMPILE_OPTIONS_H
#define ROCKE_COMPILE_OPTIONS_H

#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Python COMGR_FRESH_PROCESS_ATTR: a kernel whose codegen depends on the
 * compiles before it in the process. */
#define ROCKE_COMGR_FRESH_PROCESS_ATTR "comgr_fresh_process"

/* Writes up to `cap` options (pointers to static strings or to strings owned by
 * the kernel's arena) and returns their count, or -1 with `err` set (a concrete
 * flavor is required; an llvm_options attr that is not a list of strings is
 * rejected: Python sets it as a tuple of strings). */
int rocke_comgr_options_for_kernel(const rocke_kernel_def_t* kernel,
                                   rocke_llvm_flavor_t flavor,
                                   const char** out,
                                   int cap,
                                   char* err,
                                   size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_COMPILE_OPTIONS_H */
