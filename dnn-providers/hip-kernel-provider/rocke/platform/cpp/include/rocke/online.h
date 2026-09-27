/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/online.h -- one-call convenience entry points for the ONLINE portable-IR
 * path: hand a serialized artifact (a CBOR recipe / bundle, or portable-IR JSON)
 * to the pure-C backend and get AMDGPU LLVM IR text back, in-process. These wrap
 * (recipe VM | IR import) + rocke_lower_kernel_to_llvm so a Python/ctypes binding
 * only ever deals with char* in / char* out -- no need to expose the
 * rocke_ir_builder_t / rocke_kernel_def_t layout across the FFI boundary.
 *
 * Each call optionally reports a phase split (build-IR vs lower) in milliseconds
 * so callers can attribute the online handoff cost precisely.
 *
 * The returned *out_ll is malloc'd; free it with rocke_online_free().
 */
#ifndef ROCKE_ONLINE_H
#define ROCKE_ONLINE_H

#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/recipe_vm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* recipe CBOR (+ runtime specs) -> LLVM IR text.
 *   out_build_ms : nullable; time to decode CBOR + VM-expand the IR.
 *   out_lower_ms : nullable; time to lower the IR to .ll. */
rocke_status_t rocke_online_recipe_cbor_to_llvm(const unsigned char* data,
                                                size_t len,
                                                const rocke_recipe_spec_int_t* ints,
                                                int n_ints,
                                                const rocke_recipe_spec_str_t* strs,
                                                int n_strs,
                                                const char* arch,
                                                char** out_ll,
                                                double* out_build_ms,
                                                double* out_lower_ms,
                                                char* err,
                                                size_t err_cap);

/* one recipe out of a CBOR bundle (selected by key, and arch if non-NULL) -> .ll */
rocke_status_t rocke_online_bundle_cbor_to_llvm(const unsigned char* data,
                                                size_t len,
                                                const char* key,
                                                const char* arch,
                                                const rocke_recipe_spec_int_t* ints,
                                                int n_ints,
                                                const rocke_recipe_spec_str_t* strs,
                                                int n_strs,
                                                char** out_ll,
                                                double* out_build_ms,
                                                double* out_lower_ms,
                                                char* err,
                                                size_t err_cap);

/* portable-IR JSON (schema "rocke.ir/v1", concrete graph) -> .ll */
rocke_status_t rocke_online_ir_json_to_llvm(const char* text,
                                            const char* arch,
                                            char** out_ll,
                                            double* out_build_ms,
                                            double* out_lower_ms,
                                            char* err,
                                            size_t err_cap);

/* Free a string returned by the calls above. */
void rocke_online_free(char* p);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_ONLINE_H */
