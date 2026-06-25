/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/lower_llvm.h -- PUBLIC API for the C99 port of
 * ck_dsl.core.lower_llvm (lower_kernel_to_llvm).
 *
 * Lowers a built CK DSL IR kernel (ckc_kernel_def_t) to AMDGPU LLVM IR
 * text. This is the fast-compile path: the emitted ``.ll`` is fed to
 * ``clang -x ir`` / libamd_comgr to produce a HSA code object.
 *
 * The Python entry point::
 *
 *     def lower_kernel_to_llvm(kernel, *, llvm_flavor=None, arch=None) -> str
 *
 * becomes ::
 *
 *     ckc_status_t ckc_lower_kernel_to_llvm(const ckc_kernel_def_t *kernel,
 *                                           ckc_llvm_flavor_t flavor,
 *                                           const char *arch,
 *                                           char **out_text);
 *
 * The result is a freshly malloc'd, NUL-terminated C string handed to the
 * caller (mirroring Python returning a new str); the caller frees it with
 * free(). On failure the function returns a non-CKC_OK status and *out_text
 * is left NULL.
 *
 * Error model: unlike the IR builder, the lowerer does not own a builder, so
 * errors are reported purely via the return status. Callers wanting a message
 * pass a non-NULL err buffer to the *_ex variant.
 */
#ifndef CKC_LOWER_LLVM_H
#define CKC_LOWER_LLVM_H

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------- LLVM flavor */

/* A small set of AMDGPU intrinsic signatures changed between LLVM 20
 * (ROCm 7.0/7.1) and LLVM 21+ (ROCm 7.2 ships LLVM 22). The flavor pins the
 * declaration text emitted up front (comgr verifies declares before the
 * auto-upgrade pass). Mirrors LLVM_FLAVOR_LLVM20 / LLVM_FLAVOR_LLVM22. */
typedef enum ckc_llvm_flavor
{
    CKC_LLVM_FLAVOR_AUTO = 0, /* resolve from env / ROCm version at call time */
    CKC_LLVM_FLAVOR_LLVM20, /* "llvm20" */
    CKC_LLVM_FLAVOR_LLVM22 /* "llvm22" (modern default) */
} ckc_llvm_flavor_t;

/* Canonical flavor string ("llvm20"/"llvm22"), or "" for AUTO. */
const char* ckc_llvm_flavor_name(ckc_llvm_flavor_t flavor);
/* Reverse: "llvm20"/"llvm22" -> flavor; CKC_LLVM_FLAVOR_AUTO if unknown. */
ckc_llvm_flavor_t ckc_llvm_flavor_from_name(const char* name);

/* ------------------------------------------------------------ entry point */

/* Lower `kernel` to AMDGPU LLVM IR text.
 *
 *   flavor : CKC_LLVM_FLAVOR_AUTO resolves via $CK_DSL_LLVM_FLAVOR, then
 *            /opt/rocm/.info/version, then defaults to LLVM22. (The Python
 *            torch.version.hip step is not portable to libc-only C; see the
 *            'unported' notes.)
 *   arch   : ISA backend gfx string ("gfx942","gfx950",...). NULL => "gfx950"
 *            (the byte-identical baseline).
 *   out_text : on CKC_OK, receives a malloc'd NUL-terminated string the caller
 *              frees with free(); set to NULL on failure.
 *
 * Returns CKC_OK on success, or a ckc_status_t error code:
 *   CKC_ERR_VALUE   bad flavor / unsupported attr value
 *   CKC_ERR_NOTIMPL op or type with no LLVM lowering
 *   CKC_ERR_KEY     unknown arch backend
 *   CKC_ERR_OOM     allocation failure
 */
ckc_status_t ckc_lower_kernel_to_llvm(const ckc_kernel_def_t* kernel,
                                      ckc_llvm_flavor_t flavor,
                                      const char* arch,
                                      char** out_text);

/* Variant that also writes a human-readable diagnostic on failure. `err` may
 * be NULL (then behaves like the base form); otherwise it must point to a
 * buffer of at least `err_cap` bytes (CKC_ERR_MSG_CAP is a good size). */
ckc_status_t ckc_lower_kernel_to_llvm_ex(const ckc_kernel_def_t* kernel,
                                         ckc_llvm_flavor_t flavor,
                                         const char* arch,
                                         char** out_text,
                                         char* err,
                                         size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_LOWER_LLVM_H */
