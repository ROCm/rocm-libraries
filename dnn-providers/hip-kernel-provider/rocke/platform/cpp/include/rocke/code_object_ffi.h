// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * Flat C ABI for the direct rocKE code-object compiler. The entry point
 * consumes serialized ck.dsl.ir/v1 and performs instruction encoding plus
 * ELF/AMDHSA writing entirely in Rust, with no LLVM dependency.
 *
 * Conventions (rust/rocke-codeobj/src/native.rs):
 *   - Check rocke_codeobj_ffi_abi_version() == ROCKE_CODEOBJ_FFI_ABI_VERSION
 *     before the first compile.
 *   - The caller sets every struct_size to sizeof(the struct). The library
 *     compares it with its own layout: a request mismatch fails with status 1
 *     and a "struct_size" diagnostic; a result mismatch (or a NULL result) is
 *     returned as 1 without touching the result.
 *   - serialized_ir is UTF-8 text of serialized_ir_size bytes; it need not be
 *     NUL-terminated and must stay readable for the duration of the call.
 *   - A compile returns result->status (0 success, 1 failure). On failure the
 *     result carries only diagnostics ("<kind>: <message>").
 *   - hsaco and diagnostics are owned by the result until
 *     rocke_codeobj_compile_result_dispose_v1; release every result a compile
 *     initialized, on success or failure.
 */

#ifndef ROCKE_CODE_OBJECT_FFI_H
#define ROCKE_CODE_OBJECT_FFI_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define ROCKE_CODEOBJ_FFI_API __declspec(dllimport)
#else
#define ROCKE_CODEOBJ_FFI_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_CODEOBJ_FFI_ABI_VERSION 1u

typedef struct rocke_direct_gemm_compile_request_v1
{
    uint32_t struct_size;
    const uint8_t* serialized_ir;
    size_t serialized_ir_size;
} rocke_direct_gemm_compile_request_v1;

/* Request for rocke_codeobj_compile_rocke_ir_v1. native.rs takes the same
 * RockeDirectGemmCompileRequestV1 layout for both entry points, so this is the
 * same type and struct_size is sizeof(rocke_direct_gemm_compile_request_v1). */
typedef rocke_direct_gemm_compile_request_v1 rocke_codeobj_compile_request_v1;

typedef struct rocke_codeobj_compile_result_v1
{
    uint32_t struct_size;
    int32_t status;
    uint8_t* hsaco;
    size_t hsaco_size;
    uint8_t* diagnostics;
    size_t diagnostics_size;
    double bc_seconds;
    double link_seconds;
    double relocatable_seconds;
    double executable_seconds;
} rocke_codeobj_compile_result_v1;

ROCKE_CODEOBJ_FFI_API uint32_t rocke_codeobj_ffi_abi_version(void);

/* Strictly matches the qualified gfx1151 f16 RCR WMMA GEMM graph and directly
 * encodes a loadable HSACO. No LLVM/Clang/LLD/COMGR/tool process is used. */
ROCKE_CODEOBJ_FFI_API int32_t rocke_codeobj_compile_gfx1151_wmma_gemm_v1(
    const rocke_direct_gemm_compile_request_v1* request, rocke_codeobj_compile_result_v1* result);

/* Compiles one serialized rocKE kernel (ck.dsl.ir/v1) directly to a gfx1151
 * HSACO: parse, optimize, instruction selection, register allocation,
 * encoding and ELF/AMDHSA writing, all in Rust. Unsupported IR is rejected with
 * a diagnostic; nothing falls back to another compiler. On success
 * executable_seconds is the in-library compile time, and diagnostics holds an
 * informational resource summary, e.g.
 *   instructions=N code_bytes=N vgprs=N sgprs=N lds_bytes=N
 *   phases_ms=parse:T,optimize:T,isel:T,regalloc:T,post:T,encode:T,elf:T
 * (milliseconds; the phase list follows the compiler and may change). */
ROCKE_CODEOBJ_FFI_API int32_t rocke_codeobj_compile_rocke_ir_v1(
    const rocke_codeobj_compile_request_v1* request, rocke_codeobj_compile_result_v1* result);

ROCKE_CODEOBJ_FFI_API void
    rocke_codeobj_compile_result_dispose_v1(rocke_codeobj_compile_result_v1* result);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_CODE_OBJECT_FFI_H */
