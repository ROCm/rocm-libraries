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

/* Request for rocke_codeobj_compile_rocke_ir_v2: the v1 request plus the
 * target to compile for. */
typedef struct rocke_codeobj_compile_request_v2
{
    uint32_t struct_size; /* sizeof(rocke_codeobj_compile_request_v2) */
    const uint8_t* serialized_ir;
    size_t serialized_ir_size;
    const char* target; /* NUL-terminated target name: "gfx1151" or "gfx950" */
} rocke_codeobj_compile_request_v2;

/* rocke_codeobj_compile_rocke_ir_v1 for request->target. A NULL, unknown or
 * unsupported target fails with a "request: ..." diagnostic; a kernel using an
 * operation the compiler does not lower for the target fails with a
 * "ROCKE_DIRECT_UNSUPPORTED: ..." one. */
ROCKE_CODEOBJ_FFI_API int32_t rocke_codeobj_compile_rocke_ir_v2(
    const rocke_codeobj_compile_request_v2* request, rocke_codeobj_compile_result_v1* result);

/* The targets the library compiles for: a static NUL-terminated,
 * comma-separated list of names (e.g. "gfx1151,gfx950"). Do not free it. */
ROCKE_CODEOBJ_FFI_API const char* rocke_codeobj_supported_targets_v1(void);

ROCKE_CODEOBJ_FFI_API void
    rocke_codeobj_compile_result_dispose_v1(rocke_codeobj_compile_result_v1* result);

/*
 * Compile sessions.
 *
 * A session is an opaque object the caller creates, owns and frees. It holds
 * what consecutive compiles reuse: a pool of persistent worker threads that
 * parse large inputs in parallel (so a compile pays no thread start), and the
 * state of an input being streamed. The library keeps no global or
 * thread-local state; everything lives in the session.
 *
 * Results are exactly those of rocke_codeobj_compile_rocke_ir_v1 on the same
 * input: the same HSACO bytes, and the same status and "<kind>: <message>"
 * diagnostics on failure.
 *
 * Thread safety: a session may be used from any thread, but calls on ONE
 * session must not overlap in time. A call that would overlap another on the
 * same session fails without touching the session (compile/finish: status 1
 * with a "ROCKE_DIRECT_SESSION: ..." diagnostic; feed: returns 1). Different
 * sessions are independent and may be used concurrently, and
 * rocke_codeobj_compile_rocke_ir_v1 needs no session at all. Free a session
 * only when no call is using it; freeing stops and joins its threads.
 *
 * Streaming: call rocke_codeobj_session_feed_v1 any number of times with
 * consecutive pieces of one serialized kernel (pieces may split lines or
 * UTF-8 sequences anywhere), then rocke_codeobj_session_finish_v1, which
 * compiles everything fed since the last finish and ends the stream. While
 * the producer keeps writing, completed lines are parsed on the session's
 * threads. feed copies the bytes; the caller may reuse its buffer at once.
 */
typedef struct rocke_codeobj_session_v1 rocke_codeobj_session_v1;

typedef struct rocke_codeobj_session_options_v1
{
    uint32_t struct_size; /* sizeof(rocke_codeobj_session_options_v1) */
    uint32_t worker_threads; /* threads to start; 0 picks a default from the machine */
} rocke_codeobj_session_options_v1;

/* Creates a session in *session (options may be NULL for the defaults).
 * Returns 0, or 1 with *session = NULL when session is NULL, struct_size does
 * not match, or the session cannot be created. */
ROCKE_CODEOBJ_FFI_API int32_t rocke_codeobj_session_create_v1(
    const rocke_codeobj_session_options_v1* options, rocke_codeobj_session_v1** session);

typedef struct rocke_codeobj_session_options_v2
{
    uint32_t struct_size; /* sizeof(rocke_codeobj_session_options_v2) */
    uint32_t worker_threads; /* threads to start; 0 picks a default from the machine */
    const char* target; /* NUL-terminated name of the target every compile is for */
} rocke_codeobj_session_options_v2;

/* rocke_codeobj_session_create_v1 with a target: the session's compile and
 * finish entry points compile for it. NULL options take the defaults
 * (gfx1151); a NULL, unknown or unsupported target fails as a mismatched
 * struct_size does. */
ROCKE_CODEOBJ_FFI_API int32_t rocke_codeobj_session_create_v2(
    const rocke_codeobj_session_options_v2* options, rocke_codeobj_session_v1** session);

/* Stops and joins the session's threads and frees it; NULL is ignored. */
ROCKE_CODEOBJ_FFI_API void rocke_codeobj_session_free_v1(rocke_codeobj_session_v1* session);

/* rocke_codeobj_compile_rocke_ir_v1 using the session's threads. Same request
 * and result conventions; dispose the result as usual. */
ROCKE_CODEOBJ_FFI_API int32_t
    rocke_codeobj_session_compile_v1(rocke_codeobj_session_v1* session,
                                     const rocke_codeobj_compile_request_v1* request,
                                     rocke_codeobj_compile_result_v1* result);

/* Appends size bytes at data to the input being streamed (starting a stream
 * if none is). Returns 0, or 1 without effect for a NULL session, NULL data
 * with size > 0, or an overlapping call. */
ROCKE_CODEOBJ_FFI_API int32_t rocke_codeobj_session_feed_v1(rocke_codeobj_session_v1* session,
                                                            const uint8_t* data,
                                                            size_t size);

/* Compiles everything fed since the last finish, as
 * rocke_codeobj_compile_rocke_ir_v1 would compile it whole (an empty or
 * non-UTF-8 input fails the same way), and ends the stream. Result
 * conventions as for the other entry points; executable_seconds counts from
 * this call, not the overlapped parsing before it. */
ROCKE_CODEOBJ_FFI_API int32_t rocke_codeobj_session_finish_v1(
    rocke_codeobj_session_v1* session, rocke_codeobj_compile_result_v1* result);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_CODE_OBJECT_FFI_H */
