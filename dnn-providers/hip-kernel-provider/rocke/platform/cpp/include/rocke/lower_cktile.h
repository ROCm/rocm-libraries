/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/lower_cktile.h -- C99 port of rocke.core.lower_cktile.
 *
 * Lower a rocke *instance spec* to CK Tile-style C++ source.
 *
 * Unlike the other lowerers in the C port (lower_hip / lower_llvm), this module
 * does NOT walk the post-IR rocke_kernel_def_t. It mirrors the Python module,
 * which operates *before* the IR exists: it consumes the high-level instance
 * specs (UniversalGemmSpec / ImplicitGemmConvSpec) and emits a self-contained
 * CK Tile C++ source that composes the same CK Tile templates a hand-written
 * CK Tile kernel would.
 *
 * Because those high-level specs are NOT part of the FROZEN IR contract in
 * rocke/ir.h (they are inputs, not IR nodes), this header declares plain C99
 * mirror structs for them. They are faithful field-for-field translations of
 * the Python dataclasses (gemm_universal.py / conv_implicit_gemm.py) restricted
 * to the fields the CK Tile lowering actually reads.
 *
 * Emission uses rocke_strbuf for all text. The returned string is owned by the
 * caller-provided rocke_strbuf_t (use rocke_strbuf_cstr / rocke_strbuf_detach).
 *
 * Error model mirrors the rest of the C port: functions return a rocke_status_t
 * and write a human-readable message into `err` (ROCKE_ERR_MSG_CAP bytes). The
 * Python NotImplementedError / TypeError / ValueError cases map to
 * ROCKE_ERR_NOTIMPL / ROCKE_ERR_TYPE / ROCKE_ERR_VALUE respectively.
 */
#ifndef ROCKE_LOWER_CKTILE_H
#define ROCKE_LOWER_CKTILE_H

#include <stdbool.h>

#include "rocke/ir.h" /* rocke_status_t, ROCKE_ERR_MSG_CAP */
#include "rocke/strbuf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ specs */

/* Mirror of gemm_universal.TileSpec (only CK-Tile-relevant fields). */
typedef struct rocke_cktile_tile_spec
{
    int tile_m;
    int tile_n;
    int tile_k;
    int warp_m;
    int warp_n;
    int warp_k; /* default 1 */
    int warp_tile_m; /* default 32 */
    int warp_tile_n; /* default 32 */
    int warp_tile_k; /* default 16 */
} rocke_cktile_tile_spec_t;

/* Mirror of gemm_universal.TraitSpec (CK-Tile-relevant subset). pipeline /
 * scheduler / epilogue are the short string names ("mem"/"compv3"/"compv4",
 * "intrawave"/"interwave", "cshuffle"/"default"). */
typedef struct rocke_cktile_trait_spec
{
    const char* pipeline; /* default "compv4"   */
    const char* scheduler; /* default "intrawave"*/
    const char* epilogue; /* default "cshuffle" */
    bool pad_m;
    bool pad_n;
    bool pad_k;
    bool persistent;
} rocke_cktile_trait_spec_t;

/* Mirror of gemm_universal.DataSpec (dtype short names + 3-letter layout). */
typedef struct rocke_cktile_data_spec
{
    const char* dtype_a; /* "fp16"/"f16"/"bf16"/... */
    const char* dtype_b;
    const char* dtype_c;
    const char* dtype_acc; /* "fp32"/"f32" */
    const char* layout; /* e.g. "RCR" */
} rocke_cktile_data_spec_t;

/* Mirror of gemm_universal.UniversalGemmSpec (CK-Tile-relevant subset). */
typedef struct rocke_cktile_gemm_spec
{
    const char* name;
    rocke_cktile_tile_spec_t tile;
    rocke_cktile_trait_spec_t trait;
    rocke_cktile_data_spec_t data;
    bool batched;
    /* Optional precomputed mangled name; NULL => compute via kernel_name(). */
    const char* kernel_name;
} rocke_cktile_gemm_spec_t;

/* Mirror of conv_implicit_gemm.ConvProblem. */
typedef struct rocke_cktile_conv_problem
{
    int N, Hi, Wi, C, K, R, S;
    int sH, sW; /* default 1 */
    int pH, pW; /* default 0 */
    int dH, dW; /* default 1 */
} rocke_cktile_conv_problem_t;

/* Mirror of conv_implicit_gemm.ImplicitGemmConvSpec (CK-Tile-relevant subset).
 * acc_epilogue is intentionally omitted: the CK Tile lowering never reads it
 * except for kernel_name() mangling -- see TODO(port) in lower_cktile.c. */
typedef struct rocke_cktile_conv_spec
{
    const char* name; /* default "conv_igemm" */
    rocke_cktile_conv_problem_t problem;
    int tile_m;
    int tile_n;
    int tile_k;
    int warp_m;
    int warp_n;
    int warp_tile_m;
    int warp_tile_n;
    int warp_tile_k;
    const char* pipeline; /* default "mem" */
    const char* epilogue; /* default "default" */
    bool async_dma;
    const char* acc_epilogue_tag; /* "" if identity; see TODO */
    /* Optional precomputed mangled name; NULL => compute via kernel_name(). */
    const char* kernel_name;
} rocke_cktile_conv_spec_t;

/* Tagged dispatch input for lower_spec_to_cktile (the C analog of Python's
 * isinstance dispatch). */
typedef enum rocke_cktile_spec_kind
{
    ROCKE_CKTILE_SPEC_GEMM = 0,
    ROCKE_CKTILE_SPEC_CONV
} rocke_cktile_spec_kind_t;

typedef struct rocke_cktile_spec
{
    rocke_cktile_spec_kind_t kind;
    union
    {
        const rocke_cktile_gemm_spec_t* gemm;
        const rocke_cktile_conv_spec_t* conv;
    } u;
} rocke_cktile_spec_t;

/* ------------------------------------------------------------------ API */

/* Emit a CK Tile C++ source for a UniversalGemmSpec into `out`.
 * `kernel_name_override` (or spec->kernel_name) overrides the mangled name.
 * Returns ROCKE_OK on success; ROCKE_ERR_NOTIMPL / ROCKE_ERR_VALUE on the Python
 * NotImplementedError / ValueError paths (message in `err`). */
rocke_status_t rocke_lower_universal_gemm_to_cktile(const rocke_cktile_gemm_spec_t* spec,
                                                    const char* kernel_name_override,
                                                    rocke_strbuf_t* out,
                                                    char err[ROCKE_ERR_MSG_CAP]);

/* Emit a CK Tile grouped-convolution-forward source for an
 * ImplicitGemmConvSpec into `out`. */
rocke_status_t rocke_lower_implicit_gemm_conv_to_cktile(const rocke_cktile_conv_spec_t* spec,
                                                        const char* kernel_name_override,
                                                        rocke_strbuf_t* out,
                                                        char err[ROCKE_ERR_MSG_CAP]);

/* Dispatch to the right CK Tile emitter for `spec`. */
rocke_status_t rocke_lower_spec_to_cktile(const rocke_cktile_spec_t* spec,
                                          const char* kernel_name_override,
                                          rocke_strbuf_t* out,
                                          char err[ROCKE_ERR_MSG_CAP]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_LOWER_CKTILE_H */
