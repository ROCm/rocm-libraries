// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/instances/test_attention_dense_spec.cpp -- C parity for the shared
 * dense attention spec contract (rocke/instance_attention_dense_spec.h).
 *
 * The validator and preflight reasons never reach the emitted IR, so the
 * byte-identity gate cannot see a C/Python divergence in them; the kernel name
 * does reach it, but only for the specs an emit pair builds. The vectors are
 * pinned here instead.
 *
 * Every expected value below was produced by the Python implementation
 * (kernels.common.attention_dense_spec: AttentionDenseSpec.__post_init__,
 * check_dense_spec_preflight, kernel_name, resolved_persist_decode) for the
 * same base-class specs, on CPython 3.14. Each reject reason is reached at
 * least once; regenerate them from there if the Python reference changes.
 */
#include <cstdio>
#include <cstring>

#include "rocke/instance_attention_dense_spec.h"

namespace
{

int g_failures = 0;

#define CHECK_STR(got, want)                                                  \
    do                                                                        \
    {                                                                         \
        if(std::strcmp((got), (want)) != 0)                                   \
        {                                                                     \
            std::fprintf(stderr,                                              \
                         "FAIL vector %zu (%s:%d)\n  got:  %s\n  want: %s\n", \
                         i,                                                   \
                         __FILE__,                                            \
                         __LINE__,                                            \
                         (got),                                               \
                         (want));                                             \
            ++g_failures;                                                     \
        }                                                                     \
    } while(0)

#define CHECK(cond)                                                                              \
    do                                                                                           \
    {                                                                                            \
        if(!(cond))                                                                              \
        {                                                                                        \
            std::fprintf(stderr, "FAIL vector %zu: %s (%s:%d)\n", i, #cond, __FILE__, __LINE__); \
            ++g_failures;                                                                        \
        }                                                                                        \
    } while(0)

struct Vector
{
    rocke_attention_dense_spec_t spec;
    bool valid;
    const char* reason;
    bool preflight_ok;
    const char* preflight_reason;
    const char* kernel_name; /* NULL when the spec is invalid */
    const char* resolved_persist_decode;
};

/* Spec fields in AttentionDenseSpec declaration order. */
const Vector k_vectors[] = {
    {{0, 512, 500, 32,  8, 128,        1, "fp16", -64, 0,       0, 128, 64,
      4, 16,  1,   256, 1, "gqa_pair", 1, 0,      16,  1048576, 0, 0},
     0,
     "seqlen_kv must be a multiple of block_n=64, got 500",
     0,
     "invalid AttentionDenseSpec: seqlen_kv must be a multiple of block_n=64, got 500",
     NULL,
     NULL},
    {{1, 2,  0, 8,  3, 128,    0, "fp16", -64,   0,  1, 128, 0,
      2, -8, 1, -1, 0, "auto", 1, 1,      65536, 64, 1, 0},
     0,
     "block_n must be a positive multiple of 32, got 0",
     0,
     "invalid AttentionDenseSpec: block_n must be a positive multiple of 32, got 0",
     NULL,
     NULL},
    {{1048576, 16384, 16384, 1048576, 0, 96,     1, "fp32", -64, 1,  0, 256, -64,
      9,       8,     0,     256,     0, "auto", 0, 0,      16,  -1, 1, 0},
     0,
     "dtype must be one of ['bf16', 'fp16'], got fp32",
     0,
     "invalid AttentionDenseSpec: dtype must be one of ['bf16', 'fp16'], got fp32",
     NULL,
     NULL},
    {{-1, 2048, 2048, 16, 1, 64,   0, "bf16", 64, 1,  1, 256, 64,
      1,  0,    0,    8,  1, NULL, 0, 0,      64, 64, 0, 0},
     0,
     "ragged is not supported with varlen",
     0,
     "invalid AttentionDenseSpec: ragged is not supported with varlen",
     NULL,
     NULL},
    {{1, 256, 256, 4,   1048576, 256,    1, NULL, 128, 0,    0, 64, 128,
      0, 8,   1,   256, 1,       "auto", 0, 0,    32,  1024, 0, 0},
     0,
     "dtype must be one of ['bf16', 'fp16'], got None",
     0,
     "invalid AttentionDenseSpec: dtype must be one of ['bf16', 'fp16'], got None",
     NULL,
     NULL},
    {{2147483647, 500, 1024, 16,  8, 128,        1, "fp16", 64, 1, 0, 256, 64,
      2,          0,   0,    256, 0, "qb_major", 1, 0,      16, 0, 1, 0},
     0,
     "ragged is not supported with sliding_window",
     0,
     "invalid AttentionDenseSpec: ragged is not supported with sliding_window",
     NULL,
     NULL},
    {{2147483647, 1024, 1024, 8,   8, 128,  1, "fp16", 100, 0,    0, 128, 64,
      1,          0,    1,    304, 1, NULL, 0, 0,      16,  1024, 1, 1},
     0,
     "persist_decode must be one of ['auto', 'hkv_major', 'qb_major'], got None",
     0,
     "invalid AttentionDenseSpec: persist_decode must be one of ['auto', 'hkv_major', 'qb_major'], "
     "got None",
     NULL,
     NULL},
    {{2147483647, -256, 0, 4, 1048576, 128,        0, "fp16", -64, 1,       0, 0, 0,
      9,          -8,   0, 0, 0,       "gqa_pair", 0, 0,      128, 1048576, 1, 0},
     0,
     "block_m must be positive, got 0",
     0,
     "invalid AttentionDenseSpec: block_m must be positive, got 0",
     NULL,
     NULL},
    {{2147483647, 64, 512, 32, 8, 256,    1, "fp16", 100, 0,  0, 16, 256,
      0,          4,  0,   1,  1, "auto", 0, 0,      48,  -1, 0, 0},
     0,
     "head_size must be 64 or 128, got 256",
     0,
     "invalid AttentionDenseSpec: head_size must be 64 or 128, got 256",
     NULL,
     NULL},
    {{0, 500, 500, 16,  1, 128,    0, "fp16", 128, 0,    0, 128, 128,
      2, 0,   0,   304, 0, "auto", 1, 0,      128, 1024, 0, 0},
     0,
     "seqlen_q must be a multiple of block_m=128, got 500",
     0,
     "invalid AttentionDenseSpec: seqlen_q must be a multiple of block_m=128, got 500",
     NULL,
     NULL},
    {{-1, 512, 256, -8,  8, 64,         1, "fp16", 0,  0,  1, 16, 64,
      8,  0,   1,   256, 0, "qb_major", 1, 0,      32, 64, 1, 0},
     0,
     "varlen is not supported with persistent=True",
     0,
     "invalid AttentionDenseSpec: varlen is not supported with persistent=True",
     NULL,
     NULL},
    {{1, 500, 64, 32,  1, 64,     1, "fp16", -64, 1, 0, 96, 128,
      8, 8,   0,  256, 0, "auto", 0, 0,      16,  0, 0, 0},
     0,
     "sliding_window must be >= 0, got -64",
     0,
     "invalid AttentionDenseSpec: sliding_window must be >= 0, got -64",
     NULL,
     NULL},
    {{1048576, 2048, 512, 8, 1, 128,    1, "fp16", 128, 1, 1, 128, 64,
      2,       8,    0,   4, 0, "it's", 1, 1,      0,   0, 0, 1},
     0,
     "causal_bottom_right requires seqlen_q <= seqlen_kv, got 2048 > 512",
     0,
     "invalid AttentionDenseSpec: causal_bottom_right requires seqlen_q <= seqlen_kv, got 2048 > "
     "512",
     NULL,
     NULL},
    {{2147483647, 300, 1024, 16,  8, 64,     0, "bf16", -64, 0,    1, 128, 32,
      2,          0,   0,    304, 0, "x\"y", 1, 1,      32,  1024, 0, 1},
     0,
     "causal_bottom_right requires causal=True",
     0,
     "invalid AttentionDenseSpec: causal_bottom_right requires causal=True",
     NULL,
     NULL},
    {{2147483647, 512, 1024, 16, 2, 64,     1, "fp16", 256,   0,    1, 128, 64,
      1,          0,   0,    4,  0, "auto", 1, 0,      65536, 1024, 0, 1},
     1,
     "",
     0,
     "K/V extent is 562949953159168 B, at or past the 32-bit buffer-resource limit (2147483648 B)",
     "rocke_attention_dense_d64_hq16_kv2_bn64_fp16_bm128_kpad0_sq512_sk1024_causal_br_swa256_"
     "varlen_lazyrs",
     "hkv_major"},
    {{2147483647, 256, 256, 0, 2, 128,         1, "bf16", 128, 0,    1, 16, 64,
      8,          16,  0,   1, 1, "hkv_major", 1, 0,      16,  1024, 0, 1},
     1,
     "",
     0,
     "num_query_heads must be positive, got 0",
     "rocke_attention_dense_d128_hq0_kv2_bn64_bf16_bm16_sq256_sk256_causal_br_swa128_varlen_lazyrs",
     "hkv_major"},
    {{-1, 500, 512, 16, 2, 128,    1, "fp16", 0,   1,  0, 256, 64,
      9,  4,   1,   1,  1, "auto", 0, 0,      128, 64, 1, 1},
     0,
     "lds_k_group_pad must be a non-negative multiple of 8 bf16 elements (16 bytes) so the K group "
     "pitch stays ds_read_b128-aligned, got 4",
     0,
     "invalid AttentionDenseSpec: lds_k_group_pad must be a non-negative multiple of 8 bf16 "
     "elements (16 bytes) so the K group pitch stays ds_read_b128-aligned, got 4",
     NULL,
     NULL},
    {{4, 2048, 2048, 8,   8, 64,         0, "bf16", 0, 0,  0, 256, 64,
      2, 8,    1,    304, 1, "gqa_pair", 0, 0,      8, 64, 0, 0},
     0,
     "persist_decode must be one of ['auto', 'hkv_major', 'qb_major'], got 'gqa_pair'",
     0,
     "invalid AttentionDenseSpec: persist_decode must be one of ['auto', 'hkv_major', 'qb_major'], "
     "got 'gqa_pair'",
     NULL,
     NULL},
    {{1, 512, 2048, 8,  8, 64,          1, "bf16", 100, 0,  1, 128, 128,
      2, 0,   0,    64, 1, "hkv_major", 1, 0,      64,  64, 1, 1},
     0,
     "sliding_window (100) must be a multiple of block_n=128",
     0,
     "invalid AttentionDenseSpec: sliding_window (100) must be a multiple of block_n=128",
     NULL,
     NULL},
    {{1, 512, 2048, 16, 8, 64,         1, "fp16", 0, 1,    0, 256, 128,
      2, 8,   0,    4,  0, "qb_major", 1, 1,      0, 1024, 0, 0},
     0,
     "paged=True requires block_size > 0",
     0,
     "invalid AttentionDenseSpec: paged=True requires block_size > 0",
     NULL,
     NULL},
    {{1, 256, 1024, 8, 0, 128,        1, "fp16", 0,  0,       0, 256, 64,
      1, 0,   0,    4, 1, "gqa_pair", 1, 0,      48, 1048576, 0, 1},
     0,
     "num_query_heads (8) must be a positive multiple of num_kv_heads (0)",
     0,
     "invalid AttentionDenseSpec: num_query_heads (8) must be a positive multiple of num_kv_heads "
     "(0)",
     NULL,
     NULL},
    {{2147483647, 2048, 1024, 16,  2, 128,         0, "fp16", 0,  0, 0, 256, 128,
      8,          0,    0,    304, 0, "hkv_major", 1, 1,      48, 0, 0, 0},
     0,
     "paged block_size (48) must be a power of two",
     0,
     "invalid AttentionDenseSpec: paged block_size (48) must be a power of two",
     NULL,
     NULL},
    {{1, -256, 64, 8, 3, 128,        1, "bf16", 64, 1,       0, 16, 128,
      2, 8,    0,  0, 0, "gqa_pair", 0, 0,      8,  1048576, 0, 0},
     0,
     "ragged requires positive seqlen_q/seqlen_kv",
     0,
     "invalid AttentionDenseSpec: ragged requires positive seqlen_q/seqlen_kv",
     NULL,
     NULL},
    {{1048576, 512, 512, 16, 8, 128,         1, "fp16", 0,  0,  0, 256, 64,
      2,       8,   1,   8,  0, "hkv_major", 1, 1,      16, -1, 0, 0},
     0,
     "paged=True requires num_kv_blocks > 0",
     0,
     "invalid AttentionDenseSpec: paged=True requires num_kv_blocks > 0",
     NULL,
     NULL},
    {{0, 256, 2048, 32, 2, 128,    1, "fp16", 0,   1,       0, 256, 64,
      1, 8,   0,    64, 0, "auto", 0, 1,      128, 1048576, 0, 0},
     0,
     "block_n (64) must be a multiple of page block_size (128)",
     0,
     "invalid AttentionDenseSpec: block_n (64) must be a multiple of page block_size (128)",
     NULL,
     NULL},
    {{1, 300, 500, 16, 1, 128,    1, "fp16", 0,  1,    0, 128, 32,
      2, 16,  1,   4,  0, "auto", 1, 1,      16, 1024, 0, 0},
     0,
     "paged + persistent not yet implemented (persistent builder is contiguous-only)",
     0,
     "invalid AttentionDenseSpec: paged + persistent not yet implemented (persistent builder is "
     "contiguous-only)",
     NULL,
     NULL},
    {{2147483647, 2048, 500, 16,  2, 64,     0, "bf16", 0, 1,    1, 256, 32,
      2,          16,   1,   256, 0, "x\"y", 1, 0,      8, 1024, 0, 0},
     0,
     "ragged is self-attention only (seqlen_q == seqlen_kv) unless causal (top-left or "
     "bottom-right) is set, got 2048 != 500",
     0,
     "invalid AttentionDenseSpec: ragged is self-attention only (seqlen_q == seqlen_kv) unless "
     "causal (top-left or bottom-right) is set, got 2048 != 500",
     NULL,
     NULL},
    {{0, 2048, 512, 8,   1, 64,         1, "bf16", 0, 0,  0, 128, 128,
      4, 0,    0,   304, 1, "qb_major", 1, 0,      8, -1, 0, 0},
     1,
     "",
     0,
     "batch must be positive, got 0",
     "rocke_attention_dense_d64_hq8_kv1_bn128_bf16_bm128_kpad0_sq2048_sk512_causal_lazyrs",
     "qb_major"},
    {{0, 1024, 1024, 8,   8, 128,         0, "fp16", 256, 0,  1, 128, 64,
      4, 0,    1,    304, 0, "hkv_major", 1, 1,      128, 64, 0, 0},
     0,
     "sliding_window>0 requires causal=True",
     0,
     "invalid AttentionDenseSpec: sliding_window>0 requires causal=True",
     NULL,
     NULL},
    {{0, 256, 2048, 16, 8, 64,     1, "bf16", 0,     0,       0, 256, 256,
      9, 8,   1,    1,  1, "auto", 0, 1,      65536, 1048576, 1, 1},
     0,
     "waves_per_eu must be in [1, 8], got 9",
     0,
     "invalid AttentionDenseSpec: waves_per_eu must be in [1, 8], got 9",
     NULL,
     NULL},
    {{2, 512, 1024, 16, 8, 128,        1, "bf16", 256, 0,  0, 256, 128,
      4, 16,  1,    4,  1, "qb_major", 1, 1,      64,  64, 1, 0},
     0,
     "paged multi-sequence (batch>1) not yet implemented",
     0,
     "invalid AttentionDenseSpec: paged multi-sequence (batch>1) not yet implemented",
     NULL,
     NULL},
    {{0, 2048, 1024, 8, 1, 64,     1, "bf16", 0,  0,    1, 256, 64,
      1, 0,    0,    4, 0, "auto", 1, 0,      64, 1024, 1, 0},
     0,
     "use_sinks is not yet supported with varlen",
     0,
     "invalid AttentionDenseSpec: use_sinks is not yet supported with varlen",
     NULL,
     NULL},
    {{1, 4096, 64, 0,  2, 128,        0, "fp16", 0,  0,       1, 16, 64,
      0, 0,    0,  -1, 0, "qb_major", 0, 1,      48, 1048576, 0, 0},
     0,
     "varlen requires causal=True",
     0,
     "invalid AttentionDenseSpec: varlen requires causal=True",
     NULL,
     NULL},
    {{4, 1024, 2048, 8, -2, 64,     1, "fp16", 0, 0, 0, 256, 32,
      1, 8,    1,    0, 0,  "it's", 1, 1,      8, 0, 0, 0},
     0,
     "num_persistent must be positive, got 0",
     0,
     "invalid AttentionDenseSpec: num_persistent must be positive, got 0",
     NULL,
     NULL},
    {{1, 256, 1024, 16, 2, 64,     1, "fp16", 0, 0,    0, 256, 128,
      2, 16,  1,    4,  0, "auto", 1, 1,      8, 1024, 0, 0},
     0,
     "paged block_size (8) must be >= and a multiple of ROWS_PER_WAVE (16)",
     0,
     "invalid AttentionDenseSpec: paged block_size (8) must be >= and a multiple of ROWS_PER_WAVE "
     "(16)",
     NULL,
     NULL},
    {{1, 1024, 1024, 16, 1, 128,        1, "fp16", 0,  0,       0, 128, 64,
      1, 16,   1,    64, 0, "qb_major", 1, 1,      64, 1048576, 1, 1},
     0,
     "paged cache 17179869184 B exceeds i32 addressing (2 GiB)",
     0,
     "invalid AttentionDenseSpec: paged cache 17179869184 B exceeds i32 addressing (2 GiB)",
     NULL,
     NULL},
    {{1, 1024, 1024, 32,  1, 64,         1, "fp16", 0,  1,    0, 256, 32,
      1, 0,    0,    304, 1, "qb_major", 0, 1,      32, 1024, 0, 0},
     0,
     "paged not yet implemented for head_size != 128",
     0,
     "invalid AttentionDenseSpec: paged not yet implemented for head_size != 128",
     NULL,
     NULL},
    {{2, 64, 64, 32, 8, 128,         1, "fp16", 64, 0,    0, 16, 64,
      1, 16, 1,  1,  1, "hkv_major", 1, 0,      32, 1024, 0, 1},
     1,
     "",
     0,
     "block_n must divide the 16-row query tile (got block_n=64; the spec also requires block_n % "
     "32 == 0, so use 32, 64, 128 or 256). Load-bearing for causal=True, where n_per = 16 // "
     "block_n floors and drops keys",
     "rocke_attention_dense_d128_hq32_kv8_bn64_fp16_bm16_sq64_sk64_causal_br_swa64_lazyrs_persist1_"
     "hkvmaj_intl",
     "hkv_major"},
    {{1, 256, 1024, 16,  8, 128,    1, "bf16", 0,  1,    0, 128, 32,
      1, 16,  0,    304, 1, "auto", 1, 1,      32, 1024, 0, 0},
     0,
     "paged not yet implemented for plain-causal (sliding_window>0 only)",
     0,
     "invalid AttentionDenseSpec: paged not yet implemented for plain-causal (sliding_window>0 "
     "only)",
     NULL,
     NULL},
    {{2147483647, 1048576, 512, 0, 2, 128,    1, "fp16", 0,   1,  0, 16, 256,
      1,          8,       0,   8, 1, "auto", 0, 1,      128, -1, 1, 0},
     0,
     "division by zero",
     0,
     "invalid AttentionDenseSpec: division by zero",
     NULL,
     NULL},
    {{4, -256, 512, 0, -2, 64,     0, "fp16", 0,  0,    0, 16, 64,
      8, 8,    0,   1, 1,  "auto", 1, 0,      32, 1024, 1, 0},
     1,
     "",
     0,
     "seqlen_q must be positive, got -256",
     "rocke_attention_dense_d64_hq0_kv-2_bn64_fp16_bm16_kpad8_sq-256_sk512_full_sinks_lazyrs",
     "qb_major"},
    {{4, 256, 8192, 16,  -2, 128,    1, "fp16", 0,   1,  0, 128, 64,
      2, 8,   0,    304, 1,  "auto", 1, 0,      128, 64, 0, 0},
     1,
     "",
     0,
     "num_kv_heads must be positive, got -2",
     "rocke_attention_dense_d128_hq16_kv-2_bn64_fp16_bm128_sq256_sk8192_causal_ragged_lazyrs",
     "qb_major"},
    {{1, 512, 512, 32,  1, 128,    1, "fp16", 256, 0,  0, 128, 64,
      2, 8,   0,   304, 1, "auto", 1, 1,      32,  64, 1, 0},
     0,
     "use_sinks is not yet supported with paged KV",
     0,
     "invalid AttentionDenseSpec: use_sinks is not yet supported with paged KV",
     NULL,
     NULL},
    {{1, 256, 1024, 8, 1, 128,         1, "bf16", 0,  0,  1, 128, 128,
      2, 0,   0,    4, 1, "hkv_major", 1, 1,      32, 64, 0, 1},
     0,
     "paged varlen not yet implemented (single-seq only)",
     0,
     "invalid AttentionDenseSpec: paged varlen not yet implemented (single-seq only)",
     NULL,
     NULL},
    {{2147483647, 2048, 2048, 8, 1, 128,    1, "fp16", 0,  1, 0, 256, 32,
      1,          8,    0,    4, 0, "it's", 0, 0,      48, 0, 1, 0},
     0,
     "persist_decode must be one of ['auto', 'hkv_major', 'qb_major'], got \"it's\"",
     0,
     "invalid AttentionDenseSpec: persist_decode must be one of ['auto', 'hkv_major', 'qb_major'], "
     "got \"it's\"",
     NULL,
     NULL},
    {{0, 256, 2048, 16,  1, 128,    0, "bf16", 64, 0,  0, 256, 128,
      2, 0,   0,    304, 1, "x\"y", 1, 0,      64, -1, 1, 0},
     0,
     "persist_decode must be one of ['auto', 'hkv_major', 'qb_major'], got 'x\"y'",
     0,
     "invalid AttentionDenseSpec: persist_decode must be one of ['auto', 'hkv_major', 'qb_major'], "
     "got 'x\"y'",
     NULL,
     NULL},
    {{1, 1024, 1024, 16, 2, 64,     1, "bf16", 0,  1,  0, 128, 32,
      2, 8,    1,    4,  1, "auto", 1, 0,      32, -1, 1, 0},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d64_hq16_kv2_bn32_bf16_bm128_kpad8_sq1024_sk1024_causal_ragged_sinks_"
     "lazyrs_persist4_hkvmaj_intl",
     "hkv_major"},
    {{1, 512, 2048, 16,  1, 64,          1, "fp16", 0,  1,  0, 256, 128,
      1, 0,   1,    256, 1, "hkv_major", 1, 0,      64, 64, 1, 1},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d64_hq16_kv1_bn128_fp16_kpad0_sq512_sk2048_causal_br_ragged_sinks_"
     "lazyrs_persist256_hkvmaj_intl",
     "hkv_major"},
    {{4, 1024, 1024, 8,   2, 64,          1, "fp16", 128, 0, 0, 128, 64,
      1, 8,    1,    304, 0, "hkv_major", 1, 0,      8,   0, 1, 1},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d64_hq8_kv2_bn64_fp16_bm128_kpad8_sq1024_sk1024_causal_br_swa128_sinks_"
     "lazyrs_persist304_hkvmaj",
     "hkv_major"},
    {{1, 256, 1024, 8,   8, 64,          1, "bf16", 0,  0,       0, 128, 64,
      4, 0,   1,    256, 1, "hkv_major", 1, 0,      64, 1048576, 1, 1},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d64_hq8_kv8_bn64_bf16_bm128_kpad0_sq256_sk1024_causal_br_sinks_lazyrs_"
     "persist256_hkvmaj_intl",
     "hkv_major"},
    {{1, 2048, 1024, 32, 2, 128,    1, "bf16", 128, 0,    0, 128, 64,
      4, 16,   1,    64, 1, "auto", 1, 0,      8,   1024, 1, 0},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d128_hq32_kv2_bn64_bf16_bm128_sq2048_sk1024_causal_swa128_sinks_lazyrs_"
     "persist64_hkvmaj_intl",
     "hkv_major"},
    {{2, 512, 1024, 16,  8, 64,     1, "bf16", 64, 0,    0, 128, 32,
      1, 0,   1,    304, 1, "auto", 1, 0,      48, 1024, 1, 1},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d64_hq16_kv8_bn32_bf16_bm128_kpad0_sq512_sk1024_causal_br_swa64_sinks_"
     "lazyrs_persist304_intl",
     "qb_major"},
    {{1, 2048, 2048, 32, 8, 64,         1, "fp16", 256, 0,  0, 128, 64,
      2, 16,   1,    64, 1, "qb_major", 1, 0,      16,  -1, 1, 0},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d64_hq32_kv8_bn64_fp16_bm128_kpad16_sq2048_sk2048_causal_swa256_sinks_"
     "lazyrs_persist64_intl",
     "qb_major"},
    {{4, 256, 512, 16, 2, 128,         1, "fp16", 0, 1, 0, 128, 128,
      4, 16,  1,   64, 1, "hkv_major", 1, 0,      0, 0, 1, 0},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d128_hq16_kv2_bn128_fp16_bm128_sq256_sk512_causal_ragged_sinks_lazyrs_"
     "persist64_hkvmaj_intl",
     "hkv_major"},
    {{1, 1024, 1024, 32, 1, 128,         1, "fp16", 0,  1,  0, 128, 64,
      1, 0,    1,    8,  0, "hkv_major", 1, 0,      32, 64, 1, 1},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d128_hq32_kv1_bn64_fp16_bm128_sq1024_sk1024_causal_br_ragged_sinks_"
     "lazyrs_persist8_hkvmaj",
     "hkv_major"},
    {{1, 256, 500, 32,  8, 128,         1, "bf16", 0, 1,       0, 128, 64,
      4, 16,  1,   304, 1, "hkv_major", 1, 0,      8, 1048576, 0, 1},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d128_hq32_kv8_bn64_bf16_bm128_sq256_sk500_causal_br_ragged_lazyrs_"
     "persist304_hkvmaj_intl",
     "hkv_major"},
    {{2, 500, 500, 32,  8, 128,         1, "bf16", 0,     1,    0, 128, 32,
      4, 0,   1,   304, 0, "hkv_major", 1, 0,      65536, 1024, 1, 0},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d128_hq32_kv8_bn32_bf16_bm128_sq500_sk500_causal_ragged_sinks_lazyrs_"
     "persist304_hkvmaj",
     "hkv_major"},
    {{1, 300, 300, 8, 2, 64,         0, "fp16", 0,  1,       0, 128, 32,
      4, 16,  1,   4, 1, "qb_major", 1, 0,      48, 1048576, 1, 0},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d64_hq8_kv2_bn32_fp16_bm128_kpad16_sq300_sk300_full_ragged_sinks_"
     "lazyrs_persist4_intl",
     "qb_major"},
    {{4, 256, 256, 8, 1, 64,     1, "fp16", 0, 0,  0, 128, 64,
      2, 16,  1,   4, 0, "auto", 1, 0,      0, -1, 0, 1},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d64_hq8_kv1_bn64_fp16_bm128_kpad16_sq256_sk256_causal_br_lazyrs_"
     "persist4_hkvmaj",
     "hkv_major"},
    {{1, 256, 1024, 32,  2, 64,     1, "fp16", 64, 0,       1, 128, 64,
      4, 8,   0,    304, 0, "auto", 1, 0,      8,  1048576, 0, 1},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d64_hq32_kv2_bn64_fp16_bm128_kpad8_sq256_sk1024_causal_br_swa64_varlen_"
     "lazyrs",
     "qb_major"},
    {{1, 64, 512, 8, 8, 128,        1, "fp16", 128, 0,  0, 64, 64,
      2, 8,  1,   8, 0, "qb_major", 1, 0,      128, 64, 0, 1},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d128_hq8_kv8_bn64_fp16_bm64_sq64_sk512_causal_br_swa128_lazyrs_"
     "persist8",
     "qb_major"},
    {{1, 512, 8192, 32, 1, 128,         1, "fp16", 256, 0,       0, 256, 64,
      2, 8,   0,    8,  0, "hkv_major", 1, 0,      0,   1048576, 1, 0},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d128_hq32_kv1_bn64_fp16_sq512_sk8192_causal_swa256_sinks_lazyrs",
     "hkv_major"},
    {{1, 512, 2048, 16,  2, 128,         1, "bf16", 0,  0, 0, 128, 128,
      2, 16,  1,    304, 0, "hkv_major", 0, 0,      32, 0, 1, 0},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d128_hq16_kv2_bn128_bf16_bm128_sq512_sk2048_causal_sinks_persist304_"
     "hkvmaj",
     "hkv_major"},
    {{4, 2048, 2048, 8,   2, 128,        0, "bf16", 0, 0,    0, 128, 32,
      2, 0,    0,    256, 0, "qb_major", 0, 0,      0, 1024, 0, 0},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d128_hq8_kv2_bn32_bf16_bm128_sq2048_sk2048_full",
     "qb_major"},
    {{1, 512, 512, 8,   2, 128,    1, "fp16", 256, 0,  0, 256, 64,
      2, 8,   0,   256, 0, "auto", 1, 1,      16,  64, 0, 0},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d128_hq8_kv2_bn64_fp16_sq512_sk512_causal_swa256_pgd16_nb64_lazyrs",
     "qb_major"},
    {{1, 512, 512, 8,   2, 128,    1, "fp16", 256, 0,  0, 16, 64,
      2, 8,   0,   256, 0, "auto", 1, 1,      16,  64, 0, 0},
     0,
     "division by zero",
     0,
     "invalid AttentionDenseSpec: division by zero",
     NULL,
     NULL},
    {{1, 2048, 2048, 8,   2, 128,    1, "fp16", 256, 0,  0, 2048, 32,
      2, 8,    0,    256, 0, "auto", 1, 1,      16,  64, 0, 0},
     0,
     "division by zero",
     0,
     "invalid AttentionDenseSpec: division by zero",
     NULL,
     NULL},
    {{1, 65536, 65536, 128, 128, 128,    0, "bf16", 0, 0, 0, 256, 64,
      2, 8,     0,     256, 0,   "auto", 1, 0,      0, 0, 0, 0},
     1,
     "",
     0,
     "K/V extent is 2147483648 B, at or past the 32-bit buffer-resource limit (2147483648 B)",
     "rocke_attention_dense_d128_hq128_kv128_bn64_bf16_sq65536_sk65536_full_lazyrs",
     "qb_major"},
    {{1, 65536, 65536, 127, 127, 128,    0, "bf16", 0, 0, 0, 256, 64,
      2, 8,     0,     256, 0,   "auto", 1, 0,      0, 0, 0, 0},
     1,
     "",
     1,
     "",
     "rocke_attention_dense_d128_hq127_kv127_bn64_bf16_sq65536_sk65536_full_lazyrs",
     "qb_major"},
    {{2147483647, 2147483520, 2147483520, 2147483647, 2147483647, 128, 0, "bf16", 0,
      0,          0,          128,        64,         2,          8,   0, 256,    0,
      "auto",     1,          0,          0,          0,          0,   0},
     1,
     "",
     0,
     "K/V extent is 2535301046979548251017181102080 B, at or past the 32-bit buffer-resource limit "
     "(2147483648 B)",
     "rocke_attention_dense_d128_hq2147483647_kv2147483647_bn64_bf16_bm128_sq2147483520_"
     "sk2147483520_full_lazyrs",
     "qb_major"},
};

const rocke_attention_dense_traits_t* traits()
{
    static const rocke_attention_dense_traits_t t = rocke_attention_dense_traits_base();
    return &t;
}

bool validate_base(const void* ctx, char* reason, size_t cap)
{
    return rocke_attention_dense_spec_validate(
        static_cast<const rocke_attention_dense_spec_t*>(ctx), traits(), reason, cap);
}

void check_vectors()
{
    for(size_t i = 0; i < sizeof(k_vectors) / sizeof(k_vectors[0]); ++i)
    {
        const Vector& v = k_vectors[i];
        char reason[ROCKE_ATTENTION_DENSE_REASON_CAP] = "";
        const bool ok
            = rocke_attention_dense_spec_validate(&v.spec, traits(), reason, sizeof(reason));
        CHECK(ok == v.valid);
        CHECK_STR(ok ? "" : reason, v.reason);

        char pre[ROCKE_ATTENTION_DENSE_REASON_CAP] = "x";
        const bool pok = rocke_attention_dense_spec_preflight(
            &v.spec, traits()->type_name, validate_base, &v.spec, pre, sizeof(pre));
        CHECK(pok == v.preflight_ok);
        CHECK_STR(pre, v.preflight_reason);

        if(v.kernel_name == NULL)
            continue;
        rocke_attention_dense_name_parts_t parts;
        rocke_attention_dense_name_parts_base(&v.spec, &parts);
        char name[ROCKE_ATTENTION_DENSE_NAME_CAP];
        CHECK(rocke_attention_dense_kernel_name(&v.spec, &parts, name, sizeof(name)) == ROCKE_OK);
        CHECK_STR(name, v.kernel_name);
        CHECK_STR(rocke_attention_dense_spec_resolved_persist_decode(&v.spec),
                  v.resolved_persist_decode);
    }
}

void check_defaults_and_tables()
{
    const size_t i = 0;
    const rocke_attention_dense_spec_t s = rocke_attention_dense_spec_default();
    CHECK(s.causal && !s.ragged && !s.varlen && s.lazy_rescale && !s.persistent);
    CHECK(s.block_m == 256 && s.block_n == 64 && s.waves_per_eu == 2);
    CHECK(s.lds_k_group_pad == 8 && s.num_persistent == 256);
    CHECK_STR(s.dtype, "bf16");
    CHECK_STR(s.persist_decode, "auto");
    CHECK_STR(ROCKE_DENSE_TILE_GEOMETRIES[0].name, "default");
    CHECK(ROCKE_DENSE_TILE_GEOMETRIES[0].block_m == 256
          && ROCKE_DENSE_TILE_GEOMETRIES[0].block_n == 64);
    CHECK_STR(ROCKE_DENSE_TILE_GEOMETRIES[1].name, "bm128");
    CHECK(ROCKE_DENSE_TILE_GEOMETRIES[1].block_m == 128
          && ROCKE_DENSE_TILE_GEOMETRIES[1].block_n == 64);
}

void check_name_buffer_too_small()
{
    const size_t i = 0;
    rocke_attention_dense_spec_t s = rocke_attention_dense_spec_default();
    s.batch = 1;
    s.seqlen_q = s.seqlen_kv = 512;
    s.num_query_heads = 8;
    s.num_kv_heads = 2;
    s.head_size = 128;
    rocke_attention_dense_name_parts_t parts;
    rocke_attention_dense_name_parts_base(&s, &parts);
    char name[8];
    CHECK(rocke_attention_dense_kernel_name(&s, &parts, name, sizeof(name)) == ROCKE_ERR_VALUE);
    CHECK(rocke_attention_dense_kernel_name(NULL, &parts, name, sizeof(name)) == ROCKE_ERR_VALUE);
}

} // namespace

int main()
{
    check_vectors();
    check_defaults_and_tables();
    check_name_buffer_too_small();
    if(g_failures != 0)
    {
        std::fprintf(stderr, "rocke_attention_dense_spec: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("rocke_attention_dense_spec: OK\n");
    return 0;
}
