// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_xf32_bf16x3_lowering.cpp -- C++ engine coverage for the XF32 BF16x3
 * emulated MMA atoms (AICK-2208).
 *
 * The Python-side twin is tests/core/test_xf32_bf16x3_lowering.py, which also
 * carries the engine-parity check. That check needs the rocke_engine pybind
 * module and skips where it is absent, so the C engine would otherwise be
 * ungated in exactly the environments that build only the archive. This file
 * closes that hole: it parses serialized IR (the same artifact the Python test
 * lowers) and pins the emitted text directly.
 *
 * Input IR is embedded rather than generated because rocke_b_mma takes a bare
 * op_id and resolves the fragment lengths from the static frag table, which
 * does not carry the emulated op_ids. Serialized IR spells the vector types
 * outright, so it needs no such lookup -- and it is the artifact the engine is
 * actually contracted to lower.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"

/* One tile.mma on xf32_bf16x3_f32_16x16x32: <8 x float> operands, <4 x float>
 * accumulator. */
static const char* const k_ir_16x16x32
    = "ckdsl.ir v1\n"
      "kernel @emulated_mma_smoke {\n"
      "  attrs { max_workgroup_size = i:64 }\n"
      "  params {\n"
      "  }\n"
      "  region @entry {\n"
      "    %c1 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:1.0 }\n"
      "    %splat2 : vec<f32x8> = vector.splat ( %c1 ) { vec = i:8 }\n"
      "    %c3 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:2.0 }\n"
      "    %splat4 : vec<f32x8> = vector.splat ( %c3 ) { vec = i:8 }\n"
      "    %c5 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:0.0 }\n"
      "    %splat6 : vec<f32x4> = vector.splat ( %c5 ) { vec = i:4 }\n"
      "    %acc7 : vec<f32x4> = tile.mma ( %splat2, %splat4, %splat6 ) "
      "{ op_id = s:\"xf32_bf16x3_f32_16x16x32\" }\n"
      "    cf.return (  )\n"
      "  }\n"
      "}\n";

/* The 32x32x16 twin: same <8 x float> operands, <16 x float> accumulator. */
static const char* const k_ir_32x32x16
    = "ckdsl.ir v1\n"
      "kernel @emulated_mma_smoke {\n"
      "  attrs { max_workgroup_size = i:64 }\n"
      "  params {\n"
      "  }\n"
      "  region @entry {\n"
      "    %c1 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:1.0 }\n"
      "    %splat2 : vec<f32x8> = vector.splat ( %c1 ) { vec = i:8 }\n"
      "    %c3 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:2.0 }\n"
      "    %splat4 : vec<f32x8> = vector.splat ( %c3 ) { vec = i:8 }\n"
      "    %c5 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:0.0 }\n"
      "    %splat6 : vec<f32x16> = vector.splat ( %c5 ) { vec = i:16 }\n"
      "    %acc7 : vec<f32x16> = tile.mma ( %splat2, %splat4, %splat6 ) "
      "{ op_id = s:\"xf32_bf16x3_f32_32x32x16\" }\n"
      "    cf.return (  )\n"
      "  }\n"
      "}\n";

#define CHECK(cond, what)                                       \
    do                                                          \
    {                                                           \
        if(!(cond))                                             \
        {                                                       \
            fprintf(stderr, "FAIL %s: %s\n", __func__, (what)); \
            failed = 1;                                         \
        }                                                       \
    } while(0)

/* Lower serialized IR for gfx950; caller frees. NULL on any failure. */
static char* lower(const char* ir_text)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "xf32_bf16x3_lowering") != ROCKE_OK)
    {
        return NULL;
    }
    rocke_kernel_def_t* kernel = NULL;
    if(rocke_ir_parse(ir_text, &b, &kernel) != ROCKE_OK || !kernel)
    {
        fprintf(stderr, "parse failed: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return NULL;
    }
    char* out = NULL;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
        kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx950", &out, err, sizeof(err));
    if(st != ROCKE_OK || !out)
    {
        fprintf(stderr, "lower failed: %s\n", err[0] ? err : "(no message)");
        rocke_ir_builder_free(&b);
        return NULL;
    }
    rocke_ir_builder_free(&b);
    return out;
}

/* Count non-overlapping occurrences of needle. */
static int count(const char* haystack, const char* needle)
{
    int n = 0;
    size_t len = strlen(needle);
    for(const char* p = strstr(haystack, needle); p; p = strstr(p + len, needle))
    {
        ++n;
    }
    return n;
}

/* Each operand is split into big / small by four lines: narrow, widen, subtract,
 * narrow the remainder. */
static int test_operands_are_split(void)
{
    int failed = 0;
    const struct
    {
        const char* ir;
        const char* f32v;
        const char* bf16v;
    } cases[] = {{k_ir_16x16x32, "<8 x float>", "<8 x bfloat>"},
                 {k_ir_32x32x16, "<8 x float>", "<8 x bfloat>"}};
    for(size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        char* ll = lower(cases[i].ir);
        CHECK(ll != NULL, "lowering produced no text");
        if(!ll)
        {
            continue;
        }
        char line[128];
        /* a: big = rne_bf16(%splat2), then the error term. */
        snprintf(line, sizeof(line), "fptrunc %s %%splat2 to %s", cases[i].f32v, cases[i].bf16v);
        CHECK(strstr(ll, line) != NULL, "operand a is not narrowed to bf16");
        snprintf(line, sizeof(line), "fsub %s %%splat2,", cases[i].f32v);
        CHECK(strstr(ll, line) != NULL, "operand a's rounding error is not taken");
        /* b: the same, from the other splat. */
        snprintf(line, sizeof(line), "fptrunc %s %%splat4 to %s", cases[i].f32v, cases[i].bf16v);
        CHECK(strstr(ll, line) != NULL, "operand b is not narrowed to bf16");
        snprintf(line, sizeof(line), "fsub %s %%splat4,", cases[i].f32v);
        CHECK(strstr(ll, line) != NULL, "operand b's rounding error is not taken");
        /* Four narrowings and two widenings in total -- no more, no fewer. */
        snprintf(line, sizeof(line), " to %s", cases[i].bf16v);
        CHECK(count(ll, line) == 4, "expected exactly four fp32->bf16 narrowings");
        snprintf(line, sizeof(line), "fpext %s", cases[i].bf16v);
        CHECK(count(ll, line) == 2, "expected exactly two bf16->fp32 widenings");
        /* RNE, not truncation: fptrunc is LLVM's only fp32->bf16 rounding. */
        CHECK(count(ll, "fptrunc") == 4, "the split must use fptrunc (RNE)");
        free(ll);
    }
    return failed;
}

/* Three MMAs on the physical bf16 atom, threading one accumulator, in the order
 * small*big, big*small, big*big. The order is load-bearing: fp32 accumulation is
 * not associative. */
static int test_three_terms_in_order(void)
{
    int failed = 0;
    const struct
    {
        const char* ir;
        const char* intrinsic;
    } cases[] = {{k_ir_16x16x32, "@llvm.amdgcn.mfma.f32.16x16x32.bf16"},
                 {k_ir_32x32x16, "@llvm.amdgcn.mfma.f32.32x32x16.bf16"}};
    for(size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        char* ll = lower(cases[i].ir);
        CHECK(ll != NULL, "lowering produced no text");
        if(!ll)
        {
            continue;
        }
        /* One declare plus three calls. */
        CHECK(count(ll, cases[i].intrinsic) == 4, "expected exactly three MMA calls");
        /* The seed accumulator enters the first term and nothing else; the
         * caller's value is written by the last term and nothing else. */
        CHECK(count(ll, "%splat6") == 2, "the seed accumulator feeds exactly one term");
        CHECK(count(ll, "%acc7 = call") == 1, "the result is produced by one call");
        /* Two intermediate accumulators thread the first two terms. */
        CHECK(count(ll, "%xf32_acc.") == 4,
              "expected two intermediate accumulators, each defined and used once");
        /* small*big, big*small, big*big -- in that order. */
        const char* call_lines[3] = {NULL, NULL, NULL};
        int num_calls = 0;
        for(const char* p = ll; *p && num_calls < 3;)
        {
            const char* eol = strchr(p, '\n');
            const char* end = eol ? eol : p + strlen(p);
            const char* call_hit = strstr(p, " = call ");
            const char* atom_hit = strstr(p, cases[i].intrinsic);
            if(call_hit && call_hit < end && atom_hit && atom_hit < end)
            {
                call_lines[num_calls++] = p;
            }
            p = eol ? eol + 1 : end;
        }
        CHECK(num_calls == 3, "expected three call sites");
        if(num_calls == 3)
        {
            const char* const want[3][2] = {{"%xf32_a_small.", "%xf32_b_big."},
                                            {"%xf32_a_big.", "%xf32_b_small."},
                                            {"%xf32_a_big.", "%xf32_b_big."}};
            const char* const order[3] = {
                "term 1 must be small*big", "term 2 must be big*small", "term 3 must be big*big"};
            for(int t = 0; t < 3; ++t)
            {
                const char* eol = strchr(call_lines[t], '\n');
                const char* a_hit = strstr(call_lines[t], want[t][0]);
                const char* b_hit = strstr(call_lines[t], want[t][1]);
                CHECK(a_hit && eol && a_hit < eol && b_hit && b_hit < eol, order[t]);
            }
        }
        /* small*small is deliberately never formed: a_small is consumed once. */
        CHECK(count(ll, "bfloat> %xf32_a_small.") == 1, "a_small feeds exactly one term");
        CHECK(count(ll, "bfloat> %xf32_b_small.") == 1, "b_small feeds exactly one term");
        free(ll);
    }
    return failed;
}

/* The expansion is complete: no emulated op_id reaches the emitted IR, and the
 * physical atom is the one the catalog names. */
static int test_no_emulated_atom_survives(void)
{
    int failed = 0;
    const char* const irs[2] = {k_ir_16x16x32, k_ir_32x32x16};
    const char* const op_ids[2] = {"xf32_bf16x3_f32_16x16x32", "xf32_bf16x3_f32_32x32x16"};
    for(int i = 0; i < 2; ++i)
    {
        char* ll = lower(irs[i]);
        CHECK(ll != NULL, "lowering produced no text");
        if(!ll)
        {
            continue;
        }
        CHECK(strstr(ll, op_ids[i]) == NULL, "an emulated op_id leaked into the IR");
        free(ll);
    }
    return failed;
}

/* Native atoms are untouched -- the guard fires only for emulated ones. */
static int test_native_atom_is_unchanged(void)
{
    int failed = 0;
    static const char* const k_native_ir
        = "ckdsl.ir v1\n"
          "kernel @native_mma_smoke {\n"
          "  attrs { max_workgroup_size = i:64 }\n"
          "  params {\n"
          "  }\n"
          "  region @entry {\n"
          "    %c1 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:0.0 }\n"
          "    %splat2 : vec<f32x4> = vector.splat ( %c1 ) { vec = i:4 }\n"
          "    %c3 : bf16 = arith.constant (  ) { ity = s:\"bf16\", value = f:1.0 }\n"
          "    %splat4 : vec<bf16x8> = vector.splat ( %c3 ) { vec = i:8 }\n"
          "    %acc5 : vec<f32x4> = tile.mma ( %splat4, %splat4, %splat2 ) "
          "{ op_id = s:\"mfma_f32_16x16x32_bf16\" }\n"
          "    cf.return (  )\n"
          "  }\n"
          "}\n";
    char* ll = lower(k_native_ir);
    CHECK(ll != NULL, "lowering produced no text");
    if(ll)
    {
        CHECK(count(ll, "@llvm.amdgcn.mfma.f32.16x16x32.bf16") == 2,
              "a native atom must lower to one declare and one call");
        CHECK(count(ll, "fptrunc") == 0, "a native atom must not be split");
        free(ll);
    }
    return failed;
}

int main(void)
{
    int failed = 0;
    failed |= test_operands_are_split();
    failed |= test_three_terms_in_order();
    failed |= test_no_emulated_atom_survives();
    failed |= test_native_atom_is_unchanged();
    if(failed)
    {
        fprintf(stderr, "test_xf32_bf16x3_lowering: FAILED\n");
        return 1;
    }
    printf("test_xf32_bf16x3_lowering: OK\n");
    return 0;
}
