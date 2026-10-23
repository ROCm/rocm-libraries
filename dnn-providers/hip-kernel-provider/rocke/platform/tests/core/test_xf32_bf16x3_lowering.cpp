// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_xf32_bf16x3_lowering.cpp -- C++ engine coverage for the emulated TF32
 * BF16x3 MMA atoms.
 *
 * The Python-side twin is tests/core/test_xf32_bf16x3_lowering.py. This file
 * pins the recipe on the C engine's own output, by parsing serialized IR (the
 * same artifact the Python test lowers) and inspecting the emitted text. Byte
 * identity between the two engines is the parity family's job
 * (tests/instances/parity/xf32_bf16x3_mma_emit.{c,py}, run by
 * tools/check_byte_identity.py), which needs only the archive.
 *
 * Input IR is embedded rather than built through rocke_b_mma so these checks
 * hold independent of the builder: serialized IR spells the vector types
 * outright, and it is the artifact the engine is contracted to lower. The
 * builder path is pinned by tests/core/mma_frag_ssot.cpp.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"

/* One tile.mma on mfma_f32_16x16x32_xf32: vec<tf32x8> operands (an fp32 payload
 * wrapped by a bitcast), <4 x float> accumulator. */
static const char* const k_ir_16x16x32
    = "ckdsl.ir v1\n"
      "kernel @emulated_mma_smoke {\n"
      "  attrs { max_workgroup_size = i:64 }\n"
      "  params {\n"
      "  }\n"
      "  region @entry {\n"
      "    %c1 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:1.0 }\n"
      "    %splat2 : vec<f32x8> = vector.splat ( %c1 ) { vec = i:8 }\n"
      "    %tfa : vec<tf32x8> = arith.bitcast ( %splat2 ) { target = s:\"vec<tf32x8>\" }\n"
      "    %c3 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:2.0 }\n"
      "    %splat4 : vec<f32x8> = vector.splat ( %c3 ) { vec = i:8 }\n"
      "    %tfb : vec<tf32x8> = arith.bitcast ( %splat4 ) { target = s:\"vec<tf32x8>\" }\n"
      "    %c5 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:0.0 }\n"
      "    %splat6 : vec<f32x4> = vector.splat ( %c5 ) { vec = i:4 }\n"
      "    %acc7 : vec<f32x4> = tile.mma ( %tfa, %tfb, %splat6 ) "
      "{ op_id = s:\"mfma_f32_16x16x32_xf32\" }\n"
      "    cf.return (  )\n"
      "  }\n"
      "}\n";

/* The 32x32x16 twin: same vec<tf32x8> operands, <16 x float> accumulator. */
static const char* const k_ir_32x32x16
    = "ckdsl.ir v1\n"
      "kernel @emulated_mma_smoke {\n"
      "  attrs { max_workgroup_size = i:64 }\n"
      "  params {\n"
      "  }\n"
      "  region @entry {\n"
      "    %c1 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:1.0 }\n"
      "    %splat2 : vec<f32x8> = vector.splat ( %c1 ) { vec = i:8 }\n"
      "    %tfa : vec<tf32x8> = arith.bitcast ( %splat2 ) { target = s:\"vec<tf32x8>\" }\n"
      "    %c3 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:2.0 }\n"
      "    %splat4 : vec<f32x8> = vector.splat ( %c3 ) { vec = i:8 }\n"
      "    %tfb : vec<tf32x8> = arith.bitcast ( %splat4 ) { target = s:\"vec<tf32x8>\" }\n"
      "    %c5 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:0.0 }\n"
      "    %splat6 : vec<f32x16> = vector.splat ( %c5 ) { vec = i:16 }\n"
      "    %acc7 : vec<f32x16> = tile.mma ( %tfa, %tfb, %splat6 ) "
      "{ op_id = s:\"mfma_f32_32x32x16_xf32\" }\n"
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

/* True when some MMA call line mentions needle. */
static int mma_call_mentions(const char* ll, const char* needle)
{
    for(const char* p = ll; *p;)
    {
        const char* eol = strchr(p, '\n');
        const char* end = eol ? eol : p + strlen(p);
        const char* call_hit = strstr(p, " = call ");
        const char* mma_hit = strstr(p, "@llvm.amdgcn.mfma.");
        if(call_hit && call_hit < end && mma_hit && mma_hit < end)
        {
            const char* hit = strstr(p, needle);
            if(hit && hit < end)
            {
                return 1;
            }
        }
        p = eol ? eol + 1 : end;
    }
    return 0;
}

/* `<8 x i16>` constant with every lane `value`, as the lowering spells it. */
static const char* lanes8(int value)
{
    static char buf[4][128];
    static int next = 0;
    char* out = buf[next++ % 4];
    char lane[16];
    snprintf(lane, sizeof(lane), "i16 %d", value);
    out[0] = '<';
    out[1] = '\0';
    for(int i = 0; i < 8; ++i)
    {
        if(i)
        {
            strcat(out, ", ");
        }
        strcat(out, lane);
    }
    strcat(out, ">");
    return out;
}

/* Each operand is reinterpreted as fp32 -- a bitcast, never a rounding -- and then
 * split into big / small: narrow, cap at the largest finite bf16, widen, subtract
 * from the PAYLOAD, narrow the remainder. */
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
        /* The I32 carrier is viewed as fp32 once per operand, and by bitcast only. */
        CHECK(strstr(ll, "= bitcast <8 x i32> %tfa to <8 x float>") != NULL,
              "operand a's payload is not reinterpreted as fp32");
        CHECK(strstr(ll, "= bitcast <8 x i32> %tfb to <8 x float>") != NULL,
              "operand b's payload is not reinterpreted as fp32");
        /* a: big = rne_bf16(payload), then the error term. */
        CHECK(strstr(ll, "fptrunc <8 x float> %xf32_a_f32.") != NULL,
              "operand a is not narrowed to bf16");
        CHECK(strstr(ll, "fsub <8 x float> %xf32_a_f32.") != NULL,
              "operand a's rounding error is not taken from its payload");
        /* b: the same, from the other operand. */
        CHECK(strstr(ll, "fptrunc <8 x float> %xf32_b_f32.") != NULL,
              "operand b is not narrowed to bf16");
        CHECK(strstr(ll, "fsub <8 x float> %xf32_b_f32.") != NULL,
              "operand b's rounding error is not taken from its payload");
        /* Four fp32->bf16 narrowings (big and small, per operand) and two widenings
         * in total -- no more, no fewer. The cap hands big back with a bitcast, which
         * is not a rounding and does not count. */
        snprintf(line, sizeof(line), "fptrunc %s", cases[i].f32v);
        CHECK(count(ll, line) == 4, "expected exactly four fp32->bf16 narrowings");
        snprintf(line, sizeof(line), "fpext %s", cases[i].bf16v);
        CHECK(count(ll, line) == 2, "expected exactly two bf16->fp32 widenings");
        /* RNE, not truncation: fptrunc is LLVM's only fp32->bf16 rounding. */
        CHECK(count(ll, "fptrunc") == 4, "the split must use fptrunc (RNE)");
        free(ll);
    }
    return failed;
}

/* The cap on big. bf16 round-to-nearest-even overflows to infinity for every fp32
 * payload of magnitude 0x7f7f8000 or more, which turns a finite product into NaN
 * through the cross terms' inf*0. big is therefore held at the largest finite
 * bf16 (0x7f7f, with the value's own sign) while small is still taken from the
 * payload. The numbers are checked on the host model and on a device; this pins
 * that the IR is built the way those assume. */
static int test_big_is_capped_at_the_largest_finite_bf16(void)
{
    int failed = 0;
    const char* const irs[2] = {k_ir_16x16x32, k_ir_32x32x16};
    for(int i = 0; i < 2; ++i)
    {
        char* ll = lower(irs[i]);
        CHECK(ll != NULL, "lowering produced no text");
        if(!ll)
        {
            continue;
        }
        char want[256];
        /* The sign bit (0x8000 as the signed i16 literal) and 0x7f7f, per operand. */
        snprintf(want, sizeof(want), "and <8 x i16> %%xf32_a_big_bits.");
        CHECK(strstr(ll, want) != NULL, "operand a's sign is not extracted");
        snprintf(want, sizeof(want), ", %s\n", lanes8(-0x8000));
        CHECK(count(ll, want) == 2, "expected the sign mask once per operand");
        snprintf(want, sizeof(want), ", %s\n", lanes8(0x7F7F));
        CHECK(count(ll, want) == 2, "expected the largest-finite-bf16 cap once per operand");
        /* Unsigned: a negative value's pattern has the sign bit set and would
         * otherwise compare below every positive one. */
        CHECK(strstr(ll, "= icmp ugt <8 x i16> %xf32_a_big_bits.") != NULL,
              "operand a's overflow test is not an unsigned compare");
        CHECK(strstr(ll, "= icmp ugt <8 x i16> %xf32_b_big_bits.") != NULL,
              "operand b's overflow test is not an unsigned compare");
        CHECK(count(ll, "= icmp ") == 2, "expected one overflow test per operand");
        CHECK(count(ll, "= select <8 x i1>") == 2, "expected one cap select per operand");
        /* Only what is past the cap is replaced: the select keeps the capped value
         * when the compare fires and the rounded pattern otherwise. */
        CHECK(strstr(ll, "= select <8 x i1> %xf32_a_big_over.") != NULL,
              "operand a's select is not driven by its overflow test");
        /* The widening reads the capped value, not the raw rounding, and the
         * remainder is taken against the payload. */
        CHECK(strstr(ll, "= fpext <8 x bfloat> %xf32_a_big.") != NULL,
              "operand a widens something other than its capped big");
        CHECK(strstr(ll, "= fpext <8 x bfloat> %xf32_a_big_rne.") == NULL,
              "operand a widens the uncapped rounding");
        CHECK(strstr(ll, "fsub <8 x float> %xf32_a_f32.") != NULL,
              "operand a's remainder is not taken against its payload");
        /* The MMAs consume the capped value. */
        CHECK(!mma_call_mentions(ll, "%xf32_a_big_rne."),
              "an MMA consumes the uncapped rounding of a");
        CHECK(!mma_call_mentions(ll, "%xf32_b_big_rne."),
              "an MMA consumes the uncapped rounding of b");
        CHECK(mma_call_mentions(ll, "%xf32_a_big."), "no MMA consumes the capped big of a");
        CHECK(mma_call_mentions(ll, "%xf32_b_big."), "no MMA consumes the capped big of b");
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
    const char* const op_ids[2] = {"mfma_f32_16x16x32_xf32", "mfma_f32_32x32x16_xf32"};
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

/* Lower serialized IR for gfx950 and return the engine's error text; caller
 * frees. NULL when the IR unexpectedly lowers. */
static char* lower_error(const char* ir_text)
{
    rocke_ir_builder_t b;
    char* error_text = NULL;
    if(rocke_ir_builder_init(&b, "xf32_bf16x3_lowering") != ROCKE_OK)
    {
        return NULL;
    }
    rocke_kernel_def_t* kernel = NULL;
    if(rocke_ir_parse(ir_text, &b, &kernel) == ROCKE_OK && kernel)
    {
        char* out = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = '\0';
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx950", &out, err, sizeof(err));
        if(st != ROCKE_OK)
        {
            error_text = strdup(err);
        }
        free(out);
    }
    rocke_ir_builder_free(&b);
    return error_text;
}

/* The emulated atoms share the native atoms' ABI: logical TF32 operands carried
 * as I32. An fp32 vector is not one, and a TF32 vector of the wrong length is
 * not this atom's fragment -- both are rejected, and the message names what was
 * expected. */
static int test_operands_must_be_tf32_fragments(void)
{
    int failed = 0;
    static const char* const k_fp32_operands
        = "ckdsl.ir v1\n"
          "kernel @emulated_mma_bad {\n"
          "  attrs { max_workgroup_size = i:64 }\n"
          "  params {\n"
          "  }\n"
          "  region @entry {\n"
          "    %c1 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:1.0 }\n"
          "    %splat2 : vec<f32x8> = vector.splat ( %c1 ) { vec = i:8 }\n"
          "    %splat6 : vec<f32x4> = vector.splat ( %c1 ) { vec = i:4 }\n"
          "    %acc7 : vec<f32x4> = tile.mma ( %splat2, %splat2, %splat6 ) "
          "{ op_id = s:\"mfma_f32_16x16x32_xf32\" }\n"
          "    cf.return (  )\n"
          "  }\n"
          "}\n";
    static const char* const k_short_fragment
        = "ckdsl.ir v1\n"
          "kernel @emulated_mma_bad {\n"
          "  attrs { max_workgroup_size = i:64 }\n"
          "  params {\n"
          "  }\n"
          "  region @entry {\n"
          "    %c1 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:1.0 }\n"
          "    %splat2 : vec<f32x4> = vector.splat ( %c1 ) { vec = i:4 }\n"
          "    %tfa : vec<tf32x4> = arith.bitcast ( %splat2 ) { target = s:\"vec<tf32x4>\" }\n"
          "    %acc7 : vec<f32x4> = tile.mma ( %tfa, %tfa, %splat2 ) "
          "{ op_id = s:\"mfma_f32_16x16x32_xf32\" }\n"
          "    cf.return (  )\n"
          "  }\n"
          "}\n";
    const char* const cases[2] = {k_fp32_operands, k_short_fragment};
    for(int i = 0; i < 2; ++i)
    {
        char* err = lower_error(cases[i]);
        CHECK(err != NULL, "a non-TF32 operand must be rejected");
        if(err)
        {
            CHECK(strstr(err,
                         "mfma_f32_16x16x32_xf32 requires vec<tf32x8>, vec<tf32x8> and vec<f32x4>")
                      != NULL,
                  err);
            free(err);
        }
    }
    return failed;
}

/* The expansion exists only in the LLVM lowerer, so the HIP lowerer must reject an
 * emulated atom rather than emit it. The status is all this entry point exposes --
 * the message is raised inside and translated to a code at the boundary -- so what
 * is pinned here is the rejection itself: an emulated op_id leaking into the
 * native-TF32 path would come out as a bogus __builtin_amdgcn_<op_id> call and a
 * successful lowering. The wording is pinned by the Python lowerer's twin test. */
static int test_hip_rejects_emulated_atoms(void)
{
    int failed = 0;
    const char* const irs[2] = {k_ir_16x16x32, k_ir_32x32x16};
    for(int i = 0; i < 2; ++i)
    {
        rocke_ir_builder_t b;
        if(rocke_ir_builder_init(&b, "xf32_bf16x3_hip") != ROCKE_OK)
        {
            failed = 1;
            continue;
        }
        rocke_kernel_def_t* kernel = NULL;
        CHECK(rocke_ir_parse(irs[i], &b, &kernel) == ROCKE_OK && kernel, "IR must parse");
        if(kernel)
        {
            rocke_strbuf_t hip;
            rocke_lower_hip_opts_t opts;
            memset(&opts, 0, sizeof(opts));
            opts.arch = "gfx950";
            CHECK(rocke_strbuf_init(&hip, 0) == 0, "strbuf init");
            CHECK(rocke_lower_kernel_to_hip(&b, kernel, &opts, &hip) == ROCKE_ERR_NOTIMPL,
                  "an emulated atom must be rejected, not emitted");
            rocke_strbuf_free(&hip);
        }
        rocke_ir_builder_free(&b);
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
    failed |= test_big_is_capped_at_the_largest_finite_bf16();
    failed |= test_three_terms_in_order();
    failed |= test_no_emulated_atom_survives();
    failed |= test_operands_must_be_tf32_fragments();
    failed |= test_hip_rejects_emulated_atoms();
    failed |= test_native_atom_is_unchanged();
    if(failed)
    {
        fprintf(stderr, "test_xf32_bf16x3_lowering: FAILED\n");
        return 1;
    }
    printf("test_xf32_bf16x3_lowering: OK\n");
    return 0;
}
