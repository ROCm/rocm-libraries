// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_wgrad_refusals_host -- every refusal of the gfx1250 direct
 * wgrad specs in the C++ validators, by reason.
 *
 * The C++ mirror of library/tests/test_direct_wgrad_refusals.py: the same
 * case ids, base specs and one-knob changes, checked through
 * rocke_direct_conv_wgrad_is_valid_spec / rocke_direct_depthwise_wgrad_is_valid_spec.
 * Each case asserts the spec is refused and that the reason contains the
 * message of the check it targets. The byte-identity gate never sees a
 * refused spec, so this is what keeps the two validators' refusals in step.
 * Keep the two tables in step.
 *
 * It also checks the cache policy of the depthwise wgrad atomics per arch,
 * like library/tests/test_direct_depthwise_wgrad_cachepolicy.py: the gfx12
 * th/scope cascade value on gfx12 targets only, 0 on gfx11.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/abi.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#define REASON_CAP 512

/* Layout canary for the public wgrad spec, like the MMA catalog ones in
 * tests/core/test_mma_catalog_lookup.cpp: a caller allocates
 * rocke_direct_conv_wgrad_spec_t from its own header, so growing it must move
 * ROCKE_ABI_VERSION too. Editing the struct fails the build here; the fix is
 * to bump ROCKE_ABI_VERSION in rocke/abi.h and BINARY_ABI in
 * python/rocke/portable_ir/src/abi.py, then update this pin. 64-bit only. */
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(rocke_direct_conv_wgrad_spec_t) == 160 && ROCKE_ABI_VERSION >= 9,
              "rocke_direct_conv_wgrad_spec_t changed size: bump ROCKE_ABI_VERSION in "
              "rocke/abi.h, then update this pin");
#endif

static int g_failures = 0;

typedef enum
{
    BASE_W32,
    BASE_IMG,
    BASE_SC,
} wgrad_base_t;

/* A spec both validators accept on gfx1250, to change one knob of. */
static rocke_direct_conv_wgrad_spec_t wgrad_base(wgrad_base_t base)
{
    rocke_direct_conv_wgrad_spec_t s = rocke_direct_conv_wgrad_spec_default();
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.W = 37;
    p.groups = 1;
    if(base == BASE_SC)
    {
        p.H = 13;
        p.cpg = 3;
        p.kpg = 32;
    }
    else
    {
        p.H = 8;
        p.cpg = 64;
        p.kpg = 64;
    }
    s.problem = p;
    s.wave_size = 32;
    if(base == BASE_IMG)
    {
        s.image_per_block = 1;
        s.waves_k = 2;
        s.waves_c = 2;
    }
    if(base == BASE_SC)
        s.small_c = 1;
    return s;
}

static rocke_direct_depthwise_wgrad_spec_t dw_base(void)
{
    rocke_direct_depthwise_wgrad_spec_t s = rocke_direct_depthwise_wgrad_spec_default();
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = 9;
    p.W = 13;
    p.groups = 64;
    p.cpg = 1;
    p.kpg = 1;
    s.problem = p;
    s.wave_size = 32;
    s.waves = 4;
    s.rows_per_wave = 2;
    s.wo_chunk = 4;
    s.ws_replicas = 4;
    return s;
}

typedef void (*wgrad_edit_t)(rocke_direct_conv_wgrad_spec_t*);
typedef void (*dw_edit_t)(rocke_direct_depthwise_wgrad_spec_t*);

typedef struct
{
    const char* id;
    wgrad_base_t base;
    wgrad_edit_t edit;
    const char* arch;
    const char* reason;
} wgrad_case_t;

typedef struct
{
    const char* id;
    dw_edit_t edit;
    const char* arch;
    const char* reason;
} dw_case_t;

using W = rocke_direct_conv_wgrad_spec_t;
using D = rocke_direct_depthwise_wgrad_spec_t;

static const wgrad_case_t k_wgrad_cases[] = {
    /* ---- _wgrad_image_reason */
    {"img_wave_tiles_lt1",
     BASE_IMG,
     [](W* s) { s->wave_tiles_k = 0; },
     "gfx1250",
     "wave_tiles_k and wave_tiles_c must be >= 1 (got 0, 1)"},
    {"img_small_c_knob",
     BASE_W32,
     [](W* s) { s->sub_steps = 2; },
     "gfx1250",
     "sub_steps / vec_x / uniform_wave need small_c=True (got sub_steps=2, vec_x=0, "
     "uniform_wave=False)"},
    {"img_tiles_need_img",
     BASE_W32,
     [](W* s) { s->wave_tiles_c = 2; },
     "gfx1250",
     "wave_tiles_k / wave_tiles_c > 1 need image_per_block=True (got 1, 2)"},
    {"img_rps_need_img",
     BASE_W32,
     [](W* s) { s->rows_per_sync = 3; },
     "gfx1250",
     "rows_per_sync > 1 needs image_per_block=True (got 3)"},
    {"img_rb_need_img",
     BASE_W32,
     [](W* s) { s->row_blocks = 2; },
     "gfx1250",
     "row_blocks > 1 needs image_per_block=True (got 2)"},
    {"img_tdm_need_img",
     BASE_W32,
     [](W* s) { s->tdm_stages = 2; },
     "gfx1250",
     "tdm_stages needs image_per_block=True (got 2)"},
    {"img_wsr_need_img",
     BASE_W32,
     [](W* s) { s->ws_replicas = 2; },
     "gfx1250",
     "ws_replicas > 1 needs image_per_block=True (got 2)"},
    {"img_ldsepi_need_img",
     BASE_W32,
     [](W* s) { s->lds_epilogue = 1; },
     "gfx1250",
     "lds_epilogue needs image_per_block=True"},
    {"img_wsr_lt1",
     BASE_IMG,
     [](W* s) { s->ws_replicas = 0; },
     "gfx1250",
     "ws_replicas must be >= 1 (got 0)"},
    {"img_ldsepi_bytes",
     BASE_IMG,
     [](W* s) {
         s->lds_epilogue = 1;
         s->waves_k = 8;
         s->problem.KH = 7;
         s->problem.KW = 7;
     },
     "gfx1250",
     "lds_epilogue stages 903168 B of dW in LDS (> 196608)"},
    {"img_rb_lt1",
     BASE_IMG,
     [](W* s) { s->row_blocks = 0; },
     "gfx1250",
     "row_blocks must be >= 1 (got 0)"},
    {"img_tdm_range",
     BASE_IMG,
     [](W* s) { s->tdm_stages = 5; },
     "gfx1250",
     "tdm_stages must be 0 or 2..4 (got 5)"},
    {"img_tdm_waves",
     BASE_IMG,
     [](W* s) {
         s->tdm_stages = 3;
         s->rows_per_sync = 3;
     },
     "gfx1250",
     "tdm_stages > 2 needs a wave per TDM transfer of a step (6), got 4 waves"},
    {"img_rps_value",
     BASE_IMG,
     [](W* s) { s->rows_per_sync = 2; },
     "gfx1250",
     "rows_per_sync must be 1 or KH=3 (got 2)"},
    {"img_wave64",
     BASE_IMG,
     [](W* s) { s->wave_size = 64; },
     "gfx950",
     "image_per_block is implemented for wave32 (gfx1250) only (got wave_size=64)"},
    {"img_waves_q",
     BASE_IMG,
     [](W* s) { s->waves_q = 2; },
     "gfx1250",
     "image_per_block walks every wo tile itself; waves_q must be 1 (got 2)"},
    {"img_vgpr_budget",
     BASE_IMG,
     [](W* s) {
         s->wave_tiles_k = 4;
         s->wave_tiles_c = 4;
     },
     "gfx1250",
     "image_per_block body needs ~1344 VGPRs for accumulators and fragments (> 224)"},
    /* ---- is_valid_wgrad_spec: the wave32 arch gate */
    {"w32_arch",
     BASE_W32,
     [](W*) {},
     "gfx1151",
     "wave32 direct wgrad is implemented for ['gfx1250'] (got gfx1151)"},
    /* ---- _wgrad_small_c_reason */
    {"sc_and_img",
     BASE_SC,
     [](W* s) { s->image_per_block = 1; },
     "gfx1250",
     "small_c and image_per_block are separate variants"},
    {"sc_wave64",
     BASE_SC,
     [](W* s) { s->wave_size = 64; },
     "gfx950",
     "small_c is implemented for wave32 (gfx1250) only (got wave_size=64)"},
    {"sc_mfma_k",
     BASE_SC,
     [](W* s) { s->mfma_k = 16; },
     "gfx1250",
     "direct wgrad on wave32 needs mfma_k=32 (WMMA 16x16x32); got mfma_k=16"},
    {"sc_groups",
     BASE_SC,
     [](W* s) { s->problem.groups = 2; },
     "gfx1250",
     "small_c needs groups=1 (got 2)"},
    {"sc_kpg",
     BASE_SC,
     [](W* s) { s->problem.kpg = 12; },
     "gfx1250",
     "small_c loads dY as 8-channel vectors; kpg must be a multiple of 8 (got 12)"},
    {"sc_filter",
     BASE_SC,
     [](W* s) { s->problem.KW = 0; },
     "gfx1250",
     "KH and KW must be >= 1 (got 3, 0)"},
    {"sc_wave_tile",
     BASE_SC,
     [](W* s) { s->wave_tile_c = 32; },
     "gfx1250",
     "wave_tile_k and wave_tile_c must be 16 (got 16, 32)"},
    {"sc_waves_lt1",
     BASE_SC,
     [](W* s) { s->waves_c = 0; },
     "gfx1250",
     "waves_k and waves_c must be >= 1 (got 1, 0)"},
    {"sc_waves_gt16",
     BASE_SC,
     [](W* s) {
         s->waves_k = 8;
         s->waves_c = 4;
     },
     "gfx1250",
     "waves_k * waves_c must be <= 16 (got 8 * 4 = 32)"},
    {"sc_waves_q",
     BASE_SC,
     [](W* s) { s->waves_q = 2; },
     "gfx1250",
     "small_c walks its pixel range itself; waves_q must be 1 (got 2)"},
    {"sc_wave_tiles_lt1",
     BASE_SC,
     [](W* s) { s->wave_tiles_k = 0; },
     "gfx1250",
     "wave_tiles_k and wave_tiles_c must be >= 1 (got 0, 1)"},
    {"sc_image_knob",
     BASE_SC,
     [](W* s) { s->tdm_stages = 2; },
     "gfx1250",
     "rows_per_sync / tdm_stages / lds_epilogue are image_per_block knobs (got rows_per_sync=1, "
     "tdm_stages=2, lds_epilogue=False)"},
    {"sc_runtime_channels",
     BASE_SC,
     [](W* s) { s->runtime_channels = 1; },
     "gfx1250",
     "small_c bakes its channel counts; runtime_channels is not supported"},
    {"sc_rb_lt1",
     BASE_SC,
     [](W* s) { s->row_blocks = 0; },
     "gfx1250",
     "row_blocks must be >= 1 (got 0)"},
    {"sc_wsr_lt1",
     BASE_SC,
     [](W* s) { s->ws_replicas = 0; },
     "gfx1250",
     "ws_replicas must be >= 1 (got 0)"},
    {"sc_sub_steps",
     BASE_SC,
     [](W* s) { s->sub_steps = 9; },
     "gfx1250",
     "sub_steps must be in 1..8 (got 9)"},
    {"sc_vec_x", BASE_SC, [](W* s) { s->vec_x = 2; }, "gfx1250", "vec_x must be 0, 4 or 8 (got 2)"},
    {"sc_acc_budget",
     BASE_SC,
     [](W* s) {
         s->wave_tiles_k = 4;
         s->wave_tiles_c = 5;
     },
     "gfx1250",
     "small_c accumulators need 160 VGPRs (> 128)"},
    {"sc_uniform_wave_spill",
     BASE_SC,
     [](W* s) {
         s->waves_k = 4;
         s->waves_c = 4;
         s->wave_tiles_k = 2;
         s->wave_tiles_c = 2;
         s->vec_x = 4;
         s->sub_steps = 4;
         s->uniform_wave = 1;
     },
     "gfx1250",
     "uniform_wave with more than 8 waves, vec_x=4, sub_steps >= 4 and >= 4 wave tiles spills "
     "SGPRs into VGPR lanes and returns an all-zero dW (got 16 waves, sub_steps=4, 4 wave tiles)"},
    /* ---- is_valid_wgrad_spec: the small_c arch gate */
    {"sc_arch",
     BASE_SC,
     [](W*) {},
     "gfx1151",
     "small_c direct wgrad is implemented for wave32 gfx1250 (got gfx1151)"},
};

static const dw_case_t k_dw_cases[] = {
    /* ---- _depthwise_wgrad_reason */
    {"dw_dtype",
     [](D* s) { s->problem.dtype = "fp32"; },
     "gfx1250",
     "unsupported dtype 'fp32'; expected fp16 or bf16"},
    {"dw_cpg",
     [](D* s) {
         s->problem.cpg = 2;
         s->problem.kpg = 2;
     },
     "gfx1250",
     "requires cpg=kpg=1 (got 2, 2)"},
    {"dw_filter", [](D* s) { s->problem.KH = 0; }, "gfx1250", "KH and KW must be >= 1 (got 0, 3)"},
    {"dw_wave_size",
     [](D* s) { s->wave_size = 16; },
     "gfx1250",
     "wave_size must be 32 or 64 (got 16)"},
    {"dw_cpl",
     [](D* s) { s->chan_per_lane = 3; },
     "gfx1250",
     "chan_per_lane must be 2, 4 or 8 (got 3)"},
    {"dw_channels",
     [](D* s) {
         s->chan_per_lane = 4;
         s->problem.groups = 66;
     },
     "gfx1250",
     "channels 66 must be a multiple of chan_per_lane 4 (one vector load per lane)"},
    {"dw_waves_lt1",
     [](D* s) { s->wo_chunk = 0; },
     "gfx1250",
     "waves, rows_per_wave and wo_chunk must be >= 1 (got 4, 2, 0)"},
    {"dw_r_split",
     [](D* s) { s->r_split = 3; },
     "gfx1250",
     "r_split must be in 1..KH=3 and divide waves=4 (got 3)"},
    {"dw_wsr_lt1", [](D* s) { s->ws_replicas = 0; }, "gfx1250", "ws_replicas must be >= 1 (got 0)"},
    {"dw_acc_budget",
     [](D* s) {
         s->chan_per_lane = 8;
         s->problem.KH = 7;
         s->problem.KW = 7;
         s->problem.PAD = 3;
     },
     "gfx1250",
     "accumulators need 392 VGPRs (> 128)"},
    /* ---- is_valid_depthwise_wgrad_spec */
    {"dw_wave_arch", [](D*) {}, "gfx950", "wave_size 32 does not match the gfx950 wave size 64"},
    {"dw_threads",
     [](D* s) { s->waves = 64; },
     "gfx1250",
     "threads_per_block 2048 > 1024 (hardware cap) on gfx1250"},
    {"dw_lds",
     [](D* s) {
         s->chan_per_lane = 8;
         s->waves = 14;
         s->r_split = 7;
         s->problem.groups = 256;
         s->problem.KH = 7;
         s->problem.KW = 7;
         s->problem.PAD = 3;
     },
     "gfx1250",
     "epilogue LDS 702464 B > 327680 on gfx1250"},
};

static void expect_refused(const char* id, bool ok, const char* why, const char* reason)
{
    if(ok)
    {
        fprintf(stderr, "FAIL %s: accepted, expected \"%s\"\n", id, reason);
        ++g_failures;
    }
    else if(strstr(why, reason) == NULL)
    {
        fprintf(stderr, "FAIL %s: reason \"%s\", expected \"%s\"\n", id, why, reason);
        ++g_failures;
    }
}

/* The refusals of a boolean flag: the message itself names the value. */
static const char* const k_flag_cases[]
    = {"img_ldsepi_need_img", "sc_and_img", "sc_runtime_channels"};

/* The refusals that state the offending value inside the sentence, mostly a
 * measured size against its limit, rather than in a "(got ...)" tail: the
 * value text each one must carry. Every other refusal ends with "(got ...)"
 * (test_wgrad_refusal_names_the_offending_value). */
static const struct
{
    const char* id;
    const char* value;
} k_inline_value_cases[] = {
    {"img_ldsepi_bytes", "903168 B"},
    {"img_tdm_waves", "got 4 waves"},
    {"img_vgpr_budget", "~1344 VGPRs"},
    {"sc_mfma_k", "got mfma_k=16"},
    {"sc_acc_budget", "160 VGPRs"},
    {"dw_dtype", "'fp32'"},
    {"dw_channels", "channels 66"},
    {"dw_acc_budget", "392 VGPRs"},
    {"dw_wave_arch", "wave_size 32"},
    {"dw_threads", "threads_per_block 2048"},
    {"dw_lds", "LDS 702464 B"},
};

/* True when why ends with " (got ...)" and the tail holds no parenthesis. */
static bool ends_with_got(const char* why)
{
    const char* tail = NULL;
    const char* p;
    for(p = strstr(why, " (got "); p != NULL; p = strstr(p + 1, " (got "))
        tail = p;
    if(tail == NULL)
        return false;
    p = tail + strlen(" (got ");
    if(*p == ')' || strlen(p) < 2)
        return false;
    for(; p[1] != '\0'; ++p)
        if(*p == '(' || *p == ')')
            return false;
    return *p == ')';
}

static void expect_names_value(const char* id, const char* why)
{
    size_t i;
    for(i = 0; i < sizeof k_flag_cases / sizeof k_flag_cases[0]; ++i)
        if(strcmp(id, k_flag_cases[i]) == 0)
            return;
    for(i = 0; i < sizeof k_inline_value_cases / sizeof k_inline_value_cases[0]; ++i)
    {
        if(strcmp(id, k_inline_value_cases[i].id) != 0)
            continue;
        if(strstr(why, k_inline_value_cases[i].value) == NULL)
        {
            fprintf(stderr,
                    "FAIL %s: reason \"%s\" does not state \"%s\"\n",
                    id,
                    why,
                    k_inline_value_cases[i].value);
            ++g_failures;
        }
        return;
    }
    if(!ends_with_got(why))
    {
        fprintf(stderr, "FAIL %s: reason \"%s\" has no (got ...) tail\n", id, why);
        ++g_failures;
    }
}

/* Every cachepolicy attr of the tile.buffer_atomic_fadd_f32 ops under
 * `region` must be `want`; *count gets the number of atomics seen. */
static bool dw_atomics_have_policy(const rocke_region_t* region, int64_t want, int* count)
{
    int i, r;
    for(i = 0; i < region->num_ops; ++i)
    {
        const rocke_op_t* op = region->ops[i];
        if(strcmp(op->name, "tile.buffer_atomic_fadd_f32") == 0)
        {
            int64_t got = -1;
            ++*count;
            if(!rocke_attr_get_int(&op->attrs, "cachepolicy", &got) || got != want)
                return false;
        }
        for(r = 0; r < op->num_regions; ++r)
        {
            if(!dw_atomics_have_policy(op->regions[r], want, count))
                return false;
        }
    }
    return true;
}

static void check_dw_cachepolicy(const char* arch, int64_t want)
{
    rocke_direct_depthwise_wgrad_spec_t s = dw_base();
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_direct_depthwise_wgrad_new(&b, &s, arch);
    int count = 0;
    if(k == NULL)
    {
        fprintf(
            stderr, "FAIL dw_cachepolicy %s: build failed: %s\n", arch, rocke_ir_builder_error(&b));
        ++g_failures;
    }
    else if(!dw_atomics_have_policy(k->body, want, &count) || count == 0)
    {
        fprintf(stderr,
                "FAIL dw_cachepolicy %s: atomics do not all carry cachepolicy %lld (%d seen)\n",
                arch,
                (long long)want,
                count);
        ++g_failures;
    }
    rocke_ir_builder_free(&b);
}

int main(void)
{
    char why[REASON_CAP];
    const wgrad_base_t bases[] = {BASE_W32, BASE_IMG, BASE_SC};
    size_t i;

    /* Each case changes one knob of a base spec, so the bases must pass. */
    for(i = 0; i < sizeof bases / sizeof bases[0]; ++i)
    {
        rocke_direct_conv_wgrad_spec_t s = wgrad_base(bases[i]);
        why[0] = '\0';
        if(!rocke_direct_conv_wgrad_is_valid_spec(&s, "gfx1250", why, sizeof why))
        {
            fprintf(stderr, "FAIL wgrad base %zu refused: %s\n", i, why);
            ++g_failures;
        }
    }
    {
        rocke_direct_depthwise_wgrad_spec_t s = dw_base();
        why[0] = '\0';
        if(!rocke_direct_depthwise_wgrad_is_valid_spec(&s, "gfx1250", why, sizeof why))
        {
            fprintf(stderr, "FAIL depthwise wgrad base refused: %s\n", why);
            ++g_failures;
        }
    }

    for(i = 0; i < sizeof k_wgrad_cases / sizeof k_wgrad_cases[0]; ++i)
    {
        const wgrad_case_t* c = &k_wgrad_cases[i];
        rocke_direct_conv_wgrad_spec_t s = wgrad_base(c->base);
        c->edit(&s);
        why[0] = '\0';
        expect_refused(c->id,
                       rocke_direct_conv_wgrad_is_valid_spec(&s, c->arch, why, sizeof why),
                       why,
                       c->reason);
        expect_names_value(c->id, why);
        /* The image-per-block and small_c builders hand the refusal on to
         * the lowering entry's error, not just a failed status. */
        if(s.image_per_block || s.small_c)
        {
            char* ll = NULL;
            char err[REASON_CAP];
            rocke_status_t st;
            err[0] = '\0';
            st = rocke_direct_conv_wgrad_lower_to_llvm(
                &s, c->arch, ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
            expect_refused(c->id, st == ROCKE_OK, err, c->reason);
            free(ll);
        }
    }
    for(i = 0; i < sizeof k_dw_cases / sizeof k_dw_cases[0]; ++i)
    {
        const dw_case_t* c = &k_dw_cases[i];
        rocke_direct_depthwise_wgrad_spec_t s = dw_base();
        c->edit(&s);
        why[0] = '\0';
        expect_refused(c->id,
                       rocke_direct_depthwise_wgrad_is_valid_spec(&s, c->arch, why, sizeof why),
                       why,
                       c->reason);
        expect_names_value(c->id, why);
    }

    /* rocke.core.ir.CPOL_ATOMIC_CASCADE_DEV = 4 | (2 << 3). */
    check_dw_cachepolicy("gfx1250", 4 | (2 << 3));
    check_dw_cachepolicy("gfx1201", 4 | (2 << 3));
    check_dw_cachepolicy("gfx1151", 0);
    check_dw_cachepolicy("gfx1100", 0);

    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("conv_direct_wgrad_refusals_host: OK (%zu + %zu cases)\n",
           sizeof k_wgrad_cases / sizeof k_wgrad_cases[0],
           sizeof k_dw_cases / sizeof k_dw_cases[0]);
    return 0;
}
