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
 * message of the check it targets. The messages live in
 * library/tests/direct_wgrad_refusal_reasons.inc, which the Python test reads
 * too, so both engines are held to one text. The byte-identity gate never
 * sees a refused spec, so this is what keeps the two validators' refusals in
 * step. Keep the case tables in step.
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
#include "rocke/instance_conv_direct_wgrad_internal.hpp" /* the WMMA wgrad arch gates */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#define REASON_CAP 512

/* Layout canary for the public wgrad spec, like the MMA catalog ones in
 * platform/tests/core/test_mma_catalog_lookup.cpp: a caller allocates
 * rocke_direct_conv_wgrad_spec_t from its own header, so growing it must move
 * ROCKE_ABI_VERSION too. Editing the struct fails the build here; the fix is
 * to bump ROCKE_ABI_VERSION in platform/cpp/include/rocke/abi.h and BINARY_ABI
 * in platform/python/rocke/portable_ir/src/abi.py, then update this pin.
 * 64-bit only. */
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(rocke_direct_conv_wgrad_spec_t) == 160 && ROCKE_ABI_VERSION >= 15,
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
} wgrad_case_t;

typedef struct
{
    const char* id;
    dw_edit_t edit;
    const char* arch;
} dw_case_t;

using W = rocke_direct_conv_wgrad_spec_t;
using D = rocke_direct_depthwise_wgrad_spec_t;

static const wgrad_case_t k_wgrad_cases[] = {
    /* ---- _wgrad_image_reason */
    {"img_wave_tiles_lt1", BASE_IMG, [](W* s) { s->wave_tiles_k = 0; }, "gfx1250"},
    {"img_small_c_knob", BASE_W32, [](W* s) { s->sub_steps = 2; }, "gfx1250"},
    {"img_tiles_need_img", BASE_W32, [](W* s) { s->wave_tiles_c = 2; }, "gfx1250"},
    {"img_rps_need_img", BASE_W32, [](W* s) { s->rows_per_sync = 3; }, "gfx1250"},
    {"img_rb_need_img", BASE_W32, [](W* s) { s->row_blocks = 2; }, "gfx1250"},
    {"img_tdm_need_img", BASE_W32, [](W* s) { s->tdm_stages = 2; }, "gfx1250"},
    {"img_wsr_need_img", BASE_W32, [](W* s) { s->ws_replicas = 2; }, "gfx1250"},
    {"img_ldsepi_need_img", BASE_W32, [](W* s) { s->lds_epilogue = 1; }, "gfx1250"},
    {"img_wsr_lt1", BASE_IMG, [](W* s) { s->ws_replicas = 0; }, "gfx1250"},
    {"img_ldsepi_bytes",
     BASE_IMG,
     [](W* s) {
         s->lds_epilogue = 1;
         s->waves_k = 8;
         s->problem.KH = 7;
         s->problem.KW = 7;
     },
     "gfx1250"},
    {"img_rb_lt1", BASE_IMG, [](W* s) { s->row_blocks = 0; }, "gfx1250"},
    {"img_tdm_range", BASE_IMG, [](W* s) { s->tdm_stages = 5; }, "gfx1250"},
    {"img_tdm_waves",
     BASE_IMG,
     [](W* s) {
         s->tdm_stages = 3;
         s->rows_per_sync = 3;
     },
     "gfx1250"},
    {"img_rps_value", BASE_IMG, [](W* s) { s->rows_per_sync = 2; }, "gfx1250"},
    {"img_wave64", BASE_IMG, [](W* s) { s->wave_size = 64; }, "gfx950"},
    {"img_waves_q", BASE_IMG, [](W* s) { s->waves_q = 2; }, "gfx1250"},
    {"img_vgpr_budget",
     BASE_IMG,
     [](W* s) {
         s->wave_tiles_k = 4;
         s->wave_tiles_c = 4;
     },
     "gfx1250"},
    /* ---- is_valid_wgrad_spec: the wave32 arch gate */
    {"w32_arch", BASE_W32, [](W*) {}, "gfx1151"},
    /* ---- is_valid_wgrad_spec: fp32 storage is wave64 only */
    {"w32_f32", BASE_W32, [](W* s) { s->problem.dtype = "tf32"; }, "gfx1250"},
    {"sc_f32", BASE_SC, [](W* s) { s->problem.dtype = "fp32"; }, "gfx1250"},
    /* ---- _wgrad_small_c_reason */
    {"sc_and_img", BASE_SC, [](W* s) { s->image_per_block = 1; }, "gfx1250"},
    {"sc_wave64", BASE_SC, [](W* s) { s->wave_size = 64; }, "gfx950"},
    {"sc_mfma_k", BASE_SC, [](W* s) { s->mfma_k = 16; }, "gfx1250"},
    {"sc_groups", BASE_SC, [](W* s) { s->problem.groups = 2; }, "gfx1250"},
    {"sc_kpg", BASE_SC, [](W* s) { s->problem.kpg = 12; }, "gfx1250"},
    {"sc_filter", BASE_SC, [](W* s) { s->problem.KW = 0; }, "gfx1250"},
    {"sc_wave_tile", BASE_SC, [](W* s) { s->wave_tile_c = 32; }, "gfx1250"},
    {"sc_waves_lt1", BASE_SC, [](W* s) { s->waves_c = 0; }, "gfx1250"},
    {"sc_waves_gt16",
     BASE_SC,
     [](W* s) {
         s->waves_k = 8;
         s->waves_c = 4;
     },
     "gfx1250"},
    {"sc_waves_q", BASE_SC, [](W* s) { s->waves_q = 2; }, "gfx1250"},
    {"sc_wave_tiles_lt1", BASE_SC, [](W* s) { s->wave_tiles_k = 0; }, "gfx1250"},
    {"sc_image_knob", BASE_SC, [](W* s) { s->tdm_stages = 2; }, "gfx1250"},
    {"sc_runtime_channels", BASE_SC, [](W* s) { s->runtime_channels = 1; }, "gfx1250"},
    {"sc_rb_lt1", BASE_SC, [](W* s) { s->row_blocks = 0; }, "gfx1250"},
    {"sc_wsr_lt1", BASE_SC, [](W* s) { s->ws_replicas = 0; }, "gfx1250"},
    {"sc_sub_steps", BASE_SC, [](W* s) { s->sub_steps = 9; }, "gfx1250"},
    {"sc_vec_x", BASE_SC, [](W* s) { s->vec_x = 2; }, "gfx1250"},
    {"sc_acc_budget",
     BASE_SC,
     [](W* s) {
         s->wave_tiles_k = 4;
         s->wave_tiles_c = 5;
     },
     "gfx1250"},
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
     "gfx1250"},
    /* ---- is_valid_wgrad_spec: the small_c arch gate */
    {"sc_arch", BASE_SC, [](W*) {}, "gfx1151"},
};

static const dw_case_t k_dw_cases[] = {
    /* ---- _depthwise_wgrad_reason */
    {"dw_dtype", [](D* s) { s->problem.dtype = "fp32"; }, "gfx1250"},
    {"dw_cpg",
     [](D* s) {
         s->problem.cpg = 2;
         s->problem.kpg = 2;
     },
     "gfx1250"},
    {"dw_filter", [](D* s) { s->problem.KH = 0; }, "gfx1250"},
    {"dw_wave_size", [](D* s) { s->wave_size = 16; }, "gfx1250"},
    {"dw_cpl", [](D* s) { s->chan_per_lane = 3; }, "gfx1250"},
    {"dw_channels",
     [](D* s) {
         s->chan_per_lane = 4;
         s->problem.groups = 66;
     },
     "gfx1250"},
    {"dw_waves_lt1", [](D* s) { s->wo_chunk = 0; }, "gfx1250"},
    {"dw_r_split", [](D* s) { s->r_split = 3; }, "gfx1250"},
    {"dw_wsr_lt1", [](D* s) { s->ws_replicas = 0; }, "gfx1250"},
    {"dw_acc_budget",
     [](D* s) {
         s->chan_per_lane = 8;
         s->problem.KH = 7;
         s->problem.KW = 7;
         s->problem.PAD = 3;
     },
     "gfx1250"},
    /* ---- is_valid_depthwise_wgrad_spec */
    {"dw_wave_arch", [](D*) {}, "gfx950"},
    {"dw_threads", [](D* s) { s->waves = 64; }, "gfx1250"},
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
     "gfx1250"},
};

/* The refusal text of every case, shared with the Python test: both engines
 * must produce the same sentence, so a reword in one engine fails here or
 * there. */
static const struct
{
    const char* id;
    const char* reason;
} k_reasons[] = {
#define ROCKE_WGRAD_REFUSAL(id, reason) {id, reason},
#include "direct_wgrad_refusal_reasons.inc"
#undef ROCKE_WGRAD_REFUSAL
};

/* The shared text of case `id`; NULL (and a failure) when it has none. */
static const char* reason_of(const char* id)
{
    size_t i;
    for(i = 0; i < sizeof k_reasons / sizeof k_reasons[0]; ++i)
        if(strcmp(k_reasons[i].id, id) == 0)
            return k_reasons[i].reason;
    fprintf(stderr, "FAIL %s: no entry in direct_wgrad_refusal_reasons.inc\n", id);
    ++g_failures;
    return NULL;
}

static void expect_refused(const char* id, bool ok, const char* why, const char* reason)
{
    if(reason == NULL)
        return;
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

/* The dense and grouped WMMA wgrad arch gates on a gfx1250 copy without TDM
 * and with a 1 KiB LDS: no shipped arch reaches either refusal. */
static void check_wmma_arch_gates(void)
{
    char why[256];
    rocke_archtarget_t t = *rocke_archtarget_from_gfx("gfx1250");
    why[0] = '\0';
    if(!rocke_dwg::wgrad_wmma_target_ok(&t, "gfx1250", true, why, sizeof why))
    {
        fprintf(stderr, "FAIL wmma_gate_gfx1250: refused (%s)\n", why);
        ++g_failures;
    }
    t.memory.has_tdm = false;
    expect_refused("wmma_gate_no_tdm",
                   rocke_dwg::wgrad_wmma_target_ok(&t, "gfx1250", true, why, sizeof why),
                   why,
                   "no TDM (nor ds_load_tr16_b128) on gfx1250");
    t = *rocke_archtarget_from_gfx("gfx1250");
    t.lds_capacity_bytes = 1024;
    why[0] = '\0';
    expect_refused("wmma_gate_lds_capacity",
                   !rocke_dwg::wgrad_wmma_lds_over(2048, 4096, &t, "gfxX", why, sizeof why),
                   why,
                   "LDS 2048 B > 1024 on gfxX");
    why[0] = '\0';
    expect_refused("wmma_gate_lds_budget",
                   !rocke_dwg::wgrad_wmma_lds_over(8192, 4096, &t, "gfxX", why, sizeof why),
                   why,
                   "LDS 8192 B > 4 KiB");
    if(rocke_dwg::wgrad_wmma_lds_over(1024, 4096, &t, "gfxX", why, sizeof why))
    {
        fprintf(stderr, "FAIL wmma_gate_lds_fits: refused (%s)\n", why);
        ++g_failures;
    }
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
                       reason_of(c->id));
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
            expect_refused(c->id, st == ROCKE_OK, err, reason_of(c->id));
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
                       reason_of(c->id));
        expect_names_value(c->id, why);
    }

    /* Every shared text belongs to a case of one of the two tables. */
    for(i = 0; i < sizeof k_reasons / sizeof k_reasons[0]; ++i)
    {
        size_t j;
        bool found = false;
        for(j = 0; j < sizeof k_wgrad_cases / sizeof k_wgrad_cases[0]; ++j)
            found = found || strcmp(k_wgrad_cases[j].id, k_reasons[i].id) == 0;
        for(j = 0; j < sizeof k_dw_cases / sizeof k_dw_cases[0]; ++j)
            found = found || strcmp(k_dw_cases[j].id, k_reasons[i].id) == 0;
        if(!found)
        {
            fprintf(stderr, "FAIL %s: shared text has no case\n", k_reasons[i].id);
            ++g_failures;
        }
    }

    check_dw_cachepolicy("gfx1250", ROCKE_CPOL_ATOMIC_CASCADE_DEV);
    check_dw_cachepolicy("gfx1201", ROCKE_CPOL_ATOMIC_CASCADE_DEV);
    check_dw_cachepolicy("gfx1151", 0);
    check_dw_cachepolicy("gfx1100", 0);
    check_wmma_arch_gates();

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
