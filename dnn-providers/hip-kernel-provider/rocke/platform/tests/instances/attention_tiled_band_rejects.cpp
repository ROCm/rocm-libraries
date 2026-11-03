// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only test: the gfx950 C++ tiled attention builders refuse
 * causal_top_left.
 *
 * gfx950 implements top-left causal in the Python engine only. The C spec
 * structs carry causal_top_left because the gfx942 C++ bodies honor it, so a
 * gfx950 builder that ignored the field would emit the bottom-right causal
 * kernel without saying so. Each gfx950 entry point -- tiled 2D, the fastKV
 * register-P 2D experiment and the tiled 3D segment kernel -- must instead fail
 * with its documented message, and still build and lower the default spec.
 *
 * This calls the C entry points directly, so it runs without the rocke_engine
 * Python extension. platform/tests/instances/test_attention_tiled_binding_band_rejects.py
 * covers the same refusals through the binding, plus the binding-only
 * right_bound rejection (the C structs have no right_bound field).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx950_attention_tiled_2d.h"
#include "rocke/instance_gfx950_attention_tiled_2d_fastkv_regp.h"
#include "rocke/instance_gfx950_attention_tiled_3d.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#define ERR_CAP 1024

static const char ARCH[] = "gfx950";

static const char NO_TOP_LEFT[] = "causal_top_left is not implemented by the gfx950 C++ twin "
                                  "(gfx950 implements it in the Python engine only)";
static const char FASTKV_NO_TOP_LEFT[]
    = "fastKV register-P experiment does not support causal_top_left";

typedef enum
{
    ENTRY_2D,
    ENTRY_2D_FASTKV,
    ENTRY_3D
} entry_t;

static const char* entry_name(entry_t entry)
{
    switch(entry)
    {
    case ENTRY_2D:
        return "gfx950_attention_tiled_2d";
    case ENTRY_2D_FASTKV:
        return "gfx950_attention_tiled_2d_fastkv_regp";
    case ENTRY_3D:
        return "gfx950_attention_tiled_3d";
    }
    return "?";
}

/* The spec the binding test builds: head_size 128, block_size 32, 8 query and
 * 8 KV heads, bf16, every other field at its C default. */
static rocke_attention_tiled_2d_spec_t tiled_2d_spec(bool top_left)
{
    rocke_attention_tiled_2d_spec_t s = rocke_attention_tiled_2d_spec_default();
    s.head_size = 128;
    s.block_size = 32;
    s.num_query_heads = 8;
    s.num_kv_heads = 8;
    s.dtype = "bf16";
    s.causal_top_left = top_left;
    return s;
}

/* The same spec as the binding's fastKV reader fills it: the standalone
 * emitter's base flags on top of the shared fields. */
static rocke_attention_tiled_2d_spec_t fastkv_spec(bool top_left)
{
    rocke_attention_tiled_2d_spec_t s = tiled_2d_spec(top_left);
    s.num_warps = 4;
    s.has_waves_per_eu = true;
    s.waves_per_eu = 2;
    s.has_tile_size = true;
    s.tile_size = 64;
    s.block_m_per_warp = 32;
    s.use_mfma_32x32 = true;
    s.use_transposed_qk_32x32 = true;
    s.use_transposed_scalar_state = true;
    s.use_transposed_mask_once = true;
    s.use_fast_paged_kv_desc = true;
    return s;
}

static rocke_unified_attention_3d_tiled_spec_t tiled_3d_spec(bool top_left)
{
    rocke_unified_attention_3d_tiled_spec_t s = rocke_unified_attention_3d_tiled_spec_default();
    s.head_size = 128;
    s.block_size = 32;
    s.num_query_heads = 8;
    s.num_kv_heads = 8;
    s.dtype = "bf16";
    s.causal_top_left = top_left ? 1 : 0;
    return s;
}

/* Build ``entry`` into ``b``; the caller frees ``b``. */
static rocke_kernel_def_t* build(entry_t entry, bool top_left, rocke_ir_builder_t* b)
{
    switch(entry)
    {
    case ENTRY_2D:
    {
        rocke_attention_tiled_2d_spec_t s = tiled_2d_spec(top_left);
        return rocke_gfx950_build_unified_attention_2d_tiled_new(b, &s, ARCH);
    }
    case ENTRY_2D_FASTKV:
    {
        rocke_attention_tiled_2d_spec_t s = fastkv_spec(top_left);
        rocke_ir_builder_init(b, "attention_tiled_2d_fastkv_regp");
        return rocke_build_unified_attention_2d_fastkv_register_p(b, &s, ARCH);
    }
    case ENTRY_3D:
    {
        rocke_unified_attention_3d_tiled_spec_t s = tiled_3d_spec(top_left);
        rocke_ir_builder_init(b, "attention_tiled_3d_segment");
        return rocke_build_unified_attention_3d_tiled_gfx950(b, &s, ARCH);
    }
    }
    return NULL;
}

/* causal_top_left must fail the build with exactly ``want``. */
static int check_rejected(entry_t entry, const char* want)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k;
    const char* got;
    int failed = 0;

    memset(&b, 0, sizeof b);
    k = build(entry, true, &b);
    got = rocke_ir_builder_error(&b);
    if(k != NULL && rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "FAIL %s: causal_top_left built a kernel\n", entry_name(entry));
        failed = 1;
    }
    else if(got == NULL || strcmp(got, want) != 0)
    {
        fprintf(stderr,
                "FAIL %s: causal_top_left error\n  got:  %s\n  want: %s\n",
                entry_name(entry),
                got != NULL ? got : "<none>",
                want);
        failed = 1;
    }
    else
    {
        printf("ok   %s rejects causal_top_left\n", entry_name(entry));
    }
    rocke_ir_builder_free(&b);
    return failed;
}

/* The default spec (causal_top_left off) must still build and lower. */
static int check_default_lowers(entry_t entry)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k;
    char* ll = NULL;
    char err[ERR_CAP];
    rocke_status_t st;
    int failed = 0;

    memset(&b, 0, sizeof b);
    err[0] = '\0';
    k = build(entry, false, &b);
    if(k == NULL || !rocke_ir_builder_ok(&b))
    {
        fprintf(stderr,
                "FAIL %s: default spec build failed: %s\n",
                entry_name(entry),
                rocke_ir_builder_error(&b) != NULL ? rocke_ir_builder_error(&b) : "<none>");
        rocke_ir_builder_free(&b);
        return 1;
    }
    st = rocke_lower_kernel_to_llvm_ex(k, ROCKE_LLVM_FLAVOR_AUTO, ARCH, &ll, err, sizeof err);
    if(st != ROCKE_OK || ll == NULL || strstr(ll, "define ") == NULL)
    {
        fprintf(stderr,
                "FAIL %s: default spec lowering failed: status=%d err=%s\n",
                entry_name(entry),
                (int)st,
                err[0] != '\0' ? err : "<none>");
        failed = 1;
    }
    else
    {
        printf("ok   %s lowers the default spec (%zu B)\n", entry_name(entry), strlen(ll));
    }
    free(ll);
    rocke_ir_builder_free(&b);
    return failed;
}

int main(void)
{
    int failures = 0;

    failures += check_rejected(ENTRY_2D, NO_TOP_LEFT);
    failures += check_rejected(ENTRY_2D_FASTKV, FASTKV_NO_TOP_LEFT);
    failures += check_rejected(ENTRY_3D, NO_TOP_LEFT);
    failures += check_default_lowers(ENTRY_2D);
    failures += check_default_lowers(ENTRY_2D_FASTKV);
    failures += check_default_lowers(ENTRY_3D);

    if(failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
