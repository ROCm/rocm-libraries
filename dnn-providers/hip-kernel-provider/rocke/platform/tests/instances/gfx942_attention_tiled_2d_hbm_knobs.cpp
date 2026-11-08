// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only test of the C++ gfx942 tiled 2D attention builder's direct-from-HBM
 * K and V knobs.
 *
 * use_k_hbm_direct is emitted by this engine, and the byte-identity parity
 * configs compare its kernels with the Python builder's. This test pins what
 * the parity gate cannot see, because that gate only compares specs both
 * engines accept:
 *   1. the spec checks that refuse a K-from-HBM spec the kernel cannot build
 *      (the same refusals the Python spec makes at construction);
 *   2. the refusal of use_v_hbm_direct, which this engine does not emit yet and
 *      so must not drop silently;
 *   3. the K-from-HBM token in the kernel name.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx942_attention_tiled_2d.h"
#include "rocke/lower_llvm.h"

static int g_failures = 0;

static void fail(const char* what, const char* detail)
{
    fprintf(stderr, "FAIL %s: %s\n", what, (detail != NULL) ? detail : "<none>");
    ++g_failures;
}

/* The parity configs' D128 bf16 transposed-x8 geometry (two warps, 32-token
 * tiles), a spec that accepts use_k_hbm_direct. */
static rocke_attention_tiled_2d_spec_t transposed_x8_spec(void)
{
    rocke_attention_tiled_2d_spec_t spec = rocke_attention_tiled_2d_spec_default();
    spec.head_size = 128;
    spec.block_size = 32;
    spec.num_query_heads = 8;
    spec.num_kv_heads = 1;
    spec.dtype = "bf16";
    spec.use_sinks = false;
    spec.sliding_window = 0;
    spec.has_softcap = false;
    spec.num_warps = 2;
    spec.block_m_per_warp = 32;
    spec.has_tile_size = true;
    spec.tile_size = 32;
    spec.use_mfma_32x32x8 = true;
    spec.use_transposed_qk_32x32 = true;
    return spec;
}

/* A D64 bf16 four-warp geometry the fast paged-KV descriptor accepts. */
static rocke_attention_tiled_2d_spec_t fast_kv_desc_spec(void)
{
    rocke_attention_tiled_2d_spec_t spec = transposed_x8_spec();
    spec.head_size = 64;
    spec.num_query_heads = 64;
    spec.num_kv_heads = 8;
    spec.num_warps = 4;
    spec.tile_size = 64;
    spec.use_fast_paged_kv_desc = true;
    return spec;
}

/* A D64 bf16 four-warp geometry on the transposed-x8 conflict-free V store
 * path with the sliced K ring, the only path the spec accepts the ring on.
 * This engine refuses the ring itself as not yet ported, but only after the
 * spec checks, so a K-from-HBM refusal still comes first. */
static rocke_attention_tiled_2d_spec_t sliced_k_ring_spec(void)
{
    rocke_attention_tiled_2d_spec_t spec = transposed_x8_spec();
    spec.head_size = 64;
    spec.num_query_heads = 32;
    spec.num_kv_heads = 8;
    spec.num_warps = 4;
    spec.tile_size = 64;
    spec.use_conflict_free_v_store = true;
    spec.use_k_sliced_ring = true;
    return spec;
}

/* Lower `spec` for gfx942. On success returns the malloc'd .ll (caller frees)
 * and leaves err empty; on refusal returns NULL with err holding the reason. */
static char* lower(const rocke_attention_tiled_2d_spec_t* spec, char* err, size_t err_cap)
{
    char* ll = NULL;
    err[0] = '\0';
    if(rocke_gfx942_attention_tiled_2d_lower_to_llvm(
           spec, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, err_cap)
       != ROCKE_OK)
    {
        free(ll);
        return NULL;
    }
    return ll;
}

static void expect_refused(const char* what,
                           const rocke_attention_tiled_2d_spec_t* spec,
                           const char* reason)
{
    char err[1024];
    char* ll = lower(spec, err, sizeof(err));
    if(ll != NULL)
    {
        free(ll);
        fail(what, "the spec was accepted, expected a refusal");
        return;
    }
    if(strstr(err, reason) == NULL)
    {
        char detail[2048];
        snprintf(detail, sizeof(detail), "expected \"%s\" in \"%s\"", reason, err);
        fail(what, detail);
        return;
    }
    printf("ok   %s (refused: \"%s\")\n", what, reason);
}

/* Each refusal below turns one knob on in a base spec; the base must build (or,
 * for the sliced K ring, be refused for another reason), so a refusal is caused
 * by that knob and not by the geometry. */
static void expect_accepted(const char* what, const rocke_attention_tiled_2d_spec_t* spec)
{
    char err[1024];
    char* ll = lower(spec, err, sizeof(err));
    if(ll == NULL)
    {
        fail(what, err);
        return;
    }
    free(ll);
    printf("ok   %s (builds)\n", what);
}

static void test_refusals(void)
{
    rocke_attention_tiled_2d_spec_t spec;

    spec = transposed_x8_spec();
    expect_accepted("transposed_x8_base", &spec);
    spec = fast_kv_desc_spec();
    expect_accepted("fast_kv_desc_base", &spec);
    spec = sliced_k_ring_spec();
    expect_refused("sliced_k_ring_base",
                   &spec,
                   "use_k_sliced_ring: the gfx942 C twin does not emit the sliced-K ring "
                   "schedule (not yet ported)");

    spec = transposed_x8_spec();
    spec.use_transposed_qk_32x32 = false;
    spec.use_k_hbm_direct = true;
    expect_refused("k_needs_x8",
                   &spec,
                   "use_k_hbm_direct requires transposed-x8 "
                   "(use_mfma_32x32x8 + use_transposed_qk_32x32); "
                   "the 16x16 QK loop always reads K_lds");

    spec = transposed_x8_spec();
    spec.use_k_hbm_direct = true;
    spec.use_global_load_lds_k = true;
    expect_refused("k_excludes_gldlds",
                   &spec,
                   "use_k_hbm_direct is mutually exclusive with use_global_load_lds_k");

    spec = sliced_k_ring_spec();
    spec.use_k_hbm_direct = true;
    expect_refused("k_excludes_sliced_ring",
                   &spec,
                   "use_k_hbm_direct is mutually exclusive with use_k_sliced_ring");

    spec = fast_kv_desc_spec();
    spec.use_k_hbm_direct = true;
    expect_refused("fast_desc_excludes_hbm",
                   &spec,
                   "gfx942 tiled 2D spec failed - use_fast_paged_kv_desc=True with "
                   "use_k_hbm_direct=True, use_v_hbm_direct=False, expected both False");

    spec = fast_kv_desc_spec();
    spec.use_v_hbm_direct = true;
    expect_refused("fast_desc_excludes_v_hbm",
                   &spec,
                   "gfx942 tiled 2D spec failed - use_fast_paged_kv_desc=True with "
                   "use_k_hbm_direct=False, use_v_hbm_direct=True, expected both False");

    spec = transposed_x8_spec();
    spec.use_v_hbm_direct = true;
    expect_refused("v_not_ported",
                   &spec,
                   "use_v_hbm_direct: the gfx942 C twin does not emit the direct "
                   "V reads (not yet ported)");
}

static void test_k_from_hbm_is_named_in_the_kernel(void)
{
    const char* const expected_name
        = "@rocke_uattn2d_tiled_d128_b32_g8_bf16_w2_mw32_mfma32x8_stqk_khbm(";
    rocke_attention_tiled_2d_spec_t spec = transposed_x8_spec();
    char err[1024];
    char* ll;

    spec.use_k_hbm_direct = true;
    ll = lower(&spec, err, sizeof(err));
    if(ll == NULL)
    {
        fail("k_from_hbm_kernel_name", err);
        return;
    }
    if(strstr(ll, expected_name) == NULL)
        fail("k_from_hbm_kernel_name", "the kernel name lacks the khbm token");
    else
        printf("ok   k_from_hbm_kernel_name (%s)\n", expected_name);
    free(ll);
}

int main(void)
{
    test_refusals();
    test_k_from_hbm_is_named_in_the_kernel();
    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("ALL GFX942 TILED 2D HBM KNOB CHECKS PASSED\n");
    return 0;
}
