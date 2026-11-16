// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only test of the C++ tiled 2D attention builders' direct-from-HBM K and
 * V knobs.
 *
 * use_k_hbm_direct is emitted by this engine, and the byte-identity parity
 * configs compare its kernels with the Python builder's. This test pins what
 * the parity gate cannot see, because that gate only compares specs both
 * engines accept:
 *   1. the spec checks that refuse a K-from-HBM spec the kernel cannot build or
 *      cannot read correctly, such as one with i64 K/V addressing (the same
 *      refusals the Python spec makes at construction);
 *   2. the refusal of use_v_hbm_direct, which this engine does not emit yet and
 *      so must not drop silently;
 *   3. the K-from-HBM token in the kernel name;
 *   4. the refusal of both knobs on every gfx950 entry (the tiled 2D build and
 *      the fastKV register-P build). The knobs are gfx942-only and the gfx950
 *      Python spec cannot carry them, but the shared C spec struct can, and the
 *      fastKV build would otherwise reach a NULL K descriptor;
 *   5. the gfx942 spec checks on the gfx942 entry when it builds for gfx950,
 *      as the Python gfx942 spec checks itself whatever the target.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx942_attention_tiled_2d.h"
#include "rocke/instance_gfx950_attention_tiled_2d.h"
#include "rocke/instance_gfx950_attention_tiled_2d_fastkv_regp.h"
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

/* A D128 bf16 gfx950 tiled 2D spec on the default path. */
static rocke_attention_tiled_2d_spec_t gfx950_tiled_2d_spec(void)
{
    rocke_attention_tiled_2d_spec_t spec = rocke_attention_tiled_2d_spec_default();
    spec.head_size = 128;
    spec.block_size = 16;
    spec.num_query_heads = 32;
    spec.num_kv_heads = 8;
    spec.dtype = "bf16";
    spec.use_sinks = false;
    spec.sliding_window = 0;
    spec.has_softcap = false;
    return spec;
}

/* make_fastkv_register_p_spec over the experiment's bf16 D64 / 32-token block
 * shape, the gfx950 fastKV register-P build's only geometry. */
static rocke_attention_tiled_2d_spec_t fastkv_register_p_spec(void)
{
    rocke_attention_tiled_2d_spec_t spec = rocke_attention_tiled_2d_spec_default();
    spec.head_size = 64;
    spec.block_size = 32;
    spec.num_query_heads = 64;
    spec.num_kv_heads = 8;
    spec.dtype = "bf16";
    spec.use_sinks = false;
    spec.sliding_window = 0;
    spec.has_softcap = false;
    spec.use_fast_paged_kv_desc = true;
    spec.use_mfma_32x32 = true;
    spec.use_transposed_qk_32x32 = true;
    return spec;
}

/* A C entry that lowers a tiled 2D spec for one target. */
typedef rocke_status_t (*lower_fn_t)(const rocke_attention_tiled_2d_spec_t* spec,
                                     const char* arch,
                                     rocke_llvm_flavor_t flavor,
                                     char** out_ll,
                                     char* err,
                                     size_t err_cap);

typedef struct tiled_2d_entry
{
    lower_fn_t lower;
    const char* arch;
} tiled_2d_entry_t;

static const tiled_2d_entry_t GFX942_TILED_2D
    = {rocke_gfx942_attention_tiled_2d_lower_to_llvm, "gfx942"};
/* The gfx942 entry also builds for gfx950; the Python gfx942 spec checks itself
 * whatever the target, so this entry must too. */
static const tiled_2d_entry_t GFX942_TILED_2D_ON_GFX950
    = {rocke_gfx942_attention_tiled_2d_lower_to_llvm, "gfx950"};
static const tiled_2d_entry_t GFX950_TILED_2D
    = {rocke_gfx950_attention_tiled_2d_lower_to_llvm, "gfx950"};
static const tiled_2d_entry_t GFX950_FASTKV_REGISTER_P
    = {rocke_gfx950_attention_tiled_2d_fastkv_regp_lower_to_llvm, "gfx950"};

/* Lower `spec` through `entry`. On success returns the malloc'd .ll (caller
 * frees) and leaves err empty; on refusal returns NULL with *status and err
 * holding the refusal. */
static char* lower(const tiled_2d_entry_t* entry,
                   const rocke_attention_tiled_2d_spec_t* spec,
                   rocke_status_t* status,
                   char* err,
                   size_t err_cap)
{
    char* ll = NULL;
    err[0] = '\0';
    *status = entry->lower(spec, entry->arch, ROCKE_LLVM_FLAVOR_AUTO, &ll, err, err_cap);
    if(*status != ROCKE_OK)
    {
        free(ll);
        return NULL;
    }
    return ll;
}

static void expect_refused(const char* what,
                           const tiled_2d_entry_t* entry,
                           const rocke_attention_tiled_2d_spec_t* spec,
                           rocke_status_t expected_status,
                           const char* reason)
{
    char err[1024];
    rocke_status_t status;
    char* ll = lower(entry, spec, &status, err, sizeof(err));
    if(ll != NULL)
    {
        free(ll);
        fail(what, "the spec was accepted, expected a refusal");
        return;
    }
    if(status != expected_status)
    {
        char detail[1200];
        snprintf(detail,
                 sizeof(detail),
                 "expected status %d, got %d (\"%s\")",
                 (int)expected_status,
                 (int)status,
                 err);
        fail(what, detail);
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
/* The gfx942 fast paged-KV descriptor refusal, in full: the same text as the
 * Python gfx942 spec's. */
#define GFX942_FAST_DESC_HBM_REFUSAL(k_value, v_value)                                         \
    "gfx942 tiled 2D spec failed - use_fast_paged_kv_desc=True with use_k_hbm_direct=" k_value \
    ", use_v_hbm_direct=" v_value ", expected both False: the direct K and V reads address "   \
    "the cache through the paged-KV descriptor, which the fast descriptor path does not build"

static void expect_accepted(const char* what,
                            const tiled_2d_entry_t* entry,
                            const rocke_attention_tiled_2d_spec_t* spec)
{
    char err[1024];
    rocke_status_t status;
    char* ll = lower(entry, spec, &status, err, sizeof(err));
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
    expect_accepted("transposed_x8_base", &GFX942_TILED_2D, &spec);
    spec = fast_kv_desc_spec();
    expect_accepted("fast_kv_desc_base", &GFX942_TILED_2D, &spec);
    spec = sliced_k_ring_spec();
    expect_refused("sliced_k_ring_base",
                   &GFX942_TILED_2D,
                   &spec,
                   ROCKE_ERR_NOTIMPL,
                   "use_k_sliced_ring: the gfx942 C twin does not emit the sliced-K ring "
                   "schedule (not yet ported)");

    spec = transposed_x8_spec();
    spec.use_transposed_qk_32x32 = false;
    spec.use_k_hbm_direct = true;
    expect_refused("k_needs_x8",
                   &GFX942_TILED_2D,
                   &spec,
                   ROCKE_ERR_VALUE,
                   "use_k_hbm_direct requires transposed-x8 "
                   "(use_mfma_32x32x8 + use_transposed_qk_32x32); "
                   "the 16x16 QK loop always reads K_lds");

    spec = transposed_x8_spec();
    spec.use_k_hbm_direct = true;
    spec.use_global_load_lds_k = true;
    expect_refused("k_excludes_gldlds",
                   &GFX942_TILED_2D,
                   &spec,
                   ROCKE_ERR_VALUE,
                   "use_k_hbm_direct is mutually exclusive with use_global_load_lds_k");

    spec = sliced_k_ring_spec();
    spec.use_k_hbm_direct = true;
    expect_refused("k_excludes_sliced_ring",
                   &GFX942_TILED_2D,
                   &spec,
                   ROCKE_ERR_VALUE,
                   "use_k_hbm_direct is mutually exclusive with use_k_sliced_ring");

    spec = transposed_x8_spec();
    spec.use_i64_kv_addr = true;
    expect_accepted("i64_kv_addr_base", &GFX942_TILED_2D, &spec);
    spec.use_k_hbm_direct = true;
    expect_refused("k_excludes_i64_kv_addr",
                   &GFX942_TILED_2D,
                   &spec,
                   ROCKE_ERR_VALUE,
                   "gfx942 tiled 2D spec failed - use_i64_kv_addr=True with "
                   "use_k_hbm_direct=True, expected use_k_hbm_direct=False: the "
                   "direct K read is an i32 offset into one K cache buffer "
                   "resource, which cannot reach pages past 2 GiB");

    spec = fast_kv_desc_spec();
    spec.use_k_hbm_direct = true;
    expect_refused("fast_desc_excludes_hbm",
                   &GFX942_TILED_2D,
                   &spec,
                   ROCKE_ERR_VALUE,
                   GFX942_FAST_DESC_HBM_REFUSAL("True", "False"));

    spec = fast_kv_desc_spec();
    spec.use_v_hbm_direct = true;
    expect_refused("fast_desc_excludes_v_hbm",
                   &GFX942_TILED_2D,
                   &spec,
                   ROCKE_ERR_VALUE,
                   GFX942_FAST_DESC_HBM_REFUSAL("False", "True"));

    /* Built for gfx950, the gfx942 entry still checks the spec, so the fast
     * descriptor never reaches the K read it does not build. */
    spec = fast_kv_desc_spec();
    expect_accepted("fast_kv_desc_base_on_gfx950", &GFX942_TILED_2D_ON_GFX950, &spec);
    spec.use_k_hbm_direct = true;
    expect_refused("fast_desc_excludes_hbm_on_gfx950",
                   &GFX942_TILED_2D_ON_GFX950,
                   &spec,
                   ROCKE_ERR_VALUE,
                   GFX942_FAST_DESC_HBM_REFUSAL("True", "False"));

    spec = transposed_x8_spec();
    spec.use_v_hbm_direct = true;
    expect_refused("v_not_ported",
                   &GFX942_TILED_2D,
                   &spec,
                   ROCKE_ERR_NOTIMPL,
                   "use_v_hbm_direct: the gfx942 C twin does not emit the direct "
                   "V reads (not yet ported)");
}

static void test_k_from_hbm_is_named_in_the_kernel(void)
{
    const char* const expected_name
        = "@rocke_uattn2d_tiled_d128_b32_g8_bf16_w2_mw32_mfma32x8_stqk_khbm(";
    rocke_attention_tiled_2d_spec_t spec = transposed_x8_spec();
    char err[1024];
    rocke_status_t status;
    char* ll;

    spec.use_k_hbm_direct = true;
    ll = lower(&GFX942_TILED_2D, &spec, &status, err, sizeof(err));
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

/* The gfx950 refusal names both knob values, as the gfx942 fast-descriptor
 * refusal does, and says why gfx950 cannot take them. */
#define GFX950_HBM_REFUSAL(k_value, v_value)                                                \
    "gfx950 tiled 2D spec failed - use_k_hbm_direct=" k_value ", use_v_hbm_direct=" v_value \
    ", expected both False: the direct K and V reads are gfx942 tiled 2D knobs; the "       \
    "gfx950 spec does not carry them"

/* Each gfx950 entry builds its base spec and refuses it with either direct
 * read turned on. */
static void expect_gfx950_entry_refuses_hbm_knobs(const char* what,
                                                  const tiled_2d_entry_t* entry,
                                                  rocke_attention_tiled_2d_spec_t base)
{
    char name[96];
    rocke_attention_tiled_2d_spec_t spec;

    snprintf(name, sizeof(name), "%s_base", what);
    expect_accepted(name, entry, &base);

    spec = base;
    spec.use_k_hbm_direct = true;
    snprintf(name, sizeof(name), "%s_refuses_k_hbm", what);
    expect_refused(name, entry, &spec, ROCKE_ERR_VALUE, GFX950_HBM_REFUSAL("True", "False"));

    spec = base;
    spec.use_v_hbm_direct = true;
    snprintf(name, sizeof(name), "%s_refuses_v_hbm", what);
    expect_refused(name, entry, &spec, ROCKE_ERR_VALUE, GFX950_HBM_REFUSAL("False", "True"));
}

int main(void)
{
    test_refusals();
    test_k_from_hbm_is_named_in_the_kernel();
    expect_gfx950_entry_refuses_hbm_knobs(
        "gfx950_tiled_2d", &GFX950_TILED_2D, gfx950_tiled_2d_spec());
    expect_gfx950_entry_refuses_hbm_knobs(
        "gfx950_fastkv_register_p", &GFX950_FASTKV_REGISTER_P, fastkv_register_p_spec());
    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("ALL TILED 2D HBM KNOB CHECKS PASSED\n");
    return 0;
}
