// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Canonical translation unit for the C99 port of
 * rocke/instances/gfx950/attention_tiled_2d_fastkv_regp.py.
 *
 * The four ported task symbols (proxy type, make_fastkv_register_p_spec,
 * supports_fastkv_register_p_2d, build_unified_attention_2d_fastkv_register_p)
 * have a single authoritative definition in the byte-identical-call helper TU
 * (helper_instance_gfx950_attention_tiled_2d_fastkv_regp.c) and are re-exported
 * by this module's header. This file adds only the build -> lower convenience
 * entry, matching the gfx942 tiled-2D instance glue.
 *
 * See rocke/instance_gfx950_attention_tiled_2d_fastkv_regp.h for the symbol map.
 */

#include "rocke/instance_gfx950_attention_tiled_2d_fastkv_regp.h"

#include <stdio.h>
#include <string.h>

/* Write a diagnostic into a caller buffer (NUL-terminated, never overflows).
 * NULL/zero-capacity buffer is a no-op. Mirrors rocke__set_err in the sibling
 * tiled glue. */
static void rocke__fastkv_regp_diag(char* err, size_t err_cap, const char* msg)
{
    if(err == NULL || err_cap == 0)
        return;
    snprintf(err, err_cap, "%s", msg ? msg : "");
}

/* ===================================================================== *
 *  build -> lower convenience.
 * ===================================================================== *
 *
 * Init an internally-owned IRBuilder, build the fastKV register-P kernel via
 * rocke_build_unified_attention_2d_fastkv_register_p, then lower to LLVM .ll text
 * through rocke_lower_kernel_to_llvm_ex. The wrapped tiled builder names the kernel
 * def itself; the IRBuilder is seeded with a stable experiment-suffixed name. */
rocke_status_t rocke_gfx950_attention_tiled_2d_fastkv_regp_lower_to_llvm(
    const rocke_attention_tiled_2d_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        rocke__fastkv_regp_diag(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }

    /* The proxy's kernel_name() is "<base>_fastkv_regp"; the base name lives as a
     * private static in the tiled glue TU, so seed the builder with the suffix as
     * a stable, valid identifier. The emitted kernel def name is produced by the
     * tiled builder body, independent of this seed. */
    if(rocke_ir_builder_init(&b, "attention_tiled_2d_fastkv_regp") != ROCKE_OK)
    {
        rocke__fastkv_regp_diag(err, err_cap, "lower_to_llvm: builder init failed");
        return ROCKE_ERR_VALUE;
    }

    kernel = rocke_build_unified_attention_2d_fastkv_register_p(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        rocke__fastkv_regp_diag(err, err_cap, rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(
        kernel, flavor, arch ? arch : "gfx950", out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
