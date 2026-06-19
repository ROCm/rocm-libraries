// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Canonical translation unit for the C99 port of
 * ck_dsl/instances/gfx950/attention_tiled_2d_fastkv_regp.py.
 *
 * The four ported task symbols (proxy type, make_fastkv_register_p_spec,
 * supports_fastkv_register_p_2d, build_unified_attention_2d_fastkv_register_p)
 * have a single authoritative definition in the byte-identical-call helper TU
 * (helper_instance_gfx950_attention_tiled_2d_fastkv_regp.c) and are re-exported
 * by this module's header. This file adds only the build -> lower convenience
 * entry, matching the gfx942 tiled-2D instance glue.
 *
 * See ckc/instance_gfx950_attention_tiled_2d_fastkv_regp.h for the symbol map.
 */

#include "ckc/instance_gfx950_attention_tiled_2d_fastkv_regp.h"

#include <stdio.h>
#include <string.h>

/* Write a diagnostic into a caller buffer (NUL-terminated, never overflows).
 * NULL/zero-capacity buffer is a no-op. Mirrors ckc__set_err in the sibling
 * tiled glue. */
static void ckc__fastkv_regp_diag(char* err, size_t err_cap, const char* msg)
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
 * ckc_build_unified_attention_2d_fastkv_register_p, then lower to LLVM .ll text
 * through ckc_lower_kernel_to_llvm_ex. The wrapped tiled builder names the kernel
 * def itself; the IRBuilder is seeded with a stable experiment-suffixed name. */
ckc_status_t
ckc_gfx950_attention_tiled_2d_fastkv_regp_lower_to_llvm(const ckc_attention_tiled_2d_spec_t* spec,
                                                        const char* arch,
                                                        ckc_llvm_flavor_t flavor,
                                                        char** out_ll,
                                                        char* err,
                                                        size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        ckc__fastkv_regp_diag(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }

    /* The proxy's kernel_name() is "<base>_fastkv_regp"; the base name lives as a
     * private static in the tiled glue TU, so seed the builder with the suffix as
     * a stable, valid identifier. The emitted kernel def name is produced by the
     * tiled builder body, independent of this seed. */
    if(ckc_ir_builder_init(&b, "attention_tiled_2d_fastkv_regp") != CKC_OK)
    {
        ckc__fastkv_regp_diag(err, err_cap, "lower_to_llvm: builder init failed");
        return CKC_ERR_VALUE;
    }

    kernel = ckc_build_unified_attention_2d_fastkv_register_p(&b, spec, arch);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        ckc__fastkv_regp_diag(err, err_cap, ckc_ir_builder_error(&b));
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch ? arch : "gfx950", out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
