// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/instances/gfx950/attention_tiled_2d_fastkv_regp.py.
 * See rocke/helper_instance_gfx950_attention_tiled_2d_fastkv_regp.h for the
 * symbol map and the relationship to the already-ported tiled-2D surface.
 */

#include "rocke/error.hpp"
#include "rocke/helper_instance_gfx950_attention_tiled_2d_fastkv_regp.h"
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ helpers */

/* Record a Python ValueError text on the sticky-error builder. Mirrors the
 * idiom used by the rest of the attention port. A dead/NULL builder is a no-op. */
/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary catches it and records status + message on the builder, so the
 * C ABI is unchanged. [[noreturn]] keeps the existing call sites' trailing
 * return valid -- it is simply never reached. */
[[noreturn]] static void
    rocke__fastkv_regp_set_err(rocke_ir_builder_t* b, rocke_status_t st, const char* msg)
{
    (void)b;
    ckc::raise_status(st, msg ? msg : "");
}

/* gfx950 UnifiedAttention2DTiledSpec.kernel_name() (rocke/instances/gfx950/
 * attention_tiled_2d.py kernel_name). The gfx950 part list is SHORTER than the
 * gfx942 one (no wpe, no mfma32x8/qdir/kvcp/gldlds/qgrid/cfv/... parts), so the
 * fastKV register-P base name must be built from this list rather than the
 * gfx942 rocke_attention_tiled_2d_spec_kernel_name. Mirrors the Python order. */
static rocke_status_t rocke__fastkv_gfx950_base_name(const rocke_attention_tiled_2d_spec_t* s,
                                                     char* out,
                                                     size_t out_cap)
{
    char t_buf[32], hkv_buf[64], kv_buf[64], sw_buf[32], w_buf[32], mw_buf[32];
    char d_buf[16], b_buf[16];

    if(s == NULL || out == NULL)
        return ROCKE_ERR_VALUE;

    snprintf(d_buf, sizeof(d_buf), "d%d", s->head_size);
    snprintf(b_buf, sizeof(b_buf), "b%d", s->block_size);
    if(rocke_attention_tiled_2d_spec_n_blocks_per_tile(s) != 1)
        snprintf(t_buf, sizeof(t_buf), "t%d", rocke_attention_tiled_2d_spec_tile_size_eff(s));
    else
        t_buf[0] = '\0';
    snprintf(hkv_buf, sizeof(hkv_buf), "h%dkv%d", s->num_query_heads, s->num_kv_heads);
    if(s->kv_storage_dtype != NULL)
        snprintf(kv_buf, sizeof(kv_buf), "kv%s", s->kv_storage_dtype);
    else
        kv_buf[0] = '\0';
    if(s->sliding_window > 0)
        snprintf(sw_buf, sizeof(sw_buf), "sw%d", s->sliding_window);
    else
        sw_buf[0] = '\0';
    if(s->num_warps != 1)
        snprintf(w_buf, sizeof(w_buf), "w%d", s->num_warps);
    else
        w_buf[0] = '\0';
    if(s->block_m_per_warp != 16)
        snprintf(mw_buf, sizeof(mw_buf), "mw%d", s->block_m_per_warp);
    else
        mw_buf[0] = '\0';

    {
        const char* parts[] = {
            d_buf,
            b_buf,
            t_buf,
            hkv_buf,
            s->dtype ? s->dtype : "",
            kv_buf,
            s->use_sinks ? "sinks" : "",
            sw_buf,
            s->has_softcap ? "softcap" : "",
            s->use_alibi ? "alibi" : "",
            s->use_qq_bias ? "qqb" : "",
            w_buf,
            mw_buf,
            s->use_mfma_32x32 ? "mfma32" : "",
            s->use_transposed_qk_32x32 ? "stqk" : "",
            s->use_transposed_scalar_state ? "s1" : "",
            s->use_transposed_mask_once ? "mask1" : "",
            s->use_transposed_invariant_hoist ? "hoist" : "",
            s->use_transposed_half_local_pv ? "hlpv" : "",
            s->use_mfma32_skip_legacy_qreg ? "skipqreg" : "",
            s->use_transposed_mask_limit ? "mlim" : "",
            s->use_grouped_kv2_softmax ? "gkv2" : "",
            s->use_fast_paged_kv_desc ? "fastkvdesc" : "",
            s->use_early_v_schedule ? "earlyv" : "",
            s->use_agpr_alloc_zero ? "agpr0" : "",
            s->use_fp8_mfma_qk ? "fp8mfma" : "",
            s->use_fp8_mfma_pv ? "fp8pv" : "",
            s->use_register_pv ? "regpv" : "",
        };
        size_t num_parts = sizeof(parts) / sizeof(parts[0]);
        return rocke_kernel_name_join(
            "rocke_uattn2d_tiled", parts, num_parts, NULL, NULL, 0, out, out_cap, NULL);
    }
}

/* ===================================================================== *
 *  _FastKvRegisterPProxy  (Python lines 29-46)
 * ===================================================================== */

rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_t
    rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_make(
        const rocke_attention_tiled_2d_spec_t* spec)
{
    rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_t p;
    memset(&p, 0, sizeof(p));
    if(spec != NULL)
        p.spec = *spec; /* self._spec = spec */
    return p;
}

bool rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_use_register_pv(
    const rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_t* p)
{
    (void)p;
    /* @property use_register_pv -> True (independent of the wrapped spec). */
    return true;
}

rocke_status_t rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_kernel_name(const char* base,
                                                                                  char* out,
                                                                                  size_t out_cap,
                                                                                  size_t* out_len)
{
    /* f"{base}_fastkv_regp" */
    static const char kSuffix[] = "_fastkv_regp";
    size_t base_len, suffix_len, total;

    if(base == NULL || out == NULL)
        return ROCKE_ERR_VALUE;

    base_len = strlen(base);
    suffix_len = sizeof(kSuffix) - 1; /* excludes NUL */
    total = base_len + suffix_len;

    if(out_cap < total + 1) /* +1 for NUL */
        return ROCKE_ERR_VALUE;

    memcpy(out, base, base_len);
    memcpy(out + base_len, kSuffix, suffix_len + 1); /* copy NUL too */

    if(out_len != NULL)
        *out_len = total;
    return ROCKE_OK;
}

/* ===================================================================== *
 *  make_fastkv_register_p_spec(...)  (Python lines 49-94)
 * ===================================================================== */

rocke_attention_tiled_2d_spec_t
    rocke_gfx950_make_fastkv_register_p_spec(const rocke_attention_tiled_2d_spec_t* spec,
                                             bool scalar_state,
                                             bool mask_once,
                                             bool has_mask_limit,
                                             bool mask_limit,
                                             bool half_local_pv,
                                             bool skip_legacy_qreg)
{
    rocke_attention_tiled_2d_spec_t out;
    bool use_mask_limit;

    /* dataclasses.replace(spec, ...): start from a copy of the source spec. */
    memset(&out, 0, sizeof(out));
    if(spec != NULL)
        out = *spec;

    /* use_mask_limit resolution (mask_limit keyword default None -> the derived
     * boolean; otherwise the explicit override). Evaluated against the *source*
     * spec's flags, exactly as the Python expression reads ``spec.*``. */
    if(!has_mask_limit)
    {
        use_mask_limit = scalar_state && mask_once
                         && (spec != NULL ? spec->sliding_window == 0 : true)
                         && (spec != NULL ? !spec->has_softcap : true)
                         && (spec != NULL ? !spec->use_alibi : true)
                         && (spec != NULL ? !spec->use_qq_bias : true);
    }
    else
    {
        use_mask_limit = mask_limit;
    }

    /* The forced experiment knobs (replace(...) kwargs). */
    out.num_warps = 4;

    out.has_waves_per_eu = true;
    out.waves_per_eu = 2;

    out.has_tile_size = true;
    out.tile_size = 2 * (spec != NULL ? spec->block_size : 0);

    out.block_m_per_warp = 32;
    out.use_mfma_32x32 = true;
    out.use_transposed_qk_32x32 = true;
    out.use_transposed_scalar_state = scalar_state;
    out.use_transposed_mask_once = mask_once;
    out.use_transposed_half_local_pv = half_local_pv;
    out.use_mfma32_skip_legacy_qreg = skip_legacy_qreg;
    out.use_transposed_mask_limit = use_mask_limit;
    out.use_fast_paged_kv_desc = true;
    out.use_agpr_alloc_zero = false;
    out.use_register_pv = false;

    return out;
}

/* ===================================================================== *
 *  supports_fastkv_register_p_2d(...)  (Python lines 97-141)
 * ===================================================================== */

bool rocke_gfx950_supports_fastkv_register_p_2d(
    const rocke_gfx950_supports_fastkv_register_p_2d_args_t* args, char* reason, size_t reason_cap)
{
    rocke_gfx942_attention_tiled_2d_supports_args_t targs;
    bool ok;

    if(args == NULL)
        return false;

    /* ok, reason = supports_tiled_2d(... num_warps=4, kv_storage_dtype=None,
     *   tile_size = tile_size if tile_size is not None else 2*block_size,
     *   arch=arch) */
    targs = rocke_gfx942_attention_tiled_2d_supports_args_default();
    targs.head_size = args->head_size;
    targs.block_size = args->block_size;
    targs.dtype = args->dtype;
    targs.num_queries_per_kv = args->num_queries_per_kv;
    targs.use_alibi = args->use_alibi;
    targs.use_qq_bias = args->use_qq_bias;
    targs.use_fp8 = args->use_fp8;
    targs.q_dtype = args->q_dtype;
    targs.num_warps = 4;
    targs.kv_storage_dtype = NULL; /* Python kv_storage_dtype=None */
    targs.has_tile_size = true;
    targs.tile_size = args->has_tile_size ? args->tile_size : 2 * args->block_size;
    targs.arch = (args->arch != NULL) ? args->arch : "gfx950";

    ok = rocke_gfx942_attention_tiled_2d_supports(&targs, reason, reason_cap);
    if(!ok)
        return false; /* reason already populated by the tiled gate */

    /* The experiment's hard shape restriction. */
    if(!(args->dtype != NULL && strcmp(args->dtype, "bf16") == 0 && args->head_size == 64
         && args->block_size == 32 && args->num_query_heads == 64 && args->num_kv_heads == 8))
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "%s",
                     "fastKV register-P experiment is restricted to bf16 d64_b32_h64kv8");
        return false;
    }

    if(reason != NULL && reason_cap > 0)
        snprintf(reason, reason_cap, "%s", "supported");
    return true;
}

/* ===================================================================== *
 *  build_unified_attention_2d_fastkv_register_p(...)  (Python lines 144-162)
 * ===================================================================== */

rocke_kernel_def_t* rocke_build_unified_attention_2d_fastkv_register_p(
    rocke_ir_builder_t* b, const rocke_attention_tiled_2d_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_t proxy;
        rocke_attention_tiled_2d_spec_t built;

        if(b == NULL || spec == NULL)
            return NULL;
        if(b->status != ROCKE_OK) /* dead builder: no-op */
            return NULL;

        /* if not spec.use_fast_paged_kv_desc: raise ValueError(...) */
        if(!spec->use_fast_paged_kv_desc)
        {
            rocke__fastkv_regp_set_err(
                b, ROCKE_ERR_VALUE, "fastKV register-P experiment requires use_fast_paged_kv_desc");
            return NULL;
        }
        /* if not (spec.use_mfma_32x32 and spec.use_transposed_qk_32x32): raise ... */
        if(!(spec->use_mfma_32x32 && spec->use_transposed_qk_32x32))
        {
            rocke__fastkv_regp_set_err(
                b, ROCKE_ERR_VALUE, "fastKV register-P experiment requires transposed R4");
            return NULL;
        }
        /* if spec.kv_storage_dtype is not None: raise ValueError(...) */
        if(spec->kv_storage_dtype != NULL)
        {
            rocke__fastkv_regp_set_err(
                b, ROCKE_ERR_VALUE, "fastKV register-P experiment does not support FP8 KV cache");
            return NULL;
        }

        /* return build_unified_attention_2d_tiled(_FastKvRegisterPProxy(spec), arch)
         *
         * The proxy forwards every attribute to the wrapped spec except the
         * use_register_pv @property, which it forces True. The C tiled builder reads
         * a plain spec struct, so apply the single proxy override (use_register_pv ->
         * true) onto a copy and pass that through. ``arch`` NULL == Python "gfx950";
         * the tiled builder defaults NULL to its own arch and threads it onward. */
        proxy = rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_make(spec);
        built = proxy.spec;
        built.use_register_pv
            = rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_use_register_pv(&proxy);

        /* The proxy overrides kernel_name() -> "<wrapped.kernel_name()>_fastkv_regp"
         * (Python _FastKvRegisterPProxy.kernel_name). The tiled builder names the
         * kernel from b->kernel->name (set by the caller's rocke_ir_builder_init seed),
         * so rename the builder's kernel to the proxy name BEFORE building -- the name
         * flows into the @Klds/@Vlds globals and the kernel symbol. */
        if(b->kernel != NULL)
        {
            char base[1024];
            char full[1056];
            /* base = self._spec.kernel_name() over the WRAPPED (original) spec, whose
             * use_register_pv is the spec's own value (False) -- the proxy's True
             * override is NOT reflected in the kernel name (Python proxy delegates
             * kernel_name to the wrapped spec, then appends "_fastkv_regp"). Use the
             * gfx950 part list, not the gfx942 rocke_attention_tiled_2d_spec_kernel_name. */
            if(rocke__fastkv_gfx950_base_name(spec, base, sizeof(base)) == ROCKE_OK
               && rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_kernel_name(
                      base, full, sizeof(full), NULL)
                      == ROCKE_OK)
            {
                b->kernel->name = rocke_arena_strdup(&b->arena, full);
            }
        }

        return rocke_build_unified_attention_2d_tiled_scalar(
            b, &built, (arch != NULL) ? arch : "gfx950");
    });
}
