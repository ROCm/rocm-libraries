// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_flatmm.c -- C99 port of rocke/instances/common/flatmm.py.
 *
 * FlatMM is a thin wrapper around batched_gemm, which delegates to
 * build_universal_gemm with batched=True. The v1 kernel body is shared
 * verbatim, so this TU only:
 *   - converts a FlatMMSpec into the equivalent UniversalGemmSpec
 *     (FlatMMSpec.to_batched_spec() -> BatchedGemmSpec.to_universal_spec()),
 *   - validates (rejecting preshuffle_b until the v2 body lands),
 *   - delegates the build/lower to rocke_build_universal_gemm,
 *   - provides the spec convenience constructors, grid/signature, and the
 *     host-side preshuffled-B layout helpers.
 *
 * batched_gemm has no standalone C port, so its (tiny) conversion logic is
 * inlined here: a BatchedGemmSpec becomes a UniversalGemmSpec with batched=True,
 * data = DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt) where dt canonicalises
 * f16->fp16, and wave_size/block_size/name carried through. FlatMM always uses
 * the f16/fp16 dtype family (the v1 body is shared with batched_gemm's default).
 */

#include "rocke/instance_flatmm.h"

#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.preshuffle.h"
#include "rocke/helper_rocke.helpers.spec.h" /* SignatureBuilder, ceil_div_grid */
#include "rocke/instance_gemm_universal.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err (sticky-error parity) */

/* ---------------------------------------------------------------- helpers *
 * Small NUL-terminated copy into a fixed buffer (snprintf-free, matching the
 * style of the rest of the C port). Returns ROCKE_OK or ROCKE_ERR_VALUE if the
 * message does not fit. */
static rocke_status_t rocke__copy_str(char* out, size_t out_cap, const char* msg)
{
    size_t n;
    if(out == NULL || out_cap == 0)
    {
        return ROCKE_OK; /* caller opted out of the message */
    }
    if(msg == NULL)
    {
        msg = "";
    }
    n = strlen(msg);
    if(n >= out_cap)
    {
        n = out_cap - 1;
    }
    memcpy(out, msg, n);
    out[n] = '\0';
    return ROCKE_OK;
}

/* ----------------------------------------------------------- spec_default *
 * FlatMMSpec dataclass defaults (mirror of the Python field defaults). */
rocke_flatmm_spec_t rocke_flatmm_spec_default(void)
{
    rocke_flatmm_spec_t s;
    memset(&s, 0, sizeof(s));

    /* tile has no defaults in Python (required positional) -- left zeroed; the
     * caller must fill the geometry. We do mirror TileSpec's per-field defaults
     * for warp_k / warp_tile_* so a partially-filled spec matches Python. */
    s.tile.warp_k = 1;
    s.tile.warp_tile_m = 32;
    s.tile.warp_tile_n = 32;
    s.tile.warp_tile_k = 16;

    /* trait := TraitSpec() defaults, sourced from the universal default. */
    {
        rocke_gemm_universal_spec_t u = rocke_gemm_universal_spec_default();
        s.trait = u.trait;
    }

    s.wave_size = 64;
    s.block_size = 0; /* derived at finalize() */
    s.batch_size = 0;
    s.preshuffle_b = false;
    s.name = "rocke_flatmm";
    return s;
}

/* --------------------------------------------------------- spec_finalize *
 * WarpTileBlockSizeMixin._init_block_size(): block_size==0 =>
 * warp_m*warp_n*warp_k*wave_size. Idempotent. */
void rocke_flatmm_spec_finalize(rocke_flatmm_spec_t* spec)
{
    if(spec == NULL)
    {
        return;
    }
    if(spec->block_size == 0)
    {
        int wm = spec->tile.warp_m;
        int wn = spec->tile.warp_n;
        int wk = spec->tile.warp_k;
        int ws = spec->wave_size;
        spec->block_size = wm * wn * wk * ws;
    }
}

/* ------------------------------------------------------ to_universal_spec *
 * FlatMMSpec.to_batched_spec() then BatchedGemmSpec.to_universal_spec():
 *
 *   prefix = name + ("_psb" if preshuffle_b else "")
 *   BatchedGemmSpec(name=prefix, tile, trait, wave_size, block_size,
 *                   batch_size, dtype="fp16")
 *   -> UniversalGemmSpec(name=prefix, tile, trait,
 *        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt), wave_size,
 *        block_size, batched=True)  where dt = "fp16" (f16/fp16 canon).
 *
 * The prefix is composed into a static thread-unsafe buffer is avoided; instead
 * we store it via the out spec's `name` field pointing at caller-provided
 * storage is not possible (const char*). So we keep a file-static buffer keyed
 * per call is also unsafe -- instead we point `out->name` at a small bounded
 * static only when no suffix is needed, and otherwise compose into a static
 * buffer. To stay re-entrant for the common (no-suffix) case we point straight
 * at spec->name; for the preshuffle_b case (rejected at build anyway) we use a
 * static compose buffer.
 */
rocke_status_t rocke_flatmm_to_universal_spec(const rocke_flatmm_spec_t* spec,
                                              rocke_gemm_universal_spec_t* out)
{
    static char psb_name[256];
    rocke_gemm_universal_spec_t u;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    u = rocke_gemm_universal_spec_default();

    /* prefix = name + ("_psb" if preshuffle_b) */
    if(spec->preshuffle_b)
    {
        const char* base = (spec->name != NULL) ? spec->name : "rocke_flatmm";
        size_t bn = strlen(base);
        const char* suf = "_psb";
        size_t sn = strlen(suf);
        if(bn + sn >= sizeof(psb_name))
        {
            return ROCKE_ERR_VALUE;
        }
        memcpy(psb_name, base, bn);
        memcpy(psb_name + bn, suf, sn);
        psb_name[bn + sn] = '\0';
        u.name = psb_name;
    }
    else
    {
        u.name = (spec->name != NULL) ? spec->name : "rocke_flatmm";
    }

    u.tile = spec->tile;
    u.trait = spec->trait;

    /* BatchedGemmSpec._data_spec(): dt = "fp16" for the f16/fp16 family.
     * FlatMM always rides the f16 dtype default. */
    u.data.dtype_a = "fp16";
    u.data.dtype_b = "fp16";
    u.data.dtype_c = "fp16";
    /* dtype_acc / layout keep the universal default (fp32 / RCR). */

    u.wave_size = spec->wave_size;
    u.block_size = spec->block_size;
    u.batched = true;

    /* BatchedGemmSpec.__post_init__ -> _init_block_size(): finalize the derived
     * block_size on the universal spec too (idempotent). */
    rocke_gemm_universal_spec_finalize(&u);

    *out = u;
    return ROCKE_OK;
}

/* ------------------------------------------------------------ kernel_name *
 * FlatMMSpec.kernel_name() -> to_batched_spec().kernel_name() ->
 * to_universal_spec().kernel_name(). */
rocke_status_t rocke_flatmm_kernel_name(const rocke_flatmm_spec_t* spec, char* out, size_t out_cap)
{
    rocke_gemm_universal_spec_t u;
    rocke_status_t st;
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    st = rocke_flatmm_to_universal_spec(spec, &u);
    if(st != ROCKE_OK)
    {
        return st;
    }
    return rocke_gemm_universal_kernel_name(&u, out, out_cap);
}

/* ------------------------------------------------------ config32 / config16 *
 * Mirrors of CK Tile FlatmmConfig32 / FlatmmConfig16. */
static bool rocke__flatmm_dtype_ok(const char* dtype)
{
    if(dtype == NULL)
    {
        return true; /* default "f16" */
    }
    return (strcmp(dtype, "f16") == 0) || (strcmp(dtype, "fp16") == 0)
           || (strcmp(dtype, "bf16") == 0);
}

rocke_status_t rocke_flatmm_config32(const char* dtype, rocke_gemm_tile_spec_t* out)
{
    rocke_gemm_tile_spec_t t;
    if(out == NULL || !rocke__flatmm_dtype_ok(dtype))
    {
        return ROCKE_ERR_VALUE;
    }
    memset(&t, 0, sizeof(t));
    t.tile_m = 128;
    t.tile_n = 128;
    t.tile_k = 128 / 2; /* 128 bytes / sizeof(f16) */
    t.warp_m = 1;
    t.warp_n = 4;
    t.warp_k = 1;
    t.warp_tile_m = 32;
    t.warp_tile_n = 32;
    t.warp_tile_k = 16;
    *out = t;
    return ROCKE_OK;
}

rocke_status_t rocke_flatmm_config16(const char* dtype, rocke_gemm_tile_spec_t* out)
{
    rocke_gemm_tile_spec_t t;
    if(out == NULL || !rocke__flatmm_dtype_ok(dtype))
    {
        return ROCKE_ERR_VALUE;
    }
    memset(&t, 0, sizeof(t));
    t.tile_m = 128;
    t.tile_n = 128;
    t.tile_k = 128 / 2;
    t.warp_m = 1;
    t.warp_n = 4;
    t.warp_k = 1;
    t.warp_tile_m = 16;
    t.warp_tile_n = 16;
    t.warp_tile_k = 32;
    *out = t;
    return ROCKE_OK;
}

/* ----------------------------------------------------------- is_valid_spec *
 * Python:
 *   if spec.preshuffle_b: return False, "...gated...flatmm_preshuffle_b_layout..."
 *   ok, why = batched_gemm.is_valid_spec(spec.to_batched_spec(), arch)
 *   if not ok: return False, f"base batched_gemm spec invalid: {why}"
 *   return True, "ok"
 *
 * batched_gemm.is_valid_spec delegates to gemm_universal.is_valid_spec. */
bool rocke_flatmm_is_valid_spec(const rocke_flatmm_spec_t* spec,
                                const char* arch,
                                char* reason,
                                size_t reason_cap)
{
    rocke_gemm_universal_spec_t u;
    char base_reason[256];
    bool ok;

    if(spec == NULL)
    {
        rocke__copy_str(reason, reason_cap, "null flatmm spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    if(spec->preshuffle_b)
    {
        rocke__copy_str(reason,
                        reason_cap,
                        "preshuffle_b=True is gated by the v2 preshuffled-B kernel "
                        "body in gemm_universal; the host-side layout descriptor is "
                        "available today via flatmm_preshuffle_b_layout(spec, N, K)");
        return false;
    }

    if(rocke_flatmm_to_universal_spec(spec, &u) != ROCKE_OK)
    {
        rocke__copy_str(reason, reason_cap, "flatmm spec conversion failed");
        return false;
    }

    base_reason[0] = '\0';
    ok = rocke_gemm_universal_is_valid_spec(&u, arch, base_reason, sizeof(base_reason));
    if(!ok)
    {
        /* "base batched_gemm spec invalid: {why}" */
        char composed[320];
        const char* pfx = "base batched_gemm spec invalid: ";
        size_t pn = strlen(pfx);
        size_t wn = strlen(base_reason);
        if(pn + wn >= sizeof(composed))
        {
            wn = sizeof(composed) - 1 - pn;
        }
        memcpy(composed, pfx, pn);
        memcpy(composed + pn, base_reason, wn);
        composed[pn + wn] = '\0';
        rocke__copy_str(reason, reason_cap, composed);
        return false;
    }

    rocke__copy_str(reason, reason_cap, "ok");
    return true;
}

/* -------------------------------------------------------------- build_flatmm *
 * Python:
 *   ok, why = is_valid_spec(spec, arch)
 *   if not ok: raise ValueError(f"invalid flatmm spec for {arch}: {why}")
 *   return build_batched_gemm(spec.to_batched_spec(), arch)
 *
 * build_batched_gemm re-validates (raising on failure) then calls
 * build_universal_gemm. We mirror by validating once via the FlatMM gate (which
 * subsumes the batched gate) and routing the build to rocke_build_universal_gemm. */
rocke_kernel_def_t*
    rocke_build_flatmm(rocke_ir_builder_t* b, const rocke_flatmm_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        rocke_gemm_universal_spec_t u;
        char reason[320];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx950";
        }

        reason[0] = '\0';
        if(!rocke_flatmm_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            /* invalid flatmm spec for {arch}: {why} -> record on the builder. */
            char msg[480];
            const char* p1 = "invalid flatmm spec for ";
            const char* p2 = ": ";
            size_t a = strlen(arch);
            size_t r = strlen(reason);
            size_t l1 = strlen(p1);
            size_t l2 = strlen(p2);
            size_t off = 0;
            if(l1 + a + l2 + r < sizeof(msg))
            {
                memcpy(msg + off, p1, l1);
                off += l1;
                memcpy(msg + off, arch, a);
                off += a;
                memcpy(msg + off, p2, l2);
                off += l2;
                memcpy(msg + off, reason, r);
                off += r;
                msg[off] = '\0';
            }
            else
            {
                rocke__copy_str(msg, sizeof(msg), "invalid flatmm spec");
            }
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", msg);
            return NULL;
        }

        if(rocke_flatmm_to_universal_spec(spec, &u) != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "flatmm spec conversion failed");
            return NULL;
        }

        return rocke_build_universal_gemm(b, &u, arch);
    });
}

/* ----------------------------------------------------------- build_flatmm_new *
 * Init the builder with spec.kernel_name(), then build. */
rocke_kernel_def_t*
    rocke_build_flatmm_new(rocke_ir_builder_t* b, const rocke_flatmm_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_flatmm_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_flatmm(b, spec, arch);
    });
}

/* -------------------------------------------------------- lower_to_llvm *
 * build + lower to .ll convenience; owns and frees its own IRBuilder. */
rocke_status_t rocke_flatmm_lower_to_llvm(const rocke_flatmm_spec_t* spec,
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
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        rocke__copy_str(err, err_cap, "flatmm lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_flatmm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke__copy_str(err, err_cap, (m != NULL) ? m : "build_flatmm failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

/* --------------------------------------------------------------- flatmm_grid *
 * batched_gemm_grid(batch, m, n, spec): ceil_div_grid((n,tile_n),(m,tile_m),
 * (batch,1)). */
rocke_status_t
    rocke_flatmm_grid(const rocke_flatmm_spec_t* spec, int batch, int m, int n, int out[3])
{
    int totals[3];
    int tiles[3];
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    totals[0] = n;
    tiles[0] = spec->tile.tile_n;
    totals[1] = m;
    tiles[1] = spec->tile.tile_m;
    totals[2] = batch;
    tiles[2] = 1;
    return rocke_ceil_div_grid(totals, tiles, 3, out);
}

/* ---------------------------------------------------------- flatmm_signature *
 * batched_gemm_signature(spec): with FlatMM's fixed fp16 dtype, ptr_dt="fp16".
 *   .ptr(A).ptr(B).ptr(C).scalar(M).scalar(N).scalar(K)
 *   .scalar(stride_a).scalar(stride_b).scalar(stride_c)
 *   (+ .ptr(SortedTokenIds,i32).scalar(slot_size,i32) if active_tile_skip). */
rocke_status_t rocke_flatmm_signature(const rocke_flatmm_spec_t* spec,
                                      rocke_arena_t* arena,
                                      const rocke_sig_entry_t** out_items,
                                      size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t st;

    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
    {
        return st;
    }

    /* ptr_dt = "fp16" (FlatMM rides the f16/fp16 family). */
    rocke_signature_builder_ptr(&sb, "A", "fp16", NULL);
    rocke_signature_builder_ptr(&sb, "B", "fp16", NULL);
    rocke_signature_builder_ptr(&sb, "C", "fp16", NULL);
    rocke_signature_builder_scalar(&sb, "M", "i32");
    rocke_signature_builder_scalar(&sb, "N", "i32");
    rocke_signature_builder_scalar(&sb, "K", "i32");
    rocke_signature_builder_scalar(&sb, "stride_a", "i32");
    rocke_signature_builder_scalar(&sb, "stride_b", "i32");
    rocke_signature_builder_scalar(&sb, "stride_c", "i32");

    if(spec->trait.active_tile_skip)
    {
        rocke_signature_builder_ptr(&sb, "SortedTokenIds", "i32", NULL);
        rocke_signature_builder_scalar(&sb, "slot_size", "i32");
    }

    return rocke_signature_builder_build(&sb, out_items, out_count);
}

/* --------------------------------------------------------- atom introspection *
 * flatmm_atom_shape(spec) -> (warp_tile_m, warp_tile_n, warp_tile_k). */
void rocke_flatmm_atom_shape(const rocke_flatmm_spec_t* spec, int out_mnk[3])
{
    if(spec == NULL || out_mnk == NULL)
    {
        return;
    }
    out_mnk[0] = spec->tile.warp_tile_m;
    out_mnk[1] = spec->tile.warp_tile_n;
    out_mnk[2] = spec->tile.warp_tile_k;
}

/* flatmm_atom(spec) -> mfma_atom("f16", *flatmm_atom_shape(spec)). */
const rocke_mfma_atom_t* rocke_flatmm_atom(const rocke_flatmm_spec_t* spec)
{
    int mnk[3];
    if(spec == NULL)
    {
        return NULL;
    }
    rocke_flatmm_atom_shape(spec, mnk);
    return rocke_mfma_atom("f16", mnk[0], mnk[1], mnk[2]);
}

/* ---------------------------------------------------- preshuffle_b helpers *
 * flatmm_preshuffle_b_spec(spec): PreshuffleBSpec(block_n=tile_n, block_k=tile_k,
 * elem_bytes=2). */
rocke_status_t rocke_flatmm_preshuffle_b_spec(const rocke_flatmm_spec_t* spec,
                                              rocke_preshuffleb_spec_t* out)
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out->block_n = spec->tile.tile_n;
    out->block_k = spec->tile.tile_k;
    out->elem_bytes = 2;
    return ROCKE_OK;
}

/* flatmm_preshuffle_b_layout(spec, n=n, k=k):
 *   host_preshuffle_layout(flatmm_preshuffle_b_spec(spec), n=n, k=k). */
rocke_status_t rocke_flatmm_preshuffle_b_layout(
    const rocke_flatmm_spec_t* spec, int n, int k, int out_shape[4], int out_strides[4])
{
    rocke_preshuffleb_spec_t ps;
    rocke_status_t st = rocke_flatmm_preshuffle_b_spec(spec, &ps);
    if(st != ROCKE_OK)
    {
        return st;
    }
    return rocke_host_preshuffle_layout(&ps, n, k, out_shape, out_strides);
}
