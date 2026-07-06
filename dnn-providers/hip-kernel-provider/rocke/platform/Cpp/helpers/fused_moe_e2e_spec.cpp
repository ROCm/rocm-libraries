// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.fused_moe_e2e_spec.c -- C99 port of the spec /
 * tile-policy surface of rocke/instances/common/fused_moe_e2e.py.
 *
 * Byte-faithful translation of the GEMM tile factories, the launch-arch / dtype
 * helpers, and the FusedMoeForwardSpec value type with the spec-derivation
 * methods that have a C counterpart. See the header for the symbol map and the
 * documented stub points (the device-arch probe in _resolve_launch_arch, and
 * the to_sort_spec / to_fused_moe_spec methods whose target instance ports do
 * not yet exist in the C library).
 */

#include "rocke/helper_rocke.helpers.fused_moe_e2e_spec.h"

#include <string.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

/* ----------------------------------------------------------- GEMM tile policy *
 *
 * Each factory mirrors the Python TileSpec(...) call exactly. The Python
 * TileSpec leaves warp_k at its default (1); we fill it explicitly here so the
 * struct is fully populated regardless of the gemm_universal default. */

static rocke_gemm_tile_spec_t rocke_fmoe_make_tile(int tile_m,
                                                   int tile_n,
                                                   int tile_k,
                                                   int warp_m,
                                                   int warp_n,
                                                   int warp_tile_m,
                                                   int warp_tile_n,
                                                   int warp_tile_k)
{
    rocke_gemm_tile_spec_t t;
    memset(&t, 0, sizeof(t));
    t.tile_m = tile_m;
    t.tile_n = tile_n;
    t.tile_k = tile_k;
    t.warp_m = warp_m;
    t.warp_n = warp_n;
    t.warp_k = 1; /* TileSpec default */
    t.warp_tile_m = warp_tile_m;
    t.warp_tile_n = warp_tile_n;
    t.warp_tile_k = warp_tile_k;
    return t;
}

rocke_gemm_tile_spec_t rocke_fmoe_default_gemm_tile(void)
{
    /* TileSpec(tile_m=32, tile_n=128, tile_k=64, warp_m=1, warp_n=2,
     *          warp_tile_m=32, warp_tile_n=32, warp_tile_k=16) */
    return rocke_fmoe_make_tile(32, 128, 64, 1, 2, 32, 32, 16);
}

rocke_gemm_tile_spec_t rocke_fmoe_default_bf16_gemm_tile(void)
{
    /* TileSpec(tile_m=32, tile_n=32, tile_k=32, warp_m=2, warp_n=2,
     *          warp_tile_m=16, warp_tile_n=16, warp_tile_k=32) */
    return rocke_fmoe_make_tile(32, 32, 32, 2, 2, 16, 16, 32);
}

rocke_gemm_tile_spec_t rocke_fmoe_default_gemm_tile_gfx942(void)
{
    /* TileSpec(tile_m=32, tile_n=128, tile_k=64, warp_m=1, warp_n=2,
     *          warp_tile_m=32, warp_tile_n=32, warp_tile_k=8) */
    return rocke_fmoe_make_tile(32, 128, 64, 1, 2, 32, 32, 8);
}

rocke_gemm_tile_spec_t rocke_fmoe_default_bf16_gemm_tile_gfx942(void)
{
    /* TileSpec(tile_m=32, tile_n=32, tile_k=32, warp_m=2, warp_n=2,
     *          warp_tile_m=16, warp_tile_n=16, warp_tile_k=16) */
    return rocke_fmoe_make_tile(32, 32, 32, 2, 2, 16, 16, 16);
}

rocke_gemm_tile_spec_t rocke_fmoe_large_batch_gemm_tile(void)
{
    /* TileSpec(tile_m=64, tile_n=128, tile_k=64, warp_m=2, warp_n=2,
     *          warp_tile_m=32, warp_tile_n=32, warp_tile_k=16) */
    return rocke_fmoe_make_tile(64, 128, 64, 2, 2, 32, 32, 16);
}

rocke_gemm_tile_spec_t rocke_fmoe_sparse_batch_gemm_tile(void)
{
    /* TileSpec(tile_m=32, tile_n=128, tile_k=128, warp_m=1, warp_n=2,
     *          warp_tile_m=32, warp_tile_n=32, warp_tile_k=16) */
    return rocke_fmoe_make_tile(32, 128, 128, 1, 2, 32, 32, 16);
}

rocke_gemm_tile_spec_t rocke_fmoe_sparse_batch_gemm_tile_gfx942(void)
{
    /* TileSpec(tile_m=32, tile_n=128, tile_k=128, warp_m=1, warp_n=2,
     *          warp_tile_m=32, warp_tile_n=32, warp_tile_k=8) */
    return rocke_fmoe_make_tile(32, 128, 128, 1, 2, 32, 32, 8);
}

bool rocke_fmoe_tile_eq(const rocke_gemm_tile_spec_t* a, const rocke_gemm_tile_spec_t* b)
{
    if(a == NULL || b == NULL)
    {
        return a == b;
    }
    /* TileSpec.__eq__ (dataclass): all fields equal. */
    return a->tile_m == b->tile_m && a->tile_n == b->tile_n && a->tile_k == b->tile_k
           && a->warp_m == b->warp_m && a->warp_n == b->warp_n && a->warp_k == b->warp_k
           && a->warp_tile_m == b->warp_tile_m && a->warp_tile_n == b->warp_tile_n
           && a->warp_tile_k == b->warp_tile_k;
}

/* ------------------------------------------------------------ arch / dtype */

const char* rocke_fmoe_resolve_launch_arch(const char* arch)
{
    /* Python:
     *   if arch is not None: return arch
     *   try: dev = get_device_arch(); if dev: return dev
     *   except Exception: pass
     *   return "gfx950"
     *
     * Explicit arch wins. */
    if(arch != NULL)
    {
        return arch;
    }
    /* TODO(port): the Python probes the running HIP device via
     * rocke.runtime.hip_module.get_device_arch(). This codegen-only library has
     * no HIP runtime, so we take the Python no-device fallback directly. A
     * future port can wire a device probe here and return its result when
     * non-empty. */
    return "gfx950";
}

const char* rocke_fmoe_gemm_dtype_to_universal(const char* dtype)
{
    if(dtype == NULL)
    {
        return NULL;
    }
    if(strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0)
    {
        return "fp16";
    }
    if(strcmp(dtype, "bf16") == 0)
    {
        return "bf16";
    }
    /* Python: raise ValueError(...). Signal via NULL. */
    return NULL;
}

const char* rocke_b_fmoe_gemm_dtype_to_universal(rocke_ir_builder_t* b, const char* dtype)
{
    const char* u;

    if(!rocke_i_live(b))
    {
        return NULL;
    }
    u = rocke_fmoe_gemm_dtype_to_universal(dtype);
    if(u == NULL)
    {
        /* Mirror: f"unsupported gemm dtype {dtype!r}; expected f16 / bf16" with
         * Python's single-quote repr (None for a NULL string). */
        return (const char*)rocke_i_set_err(b,
                                            ROCKE_ERR_VALUE,
                                            "unsupported gemm dtype %s%s%s; expected f16 / bf16",
                                            dtype ? "'" : "",
                                            dtype ? dtype : "None",
                                            dtype ? "'" : "");
    }
    return u;
}

int rocke_fmoe_ensure_2byte_dtype(const char* dtype)
{
    if(dtype == NULL)
    {
        return -1;
    }
    if(strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0 || strcmp(dtype, "bf16") == 0)
    {
        return 2;
    }
    /* Python: raise ValueError(...). Signal via -1. */
    return -1;
}

int rocke_b_fmoe_ensure_2byte_dtype(rocke_ir_builder_t* b, const char* dtype)
{
    int n;

    if(!rocke_i_live(b))
    {
        return -1;
    }
    n = rocke_fmoe_ensure_2byte_dtype(dtype);
    if(n < 0)
    {
        /* Mirror: f"only 2-byte activation dtypes supported (got {dtype!r})". */
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "only 2-byte activation dtypes supported (got %s%s%s)",
                        dtype ? "'" : "",
                        dtype ? dtype : "None",
                        dtype ? "'" : "");
        return -1;
    }
    return n;
}

/* ---------------------------------------------------------- FusedMoeForwardSpec
 */

rocke_fmoe_forward_spec_t rocke_fmoe_forward_spec_default(void)
{
    rocke_fmoe_forward_spec_t s;
    memset(&s, 0, sizeof(s));

    /* required shape -- no Python default (left at 0; the caller must set). */
    s.tokens = 0;
    s.experts = 0;
    s.topk = 0;
    s.hidden = 0;
    s.intermediate = 0;
    s.dtype = "f16";

    s.streaming_block_size = 256;
    s.streaming_vec = 8;
    s.sort_block_size = 64;
    s.router_block_size = 64;

    s.gemm_tile = rocke_fmoe_default_gemm_tile(); /* field(default_factory=...) */
    s.arch = NULL; /* Optional[str] = None       */
    s.name = "rocke_fused_moe_forward";

    s.use_experimental_fused_gate_up_silu = false;
    s.use_experimental_interleaved_gate_up_silu = true;
    s.use_experimental_fused_down_reduce = true;
    s.use_experimental_static_scatter_gather = false;
    s.preshuffle_w_down = false;
    s.preshuffle_w_gate_up_packed = false;
    s.preshuffle_w_gate_up_interleaved = false;
    s.use_grouped_gemm = true;
    s.active_tile_skip_gemms = true;
    return s;
}

int rocke_fmoe_forward_spec_total_pairs(const rocke_fmoe_forward_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    /* @property total_pairs: return self.tokens * self.topk */
    return spec->tokens * spec->topk;
}

rocke_topk_softmax_spec_t
    rocke_fmoe_forward_spec_to_topk_softmax_spec(const rocke_fmoe_forward_spec_t* spec)
{
    /* TopkSoftmaxSpec(n_per_row=experts, k=topk, dtype="f32", out_dtype="f32",
     *                 block_size=router_block_size).
     *
     * Start from the C default so the fields the Python omits (name,
     * cross_wave_argmax) carry the TopkSoftmaxSpec dataclass defaults. */
    rocke_topk_softmax_spec_t t = rocke_topk_softmax_spec_default();
    if(spec != NULL)
    {
        t.n_per_row = spec->experts;
        t.k = spec->topk;
        t.dtype = "f32";
        t.out_dtype = "f32";
        t.block_size = spec->router_block_size;
    }
    return t;
}

/* Compose "<name>_<suffix>" into out (capacity cap). Returns ROCKE_OK or
 * ROCKE_ERR_VALUE on a NULL arg / overflow. Mirrors Python f-string composition;
 * a NULL name component is rendered as "None" (repr of None) only defensively --
 * the dataclass default name is always set. */
static rocke_status_t
    rocke_fmoe_compose_name(char* out, size_t cap, const char* base, const char* suffix)
{
    size_t bl, sl, need;
    const char* b = base ? base : "None";
    const char* s = suffix ? suffix : "None";

    if(out == NULL || cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    bl = strlen(b);
    sl = strlen(s);
    need = bl + 1 /* '_' */ + sl + 1 /* NUL */;
    if(need > cap)
    {
        return ROCKE_ERR_VALUE;
    }
    memcpy(out, b, bl);
    out[bl] = '_';
    memcpy(out + bl + 1, s, sl);
    out[bl + 1 + sl] = '\0';
    return ROCKE_OK;
}

rocke_status_t rocke_fmoe_forward_spec_to_gemm_spec(const rocke_fmoe_forward_spec_t* spec,
                                                    const char* name_suffix,
                                                    char* name_out,
                                                    size_t name_cap,
                                                    rocke_grouped_gemm_spec_t* out_spec)
{
    rocke_grouped_gemm_spec_t g;
    const char* dt;
    rocke_status_t st;

    if(spec == NULL || out_spec == NULL || name_out == NULL || name_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    /* dtype=_gemm_dtype_to_universal(self.dtype) -- ValueError on unsupported. */
    dt = rocke_fmoe_gemm_dtype_to_universal(spec->dtype);
    if(dt == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* name=f"{self.name}_{name_suffix}" */
    st = rocke_fmoe_compose_name(name_out, name_cap, spec->name, name_suffix);
    if(st != ROCKE_OK)
    {
        return st;
    }

    /* GroupedGemmSpec(name=, tile=gemm_tile, trait=TraitSpec(pad_m=True),
     *                 dtype=dt). Start from the C default so the unspecified
     * trait knobs carry the TraitSpec() defaults; only pad_m is forced True. */
    g = rocke_grouped_gemm_spec_default();
    g.name = name_out;
    g.tile = spec->gemm_tile;
    g.trait.pad_m = true;
    g.dtype = dt;
    *out_spec = g;
    return ROCKE_OK;
}

/* Shared body for to_batched_gemm_spec[_preshuffle_b]: both compose the same
 * "<name>_batched_gemm" name and a TraitSpec(pad_m=True, pad_n=True[,
 * preshuffle_b=True]) trait. */
static rocke_status_t rocke_fmoe_make_batched_spec(const rocke_fmoe_forward_spec_t* spec,
                                                   char* name_out,
                                                   size_t name_cap,
                                                   bool preshuffle_b,
                                                   rocke_batched_gemm_spec_t* out_spec)
{
    rocke_batched_gemm_spec_t bspec;
    const char* dt;
    rocke_status_t st;

    if(spec == NULL || out_spec == NULL || name_out == NULL || name_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    dt = rocke_fmoe_gemm_dtype_to_universal(spec->dtype);
    if(dt == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* name=f"{self.name}_batched_gemm" */
    st = rocke_fmoe_compose_name(name_out, name_cap, spec->name, "batched_gemm");
    if(st != ROCKE_OK)
    {
        return st;
    }

    bspec = rocke_batched_gemm_spec_default();
    bspec.name = name_out;
    bspec.tile = spec->gemm_tile;
    bspec.trait.pad_m = true;
    bspec.trait.pad_n = true;
    bspec.trait.preshuffle_b = preshuffle_b;
    bspec.dtype = dt;
    /* Mirror Python BatchedGemmSpec.__post_init__ (WarpTileBlockSizeMixin):
     * derive block_size from the warp grid so the lowered kernel has a valid
     * launch geometry. The Python dataclass runs this on construction; the C
     * value-type returns an un-finalized spec otherwise (block_size==0), which
     * fails the downstream build. */
    rocke_batched_gemm_spec_finalize(&bspec);
    *out_spec = bspec;
    return ROCKE_OK;
}

rocke_status_t rocke_fmoe_forward_spec_to_batched_gemm_spec(const rocke_fmoe_forward_spec_t* spec,
                                                            char* name_out,
                                                            size_t name_cap,
                                                            rocke_batched_gemm_spec_t* out_spec)
{
    /* trait=TraitSpec(pad_m=True, pad_n=True) */
    return rocke_fmoe_make_batched_spec(spec, name_out, name_cap, false, out_spec);
}

rocke_status_t
    rocke_fmoe_forward_spec_to_batched_gemm_spec_preshuffle_b(const rocke_fmoe_forward_spec_t* spec,
                                                              char* name_out,
                                                              size_t name_cap,
                                                              rocke_batched_gemm_spec_t* out_spec)
{
    /* trait=TraitSpec(pad_m=True, pad_n=True, preshuffle_b=True) */
    return rocke_fmoe_make_batched_spec(spec, name_out, name_cap, true, out_spec);
}
