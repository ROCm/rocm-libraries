// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_batched_gemm.c -- C99 port of
 * rocke/instances/common/batched_gemm.py.
 *
 * batched_gemm.py is a thin wrapper around gemm_universal.py: a BatchedGemmSpec
 * value type, a to_universal_spec() conversion that flips batched=True, and a
 * build_batched_gemm() that validates then delegates verbatim to
 * build_universal_gemm. The kernel body (block_id_z as batch index + per-batch
 * stride offsets) lives entirely in the universal builder, which is already
 * ported. This TU therefore only mirrors the wrapper's op/attr ordering:
 *
 *   _default / _finalize         (dataclass defaults + __post_init__)
 *   _data_spec                   (homogeneous A/B/C DataSpec)
 *   to_universal_spec            (wrap, batched=True)
 *   kernel_name                  (== universal kernel_name)
 *   is_valid_spec                (delegates to universal)
 *   build_batched_gemm           (validate -> build_universal_gemm)
 *   build_persistent_batched_gemm(force persistent + rename)
 *   batched_gemm_signature       (SignatureBuilder, +stride params)
 *   batched_gemm_grid            (ceil_div_grid)
 *   lower_to_llvm                (build + lower convenience)
 */

#include <string.h>

#include "rocke/instance_batched_gemm.h"

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

/* ===================================================================== *
 *  rocke_batched_gemm_spec_default / _finalize
 *
 *  Python:
 *      @dataclass(frozen=True)
 *      class BatchedGemmSpec(WarpTileBlockSizeMixin):
 *          name: str
 *          tile: TileSpec
 *          trait: TraitSpec = field(default_factory=TraitSpec)
 *          wave_size: int = 64
 *          block_size: int = 0
 *          batch_size: int = 0
 *          dtype: str = "fp16"
 *          def __post_init__(self): self._init_block_size()
 *
 *  We seed tile/trait from the gemm_universal default so the BatchedGemmSpec's
 *  TileSpec()/TraitSpec() defaults match exactly the dataclasses it reuses.
 * ===================================================================== */
rocke_batched_gemm_spec_t rocke_batched_gemm_spec_default(void)
{
    rocke_batched_gemm_spec_t s;
    rocke_gemm_universal_spec_t base = rocke_gemm_universal_spec_default();

    memset(&s, 0, sizeof(s));
    s.name = NULL; /* required field, no Python default */
    s.tile = base.tile; /* TileSpec() defaults */
    s.trait = base.trait; /* TraitSpec() defaults */
    s.wave_size = 64; /* default 64 */
    s.block_size = 0; /* default 0 => derived at finalize() */
    s.batch_size = 0; /* default 0 (informational) */
    s.dtype = "fp16"; /* default "fp16" */
    return s;
}

void rocke_batched_gemm_spec_finalize(rocke_batched_gemm_spec_t* spec)
{
    if(spec == NULL)
    {
        return;
    }
    /* WarpTileBlockSizeMixin._init_block_size(): only derive when still the
     * 0 sentinel (idempotent). */
    spec->block_size = rocke_warp_tile_init_block_size(
        spec->block_size, spec->tile.warp_m, spec->tile.warp_n, spec->tile.warp_k, spec->wave_size);
}

/* ===================================================================== *
 *  rocke_batched_gemm_data_spec
 *
 *  Python:
 *      def _data_spec(self) -> DataSpec:
 *          dt = "fp16" if self.dtype in ("f16", "fp16") else self.dtype
 *          return DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt)
 *
 *  DataSpec(dtype_a/b/c=dt) leaves dtype_acc / layout at their defaults
 *  ("fp32" / "RCR"); we start from the universal default DataSpec and override
 *  only A/B/C.
 * ===================================================================== */
rocke_gemm_data_spec_t rocke_batched_gemm_data_spec(const rocke_batched_gemm_spec_t* spec)
{
    rocke_gemm_data_spec_t d = rocke_gemm_universal_spec_default().data;
    const char* dt;

    /* dt = "fp16" if dtype in ("f16","fp16") else dtype */
    if(spec != NULL && spec->dtype != NULL
       && (strcmp(spec->dtype, "f16") == 0 || strcmp(spec->dtype, "fp16") == 0))
    {
        dt = "fp16";
    }
    else if(spec != NULL && spec->dtype != NULL)
    {
        dt = spec->dtype;
    }
    else
    {
        dt = "fp16";
    }

    d.dtype_a = dt;
    d.dtype_b = dt;
    d.dtype_c = dt;
    return d;
}

/* ===================================================================== *
 *  rocke_batched_gemm_to_universal_spec
 *
 *  Python:
 *      def to_universal_spec(self) -> UniversalGemmSpec:
 *          return UniversalGemmSpec(
 *              name=self.name, tile=self.tile, trait=self.trait,
 *              data=self._data_spec(), wave_size=self.wave_size,
 *              block_size=self.block_size, batched=True)
 * ===================================================================== */
rocke_gemm_universal_spec_t
    rocke_batched_gemm_to_universal_spec(const rocke_batched_gemm_spec_t* spec)
{
    rocke_gemm_universal_spec_t u = rocke_gemm_universal_spec_default();

    if(spec == NULL)
    {
        return u;
    }
    u.name = spec->name;
    u.tile = spec->tile;
    u.trait = spec->trait;
    u.data = rocke_batched_gemm_data_spec(spec);
    u.wave_size = spec->wave_size;
    u.block_size = spec->block_size;
    u.batched = true;
    return u;
}

/* ===================================================================== *
 *  rocke_batched_gemm_kernel_name
 *
 *  Python:
 *      def kernel_name(self) -> str:
 *          return self.to_universal_spec().kernel_name()
 * ===================================================================== */
rocke_status_t
    rocke_batched_gemm_kernel_name(const rocke_batched_gemm_spec_t* spec, char* out, size_t out_cap)
{
    rocke_gemm_universal_spec_t u;
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    u = rocke_batched_gemm_to_universal_spec(spec);
    return rocke_gemm_universal_kernel_name(&u, out, out_cap);
}

/* ===================================================================== *
 *  rocke_batched_gemm_is_valid_spec
 *
 *  Python:
 *      def is_valid_spec(spec, arch="gfx950"):
 *          return is_valid_gemm_spec(spec.to_universal_spec(), arch=arch)
 * ===================================================================== */
bool rocke_batched_gemm_is_valid_spec(const rocke_batched_gemm_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap)
{
    rocke_gemm_universal_spec_t u;
    if(spec == NULL)
    {
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    u = rocke_batched_gemm_to_universal_spec(spec);
    return rocke_gemm_universal_is_valid_spec(&u, arch, reason, reason_cap);
}

/* ===================================================================== *
 *  rocke_build_batched_gemm
 *
 *  Python:
 *      def build_batched_gemm(spec, arch="gfx950"):
 *          universal = spec.to_universal_spec()
 *          ok, why = is_valid_gemm_spec(universal, arch=arch)
 *          if not ok: raise ValueError(f"invalid batched_gemm spec ...: {why}")
 *          return build_universal_gemm(universal, arch=arch)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_batched_gemm(rocke_ir_builder_t* b,
                                             const rocke_batched_gemm_spec_t* spec,
                                             const char* arch)
{
    rocke_gemm_universal_spec_t universal;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* universal = spec.to_universal_spec() */
    universal = rocke_batched_gemm_to_universal_spec(spec);

    /* ok, why = is_valid_gemm_spec(universal, arch=arch); if not ok: raise */
    if(!rocke_gemm_universal_is_valid_spec(&universal, arch, NULL, 0))
    {
        return NULL;
    }

    /* return build_universal_gemm(universal, arch=arch) */
    return rocke_build_universal_gemm(b, &universal, arch);
}

/* ===================================================================== *
 *  rocke_build_batched_gemm_new -- init the builder with spec.kernel_name(),
 *  then build (mirrors rocke_build_universal_gemm_new).
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_batched_gemm_new(rocke_ir_builder_t* b,
                                                 const rocke_batched_gemm_spec_t* spec,
                                                 const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[512];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_batched_gemm_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_batched_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  rocke_build_persistent_batched_gemm
 *
 *  Python:
 *      def build_persistent_batched_gemm(spec, arch="gfx950"):
 *          universal = spec.to_universal_spec()
 *          persistent_trait = TraitSpec(
 *              pipeline=universal.trait.pipeline,
 *              scheduler=universal.trait.scheduler,
 *              epilogue=universal.trait.epilogue,
 *              pad_m=..., pad_n=..., pad_k=...,
 *              persistent=True,
 *              chiplet_swizzle=..., chiplet_wgm=..., chiplet_num_xcds=...,
 *              chiplet_chunk_size=..., waves_per_eu=...)
 *          persistent_universal = UniversalGemmSpec(
 *              name=universal.name + "_persistent",
 *              tile=universal.tile, trait=persistent_trait,
 *              data=universal.data, wave_size=universal.wave_size,
 *              block_size=universal.block_size, batched=universal.batched)
 *          ok, why = is_valid_gemm_spec(persistent_universal, arch=arch)
 *          if not ok: raise ValueError(...)
 *          return build_universal_gemm(persistent_universal, arch=arch)
 *
 *  The Python persistent_trait is a *fresh* TraitSpec that copies only the
 *  listed fields from universal.trait; every other TraitSpec field falls back
 *  to its dataclass default. We mirror that exactly: start from the default
 *  trait, copy the named fields, force persistent=True.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_persistent_batched_gemm(rocke_ir_builder_t* b,
                                                        const rocke_batched_gemm_spec_t* spec,
                                                        const char* arch)
{
    rocke_gemm_universal_spec_t universal;
    rocke_gemm_universal_spec_t persistent_universal;
    rocke_gemm_trait_spec_t persistent_trait;
    char name_buf[512];

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* universal = spec.to_universal_spec() */
    universal = rocke_batched_gemm_to_universal_spec(spec);

    /* persistent_trait = TraitSpec(<copied fields>, persistent=True) -- fresh
     * dataclass: unlisted fields take their TraitSpec() defaults. */
    persistent_trait = rocke_gemm_universal_spec_default().trait;
    persistent_trait.pipeline = universal.trait.pipeline;
    persistent_trait.scheduler = universal.trait.scheduler;
    persistent_trait.epilogue = universal.trait.epilogue;
    persistent_trait.pad_m = universal.trait.pad_m;
    persistent_trait.pad_n = universal.trait.pad_n;
    persistent_trait.pad_k = universal.trait.pad_k;
    persistent_trait.persistent = true;
    persistent_trait.chiplet_swizzle = universal.trait.chiplet_swizzle;
    persistent_trait.chiplet_wgm = universal.trait.chiplet_wgm;
    persistent_trait.chiplet_num_xcds = universal.trait.chiplet_num_xcds;
    persistent_trait.chiplet_chunk_size = universal.trait.chiplet_chunk_size;
    persistent_trait.waves_per_eu_set = universal.trait.waves_per_eu_set;
    persistent_trait.waves_per_eu = universal.trait.waves_per_eu;

    /* persistent_universal = UniversalGemmSpec(name + "_persistent", ...) */
    persistent_universal = universal;
    persistent_universal.trait = persistent_trait;

    /* name=universal.name + "_persistent" (arena-stable storage on the builder
     * is not needed here: name_buf lives until build returns, and the universal
     * builder copies the name during ir_builder_init in _new(); but this entry
     * builds into the caller-provided `b`, so we keep name_buf alive for the
     * whole call). */
    name_buf[0] = '\0';
    if(universal.name != NULL)
    {
        size_t n = strlen(universal.name);
        const char* suf = "_persistent";
        size_t sn = strlen(suf);
        if(n + sn + 1 > sizeof(name_buf))
        {
            return NULL;
        }
        memcpy(name_buf, universal.name, n);
        memcpy(name_buf + n, suf, sn);
        name_buf[n + sn] = '\0';
        persistent_universal.name = name_buf;
    }

    /* ok, why = is_valid_gemm_spec(persistent_universal, arch); if not ok: raise */
    if(!rocke_gemm_universal_is_valid_spec(&persistent_universal, arch, NULL, 0))
    {
        return NULL;
    }

    /* return build_universal_gemm(persistent_universal, arch=arch) */
    return rocke_build_universal_gemm(b, &persistent_universal, arch);
}

/* ===================================================================== *
 *  rocke_batched_gemm_signature
 *
 *  Python:
 *      def batched_gemm_signature(spec):
 *          ptr_dt = spec.dtype if spec.dtype in ("f16","fp16","bf16") else "f16"
 *          sig = (SignatureBuilder()
 *                 .ptr("A", ptr_dt).ptr("B", ptr_dt).ptr("C", ptr_dt)
 *                 .scalar("M","i32").scalar("N","i32").scalar("K","i32")
 *                 .scalar("stride_a","i32").scalar("stride_b","i32")
 *                 .scalar("stride_c","i32"))
 *          if spec.trait.active_tile_skip:
 *              sig = sig.ptr("SortedTokenIds","i32").scalar("slot_size","i32")
 *          return sig.build()
 * ===================================================================== */
rocke_status_t rocke_batched_gemm_signature(rocke_arena_t* arena,
                                            const rocke_batched_gemm_spec_t* spec,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t st;
    const char* ptr_dt;

    if(arena == NULL || spec == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* ptr_dt = spec.dtype if dtype in ("f16","fp16","bf16") else "f16" */
    if(spec->dtype != NULL
       && (strcmp(spec->dtype, "f16") == 0 || strcmp(spec->dtype, "fp16") == 0
           || strcmp(spec->dtype, "bf16") == 0))
    {
        ptr_dt = spec->dtype;
    }
    else
    {
        ptr_dt = "f16";
    }

    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
    {
        return st;
    }

    /* SignatureBuilder().ptr(A/B/C, ptr_dt).scalar(M/N/K).scalar(stride_a/b/c) */
    rocke_signature_builder_ptr(&sb, "A", ptr_dt, NULL);
    rocke_signature_builder_ptr(&sb, "B", ptr_dt, NULL);
    rocke_signature_builder_ptr(&sb, "C", ptr_dt, NULL);
    rocke_signature_builder_scalar(&sb, "M", "i32");
    rocke_signature_builder_scalar(&sb, "N", "i32");
    rocke_signature_builder_scalar(&sb, "K", "i32");
    rocke_signature_builder_scalar(&sb, "stride_a", "i32");
    rocke_signature_builder_scalar(&sb, "stride_b", "i32");
    rocke_signature_builder_scalar(&sb, "stride_c", "i32");

    /* if spec.trait.active_tile_skip:
     *     sig.ptr("SortedTokenIds","i32").scalar("slot_size","i32") */
    if(spec->trait.active_tile_skip)
    {
        rocke_signature_builder_ptr(&sb, "SortedTokenIds", "i32", NULL);
        rocke_signature_builder_scalar(&sb, "slot_size", "i32");
    }

    /* return sig.build() */
    return rocke_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  rocke_batched_gemm_grid
 *
 *  Python:
 *      def batched_gemm_grid(batch, m, n, spec):
 *          t = spec.tile
 *          return ceil_div_grid((n, t.tile_n), (m, t.tile_m), (batch, 1))
 * ===================================================================== */
rocke_status_t rocke_batched_gemm_grid(
    int batch, int m, int n, const rocke_batched_gemm_spec_t* spec, int out[3])
{
    const rocke_gemm_tile_spec_t* t;
    int totals[3];
    int tiles[3];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    t = &spec->tile;
    totals[0] = n;
    tiles[0] = t->tile_n;
    totals[1] = m;
    tiles[1] = t->tile_m;
    totals[2] = batch;
    tiles[2] = 1;

    return rocke_ceil_div_grid(totals, tiles, 3, out);
}

/* ===================================================================== *
 *  rocke_batched_gemm_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder (mirrors gemm_universal's variant).
 * ===================================================================== */
rocke_status_t rocke_batched_gemm_lower_to_llvm(const rocke_batched_gemm_spec_t* spec,
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
        if(err != NULL && err_cap > 0)
        {
            const char* m = "lower_to_llvm: null spec/out";
            size_t n = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_batched_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_batched_gemm failed";
            }
            n = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
