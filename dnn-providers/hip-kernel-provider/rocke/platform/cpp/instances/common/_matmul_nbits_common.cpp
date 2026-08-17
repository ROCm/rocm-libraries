// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/instances/common/_matmul_nbits_common.py (Milestone-1
 * surface: spec, validator, signature, grid, 64-row outer-loop driver).
 *
 * See rocke/helper_rocke.instances.common._matmul_nbits_common.h for the symbol
 * map. None of these routines emit IR; they reuse the canonical sibling helpers
 * so the produced block_size / kernel name / verdict / signature / grid are
 * byte-faithful to Python.
 */
#include "rocke/helper_rocke.instances.common._matmul_nbits_common.h"

#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_* (ArchTarget)    */
#include "rocke/helper_rocke.helpers.spec.h" /* derive_block_size, name_join */

/* WMMA atom the dequant-to-fp16 path feeds (module-level Python constants). */
static const char* const ROCKE__WMMA_C_DTYPE = "fp32";
static const char* const ROCKE__WMMA_AB_DTYPE = "fp16";

/* ===================================================================== *
 *  Module constants: FAMILIES / SUPPORTED_ARCHES
 * ===================================================================== */

const char* const* rocke_matmul_nbits_families(void)
{
    /* FAMILIES = ("large_n", "skinny_n", "decode_gemv"); NULL-terminated. */
    static const char* const families[ROCKE_MATMUL_NBITS_FAMILIES_COUNT + 1]
        = {"large_n", "skinny_n", "decode_gemv", NULL};
    return families;
}

bool rocke_matmul_nbits_family_known(const char* family)
{
    const char* const* f;
    if(family == NULL)
    {
        return false;
    }
    for(f = rocke_matmul_nbits_families(); *f != NULL; ++f)
    {
        if(strcmp(*f, family) == 0)
        {
            return true;
        }
    }
    return false;
}

bool rocke_matmul_nbits_arch_supported(const char* arch)
{
    /* SUPPORTED_ARCHES = frozenset({"gfx1151", "gfx1201"}). */
    if(arch == NULL)
    {
        return false;
    }
    return strcmp(arch, "gfx1151") == 0 || strcmp(arch, "gfx1201") == 0;
}

/* ===================================================================== *
 *  _scale_wire_dtype
 * ===================================================================== */

rocke_status_t rocke_matmul_nbits_scale_wire_dtype(const char* scale_dtype, const char** out)
{
    static const char* const F16 = "f16";
    static const char* const F32 = "f32";
    if(scale_dtype == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(strcmp(scale_dtype, "fp16") == 0 || strcmp(scale_dtype, "f16") == 0)
    {
        if(out != NULL)
        {
            *out = F16;
        }
        return ROCKE_OK;
    }
    if(strcmp(scale_dtype, "fp32") == 0 || strcmp(scale_dtype, "f32") == 0)
    {
        if(out != NULL)
        {
            *out = F32;
        }
        return ROCKE_OK;
    }
    return ROCKE_ERR_VALUE;
}

/* ===================================================================== *
 *  MatMulNBitsSpec: default / finalize / kernel_name
 * ===================================================================== */

rocke_matmul_nbits_spec_t rocke_matmul_nbits_spec_default(void)
{
    rocke_matmul_nbits_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = NULL;
    s.N = 0;
    s.K = 0;
    /* TileSpec defaults: warp_k=1, warp_tile = 32x32x16 (the rest are required
     * geometry the caller sets). Mirror the TileSpec field defaults. */
    s.tile.tile_m = 0;
    s.tile.tile_n = 0;
    s.tile.tile_k = 0;
    s.tile.warp_m = 0;
    s.tile.warp_n = 0;
    s.tile.warp_k = 1;
    s.tile.warp_tile_m = 32;
    s.tile.warp_tile_n = 32;
    s.tile.warp_tile_k = 16;
    s.group_size = ROCKE_MATMUL_NBITS_V1_GROUP_SIZE;
    s.seq_len_tile = 64;
    s.wave_size = 32;
    s.block_size = 0;
    s.scale_dtype = "fp16";
    s.zero_points = false;
    s.packing = "row_k_contiguous";
    s.family = "large_n";
    s.optimized = false;
    return s;
}

void rocke_matmul_nbits_spec_finalize(rocke_matmul_nbits_spec_t* spec)
{
    if(spec == NULL)
    {
        return;
    }
    /* WarpTileBlockSizeMixin._init_block_size(): only derive when still 0. */
    spec->block_size = rocke_warp_tile_init_block_size(
        spec->block_size, spec->tile.warp_m, spec->tile.warp_n, spec->tile.warp_k, spec->wave_size);
}

rocke_status_t
    rocke_matmul_nbits_kernel_name(const rocke_matmul_nbits_spec_t* spec, char* out, size_t out_cap)
{
    char part_nk[64];
    char part_g[32];
    char part_t[64];
    char part_w[64];
    char part_wt[64];
    char part_s[16];
    const char* parts[8];
    const char* flag_names[1];
    int flag_on[1];
    const char* scale_wire = NULL;
    const rocke_gemm_tile_spec_t* t;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_matmul_nbits_scale_wire_dtype(spec->scale_dtype, &scale_wire) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    t = &spec->tile;

    /* kernel_name_join(
     *     self.name, self.family, _WMMA_AB_DTYPE ("f16"),
     *     f"N{N}K{K}", f"g{group_size}",
     *     f"t{tm}x{tn}x{tk}", f"w{wm}x{wn}x{wk}", f"wt{wtm}x{wtn}x{wtk}",
     *     f"s{scale_wire}", flags={"zp": zero_points}) */
    snprintf(part_nk, sizeof(part_nk), "N%dK%d", spec->N, spec->K);
    snprintf(part_g, sizeof(part_g), "g%d", spec->group_size);
    snprintf(part_t, sizeof(part_t), "t%dx%dx%d", t->tile_m, t->tile_n, t->tile_k);
    snprintf(part_w, sizeof(part_w), "w%dx%dx%d", t->warp_m, t->warp_n, t->warp_k);
    snprintf(
        part_wt, sizeof(part_wt), "wt%dx%dx%d", t->warp_tile_m, t->warp_tile_n, t->warp_tile_k);
    snprintf(part_s, sizeof(part_s), "s%s", scale_wire);

    parts[0] = spec->family;
    /* Python passes _WMMA_AB_DTYPE = "fp16"; kernel_name_join emits the part
     * verbatim (no dtype canonicalisation in name_join), so keep it "fp16". */
    parts[1] = ROCKE__WMMA_AB_DTYPE;
    parts[2] = part_nk;
    parts[3] = part_g;
    parts[4] = part_t;
    parts[5] = part_w;
    parts[6] = part_wt;
    parts[7] = part_s;

    flag_names[0] = "zp";
    flag_on[0] = spec->zero_points ? 1 : 0;

    return rocke_kernel_name_join(spec->name, parts, 8, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* ===================================================================== *
 *  validate_common_spec
 * ===================================================================== */

#define ROCKE_NBITS_REJECT(...)                        \
    do                                                 \
    {                                                  \
        if(reason != NULL && reason_cap > 0)           \
        {                                              \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        }                                              \
        return false;                                  \
    } while(0)

bool rocke_matmul_nbits_validate_common_spec(const rocke_matmul_nbits_spec_t* spec,
                                             const char* arch,
                                             char* reason,
                                             size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const char* scale_wire = NULL;
    const rocke_gemm_tile_spec_t* t;
    const char* family_str;
    int expected_bs;

    if(spec == NULL)
    {
        ROCKE_NBITS_REJECT("spec is NULL");
    }
    if(arch == NULL)
    {
        arch = ROCKE_MATMUL_NBITS_V1_ARCH;
    }

    /* target = ArchTarget.from_gfx(arch)  (KeyError -> reject with str(e)). */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        ROCKE_NBITS_REJECT("'%s'", arch);
    }

    /* v1 contract gates. */
    if(!rocke_matmul_nbits_arch_supported(arch))
    {
        /* sorted(SUPPORTED_ARCHES) -> ['gfx1151', 'gfx1201']. */
        ROCKE_NBITS_REJECT("matmul_nbits v1 supports ['gfx1151', 'gfx1201'] only (got '%s')", arch);
    }
    if(!rocke_matmul_nbits_family_known(spec->family))
    {
        ROCKE_NBITS_REJECT("unknown family '%s'; expected one of ('large_n', 'skinny_n', "
                           "'decode_gemv')",
                           spec->family != NULL ? spec->family : "(null)");
    }
    if(spec->group_size != ROCKE_MATMUL_NBITS_V1_GROUP_SIZE)
    {
        ROCKE_NBITS_REJECT("matmul_nbits v1 fixes group_size=%d (got %d)",
                           ROCKE_MATMUL_NBITS_V1_GROUP_SIZE,
                           spec->group_size);
    }
    if(spec->zero_points)
    {
        ROCKE_NBITS_REJECT("matmul_nbits v1 is signed-symmetric int4 only; zero_points not "
                           "yet supported");
    }
    if(spec->packing == NULL || strcmp(spec->packing, "row_k_contiguous") != 0)
    {
        ROCKE_NBITS_REJECT("matmul_nbits v1 ships packing='row_k_contiguous' only (got '%s')",
                           spec->packing != NULL ? spec->packing : "(null)");
    }
    if(rocke_matmul_nbits_scale_wire_dtype(spec->scale_dtype, &scale_wire) != ROCKE_OK)
    {
        ROCKE_NBITS_REJECT("scale_dtype must be fp16 / fp32, got '%s'",
                           spec->scale_dtype != NULL ? spec->scale_dtype : "(null)");
    }

    /* Problem-size sanity. */
    if(spec->N <= 0 || spec->K <= 0)
    {
        ROCKE_NBITS_REJECT("N / K must be positive (got N=%d, K=%d)", spec->N, spec->K);
    }
    if(spec->K % spec->group_size)
    {
        ROCKE_NBITS_REJECT(
            "K (%d) must be divisible by group_size (%d)", spec->K, spec->group_size);
    }

    /* Wave geometry must match the target. */
    if(spec->wave_size != target->wave_size)
    {
        ROCKE_NBITS_REJECT(
            "spec wave_size %d != %s wave_size %d", spec->wave_size, arch, target->wave_size);
    }

    t = &spec->tile;
    family_str = spec->family;

    /* The decode-GEMV family is a scalar (no-WMMA) body: skip the atom lookup
     * and tile-divisibility gates. */
    if(strcmp(family_str, "decode_gemv") != 0)
    {
        /* family = "wmma" if wave_size == 32 else "mma". */
        const char* mma_family = (target->wave_size == 32) ? "wmma" : "mma";

        if(!rocke_archtarget_supports_dtype_combo(
               target, ROCKE__WMMA_AB_DTYPE, ROCKE__WMMA_AB_DTYPE, ROCKE__WMMA_C_DTYPE, mma_family))
        {
            ROCKE_NBITS_REJECT("unsupported matmul_nbits dtype fp16 on %s", arch);
        }
        if(rocke_archtarget_op_for_shape(target,
                                         mma_family,
                                         ROCKE__WMMA_AB_DTYPE,
                                         ROCKE__WMMA_AB_DTYPE,
                                         ROCKE__WMMA_C_DTYPE,
                                         t->warp_tile_m,
                                         t->warp_tile_n,
                                         t->warp_tile_k)
           == NULL)
        {
            ROCKE_NBITS_REJECT("unsupported fp16 warp_tile (%d, %d, %d) on %s",
                               t->warp_tile_m,
                               t->warp_tile_n,
                               t->warp_tile_k,
                               arch);
        }

        /* Geometry divisibility. */
        if(t->tile_m % (t->warp_m * t->warp_tile_m))
        {
            ROCKE_NBITS_REJECT("tile_m not divisible by warp_m * warp_tile_m");
        }
        if(t->tile_n % (t->warp_n * t->warp_tile_n))
        {
            ROCKE_NBITS_REJECT("tile_n not divisible by warp_n * warp_tile_n");
        }
        if(t->tile_k % t->warp_tile_k)
        {
            ROCKE_NBITS_REJECT("tile_k not divisible by warp_tile_k");
        }
        if(spec->N % t->tile_n)
        {
            ROCKE_NBITS_REJECT("N (%d) must be divisible by tile_n (%d)", spec->N, t->tile_n);
        }
        if(spec->K % t->tile_k)
        {
            ROCKE_NBITS_REJECT("K (%d) must be divisible by tile_k (%d)", spec->K, t->tile_k);
        }
    }

    /* block_size = warp_m * warp_n * warp_k * wave_size. */
    expected_bs = rocke_warp_tile_block_size(t->warp_m, t->warp_n, t->warp_k, spec->wave_size);
    if(expected_bs != spec->block_size)
    {
        ROCKE_NBITS_REJECT(
            "block_size %d != warp_m*warp_n*warp_k*wave_size = %d", spec->block_size, expected_bs);
    }
    if(spec->block_size > rocke_archtarget_max_threads_per_block(target))
    {
        ROCKE_NBITS_REJECT("block_size %d > %d (hardware cap) on %s",
                           spec->block_size,
                           rocke_archtarget_max_threads_per_block(target),
                           arch);
    }

    if(spec->seq_len_tile <= 0)
    {
        ROCKE_NBITS_REJECT("seq_len_tile must be positive (got %d)", spec->seq_len_tile);
    }

    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "ok");
    }
    return true;
}

#undef ROCKE_NBITS_REJECT

/* ===================================================================== *
 *  matmul_nbits_signature
 * ===================================================================== */

rocke_status_t rocke_matmul_nbits_signature(const rocke_matmul_nbits_spec_t* spec,
                                            rocke_arena_t* arena,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count)
{
    const char* scale_wire = NULL;
    rocke_signature_builder_t sb;
    rocke_status_t st;

    if(spec == NULL || arena == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* _scale_wire_dtype(spec.scale_dtype) -- evaluated before the chain (it is
     * an argument to .ptr("Scales", ...)); its ValueError aborts the build. */
    if(rocke_matmul_nbits_scale_wire_dtype(spec->scale_dtype, &scale_wire) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
    {
        return st;
    }

    /* SignatureBuilder().ptr("A","f16").ptr("B","i8")
     *   .ptr("Scales", scale_wire).ptr("C","f16").scalar("M","i32").build() */
    rocke_signature_builder_ptr(&sb, "A", "f16", NULL);
    rocke_signature_builder_ptr(&sb, "B", "i8", NULL);
    rocke_signature_builder_ptr(&sb, "Scales", scale_wire, NULL);
    rocke_signature_builder_ptr(&sb, "C", "f16", NULL);
    rocke_signature_builder_scalar(&sb, "M", "i32");

    return rocke_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  matmul_nbits_grid
 * ===================================================================== */

rocke_status_t
    rocke_matmul_nbits_grid(int M, const rocke_matmul_nbits_spec_t* spec, int* gx, int* gy, int* gz)
{
    const rocke_gemm_tile_spec_t* t;
    int totals[2];
    int tiles[2];
    int out[3];
    rocke_status_t st;

    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    t = &spec->tile;
    /* ceil_div_grid((N, tile_n), (M, tile_m)) -> (ceil(N/tile_n),
     * ceil(M/tile_m), 1). Non-positive tile -> ValueError -> ROCKE_ERR_VALUE. */
    totals[0] = spec->N;
    tiles[0] = t->tile_n;
    totals[1] = M;
    tiles[1] = t->tile_m;
    st = rocke_ceil_div_grid(totals, tiles, 2, out);
    if(st != ROCKE_OK)
    {
        return st;
    }
    if(gx != NULL)
    {
        *gx = out[0];
    }
    if(gy != NULL)
    {
        *gy = out[1];
    }
    if(gz != NULL)
    {
        *gz = out[2];
    }
    return ROCKE_OK;
}

/* ===================================================================== *
 *  matmul_nbits_outer_tiles
 * ===================================================================== */

rocke_status_t rocke_matmul_nbits_outer_tiles(int seq_len,
                                              const rocke_matmul_nbits_spec_t* spec,
                                              int* out_m_outer,
                                              int* out_m_tile,
                                              size_t out_cap,
                                              size_t* out_count)
{
    int st;
    int m;
    size_t count = 0;

    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* if seq_len < 0: raise ValueError. */
    if(seq_len < 0)
    {
        return ROCKE_ERR_VALUE;
    }
    st = spec->seq_len_tile;
    if(st <= 0)
    {
        /* range(0, seq_len, st) with st<=0 is undefined / non-terminating in
         * Python; guard it (validate_common_spec already rejects st<=0). */
        return ROCKE_ERR_VALUE;
    }

    /* [(m, min(st, seq_len - m)) for m in range(0, seq_len, st)]. */
    for(m = 0; m < seq_len; m += st)
    {
        int m_tile = (st < seq_len - m) ? st : (seq_len - m);
        if(out_m_outer != NULL || out_m_tile != NULL)
        {
            if(count >= out_cap)
            {
                return ROCKE_ERR_VALUE; /* buffer too small */
            }
            if(out_m_outer != NULL)
            {
                out_m_outer[count] = m;
            }
            if(out_m_tile != NULL)
            {
                out_m_tile[count] = m_tile;
            }
        }
        ++count;
    }

    if(out_count != NULL)
    {
        *out_count = count;
    }
    return ROCKE_OK;
}
