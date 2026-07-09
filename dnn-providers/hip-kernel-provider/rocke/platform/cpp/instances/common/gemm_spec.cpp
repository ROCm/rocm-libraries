// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gemm_spec.c -- C99 port of the SPEC + VALIDITY surface of
 * rocke/instances/common/gemm_universal.py.
 *
 * This translation unit owns the "host-side, IR-free" half of the universal
 * GEMM instance builder:
 *
 *   Python (gemm_universal.py)             C99 (this file)
 *   ------------------------------------   --------------------------------------
 *   UniversalGemmSpec defaults             rocke_gemm_universal_spec_default()
 *   __post_init__ / _init_block_size()     rocke_gemm_universal_spec_finalize()
 *   UniversalGemmSpec.kernel_name()        rocke_gemm_universal_kernel_name()
 *   is_valid_spec(spec, arch)              rocke_gemm_universal_is_valid_spec()
 *   TileSpec.mfmas_per_warp_m / _n         rocke_gemm_tile_mfmas_per_warp_m / _n
 *   TileSpec.k_atoms_per_tile_k            rocke_gemm_tile_k_atoms_per_tile_k
 *   _dtype_ir / _mma_family                (file-static helpers)
 *   _resolve_mma_op / _storage_dtype       (file-static helpers)
 *   _ab_lds_plan / _mfma_atom_widths       (file-static helpers)
 *
 * Everything here is a pure value producer: NONE of it calls the IR builder
 * (rocke_b_*). The validity gate's reason strings + the kernel name are formatted
 * byte-identically to Python so a sweep driver sees the same accept/reject and
 * the same kernel identifier.
 *
 * The IR-emitting half (build_universal_gemm + every phase closure) lives in the
 * sibling TUs that bind to rocke/instance_gemm_internal.h.
 */

#include "rocke/instance_gemm_universal.h"

#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_*, rocke_mmaop_* */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_io_ir_type               */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, ...  */

/* Reproduce str(KeyError(_build_target message)) for an unknown gfx target:
 *
 *   Python _build_target: raise KeyError(
 *     f"unknown gfx target {gfx!r}; known: {sorted(specs)}. "
 *     f"Add a row to {_DATA_FILE.name}.")
 *   is_valid_spec: except KeyError as e: return False, str(e)
 *
 * str(KeyError(msg)) == repr(msg); the message contains single quotes, so
 * Python wraps it in DOUBLE quotes. sorted(specs) renders as a Python list
 * literal: ['gfx...', 'gfx...'] (single-quoted tokens, ", " separated).
 * rocke_known_arches() is tuple(sorted(_load_specs())) -- the same set in the same
 * order. Matches the vetted fmha_arch.cpp reproduction. */
static void rocke_gemm__set_unknown_arch_reason(char* out, size_t out_cap, const char* gfx)
{
    int count = 0;
    const char* const* arches;
    int i;
    size_t pos = 0;
    int wrote;

    if(out == NULL || out_cap == 0)
    {
        return;
    }

    arches = rocke_known_arches(&count);

    wrote = snprintf(out + pos, out_cap - pos, "\"unknown gfx target '%s'; known: [", gfx);
    if(wrote < 0)
    {
        out[0] = '\0';
        return;
    }
    pos += (size_t)wrote;
    if(pos >= out_cap)
    {
        out[out_cap - 1] = '\0';
        return;
    }

    for(i = 0; i < count; ++i)
    {
        wrote = snprintf(out + pos, out_cap - pos, "%s'%s'", (i == 0) ? "" : ", ", arches[i]);
        if(wrote < 0)
        {
            out[out_cap - 1] = '\0';
            return;
        }
        pos += (size_t)wrote;
        if(pos >= out_cap)
        {
            out[out_cap - 1] = '\0';
            return;
        }
    }

    snprintf(out + pos, out_cap - pos, "]. Add a row to arch_specs.json.\"");
}

/* ===================================================================== *
 *  TileSpec computed properties.
 *
 *  Each mirrors a Python @property whose `divmod` raises ValueError on a
 *  non-zero remainder. The C contract (instance_gemm_universal.h) is: return
 *  -1 on that divisibility failure so the caller treats the spec as invalid.
 * ===================================================================== */

int rocke_gemm_tile_mfmas_per_warp_m(const rocke_gemm_tile_spec_t* t)
{
    int denom;
    if(t == NULL)
    {
        return -1;
    }
    denom = t->warp_m * t->warp_tile_m;
    if(denom == 0 || (t->tile_m % denom) != 0)
    {
        return -1; /* Python: raise ValueError */
    }
    return t->tile_m / denom;
}

int rocke_gemm_tile_mfmas_per_warp_n(const rocke_gemm_tile_spec_t* t)
{
    int denom;
    if(t == NULL)
    {
        return -1;
    }
    denom = t->warp_n * t->warp_tile_n;
    if(denom == 0 || (t->tile_n % denom) != 0)
    {
        return -1;
    }
    return t->tile_n / denom;
}

int rocke_gemm_tile_k_atoms_per_tile_k(const rocke_gemm_tile_spec_t* t)
{
    if(t == NULL)
    {
        return -1;
    }
    if(t->warp_tile_k == 0 || (t->tile_k % t->warp_tile_k) != 0)
    {
        return -1;
    }
    return t->tile_k / t->warp_tile_k;
}

/* ===================================================================== *
 *  Spec defaults + finalize.
 *
 *  rocke_gemm_universal_spec_default() returns a struct with every field set to
 *  the Python dataclass default (TileSpec defaults too, except the four
 *  required tile/warp dims which have no Python default -- left 0 for the caller
 *  to fill). rocke_gemm_universal_spec_finalize() runs
 *  WarpTileBlockSizeMixin._init_block_size().
 * ===================================================================== */

rocke_gemm_universal_spec_t rocke_gemm_universal_spec_default(void)
{
    rocke_gemm_universal_spec_t s;
    memset(&s, 0, sizeof(s));

    s.name = NULL; /* caller must set (Python: required field, no default) */

    /* TileSpec defaults: tile_m/n/k + warp_m/n have no Python default (required);
     * the rest carry the dataclass defaults. */
    s.tile.tile_m = 0;
    s.tile.tile_n = 0;
    s.tile.tile_k = 0;
    s.tile.warp_m = 0;
    s.tile.warp_n = 0;
    s.tile.warp_k = 1; /* default 1  */
    s.tile.warp_tile_m = 32; /* default 32 */
    s.tile.warp_tile_n = 32; /* default 32 */
    s.tile.warp_tile_k = 16; /* default 16 */

    /* TraitSpec defaults. */
    s.trait.pipeline = "compv4"; /* default "compv4"    */
    s.trait.scheduler = "intrawave"; /* default "intrawave" */
    s.trait.epilogue = "cshuffle"; /* default "cshuffle"  */
    s.trait.pad_m = false;
    s.trait.pad_n = false;
    s.trait.pad_k = false;
    s.trait.persistent = false;
    s.trait.chiplet_swizzle = false;
    s.trait.chiplet_wgm = 8; /* default 8  */
    s.trait.chiplet_num_xcds = 8; /* default 8  */
    s.trait.chiplet_chunk_size = 64; /* default 64 */
    s.trait.waves_per_eu_set = false; /* Python None */
    s.trait.waves_per_eu = 0;
    s.trait.preshuffle_b = false;
    s.trait.direct_to_lds = false;
    s.trait.dtl_cache_a = 0; /* CACHE_ALL    */
    s.trait.dtl_cache_b = 2; /* CACHE_STREAM */
    s.trait.dtl_prefetch = false;
    s.trait.active_tile_skip = false;
    s.trait.lds_k_pad = 0;
    s.trait.lds_swizzle = false;
    s.trait.emit_sched_hints_set = false; /* Python None (arch-resolved) */
    s.trait.emit_sched_hints = false;
    s.trait.split_k = 1; /* default 1 (single-K-pass body) */

    /* DataSpec defaults. */
    s.data.dtype_a = "fp16"; /* default "fp16" */
    s.data.dtype_b = "fp16"; /* default "fp16" */
    s.data.dtype_c = "fp16"; /* default "fp16" */
    s.data.dtype_acc = "fp32"; /* default "fp32" */
    s.data.layout = "RCR"; /* default "RCR"  */

    s.wave_size = 64; /* default 64 */
    s.block_size = 0; /* derived at finalize() */
    s.batched = false;

    return s;
}

void rocke_gemm_universal_spec_finalize(rocke_gemm_universal_spec_t* spec)
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
 *  kernel_name().
 *
 *  Python:
 *      kernel_name_join(
 *          self.name,
 *          self.data.dtype_a,
 *          f"t{tile_m}x{tile_n}x{tile_k}",
 *          f"w{warp_m}x{warp_n}x{warp_k}",
 *          f"wt{wt_m}x{wt_n}x{wt_k}",
 *          f"{pipeline}_{scheduler}_{epilogue}",
 *          flags={"pad": any(pad_*), "pers": persistent, "bat": batched,
 *                 "preb": preshuffle_b, "dtl": direct_to_lds,
 *                 "pref": dtl_prefetch, "actt": active_tile_skip})
 *
 *  Flag iteration order matters (CPython 3.7+ dict insertion order); we pass the
 *  same fixed order to rocke_kernel_name_join.
 * ===================================================================== */

rocke_status_t rocke_gemm_universal_kernel_name(const rocke_gemm_universal_spec_t* spec,
                                                char* out,
                                                size_t out_cap)
{
    char part_t[64];
    char part_w[64];
    char part_wt[64];
    char part_pipe[128];
    char part_spk[32];
    const char* parts[5];
    const char* flag_names[8];
    int flag_on[8];
    const rocke_gemm_tile_spec_t* t;
    const rocke_gemm_trait_spec_t* tr;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    t = &spec->tile;
    tr = &spec->trait;

    snprintf(part_t, sizeof(part_t), "t%dx%dx%d", t->tile_m, t->tile_n, t->tile_k);
    snprintf(part_w, sizeof(part_w), "w%dx%dx%d", t->warp_m, t->warp_n, t->warp_k);
    snprintf(
        part_wt, sizeof(part_wt), "wt%dx%dx%d", t->warp_tile_m, t->warp_tile_n, t->warp_tile_k);
    snprintf(part_pipe, sizeof(part_pipe), "%s_%s_%s", tr->pipeline, tr->scheduler, tr->epilogue);

    /* parts after the prefix (in Python positional order). */
    parts[0] = spec->data.dtype_a;
    parts[1] = part_t;
    parts[2] = part_w;
    parts[3] = part_wt;
    parts[4] = part_pipe;

    /* flags map, in Python insertion order. */
    flag_names[0] = "pad";
    flag_names[1] = "pers";
    flag_names[2] = "bat";
    flag_names[3] = "preb";
    flag_names[4] = "dtl";
    flag_names[5] = "pref";
    flag_names[6] = "actt";
    /* Python flag key f"spk{tr.split_k}" (dynamic name; on when split_k > 1). */
    snprintf(part_spk, sizeof(part_spk), "spk%d", tr->split_k);
    flag_names[7] = part_spk;

    flag_on[0] = (tr->pad_m || tr->pad_n || tr->pad_k) ? 1 : 0;
    flag_on[1] = tr->persistent ? 1 : 0;
    flag_on[2] = spec->batched ? 1 : 0;
    flag_on[3] = tr->preshuffle_b ? 1 : 0;
    flag_on[4] = tr->direct_to_lds ? 1 : 0;
    flag_on[5] = tr->dtl_prefetch ? 1 : 0;
    flag_on[6] = tr->active_tile_skip ? 1 : 0;
    flag_on[7] = (tr->split_k > 1) ? 1 : 0;

    return rocke_kernel_name_join(spec->name, parts, 5, flag_names, flag_on, 8, out, out_cap, NULL);
}

/* ===================================================================== *
 *  Module-level pure helpers (file-static; mirror the Python file-scope
 *  helpers of the same name). None of these emit IR.
 * ===================================================================== */

/* _dtype_ir(name): resolve a GEMM storage dtype string to its IR scalar type.
 * Python wraps io_ir_type; returns NULL for an unsupported dtype (Python
 * ValueError path -- the caller turns it into a structured reject). */
static const rocke_type_t* ck_gemm_dtype_ir(const char* name)
{
    return rocke_io_ir_type(name);
}

/* _mma_family(arch): "wmma" for the RDNA wave32 targets, "mma" (MFMA) for CDNA.
 * Python: "wmma" if ArchTarget.from_gfx(arch).wave_size == 32 else "mma".
 * `target` may be passed pre-resolved to avoid a second from_gfx lookup. */
static const char* ck_gemm_mma_family(const rocke_archtarget_t* target)
{
    if(target == NULL)
    {
        return "mma";
    }
    return (target->wave_size == 32) ? "wmma" : "mma";
}

/* _storage_dtype(spec): validate homogeneous A/B/C, fp32 acc, RCR layout, then
 * return the A dtype's IR type. On a validation failure returns NULL and (when
 * `reason`/`reason_cap` are non-NULL) writes the Python ValueError text. */
static const rocke_type_t*
    ck_gemm_storage_dtype(const rocke_gemm_universal_spec_t* spec, char* reason, size_t reason_cap)
{
    const rocke_gemm_data_spec_t* d;
    const rocke_type_t* ty;

    d = &spec->data;
    if(strcmp(d->dtype_a, d->dtype_b) != 0 || strcmp(d->dtype_a, d->dtype_c) != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "UniversalGemmSpec currently requires homogeneous A/B/C dtypes; "
                     "got A=%s, B=%s, C=%s",
                     d->dtype_a,
                     d->dtype_b,
                     d->dtype_c);
        }
        return NULL;
    }
    if(strcmp(d->dtype_acc, "fp32") != 0 && strcmp(d->dtype_acc, "f32") != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            /* Python: f"...got {d.dtype_acc!r}" -> single-quoted repr. */
            snprintf(reason,
                     reason_cap,
                     "UniversalGemmSpec only supports fp32 accumulation, got '%s'",
                     d->dtype_acc);
        }
        return NULL;
    }
    if(strcmp(d->layout, "RCR") != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "UniversalGemmSpec only supports RCR layout, got '%s'",
                     d->layout);
        }
        return NULL;
    }

    ty = ck_gemm_dtype_ir(d->dtype_a);
    if(ty == NULL && reason != NULL && reason_cap > 0)
    {
        /* io_ir_type ValueError surface (unsupported A dtype). */
        snprintf(
            reason, reason_cap, "unsupported I/O dtype '%s'; expected f16/fp16/bf16", d->dtype_a);
    }
    return ty;
}

/* _resolve_mma_op(spec, arch): resolve the MmaOp for spec on the target via
 * target.mma.op_for_shape(family, a, a, fp32, wt_m, wt_n, wt_k). Returns NULL if
 * the target has no atom for the spec's warp-tile shape + dtype. */
[[maybe_unused]] static const rocke_mmaop_t*
    ck_gemm_resolve_mma_op(const rocke_gemm_universal_spec_t* spec,
                           const rocke_archtarget_t* target)
{
    const rocke_gemm_tile_spec_t* t;
    const char* a_name;

    if(target == NULL)
    {
        return NULL;
    }
    t = &spec->tile;
    a_name = spec->data.dtype_a; /* homogeneous A/B/C (validated in storage_dtype) */
    return rocke_archtarget_op_for_shape(target,
                                         ck_gemm_mma_family(target),
                                         a_name,
                                         a_name,
                                         "fp32",
                                         t->warp_tile_m,
                                         t->warp_tile_n,
                                         t->warp_tile_k);
}

/* _ab_lds_plan(spec, arch) -> (ab_single, db, two_buf). Pure ints/bools. */
static void ck_gemm_ab_lds_plan(const rocke_gemm_universal_spec_t* spec,
                                const rocke_archtarget_t* target,
                                int* out_ab_single,
                                bool* out_db,
                                bool* out_two_buf)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    int ab_single;
    long lds_cap;
    bool db_fits_2wg;
    bool db;

    ab_single = ((t->tile_m * t->tile_k) + (t->tile_n * t->tile_k)) * 2;
    lds_cap = (target != NULL) ? (long)target->lds_capacity_bytes : 0;
    db_fits_2wg = ((long)(2 * ab_single) * 2) <= lds_cap;
    db = (strcmp(spec->trait.pipeline, "compv4") == 0)
         && (strcmp(spec->trait.epilogue, "cshuffle") != 0) && (!spec->trait.direct_to_lds)
         && db_fits_2wg;

    if(out_ab_single != NULL)
    {
        *out_ab_single = ab_single;
    }
    if(out_db != NULL)
    {
        *out_db = db;
    }
    if(out_two_buf != NULL)
    {
        *out_two_buf = (spec->trait.dtl_prefetch ? true : false) || db;
    }
}

/* _mfma_atom_widths(spec) -> (a_per_lane, b_per_lane, c_per_lane). MFMA-only
 * per-lane widths derived straight from the wave64 geometry. Kept for parity
 * with the Python module surface (the contract-driven body uses the op-sourced
 * _atom_frag_lengths, a peer). */
[[maybe_unused]] static void ck_gemm_mfma_atom_widths(const rocke_gemm_universal_spec_t* spec,
                                                      int* out_a,
                                                      int* out_b,
                                                      int* out_c)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    int waves = spec->wave_size;

    if(waves == 0)
    {
        if(out_a)
            *out_a = 0;
        if(out_b)
            *out_b = 0;
        if(out_c)
            *out_c = 0;
        return;
    }
    if(out_a)
    {
        *out_a = (t->warp_tile_m * t->warp_tile_k) / waves;
    }
    if(out_b)
    {
        *out_b = (t->warp_tile_k * t->warp_tile_n) / waves;
    }
    if(out_c)
    {
        *out_c = (t->warp_tile_m * t->warp_tile_n) / waves;
    }
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch) -> (ok, reason).
 *
 *  A faithful, ordered port of the Python predicate: each rejection writes the
 *  same single-line reason. `arch` NULL => "gfx950". On accept writes "ok".
 * ===================================================================== */

bool rocke_gemm_universal_is_valid_spec(const rocke_gemm_universal_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const rocke_gemm_tile_spec_t* t;
    const char* family;
    const char* a_name;
    int atom_m, atom_n, atom_k;
    int expected_bs;
    int ab_single;
    bool ab_dbl;
    int ab_bytes;
    int c_bytes;
    int bytes_lds;
    int threads;
    int a_total;
    int b_total;

#define CK_GEMM_REJECT(...)                            \
    do                                                 \
    {                                                  \
        if(reason != NULL && reason_cap > 0)           \
        {                                              \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        }                                              \
        return false;                                  \
    } while(0)

    if(spec == NULL)
    {
        CK_GEMM_REJECT("spec is NULL");
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* try: target = ArchTarget.from_gfx(arch) except KeyError as e: return False, str(e) */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        /* Python str(KeyError) is the full "unknown gfx target {arch!r}; known:
         * [...]. Add a row to arch_specs.json." message, reproduced verbatim from
         * rocke_known_arches() (== sorted(specs)). */
        rocke_gemm__set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }

    t = &spec->tile;

    /* _storage_dtype(spec): validates homogeneous dtypes / fp32 acc / RCR. */
    if(ck_gemm_storage_dtype(spec, reason, reason_cap) == NULL)
    {
        /* reason already written by ck_gemm_storage_dtype. */
        return false;
    }

    /* MMA atom must be in the target's catalog for this dtype combo. */
    family = ck_gemm_mma_family(target);
    a_name = spec->data.dtype_a;
    atom_m = t->warp_tile_m;
    atom_n = t->warp_tile_n;
    atom_k = t->warp_tile_k;

    if(!rocke_archtarget_supports_dtype_combo(target, a_name, a_name, "fp32", family))
    {
        CK_GEMM_REJECT("unsupported GEMM dtype '%s' on %s", a_name, arch);
    }
    if(!rocke_mma_catalog_has_shape(
           &target->mma, family, a_name, a_name, "fp32", atom_m, atom_n, atom_k))
    {
        /* Python: f"unsupported {a_name} warp_tile {atom} on {arch}" -- {atom}
         * is a tuple repr "(m, n, k)". */
        CK_GEMM_REJECT(
            "unsupported %s warp_tile (%d, %d, %d) on %s", a_name, atom_m, atom_n, atom_k, arch);
    }

    /* spec wave size must match the target's. */
    if(spec->wave_size != target->wave_size)
    {
        CK_GEMM_REJECT(
            "spec wave_size %d != %s wave_size %d", spec->wave_size, arch, target->wave_size);
    }

    /* WMMA coverage is narrower than CDNA's MFMA matrix. */
    if(strcmp(family, "wmma") == 0)
    {
        if(!(atom_m == 16 && atom_n == 16 && atom_k == 16))
        {
            CK_GEMM_REJECT("WMMA path supports only 16x16x16 (got (%d, %d, %d)) on %s",
                           atom_m,
                           atom_n,
                           atom_k,
                           arch);
        }
        if(strcmp(spec->trait.pipeline, "mem") != 0)
        {
            CK_GEMM_REJECT("WMMA path supports only the 'mem' pipeline (got '%s') on %s",
                           spec->trait.pipeline,
                           arch);
        }
        if(strcmp(spec->trait.epilogue, "default") != 0)
        {
            CK_GEMM_REJECT("WMMA path supports only the 'default' epilogue (got '%s') on %s",
                           spec->trait.epilogue,
                           arch);
        }
        /* The Python loop walks (flag, label) in this fixed order and rejects on
         * the first set flag. */
        if(spec->trait.preshuffle_b)
        {
            CK_GEMM_REJECT("WMMA path does not support preshuffle_b on %s", arch);
        }
        if(spec->trait.direct_to_lds)
        {
            CK_GEMM_REJECT("WMMA path does not support direct_to_lds on %s", arch);
        }
        if(spec->trait.dtl_prefetch)
        {
            CK_GEMM_REJECT("WMMA path does not support dtl_prefetch on %s", arch);
        }
        if(spec->trait.active_tile_skip)
        {
            CK_GEMM_REJECT("WMMA path does not support active_tile_skip on %s", arch);
        }
        if(spec->trait.chiplet_swizzle)
        {
            CK_GEMM_REJECT("WMMA path does not support chiplet_swizzle on %s", arch);
        }
    }

    /* Geometry divisibility. */
    if((t->tile_m % (t->warp_m * t->warp_tile_m)) != 0)
    {
        CK_GEMM_REJECT("tile_m not divisible by warp_m * warp_tile_m");
    }
    if((t->tile_n % (t->warp_n * t->warp_tile_n)) != 0)
    {
        CK_GEMM_REJECT("tile_n not divisible by warp_n * warp_tile_n");
    }
    if((t->tile_k % t->warp_tile_k) != 0)
    {
        CK_GEMM_REJECT("tile_k not divisible by warp_tile_k");
    }

    /* block_size = warp_m * warp_n * wave_size. */
    expected_bs = t->warp_m * t->warp_n * spec->wave_size;
    if(expected_bs != spec->block_size)
    {
        CK_GEMM_REJECT(
            "block_size %d != warp_m*warp_n*wave_size = %d", spec->block_size, expected_bs);
    }

    /* LDS budget. */
    ck_gemm_ab_lds_plan(spec, target, &ab_single, NULL, &ab_dbl);
    ab_bytes = ab_single * (ab_dbl ? 2 : 1);
    c_bytes = (strcmp(spec->trait.epilogue, "cshuffle") == 0) ? (t->tile_m * t->tile_n * 2) : 0;
    bytes_lds = ab_bytes + c_bytes;
    if(!rocke_archtarget_fits_lds(target, (long)bytes_lds))
    {
        CK_GEMM_REJECT("LDS budget %d > %d cap (AB=%d, C=%d) on %s",
                       bytes_lds,
                       target->lds_capacity_bytes,
                       ab_bytes,
                       c_bytes,
                       arch);
    }

    /* Per-WG thread cap. */
    if(spec->block_size > rocke_archtarget_max_threads_per_block(target))
    {
        CK_GEMM_REJECT("block_size %d > %d (hardware cap) on %s",
                       spec->block_size,
                       rocke_archtarget_max_threads_per_block(target),
                       arch);
    }

    /* Global -> LDS vectorised load divisibility: >= 1 element/thread/phase. */
    threads = spec->block_size;
    a_total = t->tile_m * t->tile_k;
    b_total = t->tile_n * t->tile_k;
    if(a_total < threads || b_total < threads)
    {
        CK_GEMM_REJECT("block too small for one element/thread/phase");
    }

    /* Split-K (over the production body): only static invariants are checked
     * here -- split_k >= 1, and the atomic-add epilogue is MFMA (CDNA) only.
     * The K % split_k and ks % tile_k divisibility are the caller's
     * responsibility (K is a runtime arg in the universal body). */
    {
        int sk = spec->trait.split_k;
        if(sk < 1)
        {
            CK_GEMM_REJECT("split_k must be >= 1 (got %d)", sk);
        }
        if(sk > 1 && strcmp(family, "mma") != 0)
        {
            CK_GEMM_REJECT("split_k > 1 is CDNA-only (got family '%s' on %s)", family, arch);
        }
    }

    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "ok");
    }
    return true;

#undef CK_GEMM_REJECT
}

/* ck_gemm_resolve_mma_op / ck_gemm_mfma_atom_widths are pure parity surfaces
 * mirroring the Python module's file-scope _resolve_mma_op / _mfma_atom_widths.
 * The validity gate above resolves the atom via rocke_mma_catalog_has_shape, so
 * neither is called within this TU; they are kept defined for source parity and
 * marked [[maybe_unused]] at their definitions. (A prior "anchor" function that
 * referenced them tripped clang's -Wunneeded-internal-declaration, since it was
 * itself never emitted.) */
