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
#include "rocke/instance_gemm_internal.h" /* rocke_gemm_ab_db_decision */

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

/* Python acc_agpr_regs(spec): per-lane f32 accumulator registers of one wave
 * (wave64 MFMA) = mfmas_m * mfmas_n * warp_tile_m * warp_tile_n / 64. */
int rocke_gemm_acc_agpr_regs(const rocke_gemm_universal_spec_t* spec)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    int c_per = (t->warp_tile_m * t->warp_tile_n) / 64;
    return rocke_gemm_tile_mfmas_per_warp_m(t) * rocke_gemm_tile_mfmas_per_warp_n(t) * c_per;
}

bool rocke_gemm_universal_streamk_workspace_bytes(const rocke_gemm_universal_spec_t* spec,
                                                  long long* flags_bytes,
                                                  long long* partials_bytes)
{
    long long fb = 0, pb = 0;
    bool ok = true;
    if(spec != NULL && spec->trait.streamk)
    {
        const rocke_gemm_tile_spec_t* t = &spec->tile;
        if(t->warp_k != 1)
        {
            ok = false;
        }
        else
        {
            long long pool = spec->trait.persistent_ctas;
            long long waves = (long long)t->warp_m * t->warp_n * t->warp_k;
            fb = (4 * pool * waves + 255) / 256 * 256;
            pb = pool * t->tile_m * t->tile_n * 4;
        }
    }
    if(flags_bytes != NULL)
        *flags_bytes = fb;
    if(partials_bytes != NULL)
        *partials_bytes = pb;
    return ok;
}

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
    s.trait.cshuffle_no_alias = false; /* default: alias cshuffle C onto A/B */
    s.trait.ab_double_buffer_set = false; /* Python None (automatic rule) */
    s.trait.ab_double_buffer = false;
    s.trait.interwave_setprio = false; /* default off */
    s.trait.dtl_stages = 0; /* DTL ring levers: all default off */
    s.trait.dtl_issue_spread = false;
    s.trait.dtl_frag_prefetch = false;
    s.trait.dtl_stream_barrier = false;
    s.trait.mfma_swap_ab = false;
    s.trait.iglp_opt_set = false; /* Python None */
    s.trait.iglp_opt = 0;
    s.trait.persistent_ctas = 0;
    s.trait.acc_agpr = false; /* AGPR / interleave levers: default off */
    s.trait.mfma_interleave = false;
    s.trait.acc_agpr_pin = false;
    s.trait.ilv_dtl_every = 0;
    s.trait.ilv_barrier_after = 0;
    s.trait.ilv_merge = false;
    s.trait.lds_swizzle_r = -1;
    s.trait.dtl_hoist = false;
    s.trait.streamk = false; /* persistent / stream-K / pad levers: default off */
    s.trait.dtl_tight_rsrc = false;
    s.trait.pad_fast_tile = false;
    s.trait.ilv_early_release = 0; /* early-release interleave: default off */
    s.trait.ilv_release_at = 0;
    s.trait.ilv_acquire_at = 0;
    s.trait.acc_out_vgpr = false;
    s.trait.ilv_read_span = 0;
    s.trait.ilv_dtl_span = 0;
    s.trait.lds_row_interleave = false; /* row interleave / 64-bit origins: off */
    s.trait.rilv_pad = 16; /* Python default (0 reads as 16 too) */
    s.trait.rilv_vec_epilogue = false;
    s.trait.rve_store_nt = false;
    s.trait.dtl_soff0 = false;
    s.trait.wide_addr = false;
    s._prologue = NULL; /* no fused operand prologue */

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
    char part_chip[64];
    char part_dtlc[32];
    char part_stg[32];
    char part_iglp[32];
    char part_pc[32];
    char part_de[32];
    char part_ba[32];
    char part_swr[32];
    char part_er[32];
    char part_rl[32];
    char part_aq[32];
    char part_rs[32];
    char part_ds[32];
    char part_rp[32];
    const char* parts[5];
    const char* flag_names[45];
    int flag_on[45];
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
    flag_names[8] = "noalc";
    flag_names[9] = "abdb";
    flag_names[10] = "absb";
    /* Levers that change the body but used to share one symbol; encoded only
     * when they take effect (Python: "swz", f"chip{wgm}x{xcds}c{chunk}",
     * f"ca{a}cb{b}" (DTL with non-default cache hints), "iwp"). */
    flag_names[11] = "swz";
    snprintf(part_chip,
             sizeof(part_chip),
             "chip%dx%dc%d",
             tr->chiplet_wgm,
             tr->chiplet_num_xcds,
             tr->chiplet_chunk_size);
    flag_names[12] = part_chip;
    snprintf(part_dtlc, sizeof(part_dtlc), "ca%dcb%d", tr->dtl_cache_a, tr->dtl_cache_b);
    flag_names[13] = part_dtlc;
    flag_names[14] = "iwp";
    /* DTL ring levers (Python: f"stg{n}", "isp", "frp", "sbar", "sab",
     * f"iglp{n}", f"pc{n}"), encoded only when on. */
    snprintf(part_stg, sizeof(part_stg), "stg%d", tr->dtl_stages);
    flag_names[15] = part_stg;
    flag_names[16] = "isp";
    flag_names[17] = "frp";
    flag_names[18] = "sbar";
    flag_names[19] = "sab";
    snprintf(part_iglp, sizeof(part_iglp), "iglp%d", tr->iglp_opt);
    flag_names[20] = part_iglp;
    snprintf(part_pc, sizeof(part_pc), "pc%d", tr->persistent_ctas);
    flag_names[21] = part_pc;
    flag_names[22] = "agpr"; /* AGPR / interleave */
    flag_names[23] = "ilv";
    flag_names[24] = "apin";
    snprintf(part_de, sizeof(part_de), "de%d", tr->ilv_dtl_every);
    flag_names[25] = part_de;
    snprintf(part_ba, sizeof(part_ba), "ba%d", tr->ilv_barrier_after);
    flag_names[26] = part_ba;
    flag_names[27] = "mrg";
    snprintf(part_swr, sizeof(part_swr), "swr%d", tr->lds_swizzle_r);
    flag_names[28] = part_swr;
    flag_names[29] = "dh";
    flag_names[30] = "sk"; /* stream-K */
    flag_names[31] = "trs";
    flag_names[32] = "pft";
    /* early-release interleave */
    snprintf(part_er, sizeof(part_er), "er%d", tr->ilv_early_release);
    flag_names[33] = part_er;
    snprintf(part_rl, sizeof(part_rl), "rl%d", tr->ilv_release_at);
    flag_names[34] = part_rl;
    snprintf(part_aq, sizeof(part_aq), "aq%d", tr->ilv_acquire_at);
    flag_names[35] = part_aq;
    flag_names[36] = "aov";
    snprintf(part_rs, sizeof(part_rs), "rs%d", tr->ilv_read_span);
    flag_names[37] = part_rs;
    snprintf(part_ds, sizeof(part_ds), "ds%d", tr->ilv_dtl_span);
    flag_names[38] = part_ds;
    /* row interleave / vector epilogue / DTL origin levers (Python: "rilv",
     * f"rp{rilv_pad}", "rve", "rvnt", "so0", "w64"; the Python-engine-only flags
     * between them are never on here) */
    flag_names[39] = "rilv";
    snprintf(part_rp, sizeof(part_rp), "rp%d", tr->rilv_pad ? tr->rilv_pad : 16);
    flag_names[40] = part_rp;
    flag_names[41] = "rve";
    flag_names[42] = "rvnt";
    flag_names[43] = "so0";
    flag_names[44] = "w64";

    flag_on[0] = (tr->pad_m || tr->pad_n || tr->pad_k) ? 1 : 0;
    flag_on[1] = tr->persistent ? 1 : 0;
    flag_on[2] = spec->batched ? 1 : 0;
    flag_on[3] = tr->preshuffle_b ? 1 : 0;
    flag_on[4] = tr->direct_to_lds ? 1 : 0;
    flag_on[5] = tr->dtl_prefetch ? 1 : 0;
    flag_on[6] = tr->active_tile_skip ? 1 : 0;
    flag_on[7] = (tr->split_k > 1) ? 1 : 0;
    flag_on[8] = tr->cshuffle_no_alias ? 1 : 0;
    /* Python: "abdb": ab_double_buffer is True, "absb": ab_double_buffer is False */
    flag_on[9] = (tr->ab_double_buffer_set && tr->ab_double_buffer) ? 1 : 0;
    flag_on[10] = (tr->ab_double_buffer_set && !tr->ab_double_buffer) ? 1 : 0;
    flag_on[11] = tr->lds_swizzle ? 1 : 0;
    flag_on[12] = tr->chiplet_swizzle ? 1 : 0;
    flag_on[13] = (tr->direct_to_lds && !(tr->dtl_cache_a == 0 && tr->dtl_cache_b == 2)) ? 1 : 0;
    flag_on[14] = tr->interwave_setprio ? 1 : 0;
    flag_on[15] = (tr->dtl_stages >= 2) ? 1 : 0;
    flag_on[16] = tr->dtl_issue_spread ? 1 : 0;
    flag_on[17] = tr->dtl_frag_prefetch ? 1 : 0;
    flag_on[18] = tr->dtl_stream_barrier ? 1 : 0;
    flag_on[19] = tr->mfma_swap_ab ? 1 : 0;
    flag_on[20] = tr->iglp_opt_set ? 1 : 0;
    flag_on[21] = (tr->persistent_ctas > 0) ? 1 : 0;
    flag_on[22] = tr->acc_agpr ? 1 : 0;
    flag_on[23] = tr->mfma_interleave ? 1 : 0;
    flag_on[24] = tr->acc_agpr_pin ? 1 : 0;
    flag_on[25] = (tr->ilv_dtl_every > 0) ? 1 : 0;
    flag_on[26] = (tr->ilv_barrier_after > 0) ? 1 : 0;
    flag_on[27] = tr->ilv_merge ? 1 : 0;
    flag_on[28] = (tr->lds_swizzle_r >= 0) ? 1 : 0;
    flag_on[29] = tr->dtl_hoist ? 1 : 0;
    flag_on[30] = tr->streamk ? 1 : 0;
    flag_on[31] = tr->dtl_tight_rsrc ? 1 : 0;
    flag_on[32] = tr->pad_fast_tile ? 1 : 0;
    flag_on[33] = (tr->ilv_early_release > 0) ? 1 : 0;
    flag_on[34] = (tr->ilv_release_at > 0) ? 1 : 0;
    flag_on[35] = (tr->ilv_acquire_at > 0) ? 1 : 0;
    flag_on[36] = tr->acc_out_vgpr ? 1 : 0;
    flag_on[37] = (tr->ilv_read_span > 0) ? 1 : 0;
    flag_on[38] = (tr->ilv_dtl_span > 0) ? 1 : 0;
    flag_on[39] = tr->lds_row_interleave ? 1 : 0;
    flag_on[40] = (tr->lds_row_interleave && tr->rilv_pad != 0 && tr->rilv_pad != 16) ? 1 : 0;
    flag_on[41] = tr->rilv_vec_epilogue ? 1 : 0;
    flag_on[42] = tr->rve_store_nt ? 1 : 0;
    flag_on[43] = tr->dtl_soff0 ? 1 : 0;
    flag_on[44] = tr->wide_addr ? 1 : 0;

    return rocke_kernel_name_join(
        spec->name, parts, 5, flag_names, flag_on, 45, out, out_cap, NULL);
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
static const rocke_mmaop_t* ck_gemm_resolve_mma_op(const rocke_gemm_universal_spec_t* spec,
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
                                         t->warp_tile_k,
                                         nullptr);
}

/* ab_double_buffer_decision(tile, pipeline, epilogue, direct_to_lds,
 * ab_double_buffer, lds_cap). None (ab_double_buffer_set==false) keeps the
 * historical automatic rule byte-identical: compv4, direct epilogue, not DTL, and
 * the doubled AB still fits 2 WG/CU. When set: forced (compv4 non-DTL only). */
bool rocke_gemm_ab_db_decision(const rocke_gemm_universal_spec_t* spec,
                               const rocke_archtarget_t* target)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    long ab_single;
    long lds_cap;

    if(spec->trait.pipeline == NULL || strcmp(spec->trait.pipeline, "compv4") != 0
       || spec->trait.direct_to_lds)
    {
        return false;
    }
    if(spec->trait.ab_double_buffer_set)
    {
        return spec->trait.ab_double_buffer;
    }
    ab_single = ((long)t->tile_m * t->tile_k + (long)t->tile_n * t->tile_k) * 2;
    lds_cap = (target != NULL) ? (long)target->lds_capacity_bytes : 0;
    return (spec->trait.epilogue != NULL && strcmp(spec->trait.epilogue, "cshuffle") != 0)
           && ((2 * ab_single) * 2 <= lds_cap);
}

/* _ilog2_exact(x): log2 of a positive power of two, else -1 (Python None). */
static int ck_gemm_ilog2_exact(int x)
{
    int n = 0;
    if(x <= 0 || (x & (x - 1)) != 0)
    {
        return -1;
    }
    while((x >> n) > 1)
    {
        ++n;
    }
    return n;
}

/* lds_swizzle_params(a_per_lane, b_per_lane, block_k) -> (R, W, L). Pure and
 * env-free (the gate never reads the environment; the build ctx applies the
 * CK_SWZ_* debug overrides on top). */
void rocke_gemm_lds_swizzle_params(
    int a_per_lane, int b_per_lane, int block_k, int* out_r, int* out_w, int* out_l)
{
    int elem = (a_per_lane == b_per_lane) ? a_per_lane : 0;
    int slots = (elem && (block_k % elem == 0)) ? (block_k / elem) : 0;
    int auto_l = ck_gemm_ilog2_exact(elem);
    int auto_w = ck_gemm_ilog2_exact(slots);
    int r = 3;
    int w = 1;
    int l = 4;
    if(auto_l >= 0 && auto_w >= 1)
    {
        r = 0;
        w = auto_w;
        l = auto_l;
    }
    if(out_r != NULL)
    {
        *out_r = r;
    }
    if(out_w != NULL)
    {
        *out_w = w;
    }
    if(out_l != NULL)
    {
        *out_l = l;
    }
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
    bool db;

    ab_single = ((t->tile_m * t->tile_k) + (t->tile_n * t->tile_k)) * 2;
    db = rocke_gemm_ab_db_decision(spec, target);

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

/* Per-arch CU counts: the static table rocke.helpers.split_k _ARCH_NUM_CUS /
 * _DEFAULT_NUM_CUS (arch_num_cus) that Python streamk_resident_ctas reads; the
 * stream-K residency verdicts of both engines are compared at the boundary P in
 * test_gemm_persistent_streamk.py. */
static int ck_gemm_arch_num_cus(const char* arch)
{
    static const struct
    {
        const char* gfx;
        int num_cus;
    } table[] = {
        {"gfx942", 304}, /* 8 XCDs * 38 CUs */
        {"gfx950", 256}, /* 8 XCDs * 32 CUs */
        {"gfx1151", 64}, /* RDNA gfx1151 (WMMA) */
    };
    if(arch != NULL)
        for(const auto& e : table)
            if(strcmp(arch, e.gfx) == 0)
                return e.num_cus;
    return 256; /* _DEFAULT_NUM_CUS */
}

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

    /* DirectToLDS: the emitter bakes a fixed per-lane chunk width into the
     * kernel at build time, capped at the arch's own buffer_load_lds width
     * (ArchTarget.async_lds_max_dwords). Asking for a wider one than the arch
     * has is not a build-time error -- the AMDGPU backend aborts the whole
     * process -- so an arch with too little width to satisfy tile_k, or none
     * at all, must be rejected here. */
    if(spec->trait.direct_to_lds)
    {
        int dtl_max_dwords = rocke_archtarget_async_lds_max_dwords(target);
        int dtl_halves;
        if(dtl_max_dwords < 1)
        {
            CK_GEMM_REJECT("direct_to_lds: %s has no DRAM->LDS DMA instruction", arch);
        }
        dtl_halves = (dtl_max_dwords > 4 ? 4 : dtl_max_dwords) * 2;
        if((t->tile_k % dtl_halves) != 0)
        {
            CK_GEMM_REJECT("direct_to_lds requires tile_k %% %d == 0 (got %d) with "
                           "%d dword(s)/lane on %s",
                           dtl_halves,
                           t->tile_k,
                           dtl_max_dwords,
                           arch);
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

    /* Trait values (persistent: accepted, a no-op rename; scheduler: a known
     * value, both emit the intrawave body; interwave_setprio: MFMA-only;
     * ab_double_buffer: compv4 VGPR-staged body only). */
    if(spec->trait.scheduler == NULL
       || (strcmp(spec->trait.scheduler, "intrawave") != 0
           && strcmp(spec->trait.scheduler, "interwave") != 0))
    {
        if(spec->trait.scheduler == NULL)
        {
            CK_GEMM_REJECT("unknown scheduler None");
        }
        CK_GEMM_REJECT("unknown scheduler '%s'", spec->trait.scheduler);
    }
    if(spec->trait.interwave_setprio && strcmp(family, "mma") != 0)
    {
        CK_GEMM_REJECT("interwave_setprio is MFMA-only (got family '%s' on %s)", family, arch);
    }
    if(spec->trait.ab_double_buffer_set
       && (strcmp(spec->trait.pipeline, "compv4") != 0 || spec->trait.direct_to_lds))
    {
        CK_GEMM_REJECT("ab_double_buffer applies only to the compv4 VGPR-staged pipeline "
                       "(pipeline='%s', direct_to_lds=%s); DTL uses dtl_prefetch",
                       spec->trait.pipeline,
                       spec->trait.direct_to_lds ? "True" : "False");
    }

    /* DTL ring levers: each is implemented by one body only, so asking
     * for it anywhere else is rejected rather than silently ignored (same order
     * and reasons as the Python predicate). */
    {
        const rocke_gemm_trait_spec_t* tr = &spec->trait;
        if(!(tr->dtl_stages == 0 || tr->dtl_stages == 2 || tr->dtl_stages == 3
             || tr->dtl_stages == 4))
        {
            CK_GEMM_REJECT("dtl_stages must be 0, 2, 3 or 4 (got %d)", tr->dtl_stages);
        }
        if(tr->dtl_stages != 0
           && !(strcmp(family, "mma") == 0 && tr->direct_to_lds && tr->dtl_prefetch))
        {
            CK_GEMM_REJECT("dtl_stages requires the MFMA direct_to_lds + dtl_prefetch body");
        }
        if(tr->dtl_issue_spread && tr->dtl_stages < 2)
        {
            CK_GEMM_REJECT("dtl_issue_spread requires dtl_stages >= 2");
        }
        if(tr->dtl_frag_prefetch && tr->dtl_stages < 2)
        {
            CK_GEMM_REJECT("dtl_frag_prefetch requires dtl_stages >= 2");
        }
        if(tr->dtl_stream_barrier && tr->dtl_stages < 2)
        {
            CK_GEMM_REJECT("dtl_stream_barrier requires dtl_stages >= 2");
        }
        if(tr->iglp_opt_set && tr->dtl_stages < 2)
        {
            CK_GEMM_REJECT("iglp_opt requires dtl_stages >= 2");
        }
        if(tr->persistent_ctas != 0)
        {
            if(tr->persistent_ctas < 0)
            {
                CK_GEMM_REJECT("persistent_ctas must be >= 0 (got %d)", tr->persistent_ctas);
            }
            if(tr->dtl_stages < 2)
            {
                CK_GEMM_REJECT("persistent_ctas requires dtl_stages >= 2");
            }
            if(strcmp(tr->epilogue, "default") != 0 || spec->batched || tr->split_k > 1
               || tr->active_tile_skip)
            {
                CK_GEMM_REJECT("persistent_ctas requires the default epilogue, a non-batched "
                               "body and split_k == 1");
            }
            if(tr->chiplet_swizzle
               && tr->persistent_ctas % (tr->chiplet_num_xcds > 1 ? tr->chiplet_num_xcds : 1))
            {
                CK_GEMM_REJECT("persistent_ctas %d with chiplet_swizzle must be a multiple of "
                               "chiplet_num_xcds %d (else the tiles of a workgroup leave the XCD "
                               "the swizzle assigned them to)",
                               tr->persistent_ctas,
                               tr->chiplet_num_xcds);
            }
        }
        if(tr->streamk && tr->persistent_ctas <= 0)
        {
            CK_GEMM_REJECT("streamk requires persistent_ctas > 0 (the stream-K pool)");
        }
        if(tr->streamk && spec->tile.warp_k != 1)
        {
            CK_GEMM_REJECT("streamk requires warp_k == 1 (got %d): the partial-tile "
                           "slot is sized tile_m x tile_n",
                           spec->tile.warp_k);
        }
        if(tr->dtl_tight_rsrc && (tr->dtl_stages < 2 || spec->batched))
        {
            CK_GEMM_REJECT("dtl_tight_rsrc requires dtl_stages >= 2 and a non-batched body");
        }
        if(tr->pad_fast_tile
           && (!(tr->pad_m || tr->pad_n) || strcmp(tr->epilogue, "default") != 0 || tr->split_k > 1
               || tr->mfma_swap_ab))
        {
            CK_GEMM_REJECT("pad_fast_tile requires pad_m or pad_n, the default epilogue, "
                           "split_k == 1 and no mfma_swap_ab");
        }
        if(tr->iglp_opt_set && (tr->iglp_opt < 0 || tr->iglp_opt > 3))
        {
            CK_GEMM_REJECT("iglp_opt must be 0..3 (got %d)", tr->iglp_opt);
        }
        /* Python _AGPR_ARCHS / _ILV_ARCHS / _SWR_DH_ARCHS: each AGPR / interleave lever is
         * admitted only where its bodies passed GPU numerics. */
        bool gfx942_or_950 = strcmp(arch, "gfx942") == 0 || strcmp(arch, "gfx950") == 0;
        if(tr->acc_agpr && !(strcmp(family, "mma") == 0 && gfx942_or_950))
        {
            CK_GEMM_REJECT("acc_agpr needs MFMA accumulators in AGPRs and is admitted on "
                           "gfx942 / gfx950 only (GPU numerics); got family '%s' on %s",
                           family,
                           arch);
        }
        if(tr->mfma_interleave && !gfx942_or_950)
        {
            CK_GEMM_REJECT("mfma_interleave is admitted on gfx942 / gfx950 only (GPU numerics); "
                           "got %s",
                           arch);
        }
        if((tr->lds_swizzle_r != -1 || tr->dtl_hoist) && !gfx942_or_950)
        {
            CK_GEMM_REJECT("lds_swizzle_r / dtl_hoist are admitted on gfx942 / gfx950 only (GPU "
                           "numerics); got %s",
                           arch);
        }
        if(tr->ilv_dtl_every != 0 && (tr->ilv_dtl_every < 0 || !tr->mfma_interleave))
        {
            CK_GEMM_REJECT("ilv_dtl_every must be >= 0 and requires mfma_interleave (got %d)",
                           tr->ilv_dtl_every);
        }
        if(tr->ilv_barrier_after != 0)
        {
            int n_atom = rocke_gemm_tile_mfmas_per_warp_m(&spec->tile)
                         * rocke_gemm_tile_mfmas_per_warp_n(&spec->tile);
            if(!tr->mfma_interleave
               || !(0 < tr->ilv_barrier_after && tr->ilv_barrier_after < n_atom))
            {
                CK_GEMM_REJECT("ilv_barrier_after requires mfma_interleave and 0 < k < the MFMAs "
                               "of one K atom (%d); got %d",
                               n_atom,
                               tr->ilv_barrier_after);
            }
        }
        if(tr->ilv_merge && (!tr->mfma_interleave || tr->ilv_dtl_every != 0))
        {
            CK_GEMM_REJECT("ilv_merge requires mfma_interleave and excludes ilv_dtl_every");
        }
        {
            int n_atom = rocke_gemm_tile_mfmas_per_warp_m(&spec->tile)
                         * rocke_gemm_tile_mfmas_per_warp_n(&spec->tile);
            int k_atoms = spec->tile.tile_k / spec->tile.warp_tile_k;
            if((tr->ilv_early_release != 0 || tr->ilv_read_span != 0 || tr->ilv_dtl_span != 0
                || tr->acc_out_vgpr)
               && strcmp(arch, "gfx950") != 0)
            {
                CK_GEMM_REJECT(
                    "ilv_early_release / ilv_read_span / ilv_dtl_span / acc_out_vgpr are "
                    "admitted on gfx950 only (GPU numerics); got %s",
                    arch);
            }
            if(tr->ilv_early_release != 0 && (!tr->mfma_interleave || k_atoms < 2))
            {
                CK_GEMM_REJECT("ilv_early_release requires mfma_interleave and >= 2 K atoms per "
                               "tile (got %d)",
                               k_atoms);
            }
            if(tr->ilv_early_release < 0 || tr->ilv_early_release > 3
               || (tr->ilv_early_release >= 2 && tr->ilv_barrier_after != 0))
            {
                CK_GEMM_REJECT("ilv_early_release must be 0..3; modes 2 / 3 exclude "
                               "ilv_barrier_after (got %d)",
                               tr->ilv_early_release);
            }
            if(tr->ilv_release_at != 0
               && (tr->ilv_early_release < 2
                   || !(0 < tr->ilv_release_at && tr->ilv_release_at < n_atom)))
            {
                CK_GEMM_REJECT("ilv_release_at requires ilv_early_release 2 / 3 and 0 < k < the "
                               "MFMAs of one K atom (%d); got %d",
                               n_atom,
                               tr->ilv_release_at);
            }
            if(tr->ilv_acquire_at != 0
               && (tr->ilv_early_release == 0
                   || !(tr->ilv_barrier_after < tr->ilv_acquire_at && tr->ilv_acquire_at < n_atom)))
            {
                CK_GEMM_REJECT("ilv_acquire_at requires ilv_early_release and ilv_barrier_after < "
                               "k < the MFMAs of one K atom (%d); got %d",
                               n_atom,
                               tr->ilv_acquire_at);
            }
        }
        if(tr->lds_swizzle_r != -1
           && !(tr->lds_swizzle && 0 <= tr->lds_swizzle_r && tr->lds_swizzle_r <= 4))
        {
            CK_GEMM_REJECT("lds_swizzle_r must be -1 (auto) or 0..4 with lds_swizzle (got %d)",
                           tr->lds_swizzle_r);
        }
        if(tr->dtl_hoist && tr->dtl_stages < 2)
        {
            CK_GEMM_REJECT("dtl_hoist requires dtl_stages >= 2");
        }
        if(tr->ilv_read_span != 0
           && (!tr->mfma_interleave || !(0 < tr->ilv_read_span && tr->ilv_read_span <= 100)
               || tr->ilv_dtl_every != 0))
        {
            CK_GEMM_REJECT("ilv_read_span requires mfma_interleave (without ilv_dtl_every) and "
                           "0 < pct <= 100; got %d",
                           tr->ilv_read_span);
        }
        if(tr->ilv_dtl_span != 0
           && (!tr->mfma_interleave || !(0 < tr->ilv_dtl_span && tr->ilv_dtl_span <= 100)
               || tr->ilv_dtl_every != 0))
        {
            CK_GEMM_REJECT("ilv_dtl_span requires mfma_interleave (without ilv_dtl_every) and "
                           "0 < pct <= 100; got %d",
                           tr->ilv_dtl_span);
        }
        if(tr->acc_out_vgpr && !tr->acc_agpr)
        {
            CK_GEMM_REJECT("acc_out_vgpr requires acc_agpr");
        }
        /* row interleave, its vector epilogue and the DTL origin levers (same
         * reasons as the Python predicate, in its order) */
        if(tr->lds_row_interleave)
        {
            int mm = rocke_gemm_tile_mfmas_per_warp_m(&spec->tile);
            int mn = rocke_gemm_tile_mfmas_per_warp_n(&spec->tile);
            if(strcmp(arch, "gfx950") != 0 || tr->dtl_stages < 2 || tr->lds_swizzle)
            {
                CK_GEMM_REJECT("lds_row_interleave requires gfx950 (16-byte DTL), dtl_stages >= 2 "
                               "and lds_swizzle off (the interleave replaces the XOR swizzle)");
            }
            if(!(strcmp(spec->data.dtype_a, "fp16") == 0 || strcmp(spec->data.dtype_a, "bf16") == 0)
               || strcmp(spec->data.dtype_b, spec->data.dtype_a) != 0 || t->tile_k != 64
               || t->warp_tile_m != 16 || t->warp_tile_n != 16 || t->warp_tile_k != 32 || mm % 8
               || mn % 8 || spec->wave_size != 64)
            {
                CK_GEMM_REJECT("lds_row_interleave requires 16-bit A/B, tile_k 64 (128-byte LDS "
                               "rows), the 16x16x32 atom, wave64 and mfmas_m / mfmas_n multiples "
                               "of 8 (got %s/%s, tile_k %d, atom %dx%dx%d, mfmas %dx%d)",
                               spec->data.dtype_a,
                               spec->data.dtype_b,
                               t->tile_k,
                               t->warp_tile_m,
                               t->warp_tile_n,
                               t->warp_tile_k,
                               mm,
                               mn);
            }
        }
        if(tr->rilv_vec_epilogue
           && !(tr->lds_row_interleave && strcmp(tr->epilogue, "default") == 0 && !tr->mfma_swap_ab
                && !(tr->pad_m || tr->pad_n) && tr->split_k == 1))
        {
            CK_GEMM_REJECT("rilv_vec_epilogue requires lds_row_interleave, the default epilogue, "
                           "no mfma_swap_ab, split_k == 1 and no pad_m / pad_n unless rve_pad");
        }
        if(tr->rve_store_nt && !tr->rilv_vec_epilogue)
        {
            CK_GEMM_REJECT("rve_store_nt requires rilv_vec_epilogue");
        }
        if(tr->wide_addr
           && !(tr->direct_to_lds && tr->dtl_hoist && (tr->dtl_soff0 || tr->dtl_tight_rsrc)
                && tr->rilv_vec_epilogue && !tr->persistent_ctas && !tr->streamk && tr->split_k <= 1
                && strcmp(spec->data.layout, "RCR") == 0 && !spec->batched))
        {
            CK_GEMM_REJECT("wide_addr requires the direct-to-LDS ring with dtl_hoist and "
                           "dtl_soff0 or dtl_tight_rsrc, the row-interleave vector epilogue, a "
                           "non-persistent grid without split-K / stream-K / pingpong / asm_loop "
                           "/ dtl_k_tail / dtl_base_fold / batching, the RCR layout and no "
                           "tensor GETT / rich-epilogue hook");
        }
        if(tr->dtl_soff0
           && !(tr->direct_to_lds && tr->dtl_stages >= 2 && tr->dtl_hoist && !tr->dtl_tight_rsrc))
        {
            CK_GEMM_REJECT("dtl_soff0 requires direct_to_lds, dtl_stages >= 2 and dtl_hoist, "
                           "without dtl_tight_rsrc / pingpong");
        }
        if(tr->rilv_pad != 0 && tr->rilv_pad != 16
           && !(tr->lds_row_interleave
                && (tr->rilv_pad == 32 || tr->rilv_pad == 48 || tr->rilv_pad == 64)))
        {
            CK_GEMM_REJECT("rilv_pad requires lds_row_interleave and one of 16, 32, 48, 64 bytes "
                           "(got %d)",
                           tr->rilv_pad);
        }
        if(tr->acc_agpr_pin && !tr->acc_agpr)
        {
            CK_GEMM_REJECT("acc_agpr_pin requires acc_agpr");
        }
        if(tr->mfma_interleave && !tr->dtl_frag_prefetch)
        {
            CK_GEMM_REJECT("mfma_interleave requires dtl_frag_prefetch (dtl_stages >= 2)");
        }
        if(tr->mfma_interleave && tr->dtl_issue_spread)
        {
            CK_GEMM_REJECT("mfma_interleave already spreads the DTL issues; it excludes "
                           "dtl_issue_spread");
        }
        if(tr->acc_agpr && (spec->wave_size != 64 || rocke_gemm_acc_agpr_regs(spec) > 256))
        {
            CK_GEMM_REJECT("acc_agpr needs wave64 and <= 256 accumulator registers per lane "
                           "(got %d)",
                           rocke_gemm_acc_agpr_regs(spec));
        }
        if(tr->mfma_swap_ab)
        {
            if(strcmp(family, "mma") != 0)
            {
                CK_GEMM_REJECT("mfma_swap_ab is MFMA-only (got family '%s' on %s)", family, arch);
            }
            if(strcmp(tr->epilogue, "default") != 0 || tr->pad_n || tr->split_k > 1)
            {
                CK_GEMM_REJECT("mfma_swap_ab requires the default (direct) epilogue without "
                               "pad_n or split-K");
            }
            if(t->warp_tile_m != t->warp_tile_n)
            {
                CK_GEMM_REJECT("mfma_swap_ab requires warp_tile_m == warp_tile_n");
            }
        }
    }

    /* LDS write-granularity rules (lever-combination sweep): every DTL pass must be
     * full, and the lds_swizzle XOR group (2^L halves) must be at least as wide as
     * the LDS write chunk (DTL chunk, or the VGPR path's load_vec). */
    if(strcmp(family, "mma") == 0)
    {
        int dtl_max = rocke_archtarget_async_lds_max_dwords(target);
        int dtl_h = (dtl_max > 4 ? 4 : dtl_max) * 2;
        if(spec->trait.direct_to_lds)
        {
            const char* nm[2] = {"A", "B"};
            int rows[2] = {t->tile_m, t->tile_n};
            int i;
            for(i = 0; i < 2; ++i)
            {
                int chunks = (rows[i] * t->tile_k) / dtl_h;
                if((chunks % spec->block_size) != 0)
                {
                    CK_GEMM_REJECT("direct_to_lds: the %s tile has %d %d-half chunks, not a "
                                   "multiple of block_size %d (a partial DMA pass writes past "
                                   "the LDS tile)",
                                   nm[i],
                                   chunks,
                                   dtl_h,
                                   spec->block_size);
                }
            }
        }
        if(spec->trait.lds_swizzle)
        {
            const rocke_mmaop_t* op = ck_gemm_resolve_mma_op(spec, target);
            int swz_l = 0;
            rocke_gemm_lds_swizzle_params(op != NULL ? op->a_frag_len : 0,
                                          op != NULL ? op->b_frag_len : 0,
                                          t->tile_k,
                                          NULL,
                                          NULL,
                                          &swz_l);
            int chunk;
            if(spec->trait.direct_to_lds)
            {
                chunk = dtl_h;
            }
            else
            {
                chunk = rocke_gemm_choose_load_vec(spec);
                if(chunk <= 0)
                {
                    CK_GEMM_REJECT("no usable load_vec for tile_m=%d tile_n=%d tile_k=%d "
                                   "block_size=%d elem_bytes=2",
                                   t->tile_m,
                                   t->tile_n,
                                   t->tile_k,
                                   spec->block_size);
                }
            }
            if(swz_l < 0 || ((1 << swz_l) % chunk) != 0)
            {
                CK_GEMM_REJECT("lds_swizzle: XOR group of %d halves is narrower than the %d-half "
                               "LDS write chunk (atom (%d, %d, %d)); the swizzle would split a "
                               "chunk",
                               1 << (swz_l > 0 ? swz_l : 0),
                               chunk,
                               atom_m,
                               atom_n,
                               atom_k);
            }
        }
    }

    /* LDS budget: universal_lds_bytes(...). AB rows padded by lds_k_pad (non-DTL),
     * doubled when ping-ponged; the cshuffle C tile aliases A/B (max) unless
     * cshuffle_no_alias (sum). */
    ck_gemm_ab_lds_plan(spec, target, &ab_single, NULL, &ab_dbl);
    {
        int pad = spec->trait.direct_to_lds ? 0 : spec->trait.lds_k_pad;
        int ab_single_padded = (t->tile_m + t->tile_n) * (t->tile_k + pad) * 2;
        ab_bytes = ab_single_padded * (ab_dbl ? 2 : 1);
        /* alias-scoped DTL ring: one A + one B allocation per stage */
        if(spec->trait.direct_to_lds && spec->trait.dtl_prefetch && spec->trait.dtl_stages >= 2)
        {
            ab_bytes = ab_single_padded * spec->trait.dtl_stages;
        }
    }
    c_bytes = (strcmp(spec->trait.epilogue, "cshuffle") == 0) ? (t->tile_m * t->tile_n * 2) : 0;
    if(spec->trait.cshuffle_no_alias)
    {
        bytes_lds = ab_bytes + c_bytes;
    }
    else
    {
        bytes_lds = (ab_bytes > c_bytes) ? ab_bytes : c_bytes;
    }
    if(spec->trait.lds_row_interleave)
    {
        /* rilv_pad bytes of padding per 8 LDS rows, per operand and stage */
        int rpad = spec->trait.rilv_pad ? spec->trait.rilv_pad : 16;
        ab_bytes += ((t->tile_m + t->tile_n) / 8) * rpad * spec->trait.dtl_stages;
        bytes_lds = spec->trait.cshuffle_no_alias ? ab_bytes + c_bytes
                                                  : (ab_bytes > c_bytes ? ab_bytes : c_bytes);
    }
    if(!rocke_archtarget_fits_lds(target, (long)bytes_lds))
    {
        CK_GEMM_REJECT("LDS budget %d > %d cap (AB=%d, C=%d) on %s",
                       bytes_lds,
                       target->lds_capacity_bytes,
                       ab_bytes,
                       c_bytes,
                       arch);
    }

    /* Stream-K co-residency (streamk_resident_ctas): an owner spins on
     * higher-index workgroups of the pool, so all P must be resident at once.
     * per_cu = min(LDS cap / bytes, max(1, 4 / waves)) (one wave per SIMD at the
     * 512-register worst case). */
    if(spec->trait.streamk)
    {
        int sk_waves = t->warp_m * t->warp_n * t->warp_k;
        int sk_per_lds = target->lds_capacity_bytes / (bytes_lds > 1 ? bytes_lds : 1);
        int sk_per_reg = 4 / (sk_waves > 1 ? sk_waves : 1);
        int sk_per_cu;
        int sk_cus = ck_gemm_arch_num_cus(arch);
        if(sk_per_reg < 1)
        {
            sk_per_reg = 1;
        }
        sk_per_cu = sk_per_lds < sk_per_reg ? sk_per_lds : sk_per_reg;
        if(spec->trait.persistent_ctas > sk_cus * sk_per_cu)
        {
            CK_GEMM_REJECT("streamk needs all persistent_ctas = %d workgroups co-resident, but "
                           "at most %d fit on %s (%d CUs x %d per CU: LDS %d of %d B, %d waves "
                           "at the 512-register worst case)",
                           spec->trait.persistent_ctas,
                           sk_cus * sk_per_cu,
                           arch,
                           sk_cus,
                           sk_per_cu,
                           bytes_lds,
                           target->lds_capacity_bytes,
                           sk_waves);
        }
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

/* ck_gemm_mfma_atom_widths is a pure parity surface mirroring the Python
 * module's file-scope _mfma_atom_widths; it is kept defined for source parity and
 * marked [[maybe_unused]]. ck_gemm_resolve_mma_op is used by the lds_swizzle
 * write-granularity rule of the validity gate above. */
