// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_deep_fused_conv_pool_value_types_and_spec.c -- chunked C99 port of
 * the value types + spec/signature/grid surface of
 * rocke/instances/common/deep_fused_conv_pool.py (lines 71-385).
 *
 * SCOPE (this part-file):
 *   - ConvAccumulatorEpilogue slice: rocke_conv_acc_epilogue_relu /
 *     rocke_conv_acc_epilogue_is_identity (+ the static tag() helper used by
 *     kernel_name).
 *   - FusedConvPoolProblem (lines 71-105): ctor + conv1_channels / pool_ho /
 *     pool_wo / total_out / short().
 *   - DeepFusedConvPoolSpec (lines 108-204): default ctor + block_size /
 *     effective_conv1_tile_k / kernel_name().
 *   - make_deep_fused_conv_pool_spec (lines 207-283): the tile_m auto-derive.
 *   - is_valid_spec (lines 286-356): the deep-fusion constraint chain + the
 *     leading conv-gate delegation.
 *   - deep_fused_conv_pool_signature (lines 359-378): A/B/Y/W1 + *_bytes.
 *   - deep_fused_conv_pool_grid (lines 381-385).
 *
 * Python integer semantics: every // here operates on non-negative operands
 * (valid conv/pool shapes), so C truncating division matches Python floor
 * division exactly. Shape products that could overflow int32 (total_out) are
 * computed in 64-bit, mirroring Python's arbitrary-precision int.
 *
 * This is PURE COMPUTE: no IR builder is touched here (the scalar-apply +
 * builder-emit helpers live in peer part-files). It depends only on the
 * conv_implicit_gemm value-type port (ConvProblem) and the spec helper
 * (kernel_name_join / SignatureBuilder). The leading underlying-conv gate
 * (is_valid_conv_spec via spec.conv_spec()) is a NAMED GAP: the conv validator
 * rocke_implicit_gemm_conv_is_valid_spec is now public, but spec.conv_spec()'s C
 * port (rocke_deep_fused_conv_pool_spec_conv_spec) requires an ir_builder_t for
 * kernel_name_join, while this is_valid_spec entry is a pure-compute validator
 * with no builder; wiring one in would change the validator's contract. The
 * deep-fusion-specific checks below are byte-faithful; the conv-gate delegation
 * is deferred until a builder-free conv_spec()/validate path exists. */
#include "rocke/instance_deep_fused_conv_pool_internal.h"

/* INTEGRATION: spec.conv_spec() builds a full ImplicitGemmConvSpec value; the
 * deep_fused internal header only forward-uses it as an opaque pointer, so the
 * complete struct + rocke_implicit_gemm_conv_spec_default ctor come from the conv
 * peer's public header here. */
#include "rocke/instance_conv_implicit_gemm.h"

#include "rocke/ir_internal.h" /* rocke_i_set_err (sticky-error helper) */

#include <stdarg.h> /* va_list (interpolated reject reasons) */
#include <stdio.h> /* snprintf / vsnprintf */
#include <string.h> /* strcmp */

/* ------------------------------------------------------------------ *
 * ConvAccumulatorEpilogue value type (slice)
 * ------------------------------------------------------------------ */

/* ConvAccumulatorEpilogue(relu=True): the deep-fusion default for both the
 * conv0 acc epilogue and the conv1 epilogue (bias=0.0, scale=1.0, relu=True,
 * clamp_min=None, clamp_max=None). */
rocke_conv_acc_epilogue_t rocke_conv_acc_epilogue_relu(void)
{
    rocke_conv_acc_epilogue_t e;
    e.relu = true;
    e.bias = 0.0;
    e.scale = 1.0;
    e.has_clamp_min = false;
    e.clamp_min = 0.0;
    e.has_clamp_max = false;
    e.clamp_max = 0.0;
    return e;
}

/* ConvAccumulatorEpilogue.is_identity():
 *   bias==0.0 and scale==1.0 and not relu and
 *   clamp_min is None and clamp_max is None
 *
 * INTEGRATION NOTE: ConvAccumulatorEpilogue is the SHARED value type owned by
 * the conv_implicit_gemm peer, which already provides the canonical
 * rocke_conv_acc_epilogue_is_identity (declared identically in both public
 * headers). Defining it here too produced a multiple-definition link error
 * against instance_conv_implicit_gemm_conv_spec_descriptors.c, so the
 * deep-fused port reuses the peer's definition (same signature, byte-identical
 * predicate) rather than redefining it. */

/* ConvAccumulatorEpilogue.tag() (conv_implicit_gemm.py lines 166-180):
 *   if is_identity(): return ""
 *   pieces = []
 *   if bias != 0.0:   pieces.append(f"bias{bias:g}")
 *   if scale != 1.0:  pieces.append(f"scale{scale:g}")
 *   if relu:          pieces.append("relu")
 *   if clamp_min is not None or clamp_max is not None:
 *       lo = "-inf" if clamp_min is None else f"{clamp_min:g}"
 *       hi = "inf"  if clamp_max is None else f"{clamp_max:g}"
 *       pieces.append(f"clamp{lo}to{hi}")
 *   return "epi_" + "_".join(pieces)
 *
 * Writes the NUL-terminated tag into `out` (capacity out_cap). ROCKE_OK /
 * ROCKE_ERR_VALUE on a too-small buffer. Python's `:g` float formatting maps to
 * C's "%g"; the deep-fusion shapes only ever exercise the relu-only branch
 * ("epi_relu"), but the general form is reproduced for byte-faithfulness. */
static rocke_status_t
    dfcp_epilogue_tag(const rocke_conv_acc_epilogue_t* epi, char* out, size_t out_cap)
{
    char piece[80]; /* fits "clamp" + 2x 32-byte float reprs + "to" + NUL */
    size_t pos = 0;
    bool first = true;
    int wrote;

    if(epi == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }

    if(rocke_conv_acc_epilogue_is_identity(epi))
    {
        out[0] = '\0';
        return ROCKE_OK;
    }

    /* Seed with the "epi_" prefix; pieces are joined by "_". */
    wrote = snprintf(out, out_cap, "epi_");
    if(wrote < 0 || (size_t)wrote >= out_cap)
    {
        return ROCKE_ERR_VALUE;
    }
    pos = (size_t)wrote;

#define DFCP_APPEND_PIECE()                                              \
    do                                                                   \
    {                                                                    \
        const char* sep = first ? "" : "_";                              \
        int w2 = snprintf(out + pos, out_cap - pos, "%s%s", sep, piece); \
        if(w2 < 0 || (size_t)w2 >= out_cap - pos)                        \
        {                                                                \
            return ROCKE_ERR_VALUE;                                      \
        }                                                                \
        pos += (size_t)w2;                                               \
        first = false;                                                   \
    } while(0)

    if(epi->bias != 0.0)
    {
        snprintf(piece, sizeof(piece), "bias%g", epi->bias);
        DFCP_APPEND_PIECE();
    }
    if(epi->scale != 1.0)
    {
        snprintf(piece, sizeof(piece), "scale%g", epi->scale);
        DFCP_APPEND_PIECE();
    }
    if(epi->relu)
    {
        snprintf(piece, sizeof(piece), "relu");
        DFCP_APPEND_PIECE();
    }
    if(epi->has_clamp_min || epi->has_clamp_max)
    {
        char lo[32];
        char hi[32];
        if(epi->has_clamp_min)
        {
            snprintf(lo, sizeof(lo), "%g", epi->clamp_min);
        }
        else
        {
            snprintf(lo, sizeof(lo), "-inf");
        }
        if(epi->has_clamp_max)
        {
            snprintf(hi, sizeof(hi), "%g", epi->clamp_max);
        }
        else
        {
            snprintf(hi, sizeof(hi), "inf");
        }
        snprintf(piece, sizeof(piece), "clamp%sto%s", lo, hi);
        DFCP_APPEND_PIECE();
    }

#undef DFCP_APPEND_PIECE
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * FusedConvPoolProblem
 * ------------------------------------------------------------------ */

/* FusedConvPoolProblem(conv, conv1_k=0, pool_y=2, pool_x=2,
 *                      pool_stride_h=2, pool_stride_w=2). */
rocke_fused_conv_pool_problem_t rocke_fused_conv_pool_problem_make(rocke_conv_problem_t conv,
                                                                   int conv1_k,
                                                                   int pool_y,
                                                                   int pool_x,
                                                                   int pool_stride_h,
                                                                   int pool_stride_w)
{
    rocke_fused_conv_pool_problem_t p;
    p.conv = conv;
    p.conv1_k = conv1_k;
    p.pool_y = pool_y;
    p.pool_x = pool_x;
    p.pool_stride_h = pool_stride_h;
    p.pool_stride_w = pool_stride_w;
    return p;
}

/* conv1_channels:  conv.K if conv1_k <= 0 else conv1_k */
int rocke_fused_conv_pool_problem_conv1_channels(const rocke_fused_conv_pool_problem_t* p)
{
    return (p->conv1_k <= 0) ? p->conv.K : p->conv1_k;
}

/* pool_ho:  (conv.Ho - pool_y) // pool_stride_h + 1 */
int rocke_fused_conv_pool_problem_pool_ho(const rocke_fused_conv_pool_problem_t* p)
{
    return (rocke_conv_problem_ho(&p->conv) - p->pool_y) / p->pool_stride_h + 1;
}

/* pool_wo:  (conv.Wo - pool_x) // pool_stride_w + 1 */
int rocke_fused_conv_pool_problem_pool_wo(const rocke_fused_conv_pool_problem_t* p)
{
    return (rocke_conv_problem_wo(&p->conv) - p->pool_x) / p->pool_stride_w + 1;
}

/* total_out:  conv.N * pool_ho * pool_wo * conv1_channels (64-bit). */
long long rocke_fused_conv_pool_problem_total_out(const rocke_fused_conv_pool_problem_t* p)
{
    long long n = (long long)p->conv.N;
    long long pho = (long long)rocke_fused_conv_pool_problem_pool_ho(p);
    long long pwo = (long long)rocke_fused_conv_pool_problem_pool_wo(p);
    long long k1 = (long long)rocke_fused_conv_pool_problem_conv1_channels(p);
    return n * pho * pwo * k1;
}

/* short() ->
 *   f"{conv.short()}_K1{conv1_channels}_pool{pool_y}x{pool_x}_s{psh}x{psw}" */
rocke_status_t rocke_fused_conv_pool_problem_short(const rocke_fused_conv_pool_problem_t* p,
                                                   char* out,
                                                   size_t out_cap,
                                                   size_t* out_len)
{
    char conv_short[128];
    rocke_status_t st;
    int written;

    if(p == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_conv_problem_short(&p->conv, conv_short, sizeof(conv_short), NULL);
    if(st != ROCKE_OK)
    {
        return st;
    }

    written = snprintf(out,
                       out_cap,
                       "%s_K1%d_pool%dx%d_s%dx%d",
                       conv_short,
                       rocke_fused_conv_pool_problem_conv1_channels(p),
                       p->pool_y,
                       p->pool_x,
                       p->pool_stride_h,
                       p->pool_stride_w);
    if(written < 0 || (size_t)written >= out_cap)
    {
        return ROCKE_ERR_VALUE;
    }
    if(out_len != NULL)
    {
        *out_len = (size_t)written;
    }
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * DeepFusedConvPoolSpec
 * ------------------------------------------------------------------ */

/* The DeepFusedConvPoolSpec dataclass field defaults (lines 118-138). The
 * caller fills `problem`. */
rocke_deep_fused_conv_pool_spec_t rocke_deep_fused_conv_pool_spec_default(void)
{
    rocke_deep_fused_conv_pool_spec_t s;
    /* problem set to a minimal valid ConvProblem; caller overwrites. */
    s.problem.conv = rocke_conv_problem_default(1, 1, 1, 1, 1, 1, 1);
    s.problem.conv1_k = 0;
    s.problem.pool_y = 2;
    s.problem.pool_x = 2;
    s.problem.pool_stride_h = 2;
    s.problem.pool_stride_w = 2;
    s.name = "rocke_deep_fused_conv_pool";
    s.tile_m = 128;
    s.tile_n = 32;
    s.tile_k = 16;
    s.conv1_tile_k = 0;
    s.pool_tile_h = 4;
    s.pool_tile_w = 8;
    s.warp_m = 2;
    s.warp_n = 1;
    s.warp_tile_m = 32;
    s.warp_tile_n = 32;
    s.warp_tile_k = 16;
    s.wave_size = 64;
    s.pipeline = "mem";
    s.async_dma = false;
    s.unroll_k = false;
    s.acc_epilogue = rocke_conv_acc_epilogue_relu();
    s.conv1_epilogue = rocke_conv_acc_epilogue_relu();
    s.cache_input_footprint = false;
    s.direct_conv0_from_input_cache = false;
    return s;
}

/* block_size:  warp_m * warp_n * wave_size */
int rocke_deep_fused_conv_pool_spec_block_size(const rocke_deep_fused_conv_pool_spec_t* spec)
{
    return spec->warp_m * spec->warp_n * spec->wave_size;
}

/* effective_conv1_tile_k:  tile_k if conv1_tile_k <= 0 else conv1_tile_k */
int rocke_deep_fused_conv_pool_spec_effective_conv1_tile_k(
    const rocke_deep_fused_conv_pool_spec_t* spec)
{
    return (spec->conv1_tile_k <= 0) ? spec->tile_k : spec->conv1_tile_k;
}

/* kernel_name() (lines 148-171):
 *   conv1_k_part = f"c1k{eff_c1k}" if eff_c1k != tile_k else ""
 *   kernel_name_join(name, problem.short(), f"t{m}x{n}x{k}", conv1_k_part,
 *                    f"pt{pth}x{ptw}", f"w{wm}x{wn}",
 *                    f"a{wtm}x{wtn}x{wtk}", f"{pipeline}_{'async'|'sync'}",
 *                    acc_epilogue.tag(), conv1_epilogue.tag(),
 *                    "cshuffle_conv1_pool",
 *                    flags={"icache":cache_input_footprint,
 *                           "directa":direct_conv0_from_input_cache,
 *                           "unrollk":unroll_k}) */
rocke_status_t rocke_deep_fused_conv_pool_spec_kernel_name(
    const rocke_deep_fused_conv_pool_spec_t* spec, char* out, size_t out_cap)
{
    char short_part[160];
    char tile_part[48];
    char conv1_k_part[24];
    char pt_part[24];
    char w_part[24];
    char a_part[32];
    char pipe_part[32];
    char acc_tag[64];
    char conv1_tag[64];
    const char* parts[10];
    const char* flag_names[3];
    int flag_on[3];
    int eff_c1k;
    int wrote;
    rocke_status_t st;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_fused_conv_pool_problem_short(&spec->problem, short_part, sizeof(short_part), NULL);
    if(st != ROCKE_OK)
    {
        return st;
    }

    wrote = snprintf(
        tile_part, sizeof(tile_part), "t%dx%dx%d", spec->tile_m, spec->tile_n, spec->tile_k);
    if(wrote < 0 || (size_t)wrote >= sizeof(tile_part))
    {
        return ROCKE_ERR_VALUE;
    }

    eff_c1k = rocke_deep_fused_conv_pool_spec_effective_conv1_tile_k(spec);
    if(eff_c1k != spec->tile_k)
    {
        wrote = snprintf(conv1_k_part, sizeof(conv1_k_part), "c1k%d", eff_c1k);
        if(wrote < 0 || (size_t)wrote >= sizeof(conv1_k_part))
        {
            return ROCKE_ERR_VALUE;
        }
    }
    else
    {
        conv1_k_part[0] = '\0';
    }

    wrote = snprintf(pt_part, sizeof(pt_part), "pt%dx%d", spec->pool_tile_h, spec->pool_tile_w);
    if(wrote < 0 || (size_t)wrote >= sizeof(pt_part))
    {
        return ROCKE_ERR_VALUE;
    }

    wrote = snprintf(w_part, sizeof(w_part), "w%dx%d", spec->warp_m, spec->warp_n);
    if(wrote < 0 || (size_t)wrote >= sizeof(w_part))
    {
        return ROCKE_ERR_VALUE;
    }

    wrote = snprintf(a_part,
                     sizeof(a_part),
                     "a%dx%dx%d",
                     spec->warp_tile_m,
                     spec->warp_tile_n,
                     spec->warp_tile_k);
    if(wrote < 0 || (size_t)wrote >= sizeof(a_part))
    {
        return ROCKE_ERR_VALUE;
    }

    wrote = snprintf(
        pipe_part, sizeof(pipe_part), "%s_%s", spec->pipeline, spec->async_dma ? "async" : "sync");
    if(wrote < 0 || (size_t)wrote >= sizeof(pipe_part))
    {
        return ROCKE_ERR_VALUE;
    }

    /* acc_epilogue.tag() / conv1_epilogue.tag() -- byte-faithful tag() from
     * conv_implicit_gemm.py (relu-only deep-fusion default -> "epi_relu"). */
    st = dfcp_epilogue_tag(&spec->acc_epilogue, acc_tag, sizeof(acc_tag));
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = dfcp_epilogue_tag(&spec->conv1_epilogue, conv1_tag, sizeof(conv1_tag));
    if(st != ROCKE_OK)
    {
        return st;
    }

    parts[0] = short_part;
    parts[1] = tile_part;
    parts[2] = conv1_k_part;
    parts[3] = pt_part;
    parts[4] = w_part;
    parts[5] = a_part;
    parts[6] = pipe_part;
    parts[7] = acc_tag;
    parts[8] = conv1_tag;
    parts[9] = "cshuffle_conv1_pool";

    flag_names[0] = "icache";
    flag_names[1] = "directa";
    flag_names[2] = "unrollk";
    flag_on[0] = spec->cache_input_footprint ? 1 : 0;
    flag_on[1] = spec->direct_conv0_from_input_cache ? 1 : 0;
    flag_on[2] = spec->unroll_k ? 1 : 0;

    return rocke_kernel_name_join(
        spec->name, parts, 10, flag_names, flag_on, 3, out, out_cap, NULL);
}

/* ------------------------------------------------------------------ *
 * make_deep_fused_conv_pool_spec (lines 207-283)
 * ------------------------------------------------------------------ */
rocke_deep_fused_conv_pool_spec_t
    rocke_make_deep_fused_conv_pool_spec(int n,
                                         int h,
                                         int w,
                                         int c,
                                         int k0,
                                         int k1,
                                         int r,
                                         int s,
                                         int pool_tile_h,
                                         int pool_tile_w,
                                         int tile_n,
                                         int tile_k,
                                         int conv1_tile_k,
                                         int warp_m,
                                         int warp_n,
                                         int warp_tile_m,
                                         int warp_tile_n,
                                         int warp_tile_k,
                                         int wave_size,
                                         const char* name,
                                         const char* pipeline,
                                         bool unroll_k,
                                         bool async_dma,
                                         bool cache_input_footprint,
                                         bool direct_conv0_from_input_cache)
{
    rocke_deep_fused_conv_pool_spec_t spec;
    rocke_conv_problem_t conv;
    rocke_fused_conv_pool_problem_t problem;
    int conv_tile_h;
    int conv_tile_w;
    int tile_m;

    /* ConvProblem(N=n, Hi=h, Wi=w, C=c, K=k0, R=r, S=s,
     *             sH=1, sW=1, pH=1, pW=1, dH=1, dW=1) */
    conv = rocke_conv_problem_make(n, h, w, c, k0, r, s, 1, 1, 1, 1, 1, 1);

    /* FusedConvPoolProblem(conv=conv, conv1_k=k1) -- pool_* take the dataclass
     * defaults (2/2/2/2). */
    problem = rocke_fused_conv_pool_problem_make(conv, k1, 2, 2, 2, 2);

    /* conv_tile_h = pool_tile_h * pool_stride_h
     * conv_tile_w = pool_tile_w * pool_stride_w
     * tile_m = conv_tile_h * conv_tile_w */
    conv_tile_h = pool_tile_h * problem.pool_stride_h;
    conv_tile_w = pool_tile_w * problem.pool_stride_w;
    tile_m = conv_tile_h * conv_tile_w;

    spec = rocke_deep_fused_conv_pool_spec_default();
    spec.problem = problem;
    spec.name = (name != NULL) ? name : "rocke_deep_fused_conv_pool";
    spec.tile_m = tile_m;
    spec.tile_n = tile_n;
    spec.tile_k = tile_k;
    spec.conv1_tile_k = conv1_tile_k;
    spec.pool_tile_h = pool_tile_h;
    spec.pool_tile_w = pool_tile_w;
    spec.warp_m = warp_m;
    spec.warp_n = warp_n;
    spec.warp_tile_m = warp_tile_m;
    spec.warp_tile_n = warp_tile_n;
    spec.warp_tile_k = warp_tile_k;
    spec.wave_size = wave_size;
    spec.pipeline = (pipeline != NULL) ? pipeline : "mem";
    spec.unroll_k = unroll_k;
    spec.async_dma = async_dma;
    spec.cache_input_footprint = cache_input_footprint;
    spec.direct_conv0_from_input_cache = direct_conv0_from_input_cache;
    /* acc_epilogue=ConvAccumulatorEpilogue(relu=True); conv1_epilogue defaults
     * to the same relu=True in the dataclass. */
    spec.acc_epilogue = rocke_conv_acc_epilogue_relu();
    spec.conv1_epilogue = rocke_conv_acc_epilogue_relu();
    return spec;
}

/* ------------------------------------------------------------------ *
 * is_valid_spec (lines 286-356)
 * ------------------------------------------------------------------ */
/* printf-style so the reason string is byte-identical to the Python
 * ValueError f-strings (which interpolate K / tile_n / pool dims / etc.). */
static bool reject(char* reason, size_t cap, const char* fmt, ...)
{
    if(reason != NULL && cap > 0)
    {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(reason, cap, fmt, ap);
        va_end(ap);
    }
    return false;
}

bool rocke_deep_fused_conv_pool_is_valid_spec(const rocke_deep_fused_conv_pool_spec_t* spec,
                                              const char* arch,
                                              char* reason,
                                              size_t reason_cap)
{
    const rocke_fused_conv_pool_problem_t* p;
    const rocke_conv_problem_t* c;
    int conv_tile_h;
    int conv_tile_w;
    int conv1_tile_k;
    int pool_ho;
    int pool_wo;
    int conv1_channels;

    (void)arch; /* NULL => "gfx950"; the underlying conv gate is a peer. */

    if(spec == NULL)
    {
        return reject(reason, reason_cap, "spec is NULL");
    }

    /* NAMED GAP (blocked on a builder-free conv_spec()): the Python prologue first
     * runs
     *   conv_spec = spec.conv_spec()
     *   ok, why = is_valid_conv_spec(conv_spec, arch=arch)
     *   if not ok: return False, why
     * The per-family gate is_valid_conv_spec IS public now
     * (rocke_implicit_gemm_conv_is_valid_spec), but spec.conv_spec()'s only C port
     * (rocke_deep_fused_conv_pool_spec_conv_spec) takes an ir_builder_t* for
     * kernel_name_join; this validator has no builder and constructing one would
     * change its pure-compute contract. Prepend the (ok, why) delegation here once
     * a builder-free conv_spec()/validate path exists. The deep-fusion-specific
     * checks below are ported byte-faithfully. */

    p = &spec->problem;
    c = &p->conv;

    /* (pool_y, pool_x, pool_stride_h, pool_stride_w) != (2,2,2,2) */
    if(!(p->pool_y == 2 && p->pool_x == 2 && p->pool_stride_h == 2 && p->pool_stride_w == 2))
    {
        return reject(reason, reason_cap, "v1 supports only 2x2 stride-2 maxpool");
    }
    pool_ho = rocke_fused_conv_pool_problem_pool_ho(p);
    pool_wo = rocke_fused_conv_pool_problem_pool_wo(p);
    if(pool_ho <= 0 || pool_wo <= 0)
    {
        return reject(reason, reason_cap, "pool output dimensions must be positive");
    }
    if(!(strcmp(spec->pipeline, "mem") == 0 || strcmp(spec->pipeline, "compv3") == 0
         || strcmp(spec->pipeline, "compv4") == 0))
    {
        return reject(reason, reason_cap, "unsupported pipeline '%s'", spec->pipeline);
    }
    if(spec->async_dma && (spec->cache_input_footprint || spec->direct_conv0_from_input_cache))
    {
        return reject(reason,
                      reason_cap,
                      "async_dma is only supported with the default conv0 "
                      "A-load path");
    }
    if(spec->unroll_k && spec->async_dma)
    {
        return reject(reason,
                      reason_cap,
                      "unroll_k and async_dma are mutually exclusive K-loop "
                      "schedules");
    }
    if(spec->unroll_k && (spec->cache_input_footprint || spec->direct_conv0_from_input_cache))
    {
        return reject(reason,
                      reason_cap,
                      "unroll_k is only supported with the default conv0 A-load "
                      "path");
    }
    if(c->N != 1)
    {
        return reject(reason, reason_cap, "v1 tiled schedule supports only N=1 (got N=%d)", c->N);
    }
    if(spec->pool_tile_h <= 0 || spec->pool_tile_w <= 0)
    {
        return reject(reason, reason_cap, "pool_tile_h and pool_tile_w must be positive");
    }
    conv_tile_h = spec->pool_tile_h * p->pool_stride_h;
    conv_tile_w = spec->pool_tile_w * p->pool_stride_w;
    if(spec->tile_m != conv_tile_h * conv_tile_w)
    {
        return reject(reason,
                      reason_cap,
                      "tile_m=%d must equal rectangular conv tile %dx%d=%d",
                      spec->tile_m,
                      conv_tile_h,
                      conv_tile_w,
                      conv_tile_h * conv_tile_w);
    }
    if((pool_ho % spec->pool_tile_h) || (pool_wo % spec->pool_tile_w))
    {
        return reject(reason,
                      reason_cap,
                      "v1 requires pool dims (%d, %d) divisible by "
                      "pool tile (%d, %d)",
                      pool_ho,
                      pool_wo,
                      spec->pool_tile_h,
                      spec->pool_tile_w);
    }
    if(c->K > spec->tile_n)
    {
        return reject(reason,
                      reason_cap,
                      "v1 requires one CTA to own all conv channels: "
                      "K=%d > tile_n=%d",
                      c->K,
                      spec->tile_n);
    }
    conv1_channels = rocke_fused_conv_pool_problem_conv1_channels(p);
    if(conv1_channels > spec->tile_n)
    {
        return reject(reason,
                      reason_cap,
                      "v1 requires one CTA to own all conv1 channels: "
                      "K1=%d > tile_n=%d",
                      conv1_channels,
                      spec->tile_n);
    }
    if(c->K % 8)
    {
        return reject(reason, reason_cap, "v1 W1 loader requires conv0 channels divisible by 8");
    }
    if(spec->tile_m % (spec->warp_m * spec->warp_tile_m))
    {
        return reject(reason, reason_cap, "tile_m must divide warp_m * warp_tile_m");
    }
    if(spec->tile_n % (spec->warp_n * spec->warp_tile_n))
    {
        return reject(reason, reason_cap, "tile_n must divide warp_n * warp_tile_n");
    }
    conv1_tile_k = rocke_deep_fused_conv_pool_spec_effective_conv1_tile_k(spec);
    if(conv1_tile_k <= 0)
    {
        return reject(reason, reason_cap, "conv1_tile_k must be positive (got %d)", conv1_tile_k);
    }
    if(conv1_tile_k % spec->warp_tile_k)
    {
        return reject(reason, reason_cap, "conv1_tile_k must be divisible by warp_tile_k");
    }
    if(conv1_tile_k < spec->warp_tile_k)
    {
        return reject(reason, reason_cap, "conv1_tile_k must be at least one warp_tile_k");
    }

    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "ok");
    }
    return true;
}

/* ------------------------------------------------------------------ *
 * deep_fused_conv_pool_signature (lines 359-378)
 * ------------------------------------------------------------------ */
rocke_status_t rocke_deep_fused_conv_pool_signature(rocke_arena_t* arena,
                                                    const rocke_deep_fused_conv_pool_spec_t* spec,
                                                    const rocke_sig_entry_t** out_items,
                                                    size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t st;

    (void)spec; /* signature is shape-independent. */
    if(arena == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
    {
        return st;
    }
    /* SignatureBuilder().ptr("A","f16").ptr("B","f16").ptr("Y","f16")
     *   .ptr("W1","f16").scalar("W1_bytes","i32").scalar("A_bytes","i32")
     *   .scalar("B_bytes","i32").scalar("Y_bytes","i32").build()
     * W1 declared before the byte-size scalars so the HIP packed-args ABI keeps
     * all 64-bit pointer args aligned. */
    rocke_signature_builder_ptr(&sb, "A", "f16", NULL);
    rocke_signature_builder_ptr(&sb, "B", "f16", NULL);
    rocke_signature_builder_ptr(&sb, "Y", "f16", NULL);
    rocke_signature_builder_ptr(&sb, "W1", "f16", NULL);
    rocke_signature_builder_scalar(&sb, "W1_bytes", "i32");
    rocke_signature_builder_scalar(&sb, "A_bytes", "i32");
    rocke_signature_builder_scalar(&sb, "B_bytes", "i32");
    rocke_signature_builder_scalar(&sb, "Y_bytes", "i32");
    return rocke_signature_builder_build(&sb, out_items, out_count);
}

/* ------------------------------------------------------------------ *
 * deep_fused_conv_pool_grid (lines 381-385)
 * ------------------------------------------------------------------ */
rocke_status_t rocke_deep_fused_conv_pool_grid(const rocke_deep_fused_conv_pool_spec_t* spec,
                                               int out[3])
{
    const rocke_fused_conv_pool_problem_t* p;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    p = &spec->problem;
    /* (1, pool_ho // pool_tile_h, pool_wo // pool_tile_w) */
    out[0] = 1;
    out[1] = rocke_fused_conv_pool_problem_pool_ho(p) / spec->pool_tile_h;
    out[2] = rocke_fused_conv_pool_problem_pool_wo(p) / spec->pool_tile_w;
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * spec.conv_spec()  (deep_fused_conv_pool.py lines 173-204)
 * ------------------------------------------------------------------ *
 *
 * Builds the wrapped ImplicitGemmConvSpec from this deep-fusion spec. The C
 * signature returns a const pointer (the conv builder + ctx capture it for the
 * whole build), so the spec value (and its computed `name` string) are
 * allocated in the builder arena -- arena lifetime == build lifetime, matching
 * the Python local's reachability. Returns NULL with the builder error set on a
 * name-build / allocation failure (the driver propagates that as a NULL build).
 *
 * Python:
 *   epilogue = "default" if wave_size == 32 else "cshuffle"
 *   conv_name = kernel_name_join(name,
 *                  f"c1k{eff_c1k}" if eff_c1k != tile_k else "")
 *   return ImplicitGemmConvSpec(problem=problem.conv, name=conv_name,
 *       tile_m=.., tile_n=.., tile_k=.., warp_m=.., warp_n=..,
 *       warp_tile_m=.., warp_tile_n=.., warp_tile_k=.., wave_size=..,
 *       pipeline=.., epilogue=epilogue, async_dma=.., unroll_k=..,
 *       acc_epilogue=acc_epilogue)
 * Every other ImplicitGemmConvSpec field keeps its dataclass default. */
/* C++ build: cross-TU C-ABI helper (peer entry-glue TUs forward-declare it
 * as extern "C"); define with C linkage so the symbol is not mangled. No effect
 * in C. */
#ifdef __cplusplus
extern "C"
#endif
    const rocke_implicit_gemm_conv_spec_t*
    rocke_deep_fused_conv_pool_spec_conv_spec(rocke_ir_builder_t* b,
                                              const rocke_deep_fused_conv_pool_spec_t* spec)
{
    char conv1_k_part[24];
    char name_buf[256];
    const char* parts[1];
    int eff_c1k;
    int wrote;
    rocke_status_t st;
    rocke_implicit_gemm_conv_spec_t* cs;
    char* name_owned;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }

    /* conv1_k_part = f"c1k{eff_c1k}" if eff_c1k != tile_k else "" */
    eff_c1k = rocke_deep_fused_conv_pool_spec_effective_conv1_tile_k(spec);
    if(eff_c1k != spec->tile_k)
    {
        wrote = snprintf(conv1_k_part, sizeof(conv1_k_part), "c1k%d", eff_c1k);
        if(wrote < 0 || (size_t)wrote >= sizeof(conv1_k_part))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "deep fused conv/pool: conv_spec name overflow");
            return NULL;
        }
    }
    else
    {
        conv1_k_part[0] = '\0';
    }

    /* conv_name = kernel_name_join(name, conv1_k_part) */
    parts[0] = conv1_k_part;
    st = rocke_kernel_name_join(
        spec->name, parts, 1, NULL, NULL, 0, name_buf, sizeof(name_buf), NULL);
    if(st != ROCKE_OK)
    {
        rocke_i_set_err(b, st, "deep fused conv/pool: conv_spec name build failed");
        return NULL;
    }

    /* Arena-own the spec value + its name string (build-lifetime storage). */
    cs = (rocke_implicit_gemm_conv_spec_t*)rocke_arena_alloc(&b->arena, sizeof(*cs));
    name_owned = rocke_arena_strdup(&b->arena, name_buf);
    if(cs == NULL || name_owned == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "deep fused conv/pool: conv_spec arena alloc failed");
        return NULL;
    }

    *cs = rocke_implicit_gemm_conv_spec_default();
    cs->problem = spec->problem.conv;
    cs->name = name_owned;
    cs->tile_m = spec->tile_m;
    cs->tile_n = spec->tile_n;
    cs->tile_k = spec->tile_k;
    cs->warp_m = spec->warp_m;
    cs->warp_n = spec->warp_n;
    cs->warp_tile_m = spec->warp_tile_m;
    cs->warp_tile_n = spec->warp_tile_n;
    cs->warp_tile_k = spec->warp_tile_k;
    cs->wave_size = spec->wave_size;
    cs->pipeline = spec->pipeline;
    /* epilogue = "default" if wave_size == 32 else "cshuffle" */
    cs->epilogue = (spec->wave_size == 32) ? "default" : "cshuffle";
    cs->async_dma = spec->async_dma;
    cs->unroll_k = spec->unroll_k;
    cs->acc_epilogue = spec->acc_epilogue;

    return cs;
}
