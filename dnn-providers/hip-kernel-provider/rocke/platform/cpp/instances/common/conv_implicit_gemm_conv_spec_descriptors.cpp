// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_implicit_gemm_conv_spec_descriptors.c -- C99 port of the SPEC
 * value-type + epilogue helpers + arch-aware validity gate + the user-visible
 * coordinate-transform-DAG descriptor builders of
 * rocke/instances/common/conv_implicit_gemm.py (lines 142-629).
 *
 * SCOPE (this TU): the IR-free / transform-DAG surface. NO kernel-body IR.
 *
 *   Python (conv_implicit_gemm.py)             C99 (this file)
 *   ----------------------------------------   ----------------------------------
 *   ConvAccumulatorEpilogue (142-181)          rocke_conv_acc_epilogue_default /
 *     .is_identity() / .tag()                    _is_identity / _tag
 *   ImplicitGemmConvSpec (183-376)             rocke_implicit_gemm_conv_spec_default
 *     .block_size / .k_atoms_per_tile_k          + the @property accessors
 *     .mfmas_per_warp_m / _n
 *     .kernel_name() / .validate()               _kernel_name / _validate
 *   is_valid_spec(spec, arch) (383-466)        rocke_implicit_gemm_conv_is_valid_spec
 *   _conv_mma_family(arch) (469-472)           rocke_conv_mma_family
 *   _resolve_conv_op(spec, arch) (475-501)     rocke_conv_resolve_op
 *   make_a_descriptor (509-576)                rocke_conv_make_a_descriptor
 *   make_b_descriptor (579-613)                rocke_conv_make_b_descriptor
 *   make_d_descriptor (616-629)                rocke_conv_make_d_descriptor
 *
 * The descriptor builders emit the byte-identical builder-call sequence the
 * Python TensorDescriptor.naive(...).transform(...) chain produces. The pure
 * value/string helpers reproduce the Python return values bit-for-bit (the
 * reason / kernel-name strings never enter the IR but a sweep driver sees the
 * same accept/reject + identifier).
 *
 * effective_lds_layout() IS ported here (LdsLayout peer): the three accessors
 * rocke_conv_lds_layout_validate / _validate_for_async /
 * rocke_implicit_gemm_conv_spec_effective_lds_layout reproduce
 * rocke/helpers/layouts.py + ImplicitGemmConvSpec.effective_lds_layout, and
 * spec.validate() drives them in Python order (so async + lds_k_pad!=0 rejects
 * with the validate_for_async message). The lds_layout override branch is the
 * only piece still a peer port (no C constructor for the opaque field).
 */

#include "rocke/instance_conv_implicit_gemm.h"
#include "rocke/instance_conv_implicit_gemm_internal.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_*, has_shape, op_for_shape */
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, rocke_choose_load_vec */
#include "rocke/helper_rocke.helpers.transforms.h" /* descriptor DAG + transforms */

/* Reproduce str(KeyError(_build_target message)) for an unknown gfx target:
 *
 *   Python _build_target: raise KeyError(
 *     f"unknown gfx target {gfx!r}; known: {sorted(specs)}. "
 *     f"Add a row to {_DATA_FILE.name}.")
 *   is_valid_spec: except KeyError as e: return False, str(e)
 *
 * str(KeyError(msg)) == repr(msg); the single quotes in the message make Python
 * wrap it in DOUBLE quotes. sorted(specs) renders as ['gfx...', 'gfx...'].
 * rocke_known_arches() == tuple(sorted(_load_specs())). Mirrors fmha_arch.cpp. */
static void rocke_cspec__set_unknown_arch_reason(char* out, size_t out_cap, const char* gfx)
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
 *  ConvAccumulatorEpilogue   (Python lines 142-181)
 * ===================================================================== */

rocke_conv_acc_epilogue_t rocke_conv_acc_epilogue_default(void)
{
    rocke_conv_acc_epilogue_t e;
    memset(&e, 0, sizeof(e));
    /* @dataclass defaults:
     *   bias=0.0, scale=1.0, relu=False, clamp_min=None, clamp_max=None */
    e.bias = 0.0;
    e.scale = 1.0;
    e.relu = false;
    e.has_clamp_min = false;
    e.clamp_min = 0.0;
    e.has_clamp_max = false;
    e.clamp_max = 0.0;
    return e;
}

bool rocke_conv_acc_epilogue_is_identity(const rocke_conv_acc_epilogue_t* epi)
{
    if(epi == NULL)
    {
        return true;
    }
    /* return (bias == 0.0 and scale == 1.0 and not relu
     *         and clamp_min is None and clamp_max is None) */
    return (epi->bias == 0.0 && epi->scale == 1.0 && !epi->relu && !epi->has_clamp_min
            && !epi->has_clamp_max);
}

/* Format a double using Python's "%g" repr (the f"{x:g}" used by tag()).
 * The C "%g" conversion matches Python's general format for these values. */
static void rocke_conv_g(char* out, size_t cap, double v)
{
    snprintf(out, cap, "%g", v);
}

rocke_status_t
    rocke_conv_acc_epilogue_tag(const rocke_conv_acc_epilogue_t* epi, char* out, size_t out_cap)
{
    /* pieces: List[str] = []
     * if bias != 0.0: pieces.append(f"bias{bias:g}")
     * if scale != 1.0: pieces.append(f"scale{scale:g}")
     * if relu: pieces.append("relu")
     * if clamp_min is not None or clamp_max is not None:
     *     lo = "-inf" if clamp_min is None else f"{clamp_min:g}"
     *     hi = "inf"  if clamp_max is None else f"{clamp_max:g}"
     *     pieces.append(f"clamp{lo}to{hi}")
     * return "epi_" + "_".join(pieces)  (or "" when identity) */
    char body[256];
    char piece[112]; /* fits "clamp" + 2x 48-byte float reprs + "to" + NUL */
    char numbuf[48];
    size_t blen = 0;
    int wrote_any = 0;

    if(epi == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }

    if(rocke_conv_acc_epilogue_is_identity(epi))
    {
        if(out_cap < 1)
        {
            return ROCKE_ERR_VALUE;
        }
        out[0] = '\0';
        return ROCKE_OK;
    }

    body[0] = '\0';

#define ROCKE_TAG_APPEND(s)               \
    do                                    \
    {                                     \
        size_t _l = strlen(s);            \
        if(wrote_any)                     \
        {                                 \
            if(blen + 1 >= sizeof(body))  \
                return ROCKE_ERR_VALUE;   \
            body[blen++] = '_';           \
            body[blen] = '\0';            \
        }                                 \
        if(blen + _l >= sizeof(body))     \
            return ROCKE_ERR_VALUE;       \
        memcpy(body + blen, (s), _l + 1); \
        blen += _l;                       \
        wrote_any = 1;                    \
    } while(0)

    if(epi->bias != 0.0)
    {
        rocke_conv_g(numbuf, sizeof(numbuf), epi->bias);
        snprintf(piece, sizeof(piece), "bias%s", numbuf);
        ROCKE_TAG_APPEND(piece);
    }
    if(epi->scale != 1.0)
    {
        rocke_conv_g(numbuf, sizeof(numbuf), epi->scale);
        snprintf(piece, sizeof(piece), "scale%s", numbuf);
        ROCKE_TAG_APPEND(piece);
    }
    if(epi->relu)
    {
        ROCKE_TAG_APPEND("relu");
    }
    if(epi->has_clamp_min || epi->has_clamp_max)
    {
        char lo[48];
        char hi[48];
        if(!epi->has_clamp_min)
        {
            snprintf(lo, sizeof(lo), "-inf");
        }
        else
        {
            rocke_conv_g(lo, sizeof(lo), epi->clamp_min);
        }
        if(!epi->has_clamp_max)
        {
            snprintf(hi, sizeof(hi), "inf");
        }
        else
        {
            rocke_conv_g(hi, sizeof(hi), epi->clamp_max);
        }
        /* piece must hold "clamp" + lo + "to" + hi + NUL; lo/hi are sized 48 so
         * the worst case is 5+47+2+47+1 = 102 -- guaranteed to fit (no codegen
         * truncation). */
        snprintf(piece, sizeof(piece), "clamp%sto%s", lo, hi);
        ROCKE_TAG_APPEND(piece);
    }

#undef ROCKE_TAG_APPEND

    /* "epi_" + body */
    if(snprintf(out, out_cap, "epi_%s", body) < 0 || strlen("epi_") + blen >= out_cap)
    {
        return ROCKE_ERR_VALUE;
    }
    return ROCKE_OK;
}

/* ===================================================================== *
 *  ImplicitGemmConvSpec   (Python lines 183-376)
 * ===================================================================== */

rocke_implicit_gemm_conv_spec_t rocke_implicit_gemm_conv_spec_default(void)
{
    rocke_implicit_gemm_conv_spec_t s;
    memset(&s, 0, sizeof(s));

    /* problem has no Python default (required) -> zero-init via ConvProblem
     * defaulted optional fields; caller must set the required dims. */
    s.problem = rocke_conv_problem_default(0, 0, 0, 0, 0, 0, 0);

    s.name = "conv_igemm";

    s.tile_m = 64;
    s.tile_n = 64;
    s.tile_k = 64;

    s.warp_m = 2;
    s.warp_n = 2;

    s.warp_tile_m = 32;
    s.warp_tile_n = 32;
    s.warp_tile_k = 16;

    s.wave_size = 64;

    s.pipeline = "mem";
    s.epilogue = "default";
    s.async_dma = false;
    s.unroll_k = false;

    s.has_lds_k_pad = false;
    s.lds_k_pad = 0;
    s.lds_layout = NULL;

    s.chiplet_swizzle = false;
    s.chiplet_wgm = 8;
    s.chiplet_num_xcds = 8;
    s.chiplet_chunk_size = 64;

    s.has_waves_per_eu = false;
    s.waves_per_eu = 0;

    s.k0_k1_split = false;
    s.groups = 1;

    /* #8624 vector-sizes-as-args: default None => auto-select. */
    s.has_vector_size_a = false;
    s.vector_size_a = 0;
    s.has_vector_size_b = false;
    s.vector_size_b = 0;
    s.has_vector_size_c = false;
    s.vector_size_c = 0;

    /* #8624 ConvDataSpec defaults. */
    s.dtype_a = "fp16";
    s.dtype_b = "fp16";
    s.dtype_d = "fp16";
    s.dtype_acc = "fp32";

    s.acc_epilogue = rocke_conv_acc_epilogue_default();
    return s;
}

int rocke_implicit_gemm_conv_spec_block_size(const rocke_implicit_gemm_conv_spec_t* s)
{
    /* warp_m * warp_n * wave_size */
    return s->warp_m * s->warp_n * s->wave_size;
}

int rocke_implicit_gemm_conv_spec_k_atoms_per_tile_k(const rocke_implicit_gemm_conv_spec_t* s)
{
    /* tile_k // warp_tile_k. Python integer division; guard div-by-zero. */
    if(s->warp_tile_k == 0)
    {
        return -1;
    }
    return s->tile_k / s->warp_tile_k;
}

int rocke_implicit_gemm_conv_spec_mfmas_per_warp_m(const rocke_implicit_gemm_conv_spec_t* s)
{
    /* tile_m // (warp_m * warp_tile_m) */
    int denom = s->warp_m * s->warp_tile_m;
    if(denom == 0)
    {
        return -1;
    }
    return s->tile_m / denom;
}

int rocke_implicit_gemm_conv_spec_mfmas_per_warp_n(const rocke_implicit_gemm_conv_spec_t* s)
{
    /* tile_n // (warp_n * warp_tile_n) */
    int denom = s->warp_n * s->warp_tile_n;
    if(denom == 0)
    {
        return -1;
    }
    return s->tile_n / denom;
}

rocke_status_t rocke_implicit_gemm_conv_spec_kernel_name(const rocke_implicit_gemm_conv_spec_t* s,
                                                         char* out,
                                                         size_t out_cap)
{
    /* return kernel_name_join(
     *     self.name,
     *     p.short(),
     *     f"t{tile_m}x{tile_n}x{tile_k}",
     *     f"w{warp_m}x{warp_n}",
     *     f"a{warp_tile_m}x{warp_tile_n}x{warp_tile_k}",
     *     f"{pipeline}_{epilogue}",
     *     self.acc_epilogue.tag(),
     *     flags={"async": self.async_dma}) */
    char short_buf[128];
    char t_buf[48];
    char w_buf[32];
    char a_buf[48];
    char pe_buf[64];
    char tag_buf[256];
    const char* parts[6];
    const char* flag_names[1];
    int flag_on[1];
    rocke_status_t st;

    if(s == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_conv_problem_short(&s->problem, short_buf, sizeof(short_buf), NULL);
    if(st != ROCKE_OK)
    {
        return st;
    }
    snprintf(t_buf, sizeof(t_buf), "t%dx%dx%d", s->tile_m, s->tile_n, s->tile_k);
    snprintf(w_buf, sizeof(w_buf), "w%dx%d", s->warp_m, s->warp_n);
    snprintf(a_buf, sizeof(a_buf), "a%dx%dx%d", s->warp_tile_m, s->warp_tile_n, s->warp_tile_k);
    snprintf(pe_buf,
             sizeof(pe_buf),
             "%s_%s",
             s->pipeline ? s->pipeline : "",
             s->epilogue ? s->epilogue : "");

    st = rocke_conv_acc_epilogue_tag(&s->acc_epilogue, tag_buf, sizeof(tag_buf));
    if(st != ROCKE_OK)
    {
        return st;
    }

    parts[0] = short_buf;
    parts[1] = t_buf;
    parts[2] = w_buf;
    parts[3] = a_buf;
    parts[4] = pe_buf;
    parts[5] = tag_buf; /* "" when identity -> skipped by kernel_name_join */

    flag_names[0] = "async";
    flag_on[0] = s->async_dma ? 1 : 0;

    return rocke_kernel_name_join(s->name, parts, 6, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* ===================================================================== *
 *  LdsLayout port   (rocke/helpers/layouts.py lines 66-203)
 *
 *  The conv body / spec validation reach only the policies the spec's
 *  effective_lds_layout() derives (padded_k / packed_async), all swizzle=None.
 *  These three functions reproduce LdsLayout.validate(), validate_for_async()
 *  and ImplicitGemmConvSpec.effective_lds_layout() byte-faithfully (the reason
 *  strings match the Python ValueError text verbatim).
 * ===================================================================== */

#define ROCKE_LDS_REJECT(...)                          \
    do                                                 \
    {                                                  \
        if(reason != NULL && reason_cap > 0)           \
        {                                              \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        }                                              \
        return false;                                  \
    } while(0)

bool rocke_conv_lds_layout_validate(const rocke_conv_lds_layout_t* l,
                                    char* reason,
                                    size_t reason_cap)
{
    /* LdsLayout.validate():
     *   if logical_cols <= 0: raise ValueError("logical_cols must be positive")
     *   if k_pad < 0:         raise ValueError("k_pad must be >= 0")
     *   if swizzle not in (None,"xor","cyclic"): raise ValueError(...)
     *   if swizzle=="xor" and not swizzle_stages: raise ValueError(...)
     * The C layout carries no swizzle_stages (only the reachable swizzle=None
     * policy is derived); the swizzle-name check still mirrors Python. */
    if(l == NULL)
    {
        ROCKE_LDS_REJECT("lds layout is NULL");
    }
    if(l->logical_cols <= 0)
    {
        ROCKE_LDS_REJECT("logical_cols must be positive");
    }
    if(l->k_pad < 0)
    {
        ROCKE_LDS_REJECT("k_pad must be >= 0");
    }
    if(l->swizzle != NULL && strcmp(l->swizzle, "xor") != 0 && strcmp(l->swizzle, "cyclic") != 0)
    {
        ROCKE_LDS_REJECT("unsupported LDS swizzle %s", l->swizzle);
    }
    return true;
}

bool rocke_conv_lds_layout_validate_for_async(const rocke_conv_lds_layout_t* l,
                                              char* reason,
                                              size_t reason_cap)
{
    /* LdsLayout.validate_for_async():
     *   if k_pad != 0:        raise ValueError("async LDS layout must be packed: ...")
     *   if swizzle is not None: raise ValueError("async LDS layout cannot use ...") */
    if(l == NULL)
    {
        ROCKE_LDS_REJECT("lds layout is NULL");
    }
    if(l->k_pad != 0)
    {
        ROCKE_LDS_REJECT("async LDS layout must be packed: k_pad must be 0");
    }
    if(l->swizzle != NULL)
    {
        ROCKE_LDS_REJECT("async LDS layout cannot use arbitrary per-lane swizzle; "
                         "express swizzle in consumer read math instead");
    }
    return true;
}

bool rocke_implicit_gemm_conv_spec_effective_lds_layout(const rocke_implicit_gemm_conv_spec_t* s,
                                                        rocke_conv_lds_layout_t* out,
                                                        char* reason,
                                                        size_t reason_cap)
{
    rocke_conv_lds_layout_t l;

    if(s == NULL || out == NULL)
    {
        ROCKE_LDS_REJECT("spec or out is NULL");
    }

    /* effective_lds_layout():
     *   if self.lds_layout is not None:  layout = self.lds_layout
     *   elif self.lds_k_pad is not None: layout = padded_k(tile_k, lds_k_pad)
     *   elif self.async_dma:             layout = packed_async(tile_k)
     *   else: layout = padded_k(tile_k, 8 if tile_k >= 16 else 0)
     *   layout.validate(); return layout
     *
     * The lds_layout override (Python `self.lds_layout is not None`) has no C
     * constructor (the field is an opaque void* peer port); reject explicitly so
     * a caller that somehow sets it is not silently mis-handled. */
    memset(&l, 0, sizeof(l));
    l.swizzle = NULL;

    if(s->lds_layout != NULL)
    {
        ROCKE_LDS_REJECT("spec.lds_layout override is not supported in the C port "
                         "(LdsLayout peer port owns it)");
    }
    else if(s->has_lds_k_pad)
    {
        /* LdsLayout.padded_k(logical_cols=tile_k, k_pad=lds_k_pad) */
        l.logical_cols = s->tile_k;
        l.k_pad = s->lds_k_pad;
        l.requires_packed_async = false;
    }
    else if(s->async_dma)
    {
        /* LdsLayout.packed_async(logical_cols=tile_k) -> k_pad=0 */
        l.logical_cols = s->tile_k;
        l.k_pad = 0;
        l.requires_packed_async = true;
    }
    else
    {
        /* LdsLayout.padded_k(tile_k, 8 if tile_k >= 16 else 0) */
        l.logical_cols = s->tile_k;
        l.k_pad = (s->tile_k >= 16) ? 8 : 0;
        l.requires_packed_async = false;
    }
    l.row_stride = l.logical_cols + l.k_pad;

    if(!rocke_conv_lds_layout_validate(&l, reason, reason_cap))
    {
        return false;
    }

    *out = l;
    return true;
}

#undef ROCKE_LDS_REJECT

bool rocke_implicit_gemm_conv_spec_validate(const rocke_implicit_gemm_conv_spec_t* s,
                                            char* reason,
                                            size_t reason_cap)
{
    int block_size;

#define ROCKE_CSPEC_REJECT(...)                        \
    do                                                 \
    {                                                  \
        if(reason != NULL && reason_cap > 0)           \
        {                                              \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        }                                              \
        return false;                                  \
    } while(0)

    if(s == NULL)
    {
        ROCKE_CSPEC_REJECT("spec is NULL");
    }

    /* if tile_m % (warp_m * warp_tile_m) != 0: raise ValueError(...) */
    if((s->warp_m * s->warp_tile_m) == 0 || (s->tile_m % (s->warp_m * s->warp_tile_m)) != 0)
    {
        ROCKE_CSPEC_REJECT("tile_m %d not divisible by warp_m * warp_tile_m (%d * %d)",
                           s->tile_m,
                           s->warp_m,
                           s->warp_tile_m);
    }
    /* if tile_n % (warp_n * warp_tile_n) != 0: raise ValueError(...) */
    if((s->warp_n * s->warp_tile_n) == 0 || (s->tile_n % (s->warp_n * s->warp_tile_n)) != 0)
    {
        ROCKE_CSPEC_REJECT("tile_n %d not divisible by warp_n * warp_tile_n (%d * %d)",
                           s->tile_n,
                           s->warp_n,
                           s->warp_tile_n);
    }
    /* if tile_k % warp_tile_k != 0: raise ValueError(...) */
    if(s->warp_tile_k == 0 || (s->tile_k % s->warp_tile_k) != 0)
    {
        ROCKE_CSPEC_REJECT("tile_k %d not divisible by warp_tile_k %d", s->tile_k, s->warp_tile_k);
    }
    /* if block_size > 1024: raise ValueError(...) */
    block_size = rocke_implicit_gemm_conv_spec_block_size(s);
    if(block_size > 1024)
    {
        ROCKE_CSPEC_REJECT("block_size %d > 1024", block_size);
    }

    /* layout = self.effective_lds_layout()   (runs layout.validate())
     * if async_dma: layout.validate_for_async()
     * Ported faithfully: effective_lds_layout derives the policy + validates it
     * (e.g. lds_k_pad < 0 -> "k_pad must be >= 0"), then the async-only assert
     * runs BEFORE the lds_k_pad guard below -- matching Python's ordering so an
     * async spec with lds_k_pad != 0 rejects with the validate_for_async message
     * ("async LDS layout must be packed: k_pad must be 0"), not the guard's. */
    {
        rocke_conv_lds_layout_t layout;
        if(!rocke_implicit_gemm_conv_spec_effective_lds_layout(s, &layout, reason, reason_cap))
        {
            return false;
        }
        if(s->async_dma && !rocke_conv_lds_layout_validate_for_async(&layout, reason, reason_cap))
        {
            return false;
        }
    }

    /* if async_dma and lds_k_pad not in (None, 0): raise ValueError(...) */
    if(s->async_dma && s->has_lds_k_pad && s->lds_k_pad != 0)
    {
        ROCKE_CSPEC_REJECT("async_dma requires lds_k_pad to be 0/None because "
                           "raw_ptr_buffer_load_lds writes a packed lane-contiguous tile");
    }

    /* if clamp_min is not None and clamp_max is not None and clamp_min > clamp_max:
     *     raise ValueError(...) */
    if(s->acc_epilogue.has_clamp_min && s->acc_epilogue.has_clamp_max
       && s->acc_epilogue.clamp_min > s->acc_epilogue.clamp_max)
    {
        char lo[48];
        char hi[48];
        rocke_conv_g(lo, sizeof(lo), s->acc_epilogue.clamp_min);
        rocke_conv_g(hi, sizeof(hi), s->acc_epilogue.clamp_max);
        ROCKE_CSPEC_REJECT("acc_epilogue clamp_min must be <= clamp_max (got %s > %s)", lo, hi);
    }

    return true;

#undef ROCKE_CSPEC_REJECT
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)   (Python lines 383-466)
 * ===================================================================== */

bool rocke_implicit_gemm_conv_is_valid_spec(const rocke_implicit_gemm_conv_spec_t* s,
                                            const char* arch,
                                            char* reason,
                                            size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const rocke_arch_mma_catalog_t* mma;
    const char* family;
    int block_size;
    int mtpb;

#define ROCKE_CONVVS_REJECT(...)                       \
    do                                                 \
    {                                                  \
        if(reason != NULL && reason_cap > 0)           \
        {                                              \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        }                                              \
        return false;                                  \
    } while(0)

    if(s == NULL)
    {
        ROCKE_CONVVS_REJECT("spec is NULL");
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* try: target = ArchTarget.from_gfx(arch) except KeyError as e: return False, str(e) */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        /* Full Python str(KeyError) text, reproduced verbatim. */
        rocke_cspec__set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }

    /* Geometry divisibility (mirrors spec.validate). */
    if((s->warp_m * s->warp_tile_m) == 0 || (s->tile_m % (s->warp_m * s->warp_tile_m)) != 0)
    {
        ROCKE_CONVVS_REJECT("tile_m not divisible by warp_m * warp_tile_m");
    }
    if((s->warp_n * s->warp_tile_n) == 0 || (s->tile_n % (s->warp_n * s->warp_tile_n)) != 0)
    {
        ROCKE_CONVVS_REJECT("tile_n not divisible by warp_n * warp_tile_n");
    }
    if(s->warp_tile_k == 0 || (s->tile_k % s->warp_tile_k) != 0)
    {
        ROCKE_CONVVS_REJECT("tile_k not divisible by warp_tile_k");
    }
    block_size = rocke_implicit_gemm_conv_spec_block_size(s);
    mtpb = rocke_archtarget_max_threads_per_block(target);
    if(block_size > mtpb)
    {
        ROCKE_CONVVS_REJECT("block_size %d > %d (hardware cap) on %s", block_size, mtpb, arch);
    }

    /* family = "wmma" if target.wave_size == 32 else "mma" */
    family = (target->wave_size == 32) ? "wmma" : "mma";
    /* if spec.wave_size != target.wave_size: return False, ... */
    if(s->wave_size != target->wave_size)
    {
        ROCKE_CONVVS_REJECT(
            "spec wave_size %d != %s wave_size %d", s->wave_size, arch, target->wave_size);
    }

    /* MMA atom must be in the target's catalog (f16 in/out fp32 acc). */
    mma = rocke_archtarget_mma(target);
    if(!rocke_mma_catalog_has_shape(
           mma, family, "f16", "f16", "fp32", s->warp_tile_m, s->warp_tile_n, s->warp_tile_k))
    {
        ROCKE_CONVVS_REJECT("unsupported f16 warp_tile (%d, %d, %d) on %s",
                            s->warp_tile_m,
                            s->warp_tile_n,
                            s->warp_tile_k,
                            arch);
    }

    /* LDS budget: must fit before we attempt codegen.
     *   A_smem + B_smem, each (tile_m or tile_n) × row_stride × 2 bytes (f16).
     *   row_stride = tile_k + k_pad  (k_pad = 8 when tile_k >= 16, else 0).
     *   compv4 pipeline double-buffers A and B → ×2.
     *   This mirrors the smem_alloc calls in instance_conv_implicit_gemm_conv_build_glue.c
     *   and catches the overflow that would otherwise produce CODEGEN_BC_TO_RELOCATABLE. */
    {
        /* k_pad and double_buffer mirror build_glue priority order exactly:
         *   k_pad: has_lds_k_pad → lds_k_pad; async_dma → 0; else (tile_k>=16)?8:0
         *   double_buffer: compv4 || async_dma || unroll_k */
        int k_pad
            = s->has_lds_k_pad ? s->lds_k_pad : (s->async_dma ? 0 : ((s->tile_k >= 16) ? 8 : 0));
        int row_stride = s->tile_k + k_pad;
        int ab_single = (s->tile_m + s->tile_n) * row_stride * 2; /* f16 = 2 bytes */
        int double_buf
            = ((s->pipeline && strcmp(s->pipeline, "compv4") == 0) || s->async_dma || s->unroll_k)
                  ? 1
                  : 0;
        int bytes_lds = ab_single * (double_buf ? 2 : 1);
        if(!rocke_archtarget_fits_lds(target, (long)bytes_lds))
        {
            ROCKE_CONVVS_REJECT("LDS budget %d > %d cap (AB=%d, double_buf=%d) on %s",
                                bytes_lds,
                                target->lds_capacity_bytes,
                                ab_single,
                                double_buf,
                                arch);
        }
    }

    /* WMMA (RDNA wave32) narrow-subset gates. */
    if(strcmp(family, "wmma") == 0)
    {
        int is_16x16x16 = (s->warp_tile_m == 16 && s->warp_tile_n == 16 && s->warp_tile_k == 16);
        if(!is_16x16x16)
        {
            ROCKE_CONVVS_REJECT("WMMA conv supports only 16x16x16 (got (%d, %d, %d)) on %s",
                                s->warp_tile_m,
                                s->warp_tile_n,
                                s->warp_tile_k,
                                arch);
        }
        if(!(s->pipeline && strcmp(s->pipeline, "mem") == 0))
        {
            ROCKE_CONVVS_REJECT("WMMA conv supports only the 'mem' pipeline (got '%s') on %s",
                                s->pipeline ? s->pipeline : "",
                                arch);
        }
        if(!(s->epilogue && strcmp(s->epilogue, "default") == 0))
        {
            ROCKE_CONVVS_REJECT("WMMA conv supports only the 'default' epilogue (got '%s') on %s",
                                s->epilogue ? s->epilogue : "",
                                arch);
        }
        if(s->async_dma)
        {
            ROCKE_CONVVS_REJECT("WMMA conv does not support async_dma on %s", arch);
        }
        if(s->unroll_k)
        {
            ROCKE_CONVVS_REJECT("WMMA conv does not support unroll_k on %s", arch);
        }
        if(s->chiplet_swizzle)
        {
            ROCKE_CONVVS_REJECT("WMMA conv does not support chiplet_swizzle on %s", arch);
        }
        if(s->groups != 1)
        {
            ROCKE_CONVVS_REJECT("WMMA conv supports only groups=1 (got %d)", s->groups);
        }
    }

    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "ok");
    }
    return true;

#undef ROCKE_CONVVS_REJECT
}

/* ===================================================================== *
 *  _conv_mma_family(arch)   (Python lines 469-472)
 * ===================================================================== */

const char* rocke_conv_mma_family(const char* arch)
{
    /* return "wmma" if ArchTarget.from_gfx(arch).wave_size == 32 else "mma" */
    const rocke_archtarget_t* target;
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        /* Python would raise KeyError before the wave-size compare; there is no
         * builder here, so fall back to the wave64 default family. */
        return "mma";
    }
    return (target->wave_size == 32) ? "wmma" : "mma";
}

/* ===================================================================== *
 *  _resolve_conv_op(spec, arch)   (Python lines 475-501)
 * ===================================================================== */

const rocke_mmaop_t* rocke_conv_resolve_op(rocke_ir_builder_t* b,
                                           const rocke_implicit_gemm_conv_spec_t* spec,
                                           const char* arch)
{
    const rocke_archtarget_t* target;
    const rocke_mmaop_t* op;

    if(b != NULL && !rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* target = ArchTarget.from_gfx(arch) */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        /* Python raises KeyError; surface a builder error. */
        if(b != NULL && b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_KEY;
            snprintf(b->err, ROCKE_ERR_MSG_CAP, "unknown gfx target '%s'", arch);
        }
        return NULL;
    }

    /* op = target.mma.op_for_shape(family=_conv_mma_family(arch),
     *                              a/b="f16", c="fp32",
     *                              m=warp_tile_m, n=warp_tile_n, k=warp_tile_k) */
    op = rocke_archtarget_op_for_shape(target,
                                       rocke_conv_mma_family(arch),
                                       "f16",
                                       "f16",
                                       "fp32",
                                       spec->warp_tile_m,
                                       spec->warp_tile_n,
                                       spec->warp_tile_k);
    if(op == NULL)
    {
        /* raise ValueError(f"no MMA atom for conv warp_tile (...) on {arch}") */
        if(b != NULL && b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
            snprintf(b->err,
                     ROCKE_ERR_MSG_CAP,
                     "no MMA atom for conv warp_tile (%d,%d,%d) on %s",
                     spec->warp_tile_m,
                     spec->warp_tile_n,
                     spec->warp_tile_k,
                     arch);
        }
        return NULL;
    }
    return op;
}

/* ===================================================================== *
 *  Descriptor builders   (Python lines 509-629)
 * ===================================================================== */

struct rocke_tensor_descriptor* rocke_conv_make_a_descriptor(rocke_ir_builder_t* b,
                                                             const rocke_conv_problem_t* p,
                                                             bool decompose_m)
{
    /* 2-D DAG (#8355 r->y, s->x):
     *   if decompose_m: unmerge_magic('m'->[n,ho,wo],[N,Ho,Wo])
     *   embed(['ho','y']->'hi'), embed(['wo','x']->'wi'),
     *   unmerge_magic('k'->[y,x,c],[Y,X,C]), pad('y'), pad('x')
     *   naive('A_nhwc', [N,Hi,Wi,C], coords=['n','hi','wi','c'])
     * 3-D DAG (#8355 conv-3d):
     *   if decompose_m: unmerge_magic('m'->[n,do,ho,wo],[N,Do,Ho,Wo])
     *   embed(['do','z']->'di'), embed(['ho','y']->'hi'), embed(['wo','x']->'wi'),
     *   unmerge_magic('k'->[z,y,x,c],[Z,Y,X,C]), pad('z'),pad('y'),pad('x')
     *   naive('A_ndhwc', [N,Di,Hi,Wi,C], coords=['n','di','hi','wi','c']) */
    int Ho, Wo, Do;
    int lengths[5];
    const char* coords[5];
    const char* into_m[4];
    const char* up_do[2];
    const char* up_ho[2];
    const char* up_wo[2];
    const char* into_k[4];
    int dims_m[4];
    int strides_do[2];
    int strides_ho[2];
    int strides_wo[2];
    int dims_k[4];
    const rocke_transform_t* xforms[8];
    int n_x = 0;
    rocke_tensor_descriptor_t* desc;
    bool is3d;

    if(b == NULL || !rocke_ir_builder_ok(b) || p == NULL)
    {
        return NULL;
    }

    is3d = p->is_3d;
    Ho = rocke_conv_problem_ho(p);
    Wo = rocke_conv_problem_wo(p);
    Do = rocke_conv_problem_do(p);

    if(is3d)
    {
        if(decompose_m)
        {
            into_m[0] = "n";
            into_m[1] = "do";
            into_m[2] = "ho";
            into_m[3] = "wo";
            dims_m[0] = p->N;
            dims_m[1] = Do;
            dims_m[2] = Ho;
            dims_m[3] = Wo;
            xforms[n_x] = rocke_unmerge_magic(b, "m", into_m, 4, dims_m);
            if(xforms[n_x] == NULL)
            {
                return NULL;
            }
            n_x++;
        }
        /* embed(['do','z'] -> 'di', strides=[sD,dD], offset=-pD, lo=0, hi=Di) */
        up_do[0] = "do";
        up_do[1] = "z";
        strides_do[0] = p->sD;
        strides_do[1] = p->dD;
        xforms[n_x] = rocke_embed_bounded(b, up_do, 2, "di", strides_do, -p->pD, 0, p->Di);
        if(xforms[n_x] == NULL)
        {
            return NULL;
        }
        n_x++;
    }
    else if(decompose_m)
    {
        into_m[0] = "n";
        into_m[1] = "ho";
        into_m[2] = "wo";
        dims_m[0] = p->N;
        dims_m[1] = Ho;
        dims_m[2] = Wo;
        xforms[n_x] = rocke_unmerge_magic(b, "m", into_m, 3, dims_m);
        if(xforms[n_x] == NULL)
        {
            return NULL;
        }
        n_x++;
    }

    /* embed(['ho','y'] -> 'hi', strides=[sH,dH], offset=-pH, lo=0, hi=Hi) */
    up_ho[0] = "ho";
    up_ho[1] = "y";
    strides_ho[0] = p->sH;
    strides_ho[1] = p->dH;
    xforms[n_x] = rocke_embed_bounded(b, up_ho, 2, "hi", strides_ho, -p->pH, 0, p->Hi);
    if(xforms[n_x] == NULL)
    {
        return NULL;
    }
    n_x++;

    /* embed(['wo','x'] -> 'wi', strides=[sW,dW], offset=-pW, lo=0, hi=Wi) */
    up_wo[0] = "wo";
    up_wo[1] = "x";
    strides_wo[0] = p->sW;
    strides_wo[1] = p->dW;
    xforms[n_x] = rocke_embed_bounded(b, up_wo, 2, "wi", strides_wo, -p->pW, 0, p->Wi);
    if(xforms[n_x] == NULL)
    {
        return NULL;
    }
    n_x++;

    /* unmerge_magic('k' -> [z,]y,x,c, dims=[Z,]Y,X,C) */
    if(is3d)
    {
        into_k[0] = "z";
        into_k[1] = "y";
        into_k[2] = "x";
        into_k[3] = "c";
        dims_k[0] = p->Z;
        dims_k[1] = p->Y;
        dims_k[2] = p->X;
        dims_k[3] = p->C;
        xforms[n_x] = rocke_unmerge_magic(b, "k", into_k, 4, dims_k);
    }
    else
    {
        into_k[0] = "y";
        into_k[1] = "x";
        into_k[2] = "c";
        dims_k[0] = p->Y;
        dims_k[1] = p->X;
        dims_k[2] = p->C;
        xforms[n_x] = rocke_unmerge_magic(b, "k", into_k, 3, dims_k);
    }
    if(xforms[n_x] == NULL)
    {
        return NULL;
    }
    n_x++;

    if(is3d)
    {
        /* pad('z', lo=0, hi=Z) */
        xforms[n_x] = rocke_pad(b, "z", 0, p->Z);
        if(xforms[n_x] == NULL)
        {
            return NULL;
        }
        n_x++;
    }

    /* pad('y', lo=0, hi=Y) */
    xforms[n_x] = rocke_pad(b, "y", 0, p->Y);
    if(xforms[n_x] == NULL)
    {
        return NULL;
    }
    n_x++;

    /* pad('x', lo=0, hi=X) */
    xforms[n_x] = rocke_pad(b, "x", 0, p->X);
    if(xforms[n_x] == NULL)
    {
        return NULL;
    }
    n_x++;

    if(is3d)
    {
        lengths[0] = p->N;
        lengths[1] = p->Di;
        lengths[2] = p->Hi;
        lengths[3] = p->Wi;
        lengths[4] = p->C;
        coords[0] = "n";
        coords[1] = "di";
        coords[2] = "hi";
        coords[3] = "wi";
        coords[4] = "c";
        desc = rocke_tensor_descriptor_naive(b, "A_ndhwc", lengths, 5, NULL, coords, 5);
    }
    else
    {
        lengths[0] = p->N;
        lengths[1] = p->Hi;
        lengths[2] = p->Wi;
        lengths[3] = p->C;
        coords[0] = "n";
        coords[1] = "hi";
        coords[2] = "wi";
        coords[3] = "c";
        desc = rocke_tensor_descriptor_naive(b, "A_nhwc", lengths, 4, NULL, coords, 4);
    }
    if(desc == NULL)
    {
        return NULL;
    }
    return rocke_tensor_descriptor_transform(b, desc, xforms, n_x);
}

struct rocke_tensor_descriptor* rocke_conv_make_b_descriptor(rocke_ir_builder_t* b,
                                                             const rocke_conv_problem_t* p)
{
    /* 2-D: naive('B_kyxc', [K,Y,X,C], coords=['k_out','y','x','c']).transform(
     *        unmerge_magic('k_gemm'->[y,x,c],[Y,X,C]), pad('y'), pad('x'))
     * 3-D: naive('B_kzyxc', [K,Z,Y,X,C], coords=['k_out','z','y','x','c']).transform(
     *        unmerge_magic('k_gemm'->[z,y,x,c],[Z,Y,X,C]), pad('z'),pad('y'),pad('x')) */
    int lengths[5];
    const char* coords[5];
    const char* into_k[4];
    int dims_k[4];
    const rocke_transform_t* xforms[4];
    int n_x = 0;
    rocke_tensor_descriptor_t* desc;

    if(b == NULL || !rocke_ir_builder_ok(b) || p == NULL)
    {
        return NULL;
    }

    if(p->is_3d)
    {
        into_k[0] = "z";
        into_k[1] = "y";
        into_k[2] = "x";
        into_k[3] = "c";
        dims_k[0] = p->Z;
        dims_k[1] = p->Y;
        dims_k[2] = p->X;
        dims_k[3] = p->C;
        xforms[n_x] = rocke_unmerge_magic(b, "k_gemm", into_k, 4, dims_k);
        if(xforms[n_x] == NULL)
        {
            return NULL;
        }
        n_x++;
        xforms[n_x] = rocke_pad(b, "z", 0, p->Z);
        if(xforms[n_x] == NULL)
        {
            return NULL;
        }
        n_x++;
    }
    else
    {
        into_k[0] = "y";
        into_k[1] = "x";
        into_k[2] = "c";
        dims_k[0] = p->Y;
        dims_k[1] = p->X;
        dims_k[2] = p->C;
        xforms[n_x] = rocke_unmerge_magic(b, "k_gemm", into_k, 3, dims_k);
        if(xforms[n_x] == NULL)
        {
            return NULL;
        }
        n_x++;
    }
    xforms[n_x] = rocke_pad(b, "y", 0, p->Y);
    if(xforms[n_x] == NULL)
    {
        return NULL;
    }
    n_x++;
    xforms[n_x] = rocke_pad(b, "x", 0, p->X);
    if(xforms[n_x] == NULL)
    {
        return NULL;
    }
    n_x++;

    if(p->is_3d)
    {
        lengths[0] = p->K;
        lengths[1] = p->Z;
        lengths[2] = p->Y;
        lengths[3] = p->X;
        lengths[4] = p->C;
        coords[0] = "k_out";
        coords[1] = "z";
        coords[2] = "y";
        coords[3] = "x";
        coords[4] = "c";
        desc = rocke_tensor_descriptor_naive(b, "B_kzyxc", lengths, 5, NULL, coords, 5);
    }
    else
    {
        lengths[0] = p->K;
        lengths[1] = p->Y;
        lengths[2] = p->X;
        lengths[3] = p->C;
        coords[0] = "k_out";
        coords[1] = "y";
        coords[2] = "x";
        coords[3] = "c";
        desc = rocke_tensor_descriptor_naive(b, "B_kyxc", lengths, 4, NULL, coords, 4);
    }
    if(desc == NULL)
    {
        return NULL;
    }
    return rocke_tensor_descriptor_transform(b, desc, xforms, n_x);
}

struct rocke_tensor_descriptor* rocke_conv_make_d_descriptor(rocke_ir_builder_t* b,
                                                             const rocke_conv_problem_t* p)
{
    /* 2-D: naive('D_nhwk', [N,Ho,Wo,K], coords=['n','ho','wo','k_out']).transform(
     *        unmerge_magic('m'->[n,ho,wo],[N,Ho,Wo]))
     * 3-D: naive('D_ndhwk', [N,Do,Ho,Wo,K], coords=['n','do','ho','wo','k_out']).transform(
     *        unmerge_magic('m'->[n,do,ho,wo],[N,Do,Ho,Wo])) */
    int Ho, Wo, Do;
    int lengths[5];
    const char* coords[5];
    const char* into_m[4];
    int dims_m[4];
    const rocke_transform_t* xforms[1];
    rocke_tensor_descriptor_t* desc;

    if(b == NULL || !rocke_ir_builder_ok(b) || p == NULL)
    {
        return NULL;
    }

    Ho = rocke_conv_problem_ho(p);
    Wo = rocke_conv_problem_wo(p);
    Do = rocke_conv_problem_do(p);

    if(p->is_3d)
    {
        into_m[0] = "n";
        into_m[1] = "do";
        into_m[2] = "ho";
        into_m[3] = "wo";
        dims_m[0] = p->N;
        dims_m[1] = Do;
        dims_m[2] = Ho;
        dims_m[3] = Wo;
        xforms[0] = rocke_unmerge_magic(b, "m", into_m, 4, dims_m);
        if(xforms[0] == NULL)
        {
            return NULL;
        }
        lengths[0] = p->N;
        lengths[1] = Do;
        lengths[2] = Ho;
        lengths[3] = Wo;
        lengths[4] = p->K;
        coords[0] = "n";
        coords[1] = "do";
        coords[2] = "ho";
        coords[3] = "wo";
        coords[4] = "k_out";
        desc = rocke_tensor_descriptor_naive(b, "D_ndhwk", lengths, 5, NULL, coords, 5);
    }
    else
    {
        into_m[0] = "n";
        into_m[1] = "ho";
        into_m[2] = "wo";
        dims_m[0] = p->N;
        dims_m[1] = Ho;
        dims_m[2] = Wo;
        xforms[0] = rocke_unmerge_magic(b, "m", into_m, 3, dims_m);
        if(xforms[0] == NULL)
        {
            return NULL;
        }
        lengths[0] = p->N;
        lengths[1] = Ho;
        lengths[2] = Wo;
        lengths[3] = p->K;
        coords[0] = "n";
        coords[1] = "ho";
        coords[2] = "wo";
        coords[3] = "k_out";
        desc = rocke_tensor_descriptor_naive(b, "D_nhwk", lengths, 4, NULL, coords, 4);
    }
    if(desc == NULL)
    {
        return NULL;
    }
    return rocke_tensor_descriptor_transform(b, desc, xforms, 1);
}
