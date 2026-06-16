/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_conv_implicit_gemm_conv_spec_descriptors.c -- C99 port of the SPEC
 * value-type + epilogue helpers + arch-aware validity gate + the user-visible
 * coordinate-transform-DAG descriptor builders of
 * ck_dsl/instances/common/conv_implicit_gemm.py (lines 142-629).
 *
 * SCOPE (this TU): the IR-free / transform-DAG surface. NO kernel-body IR.
 *
 *   Python (conv_implicit_gemm.py)             C99 (this file)
 *   ----------------------------------------   ----------------------------------
 *   ConvAccumulatorEpilogue (142-181)          ckc_conv_acc_epilogue_default /
 *     .is_identity() / .tag()                    _is_identity / _tag
 *   ImplicitGemmConvSpec (183-376)             ckc_implicit_gemm_conv_spec_default
 *     .block_size / .k_atoms_per_tile_k          + the @property accessors
 *     .mfmas_per_warp_m / _n
 *     .kernel_name() / .validate()               _kernel_name / _validate
 *   is_valid_spec(spec, arch) (383-466)        ckc_implicit_gemm_conv_is_valid_spec
 *   _conv_mma_family(arch) (469-472)           ckc_conv_mma_family
 *   _resolve_conv_op(spec, arch) (475-501)     ckc_conv_resolve_op
 *   make_a_descriptor (509-576)                ckc_conv_make_a_descriptor
 *   make_b_descriptor (579-613)                ckc_conv_make_b_descriptor
 *   make_d_descriptor (616-629)                ckc_conv_make_d_descriptor
 *
 * The descriptor builders emit the byte-identical builder-call sequence the
 * Python TensorDescriptor.naive(...).transform(...) chain produces. The pure
 * value/string helpers reproduce the Python return values bit-for-bit (the
 * reason / kernel-name strings never enter the IR but a sweep driver sees the
 * same accept/reject + identifier).
 *
 * effective_lds_layout() is NOT ported here (the LdsLayout peer port owns it);
 * spec.validate()'s effective_lds_layout()-dependent branches are stubbed with
 * a TODO(port) note so this TU is self-contained on its scope.
 */

#include "ckc/instance_conv_implicit_gemm.h"
#include "ckc/instance_conv_implicit_gemm_internal.h"

#include <stdio.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/helper_ck_dsl.core.arch.h"   /* ckc_archtarget_*, has_shape, op_for_shape */
#include "ckc/helper_ck_dsl.helpers.atoms.h" /* ckc_mfma_atom */
#include "ckc/helper_ck_dsl.helpers.spec.h"  /* ckc_kernel_name_join, ckc_choose_load_vec */
#include "ckc/helper_ck_dsl.helpers.transforms.h" /* descriptor DAG + transforms */

/* ===================================================================== *
 *  ConvAccumulatorEpilogue   (Python lines 142-181)
 * ===================================================================== */

ckc_conv_acc_epilogue_t ckc_conv_acc_epilogue_default(void)
{
    ckc_conv_acc_epilogue_t e;
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

bool ckc_conv_acc_epilogue_is_identity(const ckc_conv_acc_epilogue_t* epi)
{
    if (epi == NULL)
    {
        return true;
    }
    /* return (bias == 0.0 and scale == 1.0 and not relu
     *         and clamp_min is None and clamp_max is None) */
    return (epi->bias == 0.0 && epi->scale == 1.0 && !epi->relu &&
            !epi->has_clamp_min && !epi->has_clamp_max);
}

/* Format a double using Python's "%g" repr (the f"{x:g}" used by tag()).
 * The C "%g" conversion matches Python's general format for these values. */
static void ckc_conv_g(char* out, size_t cap, double v)
{
    snprintf(out, cap, "%g", v);
}

ckc_status_t ckc_conv_acc_epilogue_tag(const ckc_conv_acc_epilogue_t* epi,
                                       char* out,
                                       size_t out_cap)
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
    char piece[64];
    char numbuf[48];
    size_t blen = 0;
    int wrote_any = 0;

    if (epi == NULL || out == NULL || out_cap == 0)
    {
        return CKC_ERR_VALUE;
    }

    if (ckc_conv_acc_epilogue_is_identity(epi))
    {
        if (out_cap < 1)
        {
            return CKC_ERR_VALUE;
        }
        out[0] = '\0';
        return CKC_OK;
    }

    body[0] = '\0';

#define CKC_TAG_APPEND(s)                                                   \
    do                                                                      \
    {                                                                       \
        size_t _l = strlen(s);                                             \
        if (wrote_any)                                                      \
        {                                                                  \
            if (blen + 1 >= sizeof(body))                                  \
                return CKC_ERR_VALUE;                                      \
            body[blen++] = '_';                                            \
            body[blen] = '\0';                                            \
        }                                                                 \
        if (blen + _l >= sizeof(body))                                     \
            return CKC_ERR_VALUE;                                         \
        memcpy(body + blen, (s), _l + 1);                                  \
        blen += _l;                                                        \
        wrote_any = 1;                                                     \
    } while (0)

    if (epi->bias != 0.0)
    {
        ckc_conv_g(numbuf, sizeof(numbuf), epi->bias);
        snprintf(piece, sizeof(piece), "bias%s", numbuf);
        CKC_TAG_APPEND(piece);
    }
    if (epi->scale != 1.0)
    {
        ckc_conv_g(numbuf, sizeof(numbuf), epi->scale);
        snprintf(piece, sizeof(piece), "scale%s", numbuf);
        CKC_TAG_APPEND(piece);
    }
    if (epi->relu)
    {
        CKC_TAG_APPEND("relu");
    }
    if (epi->has_clamp_min || epi->has_clamp_max)
    {
        char lo[48];
        char hi[48];
        if (!epi->has_clamp_min)
        {
            snprintf(lo, sizeof(lo), "-inf");
        }
        else
        {
            ckc_conv_g(lo, sizeof(lo), epi->clamp_min);
        }
        if (!epi->has_clamp_max)
        {
            snprintf(hi, sizeof(hi), "inf");
        }
        else
        {
            ckc_conv_g(hi, sizeof(hi), epi->clamp_max);
        }
        snprintf(piece, sizeof(piece), "clamp%sto%s", lo, hi);
        CKC_TAG_APPEND(piece);
    }

#undef CKC_TAG_APPEND

    /* "epi_" + body */
    if (snprintf(out, out_cap, "epi_%s", body) < 0 ||
        strlen("epi_") + blen >= out_cap)
    {
        return CKC_ERR_VALUE;
    }
    return CKC_OK;
}

/* ===================================================================== *
 *  ImplicitGemmConvSpec   (Python lines 183-376)
 * ===================================================================== */

ckc_implicit_gemm_conv_spec_t ckc_implicit_gemm_conv_spec_default(void)
{
    ckc_implicit_gemm_conv_spec_t s;
    memset(&s, 0, sizeof(s));

    /* problem has no Python default (required) -> zero-init via ConvProblem
     * defaulted optional fields; caller must set the required dims. */
    s.problem = ckc_conv_problem_default(0, 0, 0, 0, 0, 0, 0);

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

    s.acc_epilogue = ckc_conv_acc_epilogue_default();
    return s;
}

int ckc_implicit_gemm_conv_spec_block_size(const ckc_implicit_gemm_conv_spec_t* s)
{
    /* warp_m * warp_n * wave_size */
    return s->warp_m * s->warp_n * s->wave_size;
}

int ckc_implicit_gemm_conv_spec_k_atoms_per_tile_k(const ckc_implicit_gemm_conv_spec_t* s)
{
    /* tile_k // warp_tile_k. Python integer division; guard div-by-zero. */
    if (s->warp_tile_k == 0)
    {
        return -1;
    }
    return s->tile_k / s->warp_tile_k;
}

int ckc_implicit_gemm_conv_spec_mfmas_per_warp_m(const ckc_implicit_gemm_conv_spec_t* s)
{
    /* tile_m // (warp_m * warp_tile_m) */
    int denom = s->warp_m * s->warp_tile_m;
    if (denom == 0)
    {
        return -1;
    }
    return s->tile_m / denom;
}

int ckc_implicit_gemm_conv_spec_mfmas_per_warp_n(const ckc_implicit_gemm_conv_spec_t* s)
{
    /* tile_n // (warp_n * warp_tile_n) */
    int denom = s->warp_n * s->warp_tile_n;
    if (denom == 0)
    {
        return -1;
    }
    return s->tile_n / denom;
}

ckc_status_t ckc_implicit_gemm_conv_spec_kernel_name(const ckc_implicit_gemm_conv_spec_t* s,
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
    ckc_status_t st;

    if (s == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }

    st = ckc_conv_problem_short(&s->problem, short_buf, sizeof(short_buf), NULL);
    if (st != CKC_OK)
    {
        return st;
    }
    snprintf(t_buf, sizeof(t_buf), "t%dx%dx%d", s->tile_m, s->tile_n, s->tile_k);
    snprintf(w_buf, sizeof(w_buf), "w%dx%d", s->warp_m, s->warp_n);
    snprintf(a_buf, sizeof(a_buf), "a%dx%dx%d", s->warp_tile_m, s->warp_tile_n,
             s->warp_tile_k);
    snprintf(pe_buf, sizeof(pe_buf), "%s_%s",
             s->pipeline ? s->pipeline : "", s->epilogue ? s->epilogue : "");

    st = ckc_conv_acc_epilogue_tag(&s->acc_epilogue, tag_buf, sizeof(tag_buf));
    if (st != CKC_OK)
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

    return ckc_kernel_name_join(s->name, parts, 6, flag_names, flag_on, 1, out,
                                out_cap, NULL);
}

bool ckc_implicit_gemm_conv_spec_validate(const ckc_implicit_gemm_conv_spec_t* s,
                                          char* reason,
                                          size_t reason_cap)
{
    int block_size;

#define CKC_CSPEC_REJECT(...)                          \
    do                                                 \
    {                                                  \
        if (reason != NULL && reason_cap > 0)          \
        {                                              \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        }                                              \
        return false;                                  \
    } while (0)

    if (s == NULL)
    {
        CKC_CSPEC_REJECT("spec is NULL");
    }

    /* if tile_m % (warp_m * warp_tile_m) != 0: raise ValueError(...) */
    if ((s->warp_m * s->warp_tile_m) == 0 ||
        (s->tile_m % (s->warp_m * s->warp_tile_m)) != 0)
    {
        CKC_CSPEC_REJECT(
            "tile_m %d not divisible by warp_m * warp_tile_m (%d * %d)", s->tile_m,
            s->warp_m, s->warp_tile_m);
    }
    /* if tile_n % (warp_n * warp_tile_n) != 0: raise ValueError(...) */
    if ((s->warp_n * s->warp_tile_n) == 0 ||
        (s->tile_n % (s->warp_n * s->warp_tile_n)) != 0)
    {
        CKC_CSPEC_REJECT(
            "tile_n %d not divisible by warp_n * warp_tile_n (%d * %d)", s->tile_n,
            s->warp_n, s->warp_tile_n);
    }
    /* if tile_k % warp_tile_k != 0: raise ValueError(...) */
    if (s->warp_tile_k == 0 || (s->tile_k % s->warp_tile_k) != 0)
    {
        CKC_CSPEC_REJECT("tile_k %d not divisible by warp_tile_k %d", s->tile_k,
                         s->warp_tile_k);
    }
    /* if block_size > 1024: raise ValueError(...) */
    block_size = ckc_implicit_gemm_conv_spec_block_size(s);
    if (block_size > 1024)
    {
        CKC_CSPEC_REJECT("block_size %d > 1024", block_size);
    }

    /* layout = self.effective_lds_layout()
     * if async_dma: layout.validate_for_async()
     * The LdsLayout peer port owns effective_lds_layout / validate_for_async; the
     * remaining clamp/async branches below do not depend on it. */
    /* TODO(port): effective_lds_layout() + layout.validate_for_async() */

    /* if async_dma and lds_k_pad not in (None, 0): raise ValueError(...) */
    if (s->async_dma && s->has_lds_k_pad && s->lds_k_pad != 0)
    {
        CKC_CSPEC_REJECT(
            "async_dma requires lds_k_pad to be 0/None because "
            "raw_ptr_buffer_load_lds writes a packed lane-contiguous tile");
    }

    /* if clamp_min is not None and clamp_max is not None and clamp_min > clamp_max:
     *     raise ValueError(...) */
    if (s->acc_epilogue.has_clamp_min && s->acc_epilogue.has_clamp_max &&
        s->acc_epilogue.clamp_min > s->acc_epilogue.clamp_max)
    {
        char lo[48];
        char hi[48];
        ckc_conv_g(lo, sizeof(lo), s->acc_epilogue.clamp_min);
        ckc_conv_g(hi, sizeof(hi), s->acc_epilogue.clamp_max);
        CKC_CSPEC_REJECT("acc_epilogue clamp_min must be <= clamp_max (got %s > %s)",
                         lo, hi);
    }

    return true;

#undef CKC_CSPEC_REJECT
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)   (Python lines 383-466)
 * ===================================================================== */

bool ckc_implicit_gemm_conv_is_valid_spec(const ckc_implicit_gemm_conv_spec_t* s,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap)
{
    const ckc_archtarget_t* target;
    const ckc_arch_mma_catalog_t* mma;
    const char* family;
    int block_size;
    int mtpb;

#define CKC_CONVVS_REJECT(...)                         \
    do                                                 \
    {                                                  \
        if (reason != NULL && reason_cap > 0)          \
        {                                              \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        }                                              \
        return false;                                  \
    } while (0)

    if (s == NULL)
    {
        CKC_CONVVS_REJECT("spec is NULL");
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    /* try: target = ArchTarget.from_gfx(arch) except KeyError as e: return False, str(e) */
    target = ckc_archtarget_from_gfx(arch);
    if (target == NULL)
    {
        /* TODO(port): reproduce the exact KeyError "; known: [...]" suffix. */
        CKC_CONVVS_REJECT("unknown gfx target '%s'", arch);
    }

    /* Geometry divisibility (mirrors spec.validate). */
    if ((s->warp_m * s->warp_tile_m) == 0 ||
        (s->tile_m % (s->warp_m * s->warp_tile_m)) != 0)
    {
        CKC_CONVVS_REJECT("tile_m not divisible by warp_m * warp_tile_m");
    }
    if ((s->warp_n * s->warp_tile_n) == 0 ||
        (s->tile_n % (s->warp_n * s->warp_tile_n)) != 0)
    {
        CKC_CONVVS_REJECT("tile_n not divisible by warp_n * warp_tile_n");
    }
    if (s->warp_tile_k == 0 || (s->tile_k % s->warp_tile_k) != 0)
    {
        CKC_CONVVS_REJECT("tile_k not divisible by warp_tile_k");
    }
    block_size = ckc_implicit_gemm_conv_spec_block_size(s);
    mtpb = ckc_archtarget_max_threads_per_block(target);
    if (block_size > mtpb)
    {
        CKC_CONVVS_REJECT("block_size %d > %d (hardware cap) on %s", block_size,
                          mtpb, arch);
    }

    /* family = "wmma" if target.wave_size == 32 else "mma" */
    family = (target->wave_size == 32) ? "wmma" : "mma";
    /* if spec.wave_size != target.wave_size: return False, ... */
    if (s->wave_size != target->wave_size)
    {
        CKC_CONVVS_REJECT("spec wave_size %d != %s wave_size %d", s->wave_size, arch,
                          target->wave_size);
    }

    /* MMA atom must be in the target's catalog (f16 in/out fp32 acc). */
    mma = ckc_archtarget_mma(target);
    if (!ckc_mma_catalog_has_shape(mma, family, "f16", "f16", "fp32", s->warp_tile_m,
                                   s->warp_tile_n, s->warp_tile_k))
    {
        CKC_CONVVS_REJECT("unsupported f16 warp_tile (%d, %d, %d) on %s",
                          s->warp_tile_m, s->warp_tile_n, s->warp_tile_k, arch);
    }

    /* LDS budget: must fit before we attempt codegen.
     *   A_smem + B_smem, each (tile_m or tile_n) × row_stride × 2 bytes (f16).
     *   row_stride = tile_k + k_pad  (k_pad = 8 when tile_k >= 16, else 0).
     *   compv4 pipeline double-buffers A and B → ×2.
     *   This mirrors the smem_alloc calls in instance_conv_implicit_gemm_conv_build_glue.c
     *   and catches the overflow that would otherwise produce CODEGEN_BC_TO_RELOCATABLE. */
    {
        int k_pad      = (s->tile_k >= 16) ? 8 : 0;
        int row_stride = s->tile_k + k_pad;
        int ab_single  = (s->tile_m + s->tile_n) * row_stride * 2; /* f16 = 2 bytes */
        int double_buf = (s->pipeline && strcmp(s->pipeline, "compv4") == 0) ? 1 : 0;
        int bytes_lds  = ab_single * (double_buf ? 2 : 1);
        if (!ckc_archtarget_fits_lds(target, (long)bytes_lds))
        {
            CKC_CONVVS_REJECT("LDS budget %d > %d cap (AB=%d, double_buf=%d) on %s",
                              bytes_lds, target->lds_capacity_bytes,
                              ab_single, double_buf, arch);
        }
    }

    /* WMMA (RDNA wave32) narrow-subset gates. */
    if (strcmp(family, "wmma") == 0)
    {
        int is_16x16x16 = (s->warp_tile_m == 16 && s->warp_tile_n == 16 &&
                           s->warp_tile_k == 16);
        if (!is_16x16x16)
        {
            CKC_CONVVS_REJECT("WMMA conv supports only 16x16x16 (got (%d, %d, %d)) on %s",
                              s->warp_tile_m, s->warp_tile_n, s->warp_tile_k, arch);
        }
        if (!(s->pipeline && strcmp(s->pipeline, "mem") == 0))
        {
            CKC_CONVVS_REJECT(
                "WMMA conv supports only the 'mem' pipeline (got '%s') on %s",
                s->pipeline ? s->pipeline : "", arch);
        }
        if (!(s->epilogue && strcmp(s->epilogue, "default") == 0))
        {
            CKC_CONVVS_REJECT(
                "WMMA conv supports only the 'default' epilogue (got '%s') on %s",
                s->epilogue ? s->epilogue : "", arch);
        }
        if (s->async_dma)
        {
            CKC_CONVVS_REJECT("WMMA conv does not support async_dma on %s", arch);
        }
        if (s->unroll_k)
        {
            CKC_CONVVS_REJECT("WMMA conv does not support unroll_k on %s", arch);
        }
        if (s->chiplet_swizzle)
        {
            CKC_CONVVS_REJECT("WMMA conv does not support chiplet_swizzle on %s", arch);
        }
        if (s->groups != 1)
        {
            CKC_CONVVS_REJECT("WMMA conv supports only groups=1 (got %d)", s->groups);
        }
    }

    if (reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "ok");
    }
    return true;

#undef CKC_CONVVS_REJECT
}

/* ===================================================================== *
 *  _conv_mma_family(arch)   (Python lines 469-472)
 * ===================================================================== */

const char* ckc_conv_mma_family(const char* arch)
{
    /* return "wmma" if ArchTarget.from_gfx(arch).wave_size == 32 else "mma" */
    const ckc_archtarget_t* target;
    if (arch == NULL)
    {
        arch = "gfx950";
    }
    target = ckc_archtarget_from_gfx(arch);
    if (target == NULL)
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

const ckc_mmaop_t* ckc_conv_resolve_op(ckc_ir_builder_t* b,
                                       const ckc_implicit_gemm_conv_spec_t* spec,
                                       const char* arch)
{
    const ckc_archtarget_t* target;
    const ckc_mmaop_t* op;

    if (b != NULL && !ckc_ir_builder_ok(b))
    {
        return NULL;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    /* target = ArchTarget.from_gfx(arch) */
    target = ckc_archtarget_from_gfx(arch);
    if (target == NULL)
    {
        /* Python raises KeyError; surface a builder error. */
        if (b != NULL && b->status == CKC_OK)
        {
            b->status = CKC_ERR_KEY;
            snprintf(b->err, CKC_ERR_MSG_CAP, "unknown gfx target '%s'", arch);
        }
        return NULL;
    }

    /* op = target.mma.op_for_shape(family=_conv_mma_family(arch),
     *                              a/b="f16", c="fp32",
     *                              m=warp_tile_m, n=warp_tile_n, k=warp_tile_k) */
    op = ckc_archtarget_op_for_shape(target, ckc_conv_mma_family(arch), "f16", "f16",
                                     "fp32", spec->warp_tile_m, spec->warp_tile_n,
                                     spec->warp_tile_k);
    if (op == NULL)
    {
        /* raise ValueError(f"no MMA atom for conv warp_tile (...) on {arch}") */
        if (b != NULL && b->status == CKC_OK)
        {
            b->status = CKC_ERR_VALUE;
            snprintf(b->err, CKC_ERR_MSG_CAP,
                     "no MMA atom for conv warp_tile (%d,%d,%d) on %s",
                     spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k, arch);
        }
        return NULL;
    }
    return op;
}

/* ===================================================================== *
 *  Descriptor builders   (Python lines 509-629)
 * ===================================================================== */

struct ckc_tensor_descriptor* ckc_conv_make_a_descriptor(ckc_ir_builder_t* b,
                                                         const ckc_conv_problem_t* p,
                                                         bool decompose_m)
{
    /* transforms = []
     * if decompose_m: transforms.append(unmerge_magic('m'->[n,ho,wo],[N,Ho,Wo]))
     * transforms += [
     *   embed(['ho','r']->'hi', strides=[sH,dH], offset=-pH, lo=0, hi=Hi),
     *   embed(['wo','s']->'wi', strides=[sW,dW], offset=-pW, lo=0, hi=Wi),
     *   unmerge_magic('k'->[r,s,c],[R,S,C]),
     *   pad('r', lo=0, hi=R),
     *   pad('s', lo=0, hi=S),
     * ]
     * return TensorDescriptor.naive('A_nhwc', lengths=[N,Hi,Wi,C], dtype=F16,
     *   coord_names=['n','hi','wi','c']).transform(*transforms) */
    int Ho, Wo;
    int lengths[4];
    const char* coords[4];
    const char* into_m[3];
    const char* up_ho[2];
    const char* up_wo[2];
    const char* into_k[3];
    int dims_m[3];
    int strides_ho[2];
    int strides_wo[2];
    int dims_k[3];
    const ckc_transform_t* xforms[6];
    int n_x = 0;
    ckc_tensor_descriptor_t* desc;

    if (b == NULL || !ckc_ir_builder_ok(b) || p == NULL)
    {
        return NULL;
    }

    Ho = ckc_conv_problem_ho(p);
    Wo = ckc_conv_problem_wo(p);

    if (decompose_m)
    {
        into_m[0] = "n";
        into_m[1] = "ho";
        into_m[2] = "wo";
        dims_m[0] = p->N;
        dims_m[1] = Ho;
        dims_m[2] = Wo;
        xforms[n_x] = ckc_unmerge_magic(b, "m", into_m, 3, dims_m);
        if (xforms[n_x] == NULL)
        {
            return NULL;
        }
        n_x++;
    }

    /* embed(['ho','r'] -> 'hi', strides=[sH,dH], offset=-pH, lo=0, hi=Hi) */
    up_ho[0] = "ho";
    up_ho[1] = "r";
    strides_ho[0] = p->sH;
    strides_ho[1] = p->dH;
    xforms[n_x] = ckc_embed_bounded(b, up_ho, 2, "hi", strides_ho, -p->pH, 0, p->Hi);
    if (xforms[n_x] == NULL)
    {
        return NULL;
    }
    n_x++;

    /* embed(['wo','s'] -> 'wi', strides=[sW,dW], offset=-pW, lo=0, hi=Wi) */
    up_wo[0] = "wo";
    up_wo[1] = "s";
    strides_wo[0] = p->sW;
    strides_wo[1] = p->dW;
    xforms[n_x] = ckc_embed_bounded(b, up_wo, 2, "wi", strides_wo, -p->pW, 0, p->Wi);
    if (xforms[n_x] == NULL)
    {
        return NULL;
    }
    n_x++;

    /* unmerge_magic('k' -> [r,s,c], dims=[R,S,C]) */
    into_k[0] = "r";
    into_k[1] = "s";
    into_k[2] = "c";
    dims_k[0] = p->R;
    dims_k[1] = p->S;
    dims_k[2] = p->C;
    xforms[n_x] = ckc_unmerge_magic(b, "k", into_k, 3, dims_k);
    if (xforms[n_x] == NULL)
    {
        return NULL;
    }
    n_x++;

    /* pad('r', lo=0, hi=R) */
    xforms[n_x] = ckc_pad(b, "r", 0, p->R);
    if (xforms[n_x] == NULL)
    {
        return NULL;
    }
    n_x++;

    /* pad('s', lo=0, hi=S) */
    xforms[n_x] = ckc_pad(b, "s", 0, p->S);
    if (xforms[n_x] == NULL)
    {
        return NULL;
    }
    n_x++;

    lengths[0] = p->N;
    lengths[1] = p->Hi;
    lengths[2] = p->Wi;
    lengths[3] = p->C;
    coords[0] = "n";
    coords[1] = "hi";
    coords[2] = "wi";
    coords[3] = "c";
    desc = ckc_tensor_descriptor_naive(b, "A_nhwc", lengths, 4, NULL, coords, 4);
    if (desc == NULL)
    {
        return NULL;
    }
    return ckc_tensor_descriptor_transform(b, desc, xforms, n_x);
}

struct ckc_tensor_descriptor* ckc_conv_make_b_descriptor(ckc_ir_builder_t* b,
                                                         const ckc_conv_problem_t* p)
{
    /* return TensorDescriptor.naive('B_krsc', lengths=[K,R,S,C], dtype=F16,
     *   coord_names=['k_out','r','s','c']).transform(
     *     unmerge_magic('k_gemm' -> [r,s,c], dims=[R,S,C]),
     *     pad('r', lo=0, hi=R),
     *     pad('s', lo=0, hi=S)) */
    int lengths[4];
    const char* coords[4];
    const char* into_k[3];
    int dims_k[3];
    const ckc_transform_t* xforms[3];
    ckc_tensor_descriptor_t* desc;

    if (b == NULL || !ckc_ir_builder_ok(b) || p == NULL)
    {
        return NULL;
    }

    into_k[0] = "r";
    into_k[1] = "s";
    into_k[2] = "c";
    dims_k[0] = p->R;
    dims_k[1] = p->S;
    dims_k[2] = p->C;
    xforms[0] = ckc_unmerge_magic(b, "k_gemm", into_k, 3, dims_k);
    if (xforms[0] == NULL)
    {
        return NULL;
    }
    xforms[1] = ckc_pad(b, "r", 0, p->R);
    if (xforms[1] == NULL)
    {
        return NULL;
    }
    xforms[2] = ckc_pad(b, "s", 0, p->S);
    if (xforms[2] == NULL)
    {
        return NULL;
    }

    lengths[0] = p->K;
    lengths[1] = p->R;
    lengths[2] = p->S;
    lengths[3] = p->C;
    coords[0] = "k_out";
    coords[1] = "r";
    coords[2] = "s";
    coords[3] = "c";
    desc = ckc_tensor_descriptor_naive(b, "B_krsc", lengths, 4, NULL, coords, 4);
    if (desc == NULL)
    {
        return NULL;
    }
    return ckc_tensor_descriptor_transform(b, desc, xforms, 3);
}

struct ckc_tensor_descriptor* ckc_conv_make_d_descriptor(ckc_ir_builder_t* b,
                                                         const ckc_conv_problem_t* p)
{
    /* return TensorDescriptor.naive('D_nhwk', lengths=[N,Ho,Wo,K], dtype=F16,
     *   coord_names=['n','ho','wo','k_out']).transform(
     *     unmerge_magic('m' -> [n,ho,wo], dims=[N,Ho,Wo])) */
    int Ho, Wo;
    int lengths[4];
    const char* coords[4];
    const char* into_m[3];
    int dims_m[3];
    const ckc_transform_t* xforms[1];
    ckc_tensor_descriptor_t* desc;

    if (b == NULL || !ckc_ir_builder_ok(b) || p == NULL)
    {
        return NULL;
    }

    Ho = ckc_conv_problem_ho(p);
    Wo = ckc_conv_problem_wo(p);

    into_m[0] = "n";
    into_m[1] = "ho";
    into_m[2] = "wo";
    dims_m[0] = p->N;
    dims_m[1] = Ho;
    dims_m[2] = Wo;
    xforms[0] = ckc_unmerge_magic(b, "m", into_m, 3, dims_m);
    if (xforms[0] == NULL)
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
    desc = ckc_tensor_descriptor_naive(b, "D_nhwk", lengths, 4, NULL, coords, 4);
    if (desc == NULL)
    {
        return NULL;
    }
    return ckc_tensor_descriptor_transform(b, desc, xforms, 1);
}
