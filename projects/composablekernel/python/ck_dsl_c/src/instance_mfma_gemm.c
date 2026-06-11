/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_mfma_gemm.c -- C99 port of ck_dsl/instances/common/mfma_gemm.py.
 *
 * The MFMA-tiled GEMM kernel: the first K-packed MFMA instance (one atom per
 * CTA, no LDS staging). build_mfma_gemm reuses the seven ported helpers in
 * ckc/helper_ck_dsl.helpers.mfma_gemm_inner.h and the MfmaAtom catalog, exactly
 * as the Python imports + calls them. The build op order tracks
 * build_mfma_gemm() top-to-bottom so a reviewer can diff line by line.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/instance_mfma_gemm.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */

#include "ckc/helper_ck_dsl.core.arch.h"
#include "ckc/helper_ck_dsl.helpers.atoms.h"
#include "ckc/helper_ck_dsl.helpers.mfma_gemm_inner.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/lower_llvm.h"

/* mfma_gemm.py module constants. */
#define CKC_MFMA_GEMM_DEFAULT_NAME "ck_dsl_mfma_gemm"

/* _SUPPORTED_DTYPES = ("f16", "bf16"). */
static bool ckc_mfma_gemm_dtype_supported(const char* dtype)
{
    return dtype != NULL && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "bf16") == 0);
}

/* _SUPPORTED_ATOM_MN = ((16, 16), (32, 32)). */
static bool ckc_mfma_gemm_mn_supported(int tile_m, int tile_n)
{
    return (tile_m == 16 && tile_n == 16) || (tile_m == 32 && tile_n == 32);
}

/* _CATALOG_DTYPE = {"f16": "fp16", "fp16": "fp16", "bf16": "bf16"}.
 * Returns the catalog dtype name, or NULL on the Python `.get(...) is None`
 * miss path. */
static const char* ckc_mfma_gemm_catalog_dtype(const char* dtype)
{
    if (dtype == NULL)
    {
        return NULL;
    }
    if (strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0)
    {
        return "fp16";
    }
    if (strcmp(dtype, "bf16") == 0)
    {
        return "bf16";
    }
    return NULL;
}

/* ===================================================================== *
 *  Spec value accessors (the Python @property methods)
 * ===================================================================== */

ckc_mfma_gemm_spec_t ckc_mfma_gemm_spec_default(void)
{
    ckc_mfma_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.M = 0;
    s.N = 0;
    s.K = 0;
    s.dtype = "f16";
    s.tile_m = 16;
    s.tile_n = 16;
    s.kpack = true;
    s.name = CKC_MFMA_GEMM_DEFAULT_NAME;
    return s;
}

/* MfmaGemmSpec.atom: mfma_atom_for_dtype(dtype, tile_m, tile_n,
 *                                        prefer_packed_k=kpack). */
const ckc_mfma_atom_t* ckc_mfma_gemm_atom(const ckc_mfma_gemm_spec_t* spec)
{
    if (spec == NULL)
    {
        return NULL;
    }
    return ckc_mfma_atom_for_dtype(spec->dtype, spec->tile_m, spec->tile_n, spec->kpack);
}

/* MfmaGemmSpec.tile_k: self.atom.k. */
int ckc_mfma_gemm_tile_k(const ckc_mfma_gemm_spec_t* spec)
{
    const ckc_mfma_atom_t* atom = ckc_mfma_gemm_atom(spec);
    return atom != NULL ? atom->k : 0;
}

/* MfmaGemmSpec.block_size: one wave64 warp per CTA. */
int ckc_mfma_gemm_block_size(const ckc_mfma_gemm_spec_t* spec)
{
    (void)spec;
    return 64;
}

/* MfmaGemmSpec.kernel_name():
 *   kernel_name_join(self.name, f"M{M}N{N}K{K}", self.dtype,
 *                    f"atom{m}x{n}x{k}", flags={"kpack": self.kpack}). */
ckc_status_t ckc_mfma_gemm_kernel_name(const ckc_mfma_gemm_spec_t* spec,
                                       char* out,
                                       size_t out_cap)
{
    const ckc_mfma_atom_t* atom;
    char mnk[64];
    char atombuf[64];
    const char* parts[3];
    const char* flag_names[1];
    int flag_on[1];

    if (spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    atom = ckc_mfma_gemm_atom(spec);
    if (atom == NULL)
    {
        return CKC_ERR_VALUE;
    }

    /* f"M{M}N{N}K{K}" */
    if (snprintf(mnk, sizeof(mnk), "M%dN%dK%d", spec->M, spec->N, spec->K) < 0)
    {
        return CKC_ERR_VALUE;
    }
    /* f"atom{m}x{n}x{k}" */
    if (snprintf(atombuf, sizeof(atombuf), "atom%dx%dx%d", atom->m, atom->n, atom->k) < 0)
    {
        return CKC_ERR_VALUE;
    }

    parts[0] = mnk;
    parts[1] = spec->dtype;
    parts[2] = atombuf;
    flag_names[0] = "kpack";
    flag_on[0] = spec->kpack ? 1 : 0;

    return ckc_kernel_name_join(spec->name, parts, 3, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)
 * ===================================================================== */

/* Write `msg` into reason (capacity reason_cap), NUL-terminated. */
static void ckc_mfma_gemm_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    if (reason != NULL && reason_cap > 0)
    {
        size_t n = strlen(msg);
        if (n >= reason_cap)
        {
            n = reason_cap - 1;
        }
        memcpy(reason, msg, n);
        reason[n] = '\0';
    }
}

bool ckc_mfma_gemm_is_valid_spec(const ckc_mfma_gemm_spec_t* spec,
                                 const char* arch,
                                 char* reason,
                                 size_t reason_cap)
{
    const ckc_mfma_atom_t* atom;
    const ckc_archtarget_t* target;
    const char* cat_dtype;
    char buf[CKC_ERR_MSG_CAP];

    if (spec == NULL)
    {
        ckc_mfma_gemm_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    /* if spec.dtype not in _SUPPORTED_DTYPES: ... */
    if (!ckc_mfma_gemm_dtype_supported(spec->dtype))
    {
        snprintf(buf, sizeof(buf),
                 "mfma_gemm dtype must be one of ('f16', 'bf16'), got %s%s%s",
                 spec->dtype ? "'" : "", spec->dtype ? spec->dtype : "None",
                 spec->dtype ? "'" : "");
        ckc_mfma_gemm_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if (spec.tile_m, spec.tile_n) not in _SUPPORTED_ATOM_MN: ... */
    if (!ckc_mfma_gemm_mn_supported(spec->tile_m, spec->tile_n))
    {
        snprintf(buf, sizeof(buf),
                 "tile_m x tile_n must be one of ((16, 16), (32, 32)); got (%d, %d)",
                 spec->tile_m, spec->tile_n);
        ckc_mfma_gemm_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* atom = spec.atom */
    atom = ckc_mfma_gemm_atom(spec);
    if (atom == NULL)
    {
        /* The mfma_atom_for_dtype ValueError path -- unreachable given the two
         * gates above for the shipped dtypes, but mirror it defensively. */
        ckc_mfma_gemm_set_reason(reason, reason_cap, "no MFMA atom for spec");
        return false;
    }

    /* if spec.M % atom.m or spec.N % atom.n or spec.K % atom.k: ... */
    if ((spec->M % atom->m) || (spec->N % atom->n) || (spec->K % atom->k))
    {
        snprintf(buf, sizeof(buf),
                 "M / N / K must be divisible by the %dx%dx%d atom shape; "
                 "got M=%d, N=%d, K=%d",
                 atom->m, atom->n, atom->k, spec->M, spec->N, spec->K);
        ckc_mfma_gemm_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* try: target = ArchTarget.from_gfx(arch) except KeyError: ... */
    target = ckc_archtarget_from_gfx(arch);
    if (target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown gfx target %s%s%s",
                 arch ? "'" : "", arch ? arch : "None", arch ? "'" : "");
        ckc_mfma_gemm_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* cat_dtype = _CATALOG_DTYPE.get(spec.dtype); if None: ... */
    cat_dtype = ckc_mfma_gemm_catalog_dtype(spec->dtype);
    if (cat_dtype == NULL)
    {
        snprintf(buf, sizeof(buf), "no MFMA-catalog dtype mapping for '%s'", spec->dtype);
        ckc_mfma_gemm_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if not target.mma.has_shape(a=cat, b=cat, c="fp32", m, n, k): ...
     * op_for_shape returns NULL when the shape/dtype combo is absent. */
    if (ckc_archtarget_op_for_shape(target, "mma", cat_dtype, cat_dtype, "fp32",
                                    atom->m, atom->n, atom->k) == NULL)
    {
        snprintf(buf, sizeof(buf),
                 "%s MFMA atom %dx%dx%d (kpack=%s) not available on %s; "
                 "set kpack=False for the legacy atom on pre-CDNA4 targets",
                 spec->dtype, atom->m, atom->n, atom->k,
                 spec->kpack ? "True" : "False", arch);
        ckc_mfma_gemm_set_reason(reason, reason_cap, buf);
        return false;
    }

    ckc_mfma_gemm_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  load callbacks
 *
 *  The Python build defines two closures `_load_a` / `_load_b` capturing
 *  (A/Bp, atom, lane_decode, m_tile_base / n_tile_base, c_atom_k, K/N) and
 *  forwarding to load_a_row_major_contiguous / load_b_col_strided_scalars with
 *  k_tile_base = kt * atom.k. In C the captured environment is this struct,
 *  passed as the opaque `user` pointer to ckc_mfma_k_loop.
 * ===================================================================== */
typedef struct ckc_mfma_gemm_load_ctx
{
    ckc_value_t* A;
    ckc_value_t* Bp;
    const ckc_mfma_atom_t* atom;
    const ckc_lane_decode_t* lane_decode;
    ckc_value_t* m_tile_base;
    ckc_value_t* n_tile_base;
    ckc_value_t* c_atom_k;
    int K;
    int N;
} ckc_mfma_gemm_load_ctx_t;

/* def _load_a(b, kt):
 *     return load_a_row_major_contiguous(
 *         b, A=A, atom=atom, lane_decode=lane_decode,
 *         m_tile_base=m_tile_base, k_tile_base=b.mul(kt, c_atom_k), K=spec.K) */
static ckc_value_t* ckc_mfma_gemm_load_a_cb(ckc_ir_builder_t* b, ckc_value_t* kt, void* user)
{
    ckc_mfma_gemm_load_ctx_t* c = (ckc_mfma_gemm_load_ctx_t*)user;
    ckc_value_t* k_tile_base = ckc_b_mul(b, kt, c->c_atom_k);
    return ckc_load_a_row_major_contiguous(b, c->A, c->atom, c->lane_decode,
                                           c->m_tile_base, k_tile_base, c->K);
}

/* def _load_b(b, kt):
 *     return load_b_col_strided_scalars(
 *         b, B=Bp, atom=atom, lane_decode=lane_decode,
 *         n_tile_base=n_tile_base, k_tile_base=b.mul(kt, c_atom_k), N=spec.N) */
static ckc_value_t* ckc_mfma_gemm_load_b_cb(ckc_ir_builder_t* b, ckc_value_t* kt, void* user)
{
    ckc_mfma_gemm_load_ctx_t* c = (ckc_mfma_gemm_load_ctx_t*)user;
    ckc_value_t* k_tile_base = ckc_b_mul(b, kt, c->c_atom_k);
    return ckc_load_b_col_strided_scalars(b, c->Bp, c->atom, c->lane_decode,
                                          c->n_tile_base, k_tile_base, c->N);
}

/* ===================================================================== *
 *  build_mfma_gemm(spec, arch)
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_mfma_gemm(ckc_ir_builder_t* b,
                                      const ckc_mfma_gemm_spec_t* spec,
                                      const char* arch)
{
    const ckc_mfma_atom_t* atom;
    int BS;
    const ckc_type_t* elem_ir;
    ckc_value_t* A;
    ckc_value_t* Bp;
    ckc_value_t* C;
    ckc_value_t* lane;
    ckc_value_t* bid_n;
    ckc_value_t* bid_m;
    ckc_value_t* m_tile_base;
    ckc_value_t* n_tile_base;
    ckc_lane_decode_t lane_decode;
    ckc_value_t* c_atom_k;
    ckc_value_t* acc_final;
    ckc_mfma_gemm_load_ctx_t lctx;
    char reason[CKC_ERR_MSG_CAP];

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
    if (!ckc_mfma_gemm_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        char msg[CKC_ERR_MSG_CAP];
        snprintf(msg, sizeof(msg), "invalid mfma_gemm spec for %s: %s", arch, reason);
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "%s", msg);
        return NULL;
    }

    atom = ckc_mfma_gemm_atom(spec);
    BS = ckc_mfma_gemm_block_size(spec);

    /* The builder `b` is assumed already initialised by the caller with
     * spec.kernel_name() (per the public header contract). Set the attr the
     * Python bakes in: b.kernel.attrs["max_workgroup_size"] = BS. */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    /* elem_ir = BF16 if dtype == "bf16" else F16 */
    elem_ir = (strcmp(spec->dtype, "bf16") == 0) ? ckc_bf16() : ckc_f16();

    /* ---- kernel params -- */
    {
        ckc_param_opts_t opts;
        const ckc_type_t* ptr_elem = ckc_ptr_type(b, elem_ir, "global");

        /* A = b.param("A", PtrType(elem_ir,"global"), noalias, readonly, align16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        A = ckc_b_param(b, "A", ptr_elem, &opts);
        Bp = ckc_b_param(b, "B", ptr_elem, &opts);

        /* C = b.param("C", ..., noalias, writeonly, align16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        C = ckc_b_param(b, "C", ptr_elem, &opts);

        /* M / N / K : i32 (ABI; unused after declare) */
        (void)ckc_b_param(b, "M", ckc_i32(), NULL);
        (void)ckc_b_param(b, "N", ckc_i32(), NULL);
        (void)ckc_b_param(b, "K", ckc_i32(), NULL);
    }

    /* lane = b.thread_id_x(); bid_n = b.block_id_x(); bid_m = b.block_id_y() */
    lane = ckc_b_thread_id_x(b);
    bid_n = ckc_b_block_id_x(b);
    bid_m = ckc_b_block_id_y(b);

    /* m_tile_base = bid_m * atom.m; n_tile_base = bid_n * atom.n */
    m_tile_base = ckc_b_mul(b, bid_m, ckc_b_const_i32(b, atom->m));
    n_tile_base = ckc_b_mul(b, bid_n, ckc_b_const_i32(b, atom->n));

    /* lane_decode = decode_mfma_lanes(b, atom, lane) */
    lane_decode = ckc_decode_mfma_lanes(b, atom, lane);

    /* c_atom_k = b.const_i32(atom.k) */
    c_atom_k = ckc_b_const_i32(b, atom->k);

    /* closure environment for _load_a / _load_b */
    lctx.A = A;
    lctx.Bp = Bp;
    lctx.atom = atom;
    lctx.lane_decode = &lane_decode;
    lctx.m_tile_base = m_tile_base;
    lctx.n_tile_base = n_tile_base;
    lctx.c_atom_k = c_atom_k;
    lctx.K = spec->K;
    lctx.N = spec->N;

    /* acc_final = mfma_k_loop(b, K=spec.K, atom=atom, load_a=_load_a,
     *                         load_b=_load_b)
     * (per_tile_post_mfma=None, initial_acc=None, iv_name="kt", acc_name="acc") */
    acc_final = ckc_mfma_k_loop(b, spec->K, atom,
                                ckc_mfma_gemm_load_a_cb,
                                ckc_mfma_gemm_load_b_cb,
                                NULL,  /* per_tile_post_mfma */
                                NULL,  /* initial_acc */
                                NULL,  /* iv_name => "kt" */
                                NULL,  /* acc_name => "acc" */
                                &lctx);

    /* store_acc_to_global(b, C=C, atom=atom, lane_decode=lane_decode,
     *                     m_tile_base, n_tile_base, acc=acc_final, N=spec.N,
     *                     out_dtype=spec.dtype) */
    (void)ckc_store_acc_to_global(b, C, atom, &lane_decode,
                                  m_tile_base, n_tile_base, acc_final,
                                  spec->N, spec->dtype,
                                  false,  /* atomic_add */
                                  NULL,   /* epilogue */
                                  NULL);  /* epilogue_user */

    /* b.ret() */
    ckc_b_ret(b);

    if (!ckc_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

/* ===================================================================== *
 *  ckc_build_mfma_gemm_new -- init builder with spec.kernel_name() then build.
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_mfma_gemm_new(ckc_ir_builder_t* b,
                                          const ckc_mfma_gemm_spec_t* spec,
                                          const char* arch)
{
    char name[256];
    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (ckc_mfma_gemm_kernel_name(spec, name, sizeof(name)) != CKC_OK)
    {
        return NULL;
    }
    if (ckc_ir_builder_init(b, name) != CKC_OK)
    {
        return NULL;
    }
    return ckc_build_mfma_gemm(b, spec, arch);
}

/* ===================================================================== *
 *  mfma_gemm_grid(spec) -> (N // atom.n, M // atom.m, 1)
 * ===================================================================== */
ckc_status_t ckc_mfma_gemm_grid(const ckc_mfma_gemm_spec_t* spec, int out[3])
{
    const ckc_mfma_atom_t* atom;
    if (spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    atom = ckc_mfma_gemm_atom(spec);
    if (atom == NULL)
    {
        return CKC_ERR_VALUE;
    }
    out[0] = spec->N / atom->n;
    out[1] = spec->M / atom->m;
    out[2] = 1;
    return CKC_OK;
}

/* ===================================================================== *
 *  ckc_mfma_gemm_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
ckc_status_t ckc_mfma_gemm_lower_to_llvm(const ckc_mfma_gemm_spec_t* spec,
                                         const char* arch,
                                         ckc_llvm_flavor_t flavor,
                                         char** out_ll,
                                         char* err,
                                         size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        if (err != NULL && err_cap > 0)
        {
            const char* m = "lower_to_llvm: null spec/out";
            size_t n = strlen(m);
            if (n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_mfma_gemm_new(&b, spec, arch);
    if (kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if (err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            size_t n;
            if (m == NULL)
            {
                m = "build_mfma_gemm failed";
            }
            n = strlen(m);
            if (n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
