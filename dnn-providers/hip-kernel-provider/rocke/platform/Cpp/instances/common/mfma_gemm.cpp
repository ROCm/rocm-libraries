// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_mfma_gemm.c -- C99 port of rocke/instances/common/mfma_gemm.py.
 *
 * The MFMA-tiled GEMM kernel: the first K-packed MFMA instance (one atom per
 * CTA, no LDS staging). build_mfma_gemm reuses the seven ported helpers in
 * rocke/helper_rocke.helpers.mfma_gemm_inner.h and the MfmaAtom catalog, exactly
 * as the Python imports + calls them. The build op order tracks
 * build_mfma_gemm() top-to-bottom so a reviewer can diff line by line.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_mfma_gemm.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

/* mfma_gemm.py module constants. */
#define ROCKE_MFMA_GEMM_DEFAULT_NAME "rocke_mfma_gemm"

/* _SUPPORTED_DTYPES = ("f16", "bf16"). */
static bool rocke_mfma_gemm_dtype_supported(const char* dtype)
{
    return dtype != NULL && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "bf16") == 0);
}

/* _SUPPORTED_ATOM_MN = ((16, 16), (32, 32)). */
static bool rocke_mfma_gemm_mn_supported(int tile_m, int tile_n)
{
    return (tile_m == 16 && tile_n == 16) || (tile_m == 32 && tile_n == 32);
}

/* _CATALOG_DTYPE = {"f16": "fp16", "fp16": "fp16", "bf16": "bf16"}.
 * Returns the catalog dtype name, or NULL on the Python `.get(...) is None`
 * miss path. */
static const char* rocke_mfma_gemm_catalog_dtype(const char* dtype)
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
    return NULL;
}

/* ===================================================================== *
 *  Spec value accessors (the Python @property methods)
 * ===================================================================== */

rocke_mfma_gemm_spec_t rocke_mfma_gemm_spec_default(void)
{
    rocke_mfma_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.M = 0;
    s.N = 0;
    s.K = 0;
    s.dtype = "f16";
    s.tile_m = 16;
    s.tile_n = 16;
    s.kpack = true;
    s.name = ROCKE_MFMA_GEMM_DEFAULT_NAME;
    return s;
}

/* MfmaGemmSpec.atom: mfma_atom_for_dtype(dtype, tile_m, tile_n,
 *                                        prefer_packed_k=kpack). */
const rocke_mfma_atom_t* rocke_mfma_gemm_atom(const rocke_mfma_gemm_spec_t* spec)
{
    if(spec == NULL)
    {
        return NULL;
    }
    return rocke_mfma_atom_for_dtype(spec->dtype, spec->tile_m, spec->tile_n, spec->kpack);
}

/* MfmaGemmSpec.tile_k: self.atom.k. */
int rocke_mfma_gemm_tile_k(const rocke_mfma_gemm_spec_t* spec)
{
    const rocke_mfma_atom_t* atom = rocke_mfma_gemm_atom(spec);
    return atom != NULL ? atom->k : 0;
}

/* MfmaGemmSpec.block_size: one wave64 warp per CTA. */
int rocke_mfma_gemm_block_size(const rocke_mfma_gemm_spec_t* spec)
{
    (void)spec;
    return 64;
}

/* MfmaGemmSpec.kernel_name():
 *   kernel_name_join(self.name, f"M{M}N{N}K{K}", self.dtype,
 *                    f"atom{m}x{n}x{k}", flags={"kpack": self.kpack}). */
rocke_status_t
    rocke_mfma_gemm_kernel_name(const rocke_mfma_gemm_spec_t* spec, char* out, size_t out_cap)
{
    const rocke_mfma_atom_t* atom;
    char mnk[64];
    char atombuf[64];
    const char* parts[3];
    const char* flag_names[1];
    int flag_on[1];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    atom = rocke_mfma_gemm_atom(spec);
    if(atom == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* f"M{M}N{N}K{K}" */
    if(snprintf(mnk, sizeof(mnk), "M%dN%dK%d", spec->M, spec->N, spec->K) < 0)
    {
        return ROCKE_ERR_VALUE;
    }
    /* f"atom{m}x{n}x{k}" */
    if(snprintf(atombuf, sizeof(atombuf), "atom%dx%dx%d", atom->m, atom->n, atom->k) < 0)
    {
        return ROCKE_ERR_VALUE;
    }

    parts[0] = mnk;
    parts[1] = spec->dtype;
    parts[2] = atombuf;
    flag_names[0] = "kpack";
    flag_on[0] = spec->kpack ? 1 : 0;

    return rocke_kernel_name_join(spec->name, parts, 3, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)
 * ===================================================================== */

/* Write `msg` into reason (capacity reason_cap), NUL-terminated. */
static void rocke_mfma_gemm_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

bool rocke_mfma_gemm_is_valid_spec(const rocke_mfma_gemm_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap)
{
    const rocke_mfma_atom_t* atom;
    const rocke_archtarget_t* target;
    const char* cat_dtype;
    char buf[ROCKE_ERR_MSG_CAP];

    if(spec == NULL)
    {
        rocke_mfma_gemm_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* if spec.dtype not in _SUPPORTED_DTYPES: ... */
    if(!rocke_mfma_gemm_dtype_supported(spec->dtype))
    {
        snprintf(buf,
                 sizeof(buf),
                 "mfma_gemm dtype must be one of ('f16', 'bf16'), got %s%s%s",
                 spec->dtype ? "'" : "",
                 spec->dtype ? spec->dtype : "None",
                 spec->dtype ? "'" : "");
        rocke_mfma_gemm_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if (spec.tile_m, spec.tile_n) not in _SUPPORTED_ATOM_MN: ... */
    if(!rocke_mfma_gemm_mn_supported(spec->tile_m, spec->tile_n))
    {
        snprintf(buf,
                 sizeof(buf),
                 "tile_m x tile_n must be one of ((16, 16), (32, 32)); got (%d, %d)",
                 spec->tile_m,
                 spec->tile_n);
        rocke_mfma_gemm_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* atom = spec.atom */
    atom = rocke_mfma_gemm_atom(spec);
    if(atom == NULL)
    {
        /* The mfma_atom_for_dtype ValueError path -- unreachable given the two
         * gates above for the shipped dtypes, but mirror it defensively. */
        rocke_mfma_gemm_set_reason(reason, reason_cap, "no MFMA atom for spec");
        return false;
    }

    /* if spec.M % atom.m or spec.N % atom.n or spec.K % atom.k: ... */
    if((spec->M % atom->m) || (spec->N % atom->n) || (spec->K % atom->k))
    {
        snprintf(buf,
                 sizeof(buf),
                 "M / N / K must be divisible by the %dx%dx%d atom shape; "
                 "got M=%d, N=%d, K=%d",
                 atom->m,
                 atom->n,
                 atom->k,
                 spec->M,
                 spec->N,
                 spec->K);
        rocke_mfma_gemm_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* try: target = ArchTarget.from_gfx(arch) except KeyError: ... */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf,
                 sizeof(buf),
                 "unknown gfx target %s%s%s",
                 arch ? "'" : "",
                 arch ? arch : "None",
                 arch ? "'" : "");
        rocke_mfma_gemm_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* cat_dtype = _CATALOG_DTYPE.get(spec.dtype); if None: ... */
    cat_dtype = rocke_mfma_gemm_catalog_dtype(spec->dtype);
    if(cat_dtype == NULL)
    {
        snprintf(buf, sizeof(buf), "no MFMA-catalog dtype mapping for '%s'", spec->dtype);
        rocke_mfma_gemm_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if not target.mma.has_shape(a=cat, b=cat, c="fp32", m, n, k): ...
     * op_for_shape returns NULL when the shape/dtype combo is absent. */
    if(rocke_archtarget_op_for_shape(
           target, "mma", cat_dtype, cat_dtype, "fp32", atom->m, atom->n, atom->k)
       == NULL)
    {
        snprintf(buf,
                 sizeof(buf),
                 "%s MFMA atom %dx%dx%d (kpack=%s) not available on %s; "
                 "set kpack=False for the legacy atom on pre-CDNA4 targets",
                 spec->dtype,
                 atom->m,
                 atom->n,
                 atom->k,
                 spec->kpack ? "True" : "False",
                 arch);
        rocke_mfma_gemm_set_reason(reason, reason_cap, buf);
        return false;
    }

    rocke_mfma_gemm_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  load callbacks
 *
 *  The Python build defines two closures `_load_a` / `_load_b` capturing
 *  (A/Bp, atom, lane_decode, m_tile_base / n_tile_base, c_atom_k, K/N) and
 *  forwarding to load_a_row_major_contiguous / load_b_col_strided_scalars with
 *  k_tile_base = kt * atom.k. In C the captured environment is this struct,
 *  passed as the opaque `user` pointer to rocke_mfma_k_loop.
 * ===================================================================== */
typedef struct rocke_mfma_gemm_load_ctx
{
    rocke_value_t* A;
    rocke_value_t* Bp;
    const rocke_mfma_atom_t* atom;
    const rocke_lane_decode_t* lane_decode;
    rocke_value_t* m_tile_base;
    rocke_value_t* n_tile_base;
    rocke_value_t* c_atom_k;
    int K;
    int N;
} rocke_mfma_gemm_load_ctx_t;

/* def _load_a(b, kt):
 *     return load_a_row_major_contiguous(
 *         b, A=A, atom=atom, lane_decode=lane_decode,
 *         m_tile_base=m_tile_base, k_tile_base=b.mul(kt, c_atom_k), K=spec.K) */
static rocke_value_t*
    rocke_mfma_gemm_load_a_cb(rocke_ir_builder_t* b, rocke_value_t* kt, void* user)
{
    rocke_mfma_gemm_load_ctx_t* c = (rocke_mfma_gemm_load_ctx_t*)user;
    rocke_value_t* k_tile_base = rocke_b_mul(b, kt, c->c_atom_k);
    return rocke_load_a_row_major_contiguous(
        b, c->A, c->atom, c->lane_decode, c->m_tile_base, k_tile_base, c->K);
}

/* def _load_b(b, kt):
 *     return load_b_col_strided_scalars(
 *         b, B=Bp, atom=atom, lane_decode=lane_decode,
 *         n_tile_base=n_tile_base, k_tile_base=b.mul(kt, c_atom_k), N=spec.N) */
static rocke_value_t*
    rocke_mfma_gemm_load_b_cb(rocke_ir_builder_t* b, rocke_value_t* kt, void* user)
{
    rocke_mfma_gemm_load_ctx_t* c = (rocke_mfma_gemm_load_ctx_t*)user;
    rocke_value_t* k_tile_base = rocke_b_mul(b, kt, c->c_atom_k);
    return rocke_load_b_col_strided_scalars(
        b, c->Bp, c->atom, c->lane_decode, c->n_tile_base, k_tile_base, c->N);
}

/* ===================================================================== *
 *  build_mfma_gemm(spec, arch)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_mfma_gemm(rocke_ir_builder_t* b,
                                          const rocke_mfma_gemm_spec_t* spec,
                                          const char* arch)
{
    const rocke_mfma_atom_t* atom;
    int BS;
    const rocke_type_t* elem_ir;
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* C;
    rocke_value_t* lane;
    rocke_value_t* bid_n;
    rocke_value_t* bid_m;
    rocke_value_t* m_tile_base;
    rocke_value_t* n_tile_base;
    rocke_lane_decode_t lane_decode;
    rocke_value_t* c_atom_k;
    rocke_value_t* acc_final;
    rocke_mfma_gemm_load_ctx_t lctx;
    char reason[ROCKE_ERR_MSG_CAP];

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
    if(!rocke_mfma_gemm_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        char msg[ROCKE_ERR_MSG_CAP];
        ROCKE_ERR_SNPRINTF(msg, sizeof(msg), "invalid mfma_gemm spec for %s: %s", arch, reason);
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", msg);
        return NULL;
    }

    atom = rocke_mfma_gemm_atom(spec);
    BS = rocke_mfma_gemm_block_size(spec);

    /* The builder `b` is assumed already initialised by the caller with
     * spec.kernel_name() (per the public header contract). Set the attr the
     * Python bakes in: b.kernel.attrs["max_workgroup_size"] = BS. */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    /* elem_ir = BF16 if dtype == "bf16" else F16 */
    elem_ir = (strcmp(spec->dtype, "bf16") == 0) ? rocke_bf16() : rocke_f16();

    /* ---- kernel params -- */
    {
        rocke_param_opts_t opts;
        const rocke_type_t* ptr_elem = rocke_ptr_type(b, elem_ir, "global");

        /* A = b.param("A", PtrType(elem_ir,"global"), noalias, readonly, align16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        A = rocke_b_param(b, "A", ptr_elem, &opts);
        Bp = rocke_b_param(b, "B", ptr_elem, &opts);

        /* C = b.param("C", ..., noalias, writeonly, align16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        C = rocke_b_param(b, "C", ptr_elem, &opts);

        /* M / N / K : i32 (ABI; unused after declare) */
        (void)rocke_b_param(b, "M", rocke_i32(), NULL);
        (void)rocke_b_param(b, "N", rocke_i32(), NULL);
        (void)rocke_b_param(b, "K", rocke_i32(), NULL);
    }

    /* lane = b.thread_id_x(); bid_n = b.block_id_x(); bid_m = b.block_id_y() */
    lane = rocke_b_thread_id_x(b);
    bid_n = rocke_b_block_id_x(b);
    bid_m = rocke_b_block_id_y(b);

    /* m_tile_base = bid_m * atom.m; n_tile_base = bid_n * atom.n */
    m_tile_base = rocke_b_mul(b, bid_m, rocke_b_const_i32(b, atom->m));
    n_tile_base = rocke_b_mul(b, bid_n, rocke_b_const_i32(b, atom->n));

    /* lane_decode = decode_mfma_lanes(b, atom, lane) */
    lane_decode = rocke_decode_mfma_lanes(b, atom, lane);

    /* c_atom_k = b.const_i32(atom.k) */
    c_atom_k = rocke_b_const_i32(b, atom->k);

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
    acc_final = rocke_mfma_k_loop(b,
                                  spec->K,
                                  atom,
                                  rocke_mfma_gemm_load_a_cb,
                                  rocke_mfma_gemm_load_b_cb,
                                  NULL, /* per_tile_post_mfma */
                                  NULL, /* initial_acc */
                                  NULL, /* iv_name => "kt" */
                                  NULL, /* acc_name => "acc" */
                                  &lctx);

    /* store_acc_to_global(b, C=C, atom=atom, lane_decode=lane_decode,
     *                     m_tile_base, n_tile_base, acc=acc_final, N=spec.N,
     *                     out_dtype=spec.dtype) */
    (void)rocke_store_acc_to_global(b,
                                    C,
                                    atom,
                                    &lane_decode,
                                    m_tile_base,
                                    n_tile_base,
                                    acc_final,
                                    spec->N,
                                    spec->dtype,
                                    false, /* atomic_add */
                                    NULL, /* epilogue */
                                    NULL); /* epilogue_user */

    /* b.ret() */
    rocke_b_ret(b);

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

/* ===================================================================== *
 *  rocke_build_mfma_gemm_new -- init builder with spec.kernel_name() then build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_mfma_gemm_new(rocke_ir_builder_t* b,
                                              const rocke_mfma_gemm_spec_t* spec,
                                              const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_mfma_gemm_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_mfma_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  mfma_gemm_grid(spec) -> (N // atom.n, M // atom.m, 1)
 * ===================================================================== */
rocke_status_t rocke_mfma_gemm_grid(const rocke_mfma_gemm_spec_t* spec, int out[3])
{
    const rocke_mfma_atom_t* atom;
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    atom = rocke_mfma_gemm_atom(spec);
    if(atom == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = spec->N / atom->n;
    out[1] = spec->M / atom->m;
    out[2] = 1;
    return ROCKE_OK;
}

/* ===================================================================== *
 *  rocke_mfma_gemm_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
rocke_status_t rocke_mfma_gemm_lower_to_llvm(const rocke_mfma_gemm_spec_t* spec,
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

    kernel = rocke_build_mfma_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_mfma_gemm failed";
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
