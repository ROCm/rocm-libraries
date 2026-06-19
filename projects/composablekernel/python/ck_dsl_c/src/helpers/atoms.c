/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C99 port of mfma_atom / c_warp_params / make_c_warp_dstr_encoding from
 * ck_dsl/helpers/atoms.py. See helper_ck_dsl.helpers.atoms.h for the contract.
 *
 * None of the three ported symbols emit IR (no ckc_b_* op calls), so the
 * byte-identical op-sequence requirement is met by emitting nothing; fidelity
 * is on the returned struct/encoding values, reproduced field-for-field below.
 */

#include "ckc/helper_ck_dsl.helpers.atoms.h"

#include <string.h>

#include "ckc/ir_internal.h" /* ckc_i_set_err */

/* ------------------------------------------------------------------ catalog *
 *
 * Static reproduction of the MfmaAtom factory class-methods, in the EXACT order
 * of the Python MFMA_ATOMS tuple:
 *
 *     MFMA_ATOMS = MFMA_F16_ATOMS + MFMA_BF16_ATOMS + MFMA_FP8_ATOMS
 *                  + MFMA_MX_ATOMS
 *
 *   MFMA_F16_ATOMS  = (f16_4x4x4, f16_16x16x16, f16_16x16x32,
 *                      f16_32x32x8, f16_32x32x16)
 *   MFMA_BF16_ATOMS = (bf16_16x16x16, bf16_16x16x32, bf16_32x32x16)
 *   MFMA_FP8_ATOMS  = (fp8_16x16x32, fp8_32x32x16, bf8_16x16x32,
 *                      bf8_32x32x16, fp8_16x16x128)
 *   MFMA_MX_ATOMS   = (fp4_16x16x128, fp6_16x16x96)
 *
 * _BY_SHAPE keys each by (dtype_in, m, n, k). All keys are unique, so the linear
 * scan below is order-independent for correctness, but the catalog order is
 * preserved anyway to match the Python (the only observable order effect, the
 * "valid:" list in the ValueError, is not reproduced verbatim -- see notes).
 *
 * Fields per row: m, n, k, a_per_lane, b_per_lane, c_per_lane,
 *                 dtype_in, dtype_out, name.
 */
static const ckc_mfma_atom_t CKC_MFMA_ATOMS[] = {
    /* ---- MFMA_F16_ATOMS ---- */
    {4, 4, 4, 4, 4, 4, "f16", "f32", "mfma_f32_4x4x4_f16"},
    {16, 16, 16, 4, 4, 4, "f16", "f32", "mfma_f32_16x16x16_f16"},
    {16, 16, 32, 8, 8, 4, "f16", "f32", "mfma_f32_16x16x32_f16"},
    {32, 32, 8, 4, 4, 16, "f16", "f32", "mfma_f32_32x32x8_f16"},
    {32, 32, 16, 8, 8, 16, "f16", "f32", "mfma_f32_32x32x16_f16"},
    /* ---- MFMA_BF16_ATOMS ---- */
    {16, 16, 16, 4, 4, 4, "bf16", "f32", "mfma_f32_16x16x16_bf16"},
    {16, 16, 32, 8, 8, 4, "bf16", "f32", "mfma_f32_16x16x32_bf16"},
    {32, 32, 16, 8, 8, 16, "bf16", "f32", "mfma_f32_32x32x16_bf16"},
    /* ---- MFMA_FP8_ATOMS ---- */
    {16, 16, 32, 8, 8, 4, "fp8e4m3", "f32", "mfma_f32_16x16x32_fp8"},
    {32, 32, 16, 8, 8, 16, "fp8e4m3", "f32", "mfma_f32_32x32x16_fp8"},
    {16, 16, 32, 8, 8, 4, "bf8e5m2", "f32", "mfma_f32_16x16x32_bf8"},
    {32, 32, 16, 8, 8, 16, "bf8e5m2", "f32", "mfma_f32_32x32x16_bf8"},
    {16, 16, 128, 32, 32, 4, "fp8e4m3", "f32", "mfma_f32_16x16x128_fp8"},
    /* ---- MFMA_MX_ATOMS ---- */
    {16, 16, 128, 16, 16, 4, "fp4", "f32", "mfma_f32_16x16x128_fp4"},
    {16, 16, 96, 12, 12, 4, "fp6", "f32", "mfma_f32_16x16x96_fp6"},
};

#define CKC_NUM_MFMA_ATOMS \
    ((int)(sizeof(CKC_MFMA_ATOMS) / sizeof(CKC_MFMA_ATOMS[0])))

/* _DTYPE_ALIAS, reproduced exactly:
 *     fp8 -> fp8e4m3, fp8e4m3 -> fp8e4m3,
 *     bf8 -> bf8e5m2, bf8e5m2 -> bf8e5m2,
 *     f16 -> f16, fp16 -> f16,
 *     bf16 -> bf16, bfloat16 -> bf16.
 * Any key not in the table maps to itself (Python dict.get(dtype, dtype)). */
static const char* ckc_canon_dtype(const char* dtype)
{
    static const struct
    {
        const char* key;
        const char* val;
    } aliases[] = {
        {"fp8", "fp8e4m3"},
        {"fp8e4m3", "fp8e4m3"},
        {"bf8", "bf8e5m2"},
        {"bf8e5m2", "bf8e5m2"},
        {"f16", "f16"},
        {"fp16", "f16"},
        {"bf16", "bf16"},
        {"bfloat16", "bf16"},
    };
    int i;
    int n = (int)(sizeof(aliases) / sizeof(aliases[0]));
    if (dtype == NULL)
    {
        return NULL;
    }
    for (i = 0; i < n; ++i)
    {
        if (strcmp(dtype, aliases[i].key) == 0)
        {
            return aliases[i].val;
        }
    }
    return dtype; /* identity fallthrough */
}

/* ------------------------------------------------------------------ mfma_atom */

const ckc_mfma_atom_t* ckc_mfma_atom(const char* dtype, int m, int n, int k)
{
    const char* canon = ckc_canon_dtype(dtype);
    int i;
    if (canon == NULL)
    {
        return NULL;
    }
    for (i = 0; i < CKC_NUM_MFMA_ATOMS; ++i)
    {
        const ckc_mfma_atom_t* a = &CKC_MFMA_ATOMS[i];
        if (a->m == m && a->n == n && a->k == k &&
            strcmp(a->dtype_in, canon) == 0)
        {
            return a;
        }
    }
    return NULL; /* Python ValueError path */
}

/* ------------------------------------------------------------------ accessors *
 *
 * Faithful ports of the MfmaAtom field reads / @property bodies consumed by the
 * schedule helper's from_geometry (ck_dsl/helpers/atoms.py:429-465).
 */
const char* ckc_mfma_atom_dtype_in(const ckc_mfma_atom_t* atom)
{
    return atom->dtype_in;
}

int ckc_mfma_atom_m(const ckc_mfma_atom_t* atom)
{
    return atom->m;
}

int ckc_mfma_atom_n(const ckc_mfma_atom_t* atom)
{
    return atom->n;
}

/* @property k_per_xdlops -> self.k (CK KPerXdlops == atom's full per-inst K). */
int ckc_mfma_atom_k_per_xdlops(const ckc_mfma_atom_t* atom)
{
    return atom->k;
}

/* @property is_f4f6 -> self.dtype_in in ("fp4", "fp6"). */
bool ckc_mfma_atom_is_f4f6(const ckc_mfma_atom_t* atom)
{
    return strcmp(atom->dtype_in, "fp4") == 0 ||
           strcmp(atom->dtype_in, "fp6") == 0;
}

/* @property mfma_cycle -> C_MFMA_Inst_Cycle (blkgemmpipe_scheduler.hpp:63-74).
 *   speedup = 2 if is_f4f6 else 1
 *   n == 16: (32 if k == 128 else 16) // speedup
 *   n == 32: (64 if k ==  64 else 32) // speedup
 *   else   : NotImplementedError -> reported here as -1 (out-of-table sentinel).
 */
int ckc_mfma_atom_mfma_cycle(const ckc_mfma_atom_t* atom)
{
    int speedup = ckc_mfma_atom_is_f4f6(atom) ? 2 : 1;
    if (atom->n == 16)
    {
        return (atom->k == 128 ? 32 : 16) / speedup;
    }
    if (atom->n == 32)
    {
        return (atom->k == 64 ? 64 : 32) / speedup;
    }
    return -1; /* Python NotImplementedError path (no 16x16/32x32 XDL shape). */
}

const ckc_mfma_atom_t*
ckc_b_mfma_atom(ckc_ir_builder_t* b, const char* dtype, int m, int n, int k)
{
    const ckc_mfma_atom_t* atom;
    if (b != NULL && b->status != CKC_OK)
    {
        return NULL; /* already in error: no-op, like every other ckc_b_* */
    }
    atom = ckc_mfma_atom(dtype, m, n, k);
    if (atom == NULL)
    {
        /* Mirrors: raise ValueError(f"no MFMA atom for {key}; valid: {valid}").
         * The verbatim sorted "valid:" suffix is not reproduced (it carries no
         * IR/value semantics); the (key) prefix is. */
        return (const ckc_mfma_atom_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "no MFMA atom for ('%s', %d, %d, %d)",
            dtype != NULL ? dtype : "(null)", m, n, k);
    }
    return atom;
}

/* ------------------------------------------------------------------ c_warp_params *
 *
 * _C_WARP_PARAMS = { (16,16): (1,4,4,16), (32,32): (4,2,4,32) }
 *   tuple order is (kCM0PerLane, kCMLane, kCM1PerLane, kCNLane).
 */
static bool ckc_c_warp_params_lookup(int m,
                                     int n,
                                     int* m0,
                                     int* m_lane,
                                     int* m1,
                                     int* n_lane)
{
    if (m == 16 && n == 16)
    {
        *m0 = 1;
        *m_lane = 4;
        *m1 = 4;
        *n_lane = 16;
        return true;
    }
    if (m == 32 && n == 32)
    {
        *m0 = 4;
        *m_lane = 2;
        *m1 = 4;
        *n_lane = 32;
        return true;
    }
    return false;
}

ckc_status_t ckc_c_warp_params(const ckc_mfma_atom_t* atom,
                               int* out_kCM0PerLane,
                               int* out_kCMLane,
                               int* out_kCM1PerLane,
                               int* out_kCNLane)
{
    int m0, m_lane, m1, n_lane;
    if (atom == NULL)
    {
        return CKC_ERR_VALUE;
    }
    if (!ckc_c_warp_params_lookup(atom->m, atom->n, &m0, &m_lane, &m1, &n_lane))
    {
        return CKC_ERR_NOTIMPL; /* raise NotImplementedError */
    }
    if (m0 * m1 != atom->c_per_lane)
    {
        return CKC_ERR_VALUE; /* raise ValueError */
    }
    if (out_kCM0PerLane != NULL)
    {
        *out_kCM0PerLane = m0;
    }
    if (out_kCMLane != NULL)
    {
        *out_kCMLane = m_lane;
    }
    if (out_kCM1PerLane != NULL)
    {
        *out_kCM1PerLane = m1;
    }
    if (out_kCNLane != NULL)
    {
        *out_kCNLane = n_lane;
    }
    return CKC_OK;
}

ckc_status_t ckc_b_c_warp_params(ckc_ir_builder_t* b,
                                 const ckc_mfma_atom_t* atom,
                                 int* out_kCM0PerLane,
                                 int* out_kCMLane,
                                 int* out_kCM1PerLane,
                                 int* out_kCNLane)
{
    int m0, m_lane, m1, n_lane;
    if (b != NULL && b->status != CKC_OK)
    {
        return b->status; /* already in error: no-op */
    }
    if (atom == NULL)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "c_warp_params: NULL atom");
        return CKC_ERR_VALUE;
    }
    if (!ckc_c_warp_params_lookup(atom->m, atom->n, &m0, &m_lane, &m1, &n_lane))
    {
        ckc_i_set_err(
            b, CKC_ERR_NOTIMPL,
            "no CWarpDstrEncoding for atom %dx%d "
            "(only the 16x16 and 32x32 MFMA C tiles are supported)",
            atom->m, atom->n);
        return CKC_ERR_NOTIMPL;
    }
    if (m0 * m1 != atom->c_per_lane)
    {
        ckc_i_set_err(
            b, CKC_ERR_VALUE,
            "atom %s: kCM0PerLane*kCM1PerLane (%d) != c_per_lane (%d)",
            atom->name != NULL ? atom->name : "(null)", m0 * m1,
            atom->c_per_lane);
        return CKC_ERR_VALUE;
    }
    if (out_kCM0PerLane != NULL)
    {
        *out_kCM0PerLane = m0;
    }
    if (out_kCMLane != NULL)
    {
        *out_kCMLane = m_lane;
    }
    if (out_kCM1PerLane != NULL)
    {
        *out_kCM1PerLane = m1;
    }
    if (out_kCNLane != NULL)
    {
        *out_kCNLane = n_lane;
    }
    return CKC_OK;
}

/* ----------------------------------------------- make_c_warp_dstr_encoding *
 *
 * Python:
 *     m0, m_lane, m1, n_lane = c_warp_params(atom)
 *     return TileDistributionEncoding(
 *         Rs=(),
 *         Hs=((m0, m_lane, m1), (n_lane,)),
 *         Ps2RHs_major=((1, 2),),
 *         Ps2RHs_minor=((1, 0),),
 *         Ys2RHs_major=(1, 1),
 *         Ys2RHs_minor=(0, 2),
 *     )
 *
 * Fed verbatim into ckc_make_tile_distribution_encoding (which runs the
 * encoding __post_init__ validation and arena-owns the node).
 */
ckc_tile_distribution_encoding_t*
ckc_make_c_warp_dstr_encoding(ckc_ir_builder_t* b, const ckc_mfma_atom_t* atom)
{
    int m0, m_lane, m1, n_lane;
    ckc_status_t st;

    /* H rows: Hs[0] = (m0, m_lane, m1), Hs[1] = (n_lane,).               */
    int h0_levels[3];
    int h1_levels[1];
    ckc_h_row_t Hs[2];

    /* Single P dim: Ps2RHs_major=((1,2),), Ps2RHs_minor=((1,0),).        */
    int p0_major[2];
    int p0_minor[2];
    ckc_p_seq_t Ps[1];

    /* Two Y dims: Ys2RHs_major=(1,1), Ys2RHs_minor=(0,2).                */
    int Ys_major[2];
    int Ys_minor[2];

    if (b != NULL && b->status != CKC_OK)
    {
        return NULL; /* already in error: no-op */
    }

    /* c_warp_params(atom) -- propagates NotImplementedError/ValueError via
     * the builder sticky error, matching the Python raise. */
    st = ckc_b_c_warp_params(b, atom, &m0, &m_lane, &m1, &n_lane);
    if (st != CKC_OK)
    {
        return NULL;
    }

    h0_levels[0] = m0;
    h0_levels[1] = m_lane;
    h0_levels[2] = m1;
    h1_levels[0] = n_lane;
    Hs[0].levels = h0_levels;
    Hs[0].count = 3;
    Hs[1].levels = h1_levels;
    Hs[1].count = 1;

    p0_major[0] = 1;
    p0_major[1] = 2;
    p0_minor[0] = 1;
    p0_minor[1] = 0;
    Ps[0].major = p0_major;
    Ps[0].minor = p0_minor;
    Ps[0].count = 2;

    Ys_major[0] = 1;
    Ys_major[1] = 1;
    Ys_minor[0] = 0;
    Ys_minor[1] = 2;

    /* Rs=() -> num_R==0, Rs==NULL. The constructor copies all arrays into the
     * builder arena, so the stack locals above are safe. */
    return ckc_make_tile_distribution_encoding(b,
                                               /* Rs    */ NULL,
                                               /* num_R */ 0,
                                               /* Hs    */ Hs,
                                               /* num_X */ 2,
                                               /* Ps    */ Ps,
                                               /* num_P */ 1,
                                               /* Ys_major */ Ys_major,
                                               /* Ys_minor */ Ys_minor,
                                               /* num_Y */ 2);
}
