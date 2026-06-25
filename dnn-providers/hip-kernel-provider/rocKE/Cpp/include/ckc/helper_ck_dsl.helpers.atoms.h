/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.helpers.atoms.h -- C99 port of three symbols from
 * ck_dsl/helpers/atoms.py:
 *
 *   Python                              C99 (this header)
 *   ---------------------------------   ---------------------------------------
 *   mfma_atom(dtype, m, n, k)           ckc_mfma_atom(...) / ckc_b_mfma_atom(...)
 *   c_warp_params(atom)                 ckc_c_warp_params(...)
 *   make_c_warp_dstr_encoding(atom)     ckc_make_c_warp_dstr_encoding(...)
 *
 * SCOPE: only these three symbols (plus the MfmaAtom value type they return /
 * consume). The MfmaAtom factory class-methods, the WMMA family, emit/zero_acc/
 * lane_to_output, and the timing-trait properties are NOT in scope for this
 * phase -- but the static MFMA_ATOMS catalog is reproduced internally because
 * mfma_atom() is a lookup over it.
 *
 * NONE of these three call the IR builder (ckc_b_*):
 *
 *   - mfma_atom is a pure (dtype, m, n, k) -> MfmaAtom catalog lookup.
 *   - c_warp_params is a pure (m, n) -> (kCM0PerLane, kCMLane, kCM1PerLane,
 *     kCNLane) table lookup + a c_per_lane consistency check.
 *   - make_c_warp_dstr_encoding constructs a TileDistributionEncoding (no IR);
 *     it only needs the builder for arena ownership of the encoding node, which
 *     is the same lifetime contract the distribution helper already uses.
 *
 * Because the op sequence emitted by all three is empty, the "byte-identical op
 * sequence" requirement is trivially met; fidelity is on the returned struct /
 * encoding values, which are reproduced field-for-field from the Python.
 *
 * Error model mirrors the rest of the C port: a sticky-error builder spelling
 * (ckc_b_*) stands in for `raise`, and a builder-free pure spelling returns a
 * status / NULL for call sites that have no builder. mfma_atom's
 * `raise ValueError`, c_warp_params' `raise NotImplementedError` / `ValueError`,
 * and make_c_warp_dstr_encoding's `raise NotImplementedError` all map onto
 * ckc_status_t codes (CKC_ERR_VALUE / CKC_ERR_NOTIMPL).
 */
#ifndef CKC_HELPER_CK_DSL_HELPERS_ATOMS_H
#define CKC_HELPER_CK_DSL_HELPERS_ATOMS_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/helper_ck_dsl.helpers.distribution.h" /* ckc_tile_distribution_encoding_t */
#include "ckc/ir.h" /* ckc_status_t, ckc_ir_builder_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ MfmaAtom *
 *
 * Value type mirroring ck_dsl.helpers.atoms.MfmaAtom (frozen dataclass). It is
 * a plain value struct (no arena ownership) -- the catalog entries are static
 * and `ckc_mfma_atom` returns a pointer into that immutable static table, so
 * callers must NOT free or mutate it. The `name` and `dtype_in`/`dtype_out`
 * strings are string literals with static storage duration.
 *
 * Fields are 1:1 with the Python dataclass declaration order:
 *   m, n, k, a_per_lane, b_per_lane, c_per_lane, dtype_in, dtype_out, name.
 */
typedef struct ckc_mfma_atom
{
    int m;
    int n;
    int k;
    int a_per_lane;
    int b_per_lane;
    int c_per_lane;
    const char* dtype_in; /* e.g. "f16", "bf16", "fp8e4m3", "bf8e5m2", ... */
    const char* dtype_out; /* always "f32" for the shipped atoms            */
    const char* name; /* backend op_id, e.g. "mfma_f32_16x16x16_f16"   */
} ckc_mfma_atom_t;

/* ------------------------------------------------------------------ accessors *
 *
 * MfmaAtom field / @property accessors used by the schedule helper's
 * from_geometry (ck_dsl/helpers/schedule.py). These are the canonical
 * definitions that satisfy the forward-declared contract in
 * helper_ck_dsl.helpers.schedule.h (those forward decls are suppressed when this
 * header is included first, via CKC_HELPER_CK_DSL_HELPERS_ATOMS_H). Faithful
 * ports of the Python:
 *
 *   dtype_in     -> self.dtype_in
 *   m / n        -> self.m / self.n
 *   k_per_xdlops -> self.k
 *   is_f4f6      -> self.dtype_in in ("fp4", "fp6")
 *   mfma_cycle   -> C_MFMA_Inst_Cycle table (blkgemmpipe_scheduler.hpp:63-74);
 *                   returns -1 for shapes outside NPerXDL in {16, 32} (the
 *                   Python NotImplementedError path).
 */
const char* ckc_mfma_atom_dtype_in(const ckc_mfma_atom_t* atom);
int ckc_mfma_atom_m(const ckc_mfma_atom_t* atom);
int ckc_mfma_atom_n(const ckc_mfma_atom_t* atom);
int ckc_mfma_atom_k_per_xdlops(const ckc_mfma_atom_t* atom);
int ckc_mfma_atom_mfma_cycle(const ckc_mfma_atom_t* atom);
bool ckc_mfma_atom_is_f4f6(const ckc_mfma_atom_t* atom);

/* ------------------------------------------------------------------ mfma_atom *
 *
 * Python:
 *
 *     def mfma_atom(dtype, m, n, k) -> MfmaAtom:
 *         canon = _DTYPE_ALIAS.get(dtype, dtype)
 *         key = (canon, m, n, k)
 *         if key not in _BY_SHAPE: raise ValueError(...)
 *         return _BY_SHAPE[key]
 *
 * Lookup an atom by (dtype_in, m, n, k) over the static MFMA_ATOMS catalog,
 * after canonicalising the dtype spelling through _DTYPE_ALIAS
 * (fp8->fp8e4m3, bf8->bf8e5m2, fp16->f16, bfloat16->bf16, identity otherwise).
 *
 * Pure spelling: returns a pointer to the static catalog entry, or NULL if no
 * atom matches (the Python ValueError path). No builder, no error state. */
const ckc_mfma_atom_t* ckc_mfma_atom(const char* dtype, int m, int n, int k);

/* Builder-aware variant: same lookup, but on a miss it records CKC_ERR_VALUE +
 * a Python-matching message on the builder and returns NULL. No-op returning
 * NULL if the builder is already in an error state. */
const ckc_mfma_atom_t* ckc_b_mfma_atom(ckc_ir_builder_t* b, const char* dtype, int m, int n, int k);

/* ------------------------------------------------------------------ c_warp_params *
 *
 * Python:
 *
 *     def c_warp_params(atom) -> (kCM0PerLane, kCMLane, kCM1PerLane, kCNLane):
 *         key = (atom.m, atom.n)
 *         if key not in _C_WARP_PARAMS: raise NotImplementedError(...)
 *         m0, m_lane, m1, n_lane = _C_WARP_PARAMS[key]
 *         if m0 * m1 != atom.c_per_lane: raise ValueError(...)
 *         return m0, m_lane, m1, n_lane
 *
 * Table:
 *     (16, 16): (1, 4, 4, 16)
 *     (32, 32): (4, 2, 4, 32)
 *
 * On success writes the four constants to the out-params (any may be NULL to
 * skip) and returns CKC_OK. Returns CKC_ERR_NOTIMPL if (atom->m, atom->n) is not
 * in the table (the batched 4x4 atom and every non-square shape), and
 * CKC_ERR_VALUE if kCM0PerLane*kCM1PerLane != atom->c_per_lane. On any error the
 * out-params are left untouched. `atom` must be non-NULL. */
ckc_status_t ckc_c_warp_params(const ckc_mfma_atom_t* atom,
                               int* out_kCM0PerLane,
                               int* out_kCMLane,
                               int* out_kCM1PerLane,
                               int* out_kCNLane);

/* Builder-aware variant: identical computation; on the NotImplementedError /
 * ValueError paths it sets the builder sticky error (CKC_ERR_NOTIMPL /
 * CKC_ERR_VALUE) with a Python-matching message and returns that status. */
ckc_status_t ckc_b_c_warp_params(ckc_ir_builder_t* b,
                                 const ckc_mfma_atom_t* atom,
                                 int* out_kCM0PerLane,
                                 int* out_kCMLane,
                                 int* out_kCM1PerLane,
                                 int* out_kCNLane);

/* ------------------------------------------------ make_c_warp_dstr_encoding *
 *
 * Python:
 *
 *     def make_c_warp_dstr_encoding(atom) -> TileDistributionEncoding:
 *         m0, m_lane, m1, n_lane = c_warp_params(atom)
 *         return TileDistributionEncoding(
 *             Rs=(),
 *             Hs=((m0, m_lane, m1), (n_lane,)),
 *             Ps2RHs_major=((1, 2),),
 *             Ps2RHs_minor=((1, 0),),
 *             Ys2RHs_major=(1, 1),
 *             Ys2RHs_minor=(0, 2),
 *         )
 *
 * Builds the MFMA C-tile TileDistributionEncoding for `atom` by feeding the
 * four c_warp_params constants into ckc_make_tile_distribution_encoding (which
 * runs the encoding's __post_init__ validation and arena-owns the node).
 *
 * Unlike the Python (which takes no builder) this needs `b` for arena
 * ownership of the returned encoding -- the same lifetime contract as
 * ckc_make_tile_distribution_encoding. Returns the arena-owned encoding, or
 * NULL with the builder sticky error set if c_warp_params fails (the
 * NotImplementedError/ValueError path) or the encoding constructor fails. */
ckc_tile_distribution_encoding_t* ckc_make_c_warp_dstr_encoding(ckc_ir_builder_t* b,
                                                                const ckc_mfma_atom_t* atom);

#ifdef __cplusplus
}
#endif

#endif /* CKC_HELPER_CK_DSL_HELPERS_ATOMS_H */
