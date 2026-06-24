/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.helpers.mfma_gemm_inner.h -- C99 port of selected symbols
 * from ck_dsl/helpers/mfma_gemm_inner.py (the universal MFMA-tiled K-loop helper
 * for GEMM-shaped kernels).
 *
 * SCOPE OF THIS PORT (this phase) -- exactly these nine Python symbols:
 *
 *   Python                                  C99 (this header)
 *   --------------------------------------  -------------------------------------
 *   class LaneDecode                        ckc_lane_decode_t
 *   decode_mfma_lanes(b, atom, lane)        ckc_decode_mfma_lanes(...)
 *   load_a_row_major_contiguous(...)        ckc_load_a_row_major_contiguous(...)
 *   load_b_col_strided_scalars(...)         ckc_load_b_col_strided_scalars(...)
 *   mfma_atom_for_dtype(dtype, m, n, ...)   ckc_mfma_atom_for_dtype(...)
 *   mfma_k_loop(...)                        ckc_mfma_k_loop(...)
 *   store_acc_to_global(...)                ckc_store_acc_to_global(...)
 *   validate_arch_and_block_size(...)       ckc_validate_arch_and_block_size(...)
 *   validate_mfma_atom_in_catalog(...)      ckc_validate_mfma_atom_in_catalog(...)
 *
 * NOT in scope here (left in Python only this phase):
 *   load_smem_frag_contiguous_f16, mfma_k_loop_dynamic_K.
 *
 * BINDINGS.
 *   - The IR builder primitives are ckc/ir.h's ckc_b_* entry points (const_i32,
 *     mod/div/mul/add, global_load[_vN], global_store, global_atomic_add,
 *     zero_vec[_f32], vec_insert/vec_extract, scf_for_iter/scf_yield, mma,
 *     cast_f32_to, and the f16/bf16/fp8e4m3/bf8e5m2/f32 type singletons).
 *   - The MfmaAtom value type is ckc/helper_ck_dsl.helpers.atoms.h's
 *     ckc_mfma_atom_t (fields m, n, k, a_per_lane, b_per_lane, c_per_lane,
 *     dtype_in, dtype_out, name). mfma_atom_for_dtype resolves atoms by
 *     (dtype, m, n, k) over that catalog via ckc_mfma_atom().
 *   - validate_arch_and_block_size / validate_mfma_atom_in_catalog bind to
 *     ckc/helper_ck_dsl.core.arch.h (ArchTarget.from_gfx, max_threads_per_block,
 *     target.mma.has_shape).
 *
 * ATOM METHODS REPRODUCED INLINE (not separate symbols). The Python helper calls
 * three MfmaAtom methods that the atoms.h port does not expose (they are out of
 * scope there): atom.emit(), atom.zero_acc(), atom.lane_to_output(). Because each
 * is a tiny pure builder sequence over fields that ARE on ckc_mfma_atom_t, the
 * faithful port reproduces them inline inside ckc_mfma_k_loop /
 * ckc_store_acc_to_global, byte-for-byte:
 *     emit          -> ckc_b_mma(b, atom->name, a, b, c)
 *     zero_acc      -> ckc_b_zero_vec_f32(b, atom->c_per_lane)
 *     lane_to_output-> the 16x16 / 32x32 / 4x4 arith from atoms.py:536-591
 *
 * CALLBACKS. The Python `load_a`, `load_b`, `per_tile_post_mfma`, and `epilogue`
 * closures become explicit C function pointers carrying an opaque `user` pointer
 * (the C analog of a closure's captured environment). Signatures mirror the
 * Python call shapes one-for-one.
 *
 * ERROR MODEL. Mirrors the rest of the C port: the sticky-error builder (ckc_b_*)
 * stands in for `raise`. mfma_k_loop's `raise ValueError` (K % atom.k != 0),
 * store_acc_to_global's `raise ValueError` (atomic_add with non-f32), and the two
 * validate_* `raise`/return paths map onto ckc_status_t / sticky-error spellings.
 */
#ifndef CKC_HELPER_CK_DSL_HELPERS_MFMA_GEMM_INNER_H
#define CKC_HELPER_CK_DSL_HELPERS_MFMA_GEMM_INNER_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"                          /* ckc_ir_builder_t, ckc_value_t, ckc_status_t */
#include "ckc/helper_ck_dsl.helpers.atoms.h" /* ckc_mfma_atom_t */
#include "ckc/helper_ck_dsl.core.arch.h"     /* ckc_archtarget_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- LaneDecode *
 *
 * Python:
 *
 *     @dataclass(frozen=True)
 *     class LaneDecode:
 *         lane: Value
 *         m_in_atom: Value
 *         n_in_atom: Value
 *         k_blk: Value
 *
 * Per-lane MFMA operand coordinates for the canonical square atoms
 * (16x16x* / 32x32x*). Plain value struct -- the four SSA values are arena-owned
 * by the builder, this struct just bundles pointers to them. */
typedef struct ckc_lane_decode
{
    ckc_value_t* lane;
    ckc_value_t* m_in_atom;
    ckc_value_t* n_in_atom;
    ckc_value_t* k_blk;
} ckc_lane_decode_t;

/* ------------------------------------------------------- decode_mfma_lanes *
 *
 * Python:
 *
 *     def decode_mfma_lanes(b, atom, lane) -> LaneDecode:
 *         c_m = b.const_i32(atom.m)
 *         c_n = b.const_i32(atom.n)
 *         m_in_atom = b.mod(lane, c_m)
 *         n_in_atom = b.mod(lane, c_n)
 *         k_blk     = b.div(lane, c_m)
 *         return LaneDecode(lane, m_in_atom, n_in_atom, k_blk)
 *
 * Decompose a wave64 lane id into (m_in_atom, n_in_atom, k_blk). The returned
 * struct's fields point at fresh builder SSA values. On a dead builder every
 * field is NULL. `atom` must be non-NULL. */
ckc_lane_decode_t
ckc_decode_mfma_lanes(ckc_ir_builder_t* b, const ckc_mfma_atom_t* atom, ckc_value_t* lane);

/* ------------------------------------------------------- mfma_atom_for_dtype *
 *
 * Python:
 *
 *     def mfma_atom_for_dtype(dtype_in, m=16, n=16, *, prefer_packed_k=True):
 *         ... -> MfmaAtom  (raises ValueError on an unsupported combo)
 *
 * Pick the right atom for an in-dtype and (m, n) tile shape. The packed-K choice
 * (default) selects atom.k=32 for f16/bf16/fp8/bf8 at (16,16) and atom.k=16 at
 * (32,32); prefer_packed_k=false falls back to the legacy 16x16x16 / 32x32x8
 * f16 atoms (the non-packed bf16/fp8/bf8 shapes have no legacy fallback, exactly
 * as in Python -- those branches fall through to the ValueError).
 *
 * Returns a pointer into the static MFMA catalog (do NOT free/mutate), or NULL on
 * the Python ValueError path (unsupported dtype/shape). Pure spelling: no builder,
 * no error state. */
const ckc_mfma_atom_t*
ckc_mfma_atom_for_dtype(const char* dtype_in, int m, int n, bool prefer_packed_k);

/* Builder-aware variant: identical selection; on the ValueError path it records
 * CKC_ERR_VALUE + a Python-matching message on the builder and returns NULL.
 * No-op returning NULL if the builder is already in an error state. */
const ckc_mfma_atom_t* ckc_b_mfma_atom_for_dtype(
    ckc_ir_builder_t* b, const char* dtype_in, int m, int n, bool prefer_packed_k);

/* ---------------------------------------------- load_a_row_major_contiguous *
 *
 * Python:
 *
 *     def load_a_row_major_contiguous(b, *, A, atom, lane_decode, m_tile_base,
 *                                     k_tile_base, K) -> Value:
 *
 * Per-lane A load for row-major (M, K) layout. The K axis is contiguous so the
 * lane's a_per_lane values are at consecutive addresses; for f16/bf16 one
 * global_load_vN fills the operand, for fp8/bf8 a_per_lane scalar loads are
 * packed via zero_vec + vec_insert (no vec-load helper). Returns the per-lane
 * <a_per_lane x dtype> operand vector, or NULL on a dead builder / unsupported
 * dtype (atom->dtype_in not in f16/fp16/bf16/fp8e4m3/bf8e5m2). */
ckc_value_t* ckc_load_a_row_major_contiguous(ckc_ir_builder_t* b,
                                             ckc_value_t* A,
                                             const ckc_mfma_atom_t* atom,
                                             const ckc_lane_decode_t* lane_decode,
                                             ckc_value_t* m_tile_base,
                                             ckc_value_t* k_tile_base,
                                             int K);

/* ----------------------------------------------- load_b_col_strided_scalars *
 *
 * Python:
 *
 *     def load_b_col_strided_scalars(b, *, B, atom, lane_decode, n_tile_base,
 *                                    k_tile_base, N) -> Value:
 *
 * Per-lane B load for row-major (K, N) layout. Each K element of B is N apart, so
 * the b_per_lane values are not contiguous: b_per_lane scalar loads packed via
 * zero_vec + vec_insert. Load alignment is 2 for f16/bf16 else 1. Returns the
 * per-lane <b_per_lane x dtype> operand vector, or NULL on a dead builder /
 * unsupported dtype. */
ckc_value_t* ckc_load_b_col_strided_scalars(ckc_ir_builder_t* b,
                                            ckc_value_t* B,
                                            const ckc_mfma_atom_t* atom,
                                            const ckc_lane_decode_t* lane_decode,
                                            ckc_value_t* n_tile_base,
                                            ckc_value_t* k_tile_base,
                                            int N);

/* --------------------------------------------------------------- callbacks *
 *
 * The Python helper takes Python closures; the C port takes function pointers
 * plus an opaque `user` environment pointer.
 *
 *   load_a / load_b : Python `Callable[[IRBuilder, Value], Value]`
 *       Called once per K-tile with the loop induction value `kt`; returns the
 *       per-lane operand vector. (ckc_load_a_row_major_contiguous /
 *       ckc_load_b_col_strided_scalars are typically wrapped behind these.)
 *
 *   per_tile_post_mfma : Python
 *       `Callable[[IRBuilder, Value, Value], Value]` (b, acc, kt) -> acc
 *       Optional post-MFMA accumulator transform (per-group scale / bias). NULL
 *       => no post step (the Python `is not None` guard). */
typedef ckc_value_t* (*ckc_mfma_load_fn)(ckc_ir_builder_t* b, ckc_value_t* kt, void* user);
typedef ckc_value_t* (*ckc_mfma_post_fn)(ckc_ir_builder_t* b,
                                         ckc_value_t* acc,
                                         ckc_value_t* kt,
                                         void* user);

/* --------------------------------------------------------------- mfma_k_loop *
 *
 * Python:
 *
 *     def mfma_k_loop(b, *, K, atom, load_a, load_b, per_tile_post_mfma=None,
 *                     initial_acc=None, iv_name="kt", acc_name="acc") -> Value:
 *
 * Emit a scf.for K-loop of MFMA atoms over kt in [0, K/atom.k); per iteration:
 *   a = load_a(b, kt); b_op = load_b(b, kt);
 *   acc = atom.emit(a, b_op, acc)   [-> ckc_b_mma(name, ...)];
 *   acc = per_tile_post_mfma(b, acc, kt)   (if non-NULL);
 *   yield acc.
 * initial_acc NULL => atom.zero_acc(b) [-> ckc_b_zero_vec_f32(c_per_lane)].
 * Returns the final per-lane <c_per_lane x f32> accumulator (the for-op's first
 * result). iv_name/acc_name may be NULL (Python defaults "kt"/"acc").
 *
 * raise ValueError (K % atom.k != 0) -> builder sticky CKC_ERR_VALUE + NULL.
 * load_a/load_b are required (non-NULL); `user` is passed through to all three. */
ckc_value_t* ckc_mfma_k_loop(ckc_ir_builder_t* b,
                             int K,
                             const ckc_mfma_atom_t* atom,
                             ckc_mfma_load_fn load_a,
                             ckc_mfma_load_fn load_b,
                             ckc_mfma_post_fn per_tile_post_mfma,
                             ckc_value_t* initial_acc,
                             const char* iv_name,
                             const char* acc_name,
                             void* user);

/* -------------------------------------------------------- store epilogue cb *
 *
 * Python epilogue closure:
 *     Callable[[IRBuilder, MfmaAtom, LaneDecode, Value, Value, Value, Value,
 *               int, str], None]
 *     epilogue(b, atom, lane_decode, C, m_tile_base, n_tile_base, acc, N,
 *              out_dtype)
 * When supplied it owns the whole write-back and atomic_add is ignored. */
typedef void (*ckc_mfma_epilogue_fn)(ckc_ir_builder_t* b,
                                     const ckc_mfma_atom_t* atom,
                                     const ckc_lane_decode_t* lane_decode,
                                     ckc_value_t* C,
                                     ckc_value_t* m_tile_base,
                                     ckc_value_t* n_tile_base,
                                     ckc_value_t* acc,
                                     int N,
                                     const char* out_dtype,
                                     void* user);

/* ----------------------------------------------------------- store_acc_to_global *
 *
 * Python:
 *
 *     def store_acc_to_global(b, *, C, atom, lane_decode, m_tile_base,
 *                             n_tile_base, acc, N, out_dtype="f16",
 *                             atomic_add=False, epilogue=None) -> None:
 *
 * Write a per-lane MFMA accumulator to global C row-major. out_dtype NULL =>
 * "f16" (the Python default). out_dtype "f32" keeps the f32 accumulator; any
 * other value routes through cast_f32_to(_ir_type_for_dtype(out_dtype)).
 * atomic_add=true does global_atomic_add (requires out_dtype "f32"); else
 * global_store with align 4 (f32) / 2 (else). When `epilogue` is non-NULL it is
 * invoked instead and atomic_add is ignored.
 *
 * lane_to_output is reproduced inline per slot i (16x16 / 32x32 / 4x4 dispatch).
 *
 * Returns CKC_OK; on the Python raise paths (atomic_add with out_dtype != "f32",
 * or an unsupported out_dtype for the cast) the builder sticky error is set
 * (CKC_ERR_VALUE) and that status is returned. `epilogue_user` is passed to the
 * epilogue callback. */
ckc_status_t ckc_store_acc_to_global(ckc_ir_builder_t* b,
                                     ckc_value_t* C,
                                     const ckc_mfma_atom_t* atom,
                                     const ckc_lane_decode_t* lane_decode,
                                     ckc_value_t* m_tile_base,
                                     ckc_value_t* n_tile_base,
                                     ckc_value_t* acc,
                                     int N,
                                     const char* out_dtype,
                                     bool atomic_add,
                                     ckc_mfma_epilogue_fn epilogue,
                                     void* epilogue_user);

/* -------------------------------------------------- validate_arch_and_block_size *
 *
 * Python:
 *
 *     def validate_arch_and_block_size(arch, block_size) -> (ok, reason, target):
 *         try: target = ArchTarget.from_gfx(arch)
 *         except KeyError as e: return False, str(e), None
 *         if block_size > target.max_threads_per_block:
 *             return False, "block_size {bs} > {cap} (hardware cap) on {arch}",
 *                    target
 *         return True, "ok", target
 *
 * Shared is_valid_spec prologue for MFMA scaled-GEMM kernels. The returned
 * strings surface only through ValueError messages (never into IR), so adopting
 * this is byte-identical for emitted code.
 *
 * C spelling: returns the bool `ok`. *out_target receives the resolved target
 * (NULL on the unknown-gfx path, matching Python). *out_reason (if non-NULL)
 * receives a pointer to the reason string: a static "ok" on success, a static
 * KeyError-style "unknown gfx target ..." string on the from_gfx miss, or a
 * builder-arena-owned formatted "block_size ..." string on the cap miss. Any of
 * out_reason / out_target may be NULL to skip. `b` is used only for arena
 * ownership of the cap-miss reason string; pass a live builder. */
bool ckc_validate_arch_and_block_size(ckc_ir_builder_t* b,
                                      const char* arch,
                                      int block_size,
                                      const char** out_reason,
                                      const ckc_archtarget_t** out_target);

/* ----------------------------------------------- validate_mfma_atom_in_catalog *
 *
 * Python:
 *
 *     def validate_mfma_atom_in_catalog(atom, arch, *, where) -> None:
 *         target = ArchTarget.from_gfx(arch)
 *         if not target.mma.has_shape(a_dtype=atom.dtype_in, b_dtype=atom.dtype_in,
 *                 c_dtype=atom.dtype_out, m=atom.m, n=atom.n, k=atom.k):
 *             raise NotImplementedError("{where} MFMA atom ... not in the {arch}
 *                 MMA catalog; this configuration requires a different target.")
 *
 * Guard the selected atom against the per-arch MMA catalog. A no-op on supported
 * mantissas; raises before IR/compile for a gfx-only atom.
 *
 * C spelling: returns CKC_OK when the atom IS in the catalog. On the
 * NotImplementedError path it records CKC_ERR_NOTIMPL + a Python-matching message
 * on the builder and returns that status. (The Python does not catch from_gfx's
 * KeyError here; an unknown `arch` resolves the target to NULL, which is reported
 * as a CKC_ERR_VALUE no-target error.) `where` is the caller's kernel-name prefix
 * used in the message. */
ckc_status_t ckc_validate_mfma_atom_in_catalog(ckc_ir_builder_t* b,
                                               const ckc_mfma_atom_t* atom,
                                               const char* arch,
                                               const char* where);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_HELPERS_MFMA_GEMM_INNER_H */
