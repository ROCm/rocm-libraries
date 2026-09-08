/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.mfma_gemm_inner.h -- C99 port of selected symbols
 * from rocke/helpers/mfma_gemm_inner.py (the universal MFMA-tiled K-loop helper
 * for GEMM-shaped kernels).
 *
 * SCOPE OF THIS PORT (this phase) -- exactly these nine Python symbols:
 *
 *   Python                                  C99 (this header)
 *   --------------------------------------  -------------------------------------
 *   class LaneDecode                        rocke_lane_decode_t
 *   decode_mfma_lanes(b, atom, lane)        rocke_decode_mfma_lanes(...)
 *   load_a_row_major_contiguous(...)        rocke_load_a_row_major_contiguous(...)
 *   load_b_col_strided_scalars(...)         rocke_load_b_col_strided_scalars(...)
 *   mfma_atom_for_dtype(dtype, m, n, ...)   rocke_mfma_atom_for_dtype(...)
 *   mfma_k_loop(...)                        rocke_mfma_k_loop(...)
 *   store_acc_to_global(...)                rocke_store_acc_to_global(...)
 *   validate_arch_and_block_size(...)       rocke_validate_arch_and_block_size(...)
 *   validate_mfma_atom_in_catalog(...)      rocke_validate_mfma_atom_in_catalog(...)
 *
 * NOT in scope here (left in Python only this phase):
 *   load_smem_frag_contiguous_f16, mfma_k_loop_dynamic_K.
 *
 * BINDINGS.
 *   - The IR builder primitives are rocke/ir.h's rocke_b_* entry points (const_i32,
 *     mod/div/mul/add, global_load[_vN], global_store, global_atomic_add,
 *     zero_vec[_f32], vec_insert/vec_extract, scf_for_iter/scf_yield, mma,
 *     cast_f32_to, and the f16/bf16/fp8e4m3/bf8e5m2/f32 type singletons).
 *   - The MfmaAtom value type is rocke/helper_rocke.helpers.atoms.h's
 *     rocke_mfma_atom_t (fields m, n, k, a_per_lane, b_per_lane, c_per_lane,
 *     dtype_in, dtype_out, name). mfma_atom_for_dtype resolves atoms by
 *     (dtype, m, n, k) over that catalog via rocke_mfma_atom().
 *   - validate_arch_and_block_size / validate_mfma_atom_in_catalog bind to
 *     rocke/helper_rocke.core.arch.h (ArchTarget.from_gfx, max_threads_per_block,
 *     target.mma.has_shape).
 *
 * ATOM METHODS REPRODUCED INLINE (not separate symbols). The Python helper calls
 * three MfmaAtom methods that the atoms.h port does not expose (they are out of
 * scope there): atom.emit(), atom.zero_acc(), atom.lane_to_output(). Because each
 * is a tiny pure builder sequence over fields that ARE on rocke_mfma_atom_t, the
 * faithful port reproduces them inline inside rocke_mfma_k_loop /
 * rocke_store_acc_to_global, byte-for-byte:
 *     emit          -> rocke_b_mma(b, atom->name, a, b, c)
 *     zero_acc      -> rocke_b_zero_vec_f32(b, atom->c_per_lane)
 *     lane_to_output-> the 16x16 / 32x32 / 4x4 arith from atoms.py:536-591
 *
 * CALLBACKS. The Python `load_a`, `load_b`, `per_tile_post_mfma`, and `epilogue`
 * closures become explicit C function pointers carrying an opaque `user` pointer
 * (the C analog of a closure's captured environment). Signatures mirror the
 * Python call shapes one-for-one.
 *
 * ERROR MODEL. Mirrors the rest of the C port: the sticky-error builder (rocke_b_*)
 * stands in for `raise`. mfma_k_loop's `raise ValueError` (K % atom.k != 0),
 * store_acc_to_global's `raise ValueError` (atomic_add with non-f32), and the two
 * validate_* `raise`/return paths map onto rocke_status_t / sticky-error spellings.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_MFMA_GEMM_INNER_H
#define ROCKE_HELPER_ROCKE_HELPERS_MFMA_GEMM_INNER_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_t */
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom_t */
#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_value_t, rocke_status_t */

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
typedef struct rocke_lane_decode
{
    rocke_value_t* lane;
    rocke_value_t* m_in_atom;
    rocke_value_t* n_in_atom;
    rocke_value_t* k_blk;
} rocke_lane_decode_t;

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
rocke_lane_decode_t rocke_decode_mfma_lanes(rocke_ir_builder_t* b,
                                            const rocke_mfma_atom_t* atom,
                                            rocke_value_t* lane);

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
const rocke_mfma_atom_t*
    rocke_mfma_atom_for_dtype(const char* dtype_in, int m, int n, bool prefer_packed_k);

/* Builder-aware variant: identical selection; on the ValueError path it records
 * ROCKE_ERR_VALUE + a Python-matching message on the builder and returns NULL.
 * No-op returning NULL if the builder is already in an error state. */
const rocke_mfma_atom_t* rocke_b_mfma_atom_for_dtype(
    rocke_ir_builder_t* b, const char* dtype_in, int m, int n, bool prefer_packed_k);

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
rocke_value_t* rocke_load_a_row_major_contiguous(rocke_ir_builder_t* b,
                                                 rocke_value_t* A,
                                                 const rocke_mfma_atom_t* atom,
                                                 const rocke_lane_decode_t* lane_decode,
                                                 rocke_value_t* m_tile_base,
                                                 rocke_value_t* k_tile_base,
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
rocke_value_t* rocke_load_b_col_strided_scalars(rocke_ir_builder_t* b,
                                                rocke_value_t* B,
                                                const rocke_mfma_atom_t* atom,
                                                const rocke_lane_decode_t* lane_decode,
                                                rocke_value_t* n_tile_base,
                                                rocke_value_t* k_tile_base,
                                                int N);

/* --------------------------------------------------------------- callbacks *
 *
 * The Python helper takes Python closures; the C port takes function pointers
 * plus an opaque `user` environment pointer.
 *
 *   load_a / load_b : Python `Callable[[IRBuilder, Value], Value]`
 *       Called once per K-tile with the loop induction value `kt`; returns the
 *       per-lane operand vector. (rocke_load_a_row_major_contiguous /
 *       rocke_load_b_col_strided_scalars are typically wrapped behind these.)
 *
 *   per_tile_post_mfma : Python
 *       `Callable[[IRBuilder, Value, Value], Value]` (b, acc, kt) -> acc
 *       Optional post-MFMA accumulator transform (per-group scale / bias). NULL
 *       => no post step (the Python `is not None` guard). */
typedef rocke_value_t* (*rocke_mfma_load_fn)(rocke_ir_builder_t* b, rocke_value_t* kt, void* user);
typedef rocke_value_t* (*rocke_mfma_post_fn)(rocke_ir_builder_t* b,
                                             rocke_value_t* acc,
                                             rocke_value_t* kt,
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
 *   acc = atom.emit(a, b_op, acc)   [-> rocke_b_mma(name, ...)];
 *   acc = per_tile_post_mfma(b, acc, kt)   (if non-NULL);
 *   yield acc.
 * initial_acc NULL => atom.zero_acc(b) [-> rocke_b_zero_vec_f32(c_per_lane)].
 * Returns the final per-lane <c_per_lane x f32> accumulator (the for-op's first
 * result). iv_name/acc_name may be NULL (Python defaults "kt"/"acc").
 *
 * raise ValueError (K % atom.k != 0) -> builder sticky ROCKE_ERR_VALUE + NULL.
 * load_a/load_b are required (non-NULL); `user` is passed through to all three. */
rocke_value_t* rocke_mfma_k_loop(rocke_ir_builder_t* b,
                                 int K,
                                 const rocke_mfma_atom_t* atom,
                                 rocke_mfma_load_fn load_a,
                                 rocke_mfma_load_fn load_b,
                                 rocke_mfma_post_fn per_tile_post_mfma,
                                 rocke_value_t* initial_acc,
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
typedef void (*rocke_mfma_epilogue_fn)(rocke_ir_builder_t* b,
                                       const rocke_mfma_atom_t* atom,
                                       const rocke_lane_decode_t* lane_decode,
                                       rocke_value_t* C,
                                       rocke_value_t* m_tile_base,
                                       rocke_value_t* n_tile_base,
                                       rocke_value_t* acc,
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
 * Returns ROCKE_OK; on the Python raise paths (atomic_add with out_dtype != "f32",
 * or an unsupported out_dtype for the cast) the builder sticky error is set
 * (ROCKE_ERR_VALUE) and that status is returned. `epilogue_user` is passed to the
 * epilogue callback. */
rocke_status_t rocke_store_acc_to_global(rocke_ir_builder_t* b,
                                         rocke_value_t* C,
                                         const rocke_mfma_atom_t* atom,
                                         const rocke_lane_decode_t* lane_decode,
                                         rocke_value_t* m_tile_base,
                                         rocke_value_t* n_tile_base,
                                         rocke_value_t* acc,
                                         int N,
                                         const char* out_dtype,
                                         bool atomic_add,
                                         rocke_mfma_epilogue_fn epilogue,
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
bool rocke_validate_arch_and_block_size(rocke_ir_builder_t* b,
                                        const char* arch,
                                        int block_size,
                                        const char** out_reason,
                                        const rocke_archtarget_t** out_target);

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
 * C spelling: returns ROCKE_OK when the atom IS in the catalog. On the
 * NotImplementedError path it records ROCKE_ERR_NOTIMPL + a Python-matching message
 * on the builder and returns that status. (The Python does not catch from_gfx's
 * KeyError here; an unknown `arch` resolves the target to NULL, which is reported
 * as a ROCKE_ERR_VALUE no-target error.) `where` is the caller's kernel-name prefix
 * used in the message. */
rocke_status_t rocke_validate_mfma_atom_in_catalog(rocke_ir_builder_t* b,
                                                   const rocke_mfma_atom_t* atom,
                                                   const char* arch,
                                                   const char* where);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_MFMA_GEMM_INNER_H */
