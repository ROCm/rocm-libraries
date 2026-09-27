/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.transforms.h -- C99 port of a SUBSET of
 * rocke/helpers/transforms.py: the CK Tile-style coordinate-transform DAG.
 *
 * SCOPE OF THIS PORT (this phase) -- exactly these Python symbols:
 *   calculate_magic_numbers        rocke_calculate_magic_numbers
 *   do_magic_division              rocke_do_magic_division
 *   CoordVar                       rocke_coord_var_t
 *   Embed (transform)              rocke_embed() / ROCKE_XFORM_EMBED
 *   PassThrough (via pass_through) rocke_pass_through() / ROCKE_XFORM_PASS_THROUGH
 *   Unmerge (via unmerge)          rocke_unmerge() / ROCKE_XFORM_UNMERGE
 *   UnmergeMagicDiv (unmerge_magic)rocke_unmerge_magic() / ROCKE_XFORM_UNMERGE_MAGIC
 *   embed                          rocke_embed
 *   pass_through                   rocke_pass_through
 *   unmerge                        rocke_unmerge
 *   unmerge_magic                  rocke_unmerge_magic
 *   TensorDescriptor               rocke_tensor_descriptor_t
 *     .naive                       rocke_tensor_descriptor_naive
 *     .transform                   rocke_tensor_descriptor_transform
 *     .offset                      rocke_tensor_descriptor_offset
 *     .unmerge_lower               rocke_tensor_descriptor_unmerge_lower
 *
 * Out of scope (NOT ported here): Pad, Merge, UnmergeDivMod, XorT, Slice,
 * Freeze, Insert, Replicate, Modulo, Offset, RightPad, LeftPad, PadDynamic,
 * Indirect, the move() / offset_i64 / delta machinery, and the convenience
 * constructors for those.  Only the transform kinds reachable from the symbols
 * above are given a tag in rocke_xform_kind_t.
 *
 * The op sequence emitted into the IRBuilder by every function below is
 * byte-identical to the Python so the downstream IR op stream matches exactly.
 *
 * Lifetime: every produced node (transforms, descriptors, name arrays, coord
 * tables) is arena-owned (rocke_ir_builder_t.arena). Nothing is freed
 * individually; the arena bulk-frees the whole graph. This mirrors the Python
 * frozen-dataclass / GC lifetime exactly.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_TRANSFORMS_H
#define ROCKE_HELPER_ROCKE_HELPERS_TRANSFORMS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ magic div */

/* Python: calculate_magic_numbers(divisor) -> (multiplier, shift).
 *
 * Computes the unsigned magic-division (multiplier, shift) pair:
 *   shift      = ceil(log2(divisor))   (smallest s with 2**s >= divisor)
 *   multiplier = (((1 << shift) - divisor) << 32) // divisor + 1
 *
 * Both are compile-time constants. `multiplier` may exceed the int32 range
 * (it is a uint32 magic constant) so it is returned via a uint64 out-param.
 *
 * On divisor < 1 (the Python ValueError) the builder's sticky error is set
 * (ROCKE_ERR_VALUE) and the function returns false with *out_multiplier /
 * *out_shift left untouched. `b` may be NULL for the pure compile-time path,
 * in which case an invalid divisor just returns false (no error recorded).
 * Returns true on success. */
bool rocke_calculate_magic_numbers(rocke_ir_builder_t* b,
                                   int divisor,
                                   uint64_t* out_multiplier,
                                   int* out_shift);

/* Python: do_magic_division(b, dividend, multiplier, shift) -> Value.
 *
 * Emits `dividend // divisor` via the mul-hi magic sequence:
 *   mult_i32 = (multiplier >= 2**31) ? multiplier - 2**32 : multiplier
 *   tmp      = b.umul_hi_i32(dividend, b.const_i32(mult_i32))
 *   summed   = b.add(tmp, dividend)
 *   return shift == 0 ? summed : b.lshr(summed, b.const_i32(shift))
 *
 * `dividend` is an i32 SSA value (treated unsigned); the result is the i32
 * quotient. `multiplier` / `shift` come from rocke_calculate_magic_numbers. */
rocke_value_t* rocke_do_magic_division(rocke_ir_builder_t* b,
                                       rocke_value_t* dividend,
                                       uint64_t multiplier,
                                       int shift);

/* -------------------------------------------------------------------- CoordVar */

/* Python: @dataclass(frozen=True) class CoordVar.
 *   name:  str   -- the coord's name at this level
 *   value: Value -- i32 SSA value
 *   valid: Optional[Value] -- i1 SSA validity, or None (NULL) for "always" */
typedef struct rocke_coord_var
{
    const char* name; /* arena-owned coord name                       */
    rocke_value_t* value; /* i32 SSA value                                */
    rocke_value_t* valid; /* i1 SSA validity, or NULL (Python None)       */
} rocke_coord_var_t;

/* ------------------------------------------------------------------ transforms */

/* Kinds for the subset of transforms reachable from this port's symbols.
 * (The full Python Transform hierarchy has more; only these are ported now.) */
typedef enum rocke_xform_kind
{
    ROCKE_XFORM_PASS_THROUGH = 0, /* PassThrough: lower[0] = upper[0] (rename)  */
    ROCKE_XFORM_EMBED, /* Embed: affine lower = sum(s_i*u_i)+offset  */
    ROCKE_XFORM_UNMERGE, /* Unmerge: split 1 flat coord -> N via div/mod */
    ROCKE_XFORM_UNMERGE_MAGIC, /* UnmergeMagicDiv: split via magic division  */
    ROCKE_XFORM_PAD, /* Pad: value passes through, valid &= lo<=x<hi */
    ROCKE_XFORM_INDIRECT /* Indirect: lower = table[base + upper]      */
} rocke_xform_kind_t;

/* One node in the coord-transform DAG. A tagged record carrying the same
 * fields the corresponding Python dataclass holds. `upper`/`lower` are
 * arena-owned arrays of arena-owned name strings. For Embed, `strides` is an
 * arena-owned int array parallel to `upper`. For Unmerge*, `dims` is an
 * arena-owned int array parallel to `lower`. */
typedef struct rocke_transform
{
    rocke_xform_kind_t kind;

    const char* const* upper; /* upper-level coord names */
    int n_upper;
    const char* const* lower; /* lower-level coord names */
    int n_lower;

    /* Embed only: */
    const int* strides; /* length n_upper */
    int offset;
    int lo;
    int hi;

    /* Unmerge / UnmergeMagicDiv only: */
    const int* dims; /* length n_lower */

    /* Indirect only: physical = table[base + upper]. When `max_idx` is NULL the
     * load is unguarded (global_load_i32); otherwise it is a masked_global_load
     * clamping idx >= max_idx to `default_value`. */
    rocke_value_t* table;
    rocke_value_t* base;
    rocke_value_t* max_idx;
    int default_value;
} rocke_transform_t;

/* Python: pass_through(coord, into=None) -> PassThrough.
 * lower defaults to `coord` when `into` is NULL. */
rocke_transform_t* rocke_pass_through(rocke_ir_builder_t* b, const char* coord, const char* into);

/* Python: embed(upper, into, *, strides, offset=0, lo=None, hi=None) -> Embed.
 *
 * `upper` is an array of `n_upper` names; `strides` is an int array of the same
 * length. `lo`/`hi` use the sentinel pair from the Python __init__:
 *   lo == None -> -(1 << 30); hi == None -> (1 << 30).
 * To pass an explicit bound use rocke_embed_bounded(); rocke_embed() applies the
 * None-sentinel defaults. On len(upper) != len(strides) (the Python
 * ValueError) the builder error is set and NULL returned. */
rocke_transform_t* rocke_embed(rocke_ir_builder_t* b,
                               const char* const* upper,
                               int n_upper,
                               const char* into,
                               const int* strides,
                               int offset);

/* Explicit-bounds Embed (Python lo=/hi= supplied). Same as rocke_embed but with
 * caller-chosen lo/hi instead of the None-sentinels. */
rocke_transform_t* rocke_embed_bounded(rocke_ir_builder_t* b,
                                       const char* const* upper,
                                       int n_upper,
                                       const char* into,
                                       const int* strides,
                                       int offset,
                                       int lo,
                                       int hi);

/* Python: unmerge(upper, into, *, dims) -> Unmerge.
 * `into` (the lowers) is an array of `n_lower` names; `dims` an int array of the
 * same length. On len(lowers) != len(dims) the builder error is set, NULL
 * returned. */
rocke_transform_t* rocke_unmerge(rocke_ir_builder_t* b,
                                 const char* upper,
                                 const char* const* into,
                                 int n_lower,
                                 const int* dims);

/* Python: unmerge_magic(upper, into, *, dims) -> UnmergeMagicDiv. */
rocke_transform_t* rocke_unmerge_magic(rocke_ir_builder_t* b,
                                       const char* upper,
                                       const char* const* into,
                                       int n_lower,
                                       const int* dims);

/* Python: pad(coord, *, lo, hi) -> Pad.
 *
 * Validity-only transform: lower[0] == upper[0] (value passes through
 * unchanged) and the validity picks up ``lo <= value < hi`` AND-ed with any
 * incoming validity. Mirrors transforms.Pad.apply byte-for-byte. */
rocke_transform_t* rocke_pad(rocke_ir_builder_t* b, const char* coord, int lo, int hi);

/* Python: indirect(upper, into, *, table, base, max_idx=None, default_value=0)
 * -> Indirect. Table-lookup transform: physical = table[base + upper]. When
 * `max_idx` is NULL the load is unguarded; otherwise an OOB-safe masked load
 * clamps idx >= max_idx to `default_value`. Mirrors transforms.Indirect.apply. */
rocke_transform_t* rocke_indirect(rocke_ir_builder_t* b,
                                  const char* upper,
                                  const char* into,
                                  rocke_value_t* table,
                                  rocke_value_t* base,
                                  rocke_value_t* max_idx,
                                  int default_value);

/* ----------------------------------------------------------- TensorDescriptor */

/* Python: @dataclass class TensorDescriptor.
 *   name, base_names, base_lengths, base_strides, chain, upper_names. */
typedef struct rocke_tensor_descriptor
{
    const char* name;

    const char* const* base_names; /* naive base coord names      */
    const int* base_lengths; /* parallel to base_names      */
    const int* base_strides; /* parallel to base_names      */
    int n_base;

    const rocke_transform_t* const* chain; /* transforms, naive -> upper  */
    int n_chain;

    const char* const* upper_names; /* current user-facing coords  */
    int n_upper;
} rocke_tensor_descriptor_t;

/* Python: TensorDescriptor.naive(name, *, lengths, dtype=F16, strides=None,
 *                                coord_names=None).
 *
 * `strides == NULL` -> row-major product of lengths to the right.
 * `coord_names == NULL` -> ("d0", "d1", ...). `dtype` is accepted for parity
 * but, like the Python .dtype property, the descriptor pins F16 regardless, so
 * no dtype field is stored. On an empty lengths array, a strides/coord_names
 * length mismatch (the Python ValueErrors), the builder error is set and NULL
 * returned. */
rocke_tensor_descriptor_t*
    rocke_tensor_descriptor_naive(rocke_ir_builder_t* b,
                                  const char* name,
                                  const int* lengths,
                                  int n_lengths,
                                  const int* strides, /* or NULL */
                                  const char* const* coord_names, /* or NULL */
                                  int n_coord_names);

/* Python: TensorDescriptor.transform(*transforms) -> new TensorDescriptor.
 *
 * Appends `transforms` to the chain and recomputes upper_names as
 *   (base_names | all_uppers) - all_lowers
 * preserving the stable order (base_names first, then transform uppers in
 * appearance order). With n_transforms == 0 the original descriptor is returned
 * unchanged (Python returns `self`). */
rocke_tensor_descriptor_t*
    rocke_tensor_descriptor_transform(rocke_ir_builder_t* b,
                                      const rocke_tensor_descriptor_t* desc,
                                      const rocke_transform_t* const* transforms,
                                      int n_transforms);

/* Python: TensorDescriptor.unmerge_lower(b, **upper_values) -> {name: value}.
 *
 * Runs the chain in topological order from the supplied upper coords and
 * returns the lowered coord map (values only, not validity). Unlike offset()
 * this does NOT require every upper_name to be supplied and stops (returns the
 * partial map) when no further transform is applicable, exactly like the
 * Python `if not progress: break`.
 *
 * Inputs: parallel arrays `in_names[i]` -> `in_values[i]` (length n_in).
 * Outputs: writes up to `out_cap` entries into out_names[]/out_values[] and
 * returns the number written (the size of the produced map). Returns -1 on
 * builder failure or if out_cap is too small. The output map enumerates coords
 * in the deterministic order: all inputs first (in supplied order), then each
 * transform's produced lowers in the order they were resolved. */
int rocke_tensor_descriptor_unmerge_lower(rocke_ir_builder_t* b,
                                          const rocke_tensor_descriptor_t* desc,
                                          const char* const* in_names,
                                          rocke_value_t* const* in_values,
                                          int n_in,
                                          const char** out_names,
                                          rocke_value_t** out_values,
                                          int out_cap);

/* Python: TensorDescriptor.offset(b, **upper_values) -> (offset, valid).
 *
 * Resolves the full chain (requiring all upper_names to be present -- the
 * Python ValueError on a missing coord sets the builder error and returns
 * false) then reduces the base coords with base_strides into the linear i32
 * element offset and conjoined i1 validity.
 *
 * On success writes *out_offset (never NULL: const_i32(0) when there are no
 * terms) and *out_valid (NULL == Python None) and returns true. On any error
 * (missing upper, unresolved chain dep, base coord absent) the builder error is
 * set and false returned. */
bool rocke_transforms_descriptor_offset(rocke_ir_builder_t* b,
                                        const rocke_tensor_descriptor_t* desc,
                                        const char* const* in_names,
                                        rocke_value_t* const* in_values,
                                        int n_in,
                                        rocke_value_t** out_offset,
                                        rocke_value_t** out_valid);

/* Faithful port of TensorDescriptor.offset_i64_split (transforms.py 1463-1505).
 * Like rocke_transforms_descriptor_offset but splits the linear offset into
 * (base_i64, within_i32): the `base_coord` term is scalarised (to_sgpr_u32) and
 * widened to i64 (no 2 GiB overflow for paged caches), while the remaining base
 * terms are summed into a small i32 within-block offset. *out_base_i64 and
 * *out_within are written on success (within is const_i32(0) when no other
 * terms); *out_valid is NULL == Python None. Returns false (builder error set)
 * if base_coord is not among the descriptor's base coords. */
bool rocke_transforms_descriptor_offset_i64_split(rocke_ir_builder_t* b,
                                                  const rocke_tensor_descriptor_t* desc,
                                                  const char* base_coord,
                                                  const char* const* in_names,
                                                  rocke_value_t* const* in_values,
                                                  int n_in,
                                                  rocke_value_t** out_base_i64,
                                                  rocke_value_t** out_within,
                                                  rocke_value_t** out_valid);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_TRANSFORMS_H */
