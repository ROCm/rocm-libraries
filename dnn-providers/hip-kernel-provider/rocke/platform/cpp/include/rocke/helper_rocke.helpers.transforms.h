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
    ROCKE_XFORM_INDIRECT, /* Indirect: lower = table[base + upper]      */
    /* ---- AOT runtime (SSA Value) variants ---- */
    ROCKE_XFORM_PAD_DYNAMIC, /* PadDynamic: lo/hi are SSA Values       */
    ROCKE_XFORM_UNMERGE_MAGIC_DYNAMIC, /* UnmergeMagicDynamic: magic triples SSA */
    ROCKE_XFORM_EMBED_DYNAMIC /* EmbedDynamic: strides/offset/lo/hi SSA */
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

    /* ---- AOT dynamic (SSA Value) variants ---- */

    /* PadDynamic (ROCKE_XFORM_PAD_DYNAMIC): lo_v/hi_v are SSA Values.
     * NULL means skip that side (mirrors Python None). */
    rocke_value_t* lo_v; /* i32 SSA or NULL */
    rocke_value_t* hi_v; /* i32 SSA or NULL */

    /* Compile-time bounds for the DYNAMIC variants. Python's PadDynamic /
     * EmbedDynamic accept either an int or a Value and only call const_i32
     * inside apply(); materialising the constant at construction time would
     * give it a different SSA id and break byte-identity, so an int bound is
     * carried here and emitted at apply time instead.
     * *_const_valid selects between the int and the Value. */
    int lo_const_valid;
    int hi_const_valid;
    int lo_const;
    int hi_const;

    /* EmbedDynamic (ROCKE_XFORM_EMBED_DYNAMIC): strides/offset/lo/hi as SSA.
     * strides_v[i] parallels upper[i]; NULL slots use const_i32(1).
     * offset_v: NULL -> const_i32(0). lo_v/hi_v: NULL -> no bound check. */
    rocke_value_t* strides_v[8]; /* up to 8 upper coords */
    rocke_value_t* offset_v; /* i32 SSA or NULL      */

    /* Compile-time strides / offset for EmbedDynamic, same lazy contract as
     * lo_const / hi_const above. Python additionally *skips the multiply*
     * for a literal stride of 1 (``if isinstance(s, int) and s == 1``), so an
     * int stride is not merely a lazily-emitted constant -- it changes the
     * instruction stream, and mirroring that is what keeps the two engines
     * byte-identical. */
    int strides_const_valid[8];
    int strides_const[8];
    int offset_const_valid;
    int offset_const;

    /* UnmergeMagicDynamic (ROCKE_XFORM_UNMERGE_MAGIC_DYNAMIC):
     * n_lower-1 triples (one per lower coord except the leading quotient).
     * Stored flat: triples_mult_v[i]/triples_shift_v[i]/triples_dim_v[i]
     * correspond to into[i+1] (not into[0], which gets the final quotient). */
    rocke_value_t* triples_mult_v[7]; /* max 8 lowers -> 7 triples */
    rocke_value_t* triples_shift_v[7];
    rocke_value_t* triples_dim_v[7];
    int n_triples; /* = n_lower - 1 */
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
/* Python: pad_dynamic(coord, lo=<int>, hi=<Value>).
 * ``lo`` is a compile-time int emitted lazily at apply() time, matching the
 * Python transform; pass hi as NULL to skip that side. */
rocke_transform_t*
    rocke_pad_dynamic_lo_const(rocke_ir_builder_t* b, const char* coord, int lo, rocke_value_t* hi);

/* Python: embed_dynamic(upper, lower, strides, offset, lo=<int>, hi=<Value>).
 * Same lazy-``lo`` contract as rocke_pad_dynamic_lo_const. */
rocke_transform_t* rocke_embed_dynamic_lo_const(rocke_ir_builder_t* b,
                                                const char* const* upper,
                                                int n_upper,
                                                const char* into,
                                                rocke_value_t* const* strides,
                                                rocke_value_t* offset,
                                                int lo,
                                                rocke_value_t* hi);

/* Python: embed_dynamic with a mix of int and Value strides / offset.
 *
 * ``strides_v[i]`` non-NULL takes precedence; otherwise ``strides_const[i]``
 * is used, and a value of 1 skips the multiply exactly as Python does.
 * ``offset_v`` non-NULL likewise wins over ``offset_const``; an int offset of
 * 0 still materialises its constant (Python builds it before deciding not to
 * add it) but emits no add. */
rocke_transform_t* rocke_embed_dynamic_mixed(rocke_ir_builder_t* b,
                                             const char* const* upper,
                                             int n_upper,
                                             const char* into,
                                             rocke_value_t* const* strides_v,
                                             const int* strides_const,
                                             rocke_value_t* offset_v,
                                             int offset_const,
                                             int lo,
                                             rocke_value_t* hi);

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

/* Python: pad_dynamic(coord, *, lo=None, hi=None) -> PadDynamic.
 *
 * Like rocke_pad() but lo/hi are i32 SSA Values instead of compile-time
 * integers. Pass NULL for a side that should not be checked (mirrors
 * PadDynamic where lo/hi may be None). Mirrors transforms.PadDynamic.apply. */
rocke_transform_t* rocke_pad_dynamic(rocke_ir_builder_t* b,
                                     const char* coord,
                                     rocke_value_t* lo, /* NULL = skip lo check */
                                     rocke_value_t* hi /* NULL = skip hi check */);

/* Python: unmerge_magic_dynamic(upper, into, magic_triples) -> UnmergeMagicDynamic.
 *
 * Like rocke_unmerge_magic() but (multiplier, shift, dim) are i32 SSA Values
 * instead of compile-time integers. `magic_triples` is an array of n_lower-1
 * struct rocke_magic_triple_t (each with mult/shift/dim as rocke_value_t*).
 * The leading lower coord gets the final quotient (no triple needed for it).
 * Mirrors transforms.UnmergeMagicDynamic.apply. */
typedef struct rocke_magic_triple
{
    rocke_value_t* mult; /* magic multiplier (i32 SSA) */
    rocke_value_t* shift; /* magic shift      (i32 SSA) */
    rocke_value_t* dim; /* dimension value  (i32 SSA) -- used for remainder */
} rocke_magic_triple_t;

rocke_transform_t* rocke_unmerge_magic_dynamic(rocke_ir_builder_t* b,
                                               const char* upper,
                                               const char* const* into,
                                               int n_lower,
                                               const rocke_magic_triple_t* triples);
/* n_lower-1 triples: triples[i] is for into[i+1] (the leading coord is the
 * final quotient). into[0] gets the quotient after the last division. */

/* Python: embed_dynamic(upper, lower, strides, offset=0, lo=None, hi=None) -> EmbedDynamic.
 *
 * Like rocke_embed_bounded() but strides / offset / lo / hi are i32 SSA Values
 * (or NULL to use the corresponding None-equivalent: offset=0, lo=INT_MIN,
 * hi=INT_MAX). Mirrors transforms.EmbedDynamic.apply. */
rocke_transform_t* rocke_embed_dynamic(rocke_ir_builder_t* b,
                                       const char* const* upper,
                                       int n_upper,
                                       const char* into,
                                       rocke_value_t* const* strides,
                                       rocke_value_t* offset, /* NULL -> const_i32(0) */
                                       rocke_value_t* lo, /* NULL -> no lo check  */
                                       rocke_value_t* hi /* NULL -> no hi check  */);

/* Python: DynamicTensorDescriptor.create(name, *, coord_names, strides) and
 * DynamicTensorDescriptor.offset(b, ...) -- overrides offset() to use runtime
 * SSA strides instead of the stored integer strides of the base class.
 *
 * rocke_tensor_descriptor_naive_dynamic() constructs the equivalent: a
 * TensorDescriptor whose base_strides field is NOT used by offset(); instead
 * the runtime strides stored in `dynamic_strides` are used.  The transform
 * chain and coord resolution work identically to the static descriptor.
 *
 * The returned descriptor is heap-allocated via the builder's arena; callers
 * do not free it.  `n_coords` == `n_strides` is required. */
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

    /* Set by rocke_tensor_descriptor_naive_dynamic. When true the object is
     * really a rocke_dynamic_tensor_descriptor_t (base is its first field),
     * and rocke_transforms_descriptor_offset multiplies each base coord by the
     * runtime stride instead of the compile-time base_strides. Dispatching on
     * a flag here rather than at every call site is what lets the conv
     * builders keep one code path for both descriptor kinds. */
    int is_dynamic;
} rocke_tensor_descriptor_t;

/* Python: DynamicTensorDescriptor — TensorDescriptor with runtime SSA strides.
 * `base` must be the first field (allows safe cast from static to dynamic). */
typedef struct rocke_dynamic_tensor_descriptor
{
    rocke_tensor_descriptor_t base; /* must be first for safe cast */
    int n_dynamic; /* == base.n_base              */
    rocke_value_t* dynamic_strides[8]; /* parallel to base.base_names */
} rocke_dynamic_tensor_descriptor_t;

rocke_dynamic_tensor_descriptor_t*
    rocke_tensor_descriptor_naive_dynamic(rocke_ir_builder_t* b,
                                          const char* name,
                                          const char* const* coord_names,
                                          int n_coords,
                                          rocke_value_t* const* strides);

bool rocke_dynamic_tensor_descriptor_offset(rocke_ir_builder_t* b,
                                            const rocke_dynamic_tensor_descriptor_t* desc,
                                            const char* const* in_names,
                                            rocke_value_t* const* in_values,
                                            int n_in,
                                            rocke_value_t** out_off,
                                            rocke_value_t** out_valid);

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
