/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.rotary.h -- C99 port of rocke.helpers.rotary.
 *
 * Rotary position embedding (RoPE) helpers. RoPE rotates pairs of head-dim
 * values by a position-dependent angle:
 *
 *     pair_idx in [0, head_size / 2)
 *     theta   = inv_freq[pair_idx]                # precomputed 1 / 10000^(2k/H)
 *     cos_t   = cos(token_position * theta)
 *     sin_t   = sin(token_position * theta)
 *     out[2*i]   = in[2*i] * cos_t - in[2*i+1] * sin_t   # interleaved
 *     out[2*i+1] = in[2*i] * sin_t + in[2*i+1] * cos_t
 *
 * Two layouts are supported, exactly as in the Python:
 *   * "interleaved" (GPT-J / LLaMA-1): the paired elements are at 2*i and
 *     2*i+1 in the head-dim.
 *   * "half" (LLaMA-2/3, Qwen, ...): the head-dim is bisected; element i
 *     pairs with element i + H/2.
 * Both layouts share the same (cos_t, sin_t) table -- only the pair-index
 * function differs.
 *
 * Ported symbols (the subset requested):
 *   * RotarySpec               -> rocke_rotary_spec_t + rocke_rotary_spec_init()
 *                                 (+ pair_count / stride_pos accessors)
 *   * pair_indices             -> rocke_rotary_pair_indices()
 *   * load_cos_sin             -> rocke_rotary_load_cos_sin()
 *   * apply_rotary_pair_f32    -> rocke_rotary_apply_pair_f32()
 *
 * Error model mirrors the rest of the C port. Where the Python raises a plain
 * ValueError WITHOUT a builder in hand (RotarySpec.__post_init__,
 * pair_indices), the C spelling returns a rocke_status_t and writes nothing to
 * the (absent) builder. Where the Python raises ValueError holding a builder
 * (load_cos_sin / apply_rotary_pair_f32 type checks), the C spelling records
 * the matching ROCKE_ERR_VALUE on the sticky-error builder and returns NULL --
 * and is a NULL no-op when the builder is already failed.
 *
 * The builder-call sequence inside load_cos_sin / apply_rotary_pair_f32 is
 * reproduced in the exact same order as the Python so the emitted IR (and SSA
 * value numbering) stays byte-identical. Nested Python expressions are
 * sequenced explicitly because C argument-evaluation order is unspecified.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_ROTARY_H
#define ROCKE_HELPER_ROCKE_HELPERS_ROTARY_H

#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_value_t, rocke_status_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * RotaryLayout / RotarySpec
 * ------------------------------------------------------------------ *
 *
 * Python:
 *
 *   RotaryLayout = Literal["interleaved", "half"]
 *
 *   @dataclass(frozen=True)
 *   class RotarySpec:
 *       head_size: int
 *       layout: RotaryLayout = "half"
 *       table_stride_pos: int = 0   # 0 = compute as head_size // 2
 *
 *       def __post_init__(self):
 *           if self.head_size <= 0 or self.head_size % 2 != 0:
 *               raise ValueError(...)
 *           if self.layout not in ("interleaved", "half"):
 *               raise ValueError(...)
 *
 *       @property
 *       def pair_count(self) -> int:   return self.head_size // 2
 *       @property
 *       def stride_pos(self) -> int:   return self.table_stride_pos or self.pair_count
 */

/* RotaryLayout. The two valid string spellings become an enum; the string
 * forms are recoverable via rocke_rotary_layout_name() for diagnostics. */
typedef enum rocke_rotary_layout
{
    ROCKE_ROTARY_INTERLEAVED = 0, /* "interleaved" (GPT-J / LLaMA-1)            */
    ROCKE_ROTARY_HALF /* "half"        (LLaMA-2/3, Qwen, ...)       */
} rocke_rotary_layout_t;

/* Canonical string for a layout ("interleaved" / "half"), or NULL if the enum
 * value is out of range. Static storage; do not free. */
const char* rocke_rotary_layout_name(rocke_rotary_layout_t layout);

/* The frozen RotarySpec dataclass as a plain value struct. Populate it with
 * rocke_rotary_spec_init() so __post_init__'s validation runs; reading the
 * fields directly is fine once initialised. */
typedef struct rocke_rotary_spec
{
    int head_size; /* total head dimension H; positive + even  */
    rocke_rotary_layout_t layout; /* default "half"                           */
    int table_stride_pos; /* 0 sentinel = compute as head_size / 2    */
} rocke_rotary_spec_t;

/* RotarySpec(head_size, layout, table_stride_pos) + __post_init__.
 *
 * Fills *out and validates exactly as the Python __post_init__:
 *   * head_size must be positive and even, else ROCKE_ERR_VALUE;
 *   * layout must be ROCKE_ROTARY_INTERLEAVED or ROCKE_ROTARY_HALF, else
 *     ROCKE_ERR_VALUE.
 * There is no builder at construction time (the Python dataclass raises a bare
 * ValueError), so the diagnostic is conveyed purely through the return code;
 * nothing is emitted. To take the Python defaults pass layout =
 * ROCKE_ROTARY_HALF and table_stride_pos = 0. On failure *out is left untouched.
 * Returns ROCKE_OK or ROCKE_ERR_VALUE. */
rocke_status_t rocke_rotary_spec_init(rocke_rotary_spec_t* out,
                                      int head_size,
                                      rocke_rotary_layout_t layout,
                                      int table_stride_pos);

/* RotarySpec.pair_count property: head_size // 2. */
int rocke_rotary_spec_pair_count(const rocke_rotary_spec_t* spec);

/* RotarySpec.stride_pos property: table_stride_pos or pair_count
 * (i.e. the table_stride_pos when non-zero, else head_size // 2). */
int rocke_rotary_spec_stride_pos(const rocke_rotary_spec_t* spec);

/* ------------------------------------------------------------------ *
 * pair_indices
 * ------------------------------------------------------------------ *
 *
 * Python:
 *
 *   def pair_indices(spec, pair_idx) -> Tuple[int, int]:
 *       if pair_idx < 0 or pair_idx >= spec.pair_count:
 *           raise ValueError(...)
 *       if spec.layout == "interleaved":
 *           return (2 * pair_idx, 2 * pair_idx + 1)
 *       return (pair_idx, pair_idx + spec.pair_count)
 *
 * Compile-time index math, no builder. Returns the (lo, hi) head-dim indices
 * for rotary pair `pair_idx` via the out-params *out_lo / *out_hi. Out-of-range
 * `pair_idx` mirrors the Python ValueError as ROCKE_ERR_VALUE (out-params left
 * untouched). Returns ROCKE_OK or ROCKE_ERR_VALUE. */
rocke_status_t rocke_rotary_pair_indices(const rocke_rotary_spec_t* spec,
                                         int pair_idx,
                                         int* out_lo,
                                         int* out_hi);

/* ------------------------------------------------------------------ *
 * load_cos_sin
 * ------------------------------------------------------------------ *
 *
 * Python:
 *
 *   def load_cos_sin(b, cos_table, sin_table, *, token_pos, pair_idx, spec):
 *       if cos_table.type != sin_table.type:
 *           raise ValueError("cos / sin tables must have matching pointer type")
 *       if not isinstance(cos_table.type, PtrType):
 *           raise ValueError("cos_table must be a pointer")
 *       if cos_table.type.pointee != F32:
 *           raise ValueError("rotary tables must be ptr<f32> in v1")
 *       offset = b.add(b.mul(token_pos, b.const_i32(spec.stride_pos)), pair_idx)
 *       cos_v = b.global_load_f32(cos_table, offset)
 *       sin_v = b.global_load_f32(sin_table, offset)
 *       return cos_v, sin_v
 *
 * Loads one (cos, sin) f32 pair from the rotary tables, laid out as
 * (max_position, H/2) row-major: offset = token_pos * stride_pos + pair_idx.
 * Issues two scalar global loads. The caller promotes token_pos to a
 * wave-uniform i32 beforehand if scalar-register loads are wanted.
 *
 * The three type checks set the matching ROCKE_ERR_VALUE on the builder (the
 * Python ValueError text is reproduced) and return ROCKE_ERR_VALUE with the
 * out-params untouched. On success *out_cos / *out_sin receive the two loaded
 * f32 Values and the call returns ROCKE_OK. NULL no-op (returns the sticky
 * status) on an already-failed builder. */
rocke_status_t rocke_rotary_load_cos_sin(rocke_ir_builder_t* b,
                                         rocke_value_t* cos_table,
                                         rocke_value_t* sin_table,
                                         rocke_value_t* token_pos,
                                         rocke_value_t* pair_idx,
                                         const rocke_rotary_spec_t* spec,
                                         rocke_value_t** out_cos,
                                         rocke_value_t** out_sin);

/* ------------------------------------------------------------------ *
 * apply_rotary_pair_f32
 * ------------------------------------------------------------------ *
 *
 * Python:
 *
 *   def apply_rotary_pair_f32(b, lo, hi, cos_t, sin_t) -> Tuple[Value, Value]:
 *       if lo.type.name != "f32" or hi.type.name != "f32":
 *           raise ValueError("apply_rotary_pair_f32 expects f32 inputs")
 *       if cos_t.type.name != "f32" or sin_t.type.name != "f32":
 *           raise ValueError("apply_rotary_pair_f32 expects f32 cos / sin")
 *       new_lo = b.fsub(b.fmul(lo, cos_t), b.fmul(hi, sin_t))
 *       new_hi = b.fadd(b.fmul(lo, sin_t), b.fmul(hi, cos_t))
 *       return new_lo, new_hi
 *
 * Applies the 2x2 rotation to one (lo, hi) element pair:
 *     lo' = lo * cos_t - hi * sin_t
 *     hi' = lo * sin_t + hi * cos_t
 * Compute is in f32 (the backend folds into two v_fma_f32 per pair).
 *
 * The two f32 type checks set ROCKE_ERR_VALUE (matching the Python text) and
 * return ROCKE_ERR_VALUE with the out-params untouched. On success *out_lo /
 * *out_hi receive the rotated f32 Values and the call returns ROCKE_OK. NULL
 * no-op (returns the sticky status) on an already-failed builder. */
rocke_status_t rocke_rotary_apply_pair_f32(rocke_ir_builder_t* b,
                                           rocke_value_t* lo,
                                           rocke_value_t* hi,
                                           rocke_value_t* cos_t,
                                           rocke_value_t* sin_t,
                                           rocke_value_t** out_lo,
                                           rocke_value_t** out_hi);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_ROTARY_H */
