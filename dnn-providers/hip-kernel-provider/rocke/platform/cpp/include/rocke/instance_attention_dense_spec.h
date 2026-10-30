/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_attention_dense_spec.h -- C port of the architecture-neutral
 * dense attention spec contract, library/kernels/common/attention_dense_spec.py.
 * Shared by the gfx942 and gfx950 dense attention ports.
 *
 *   Python (attention_dense_spec.py)              C (this header)
 *   -------------------------------------------   -------------------------------------
 *   DENSE_TILE_GEOMETRIES                         ROCKE_DENSE_TILE_GEOMETRIES
 *   INT32_LIMIT                                   ROCKE_ATTENTION_DENSE_INT32_LIMIT
 *   @dataclass AttentionDenseSpec                 rocke_attention_dense_spec_t
 *                                                   + rocke_attention_dense_spec_default()
 *   AttentionDenseSpec.__post_init__              rocke_attention_dense_spec_validate()
 *   num_waves / num_queries_per_kv                rocke_attention_dense_spec_num_waves() / ...
 *   resolved_persist_decode / _nonpersist_decode  rocke_attention_dense_spec_resolved_*_decode()
 *   kernel_name() + the _*_name_parts hooks       rocke_attention_dense_kernel_name()
 *                                                   + rocke_attention_dense_name_parts_t
 *   check_dense_spec_preflight(spec)              rocke_attention_dense_spec_preflight()
 *
 * SUBCLASS HOOKS. The Python base calls overridable methods on the concrete
 * subclass. The C port passes them in explicitly:
 *   - __post_init__ consults _supports_noncausal_window,
 *     _supports_noncausal_cross_length_ragged, supported_persist_decodes and
 *     supported_nonpersist_decodes: rocke_attention_dense_traits_t, which holds
 *     only per-class answers.
 *   - The auto decode policies and the GQA-pair decode checks consult
 *     _aligned_causal, which gfx950 overrides with a per-spec value (right_bound
 *     == 0). That answer is the `lookahead` argument of validate, both
 *     resolved_*_decode functions and name_parts_base: true when the concrete
 *     spec sees keys ahead of its diagonal (gfx950: right_bound != 0), false for
 *     the base class and gfx942. It is an argument, not a traits field, so a
 *     caller cannot set it once for a type and get it wrong per spec.
 *   - kernel_name() consults the _*_name_parts methods:
 *     rocke_attention_dense_name_parts_t, which the concrete spec fills.
 *   - check_dense_spec_preflight re-runs the concrete type's validators: the
 *     base checks (run by preflight itself, on the same spec) and then the
 *     arch's own fields (rocke_attention_dense_validate_fn, optional).
 * rocke_attention_dense_traits_base() / rocke_attention_dense_name_parts_base()
 * are the base-class answers.
 *
 * DECODES. The persistent and non-persistent decode registries
 * (kernels/common/attention_dense_decode.py) are one {name, tag, check} table
 * per grid in the .cpp. A decode a traits list names but that table does not
 * implement is rejected, and never gets the default order's empty name tag.
 *
 * REASON STRINGS. Every ValueError reason is the Python text byte for byte, so a
 * C caller can log the same message the Python predicate returns. The two
 * paged geometries where Python raises ZeroDivisionError instead (block_m < 32,
 * or more waves than block_n rows) return the fixed text "division by zero";
 * CPython's own text varies by version, so those are compared by verdict only.
 * A reason longer than the caller's buffer ends in "...".
 *
 * SPEC PRECONDITION. Python cannot reach a property on a spec __post_init__
 * refused to build. In C the accessors (num_waves, num_queries_per_kv, both
 * resolved_*_decode), name_parts_base and kernel_name require a spec that has
 * passed rocke_attention_dense_spec_preflight with the same traits and
 * lookahead. They do not crash on an unvalidated spec with a zero divisor or a
 * NULL argument (the decode accessors return NULL, the other two
 * ROCKE_ERR_VALUE), but their answers are only Python's on a valid spec.
 *
 * Not ported: attention_dense_cache_key (a Python launcher-cache key over
 * dataclass fields and the spec's class object; it has no C caller).
 *
 * Pure host code: nothing here builds IR.
 */
#ifndef ROCKE_INSTANCE_ATTENTION_DENSE_SPEC_H
#define ROCKE_INSTANCE_ATTENTION_DENSE_SPEC_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h" /* rocke_status_t */

#ifdef __cplusplus
extern "C" {
#endif

/* DENSE_TILE_GEOMETRIES: name -> (block_m, block_n), in Python dict order. */
typedef struct rocke_dense_tile_geometry
{
    const char* name;
    int block_m;
    int block_n;
} rocke_dense_tile_geometry_t;

#define ROCKE_DENSE_TILE_GEOMETRY_COUNT 2
extern const rocke_dense_tile_geometry_t
    ROCKE_DENSE_TILE_GEOMETRIES[ROCKE_DENSE_TILE_GEOMETRY_COUNT];
/* DEFAULT_DENSE_TILE_GEOMETRY = DENSE_TILE_GEOMETRIES["default"]. */
#define ROCKE_DENSE_DEFAULT_BLOCK_M 256
#define ROCKE_DENSE_DEFAULT_BLOCK_N 64

/* INT32_LIMIT = 2**31: signed 32-bit ceiling for tensor extents. */
#define ROCKE_ATTENTION_DENSE_INT32_LIMIT 2147483648LL

/* Reason / message buffers: every reject reason fits. */
#define ROCKE_ATTENTION_DENSE_REASON_CAP 512
#define ROCKE_ATTENTION_DENSE_NAME_CAP 512

/* AttentionDenseSpec, field for field in Python declaration order. dtype,
 * persist_decode, and nonpersist_decode are caller-owned strings (not copied). */
typedef struct rocke_attention_dense_spec
{
    /* Problem shape and semantics. */
    int batch;
    int seqlen_q;
    int seqlen_kv;
    int num_query_heads;
    int num_kv_heads;
    int head_size;
    bool causal; /* True */
    const char* dtype; /* "bf16" */
    int sliding_window; /* 0 */
    bool ragged; /* False */
    bool varlen; /* False */

    /* Geometry and common implementation policy. */
    int block_m; /* 256 */
    int block_n; /* 64 */
    int waves_per_eu; /* 2 */
    int lds_k_group_pad; /* 8 */
    bool persistent; /* False */
    int num_persistent; /* 256 */
    bool interleave; /* False */
    const char* persist_decode; /* "auto" */
    bool lazy_rescale; /* True */

    /* Problem modes implemented by a subset of architectures. */
    bool paged; /* False */
    int block_size; /* 0 */
    int num_kv_blocks; /* 0 */
    bool use_sinks; /* False */
    bool causal_bottom_right; /* False (kw_only) */
    const char* nonpersist_decode; /* "auto" */
    int chiplet_num_xcds; /* 8 */
} rocke_attention_dense_spec_t;

/* The dataclass defaults. The six shape fields have no Python default and are
 * left 0; the caller sets them. */
rocke_attention_dense_spec_t rocke_attention_dense_spec_default(void);

/* The subclass answers __post_init__ consults. */
typedef struct rocke_attention_dense_traits
{
    /* type(spec).__name__, used by preflight's "invalid <type>: ..." reason. */
    const char* type_name;
    /* _supports_noncausal_window() */
    bool supports_noncausal_window;
    /* _supports_noncausal_cross_length_ragged() */
    bool supports_noncausal_cross_length_ragged;
    /* supported_persist_decodes(), sorted ascending (Python's reject order). */
    const char* const* persist_decodes;
    size_t num_persist_decodes;
    /* supported_nonpersist_decodes(), sorted ascending. */
    const char* const* nonpersist_decodes;
    size_t num_nonpersist_decodes;
} rocke_attention_dense_traits_t;

/* Base support: all registered persistent and non-persistent decodes, with no
 * noncausal-window or cross-length-ragged support. type_name is
 * "AttentionDenseSpec". */
rocke_attention_dense_traits_t rocke_attention_dense_traits_base(void);

/* AttentionDenseSpec.__post_init__. Returns true when the spec is valid. On
 * false, writes the reject reason (see REASON STRINGS) NUL-terminated into
 * reason (capacity reason_cap), if non-NULL. lookahead: see SUBCLASS HOOKS. */
bool rocke_attention_dense_spec_validate(const rocke_attention_dense_spec_t* spec,
                                         const rocke_attention_dense_traits_t* traits,
                                         bool lookahead,
                                         char* reason,
                                         size_t reason_cap);

/* @property num_waves: block_m // 32. See SPEC PRECONDITION. */
int rocke_attention_dense_spec_num_waves(const rocke_attention_dense_spec_t* spec);
/* @property num_queries_per_kv: num_query_heads // num_kv_heads. See SPEC
 * PRECONDITION: a validated spec has positive head counts, so the int fits. */
int rocke_attention_dense_spec_num_queries_per_kv(const rocke_attention_dense_spec_t* spec);
/* @property resolved_persist_decode: explicit value, or the auto policy using
 * causal balance, chiplet locality, and persistent work thresholds.
 * lookahead: see SUBCLASS HOOKS. NULL on an unusable spec (SPEC PRECONDITION). */
const char*
    rocke_attention_dense_spec_resolved_persist_decode(const rocke_attention_dense_spec_t* spec,
                                                       bool lookahead);
/* @property resolved_nonpersist_decode: explicit value, or the auto policy for
 * the one-CTA-per-work-item grid. lookahead and NULL: as above. */
const char*
    rocke_attention_dense_spec_resolved_nonpersist_decode(const rocke_attention_dense_spec_t* spec,
                                                          bool lookahead);

/* The _*_name_parts hooks of kernel_name(), filled by the concrete spec. */
#define ROCKE_ATTENTION_DENSE_MAX_NAME_PARTS 8
#define ROCKE_ATTENTION_DENSE_NAME_PART_CAP 32
typedef struct rocke_attention_dense_name_parts
{
    /* _head_name_parts(): hq{num_query_heads}, kv{num_kv_heads} when true. */
    bool head_parts;
    /* _shape_name_parts(): sq{seqlen_q}, sk{seqlen_kv} when true. */
    bool shape_parts;
    /* _layout_name_parts(), _band_name_parts(), _algorithm_name_parts(). */
    char layout[ROCKE_ATTENTION_DENSE_MAX_NAME_PARTS][ROCKE_ATTENTION_DENSE_NAME_PART_CAP];
    int num_layout;
    char band[ROCKE_ATTENTION_DENSE_MAX_NAME_PARTS][ROCKE_ATTENTION_DENSE_NAME_PART_CAP];
    int num_band;
    char algorithm[ROCKE_ATTENTION_DENSE_MAX_NAME_PARTS][ROCKE_ATTENTION_DENSE_NAME_PART_CAP];
    int num_algorithm;
    /* _persist_decode_name_part(); "" for none. */
    char persist_decode[ROCKE_ATTENTION_DENSE_NAME_PART_CAP];
    /* Resolved non-persistent decode tag; "" for qb_minor. */
    char nonpersist_decode[ROCKE_ATTENTION_DENSE_NAME_PART_CAP];
} rocke_attention_dense_name_parts_t;

/* The base-class name parts: head and shape tokens, no layout or band parts,
 * the lazy-rescale algorithm token, and the decode tags resolved with
 * lookahead. Returns ROCKE_ERR_VALUE when the active grid's decode is not a
 * registered one (it has no tag, and "" would alias the default order). */
rocke_status_t rocke_attention_dense_name_parts_base(const rocke_attention_dense_spec_t* spec,
                                                     bool lookahead,
                                                     rocke_attention_dense_name_parts_t* out);

/* AttentionDenseSpec.kernel_name() with the given hooks, NUL-terminated into
 * out (capacity out_cap). Returns ROCKE_OK, or ROCKE_ERR_VALUE when an argument
 * is NULL, a hook count is outside [0, ROCKE_ATTENTION_DENSE_MAX_NAME_PARTS], a
 * part does not fit its slot, or the buffer is too small: the name is the
 * kernel's identity, so it is never silently shortened. */
rocke_status_t rocke_attention_dense_kernel_name(const rocke_attention_dense_spec_t* spec,
                                                 const rocke_attention_dense_name_parts_t* parts,
                                                 char* out,
                                                 size_t out_cap);

/* An arch's validator for its OWN fields (the part of its __post_init__ after
 * super().__post_init__()), for preflight to run after the base checks. Same
 * reason contract as rocke_attention_dense_spec_validate; ctx is whatever the
 * arch needs (typically its full spec struct). */
typedef bool (*rocke_attention_dense_validate_fn)(const void* ctx, char* reason, size_t reason_cap);

/* check_dense_spec_preflight(spec) -> (ok, reason): re-run the concrete type's
 * validators -- rocke_attention_dense_spec_validate(spec, traits, lookahead),
 * then arch_validate(ctx) if non-NULL -- then positive extents, block_n
 * dividing the query tile, and the 32-bit extent bounds, in the Python order.
 * The base checks always run on `spec` itself, so no ctx can hide them. The
 * type name comes from traits->type_name. On false, writes the Python reason
 * into reason (if non-NULL). */
bool rocke_attention_dense_spec_preflight(const rocke_attention_dense_spec_t* spec,
                                          const rocke_attention_dense_traits_t* traits,
                                          bool lookahead,
                                          rocke_attention_dense_validate_fn arch_validate,
                                          const void* ctx,
                                          char* reason,
                                          size_t reason_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_DENSE_SPEC_H */
