/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_helper_rocke.instances.common.attention_unified_selectors_3d.h --
 *   C port of the tiled 3D (split-KV) selectors of
 *   library/kernels/common/attention_unified.py and of the 3D spec assembly in
 *   library/builders/common/attention_spec_builder.py.
 *
 * Every function takes the target ``arch`` explicitly, as its Python twin takes
 * ``(problem, arch)``. ``arch`` must already be canonical ("gfx942", not
 * "gfx942:sramecc+"); NULL matches no arch. A Python ``Optional[int]`` result is
 * returned as ``bool f(..., int* out)``: false is None and leaves ``*out``
 * untouched. Pure host code: no IR is emitted.
 */
#ifndef ROCKE_HELPER_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_UNIFIED_SELECTORS_3D_H
#define ROCKE_HELPER_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_UNIFIED_SELECTORS_3D_H

#include <stdbool.h>

#include "rocke/helper_helper_rocke.instances.common.attention_unified.h"
#include "rocke/instance_gfx942_attention_tiled_3d.h" /* rocke_unified_attention_3d_tiled_spec_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Python: _pre_bump_segments(problem) -- the split the segment formula gives at
 * the reference 120 CUs; the ceiling every arch's clamp bounds to. */
int rocke_unified_attn_pre_bump_segments(const rocke_unified_attention_problem_t* p);

/* Python: _gfx942_3d_decode_ctas_per_segment(problem, arch) /
 * _gfx950_3d_decode_ctas_per_segment(problem, arch) -- num_seqs * num_kv_heads
 * for a decode problem inside the measured scope of the decode split rule. */
bool rocke_unified_attn_gfx942_3d_decode_ctas_per_segment(
    const rocke_unified_attention_problem_t* p, const char* arch, int* ctas);
bool rocke_unified_attn_gfx950_3d_decode_ctas_per_segment(
    const rocke_unified_attention_problem_t* p, const char* arch, int* ctas);

/* Python: _num_segments(problem, arch). Returns 0 for a problem select_3d
 * rejects (invalid head ratio), where Python raises. */
int rocke_unified_attn_num_segments(const rocke_unified_attention_problem_t* p, const char* arch);

/* Python: _gfx942_3d_tile_size_override(problem, arch) /
 * _gfx950_3d_tile_size_override(problem, arch). */
bool rocke_unified_attn_gfx942_3d_tile_size_override(const rocke_unified_attention_problem_t* p,
                                                     const char* arch,
                                                     int* tile_size);
bool rocke_unified_attn_gfx950_3d_tile_size_override(const rocke_unified_attention_problem_t* p,
                                                     const char* arch,
                                                     int* tile_size);

/* Python: _select_3d_waves_per_eu(problem, arch). */
bool rocke_unified_attn_select_3d_waves_per_eu(const rocke_unified_attention_problem_t* p,
                                               const char* arch,
                                               int* waves_per_eu);

/* Python: _enable_gfx942_3d_wide_kv_load(problem, arch). Reads
 * HIPDNN_GFX942_3D_WKV. */
bool rocke_unified_attn_enable_gfx942_3d_wide_kv_load(const rocke_unified_attention_problem_t* p,
                                                      const char* arch);

/* Python: _enable_i64_kv_addr_3d(problem, arch) -- rocke_unified_attn_enable_i64_kv_addr,
 * except on gfx942, whose limit depends on the selected K/V loader. */
bool rocke_unified_attn_enable_i64_kv_addr_3d(const rocke_unified_attention_problem_t* p,
                                              const char* arch);

/* Python: _use_decode_grid(problem, arch). */
bool rocke_unified_attn_use_decode_grid(const rocke_unified_attention_problem_t* p,
                                        const char* arch);

/* Python: _kv_cache_policy_3d(problem, arch) -> "stream" or "all". */
const char* rocke_unified_attn_kv_cache_policy_3d(const rocke_unified_attention_problem_t* p,
                                                  const char* arch);

/* Python: _wave_page_id_load_3d(problem, arch) / _kv_lds_swizzle_3d(problem, arch). */
bool rocke_unified_attn_wave_page_id_load_3d(const rocke_unified_attention_problem_t* p,
                                             const char* arch);
bool rocke_unified_attn_kv_lds_swizzle_3d(const rocke_unified_attention_problem_t* p,
                                          const char* arch);

/* Python: attention_spec_builder._tiled_3d_spec_from_problem(problem, arch) for
 * the gfx942 / gfx950 generic path (_spec_generic_3d). gfx1250 (its own resolver,
 * no C++ gfx1250 tiled 3D) and a NULL arch return the default spec. The
 * gfx950-only Python fields use_additive_bias / causal_top_left / right_bound
 * have no slot in the C spec. The caller validates the spec
 * (rocke_unified_attention_3d_tiled_spec_validate), as Python's spec class does
 * when it is constructed. */
rocke_unified_attention_3d_tiled_spec_t
    rocke_unified_attn_tiled_3d_spec_from_problem(const rocke_unified_attention_problem_t* p,
                                                  const char* arch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_UNIFIED_SELECTORS_3D_H */
