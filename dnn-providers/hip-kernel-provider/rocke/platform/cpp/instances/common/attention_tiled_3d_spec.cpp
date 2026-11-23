// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * attention_tiled_3d_spec.cpp -- the field defaults of the tiled split-KV 3D
 * attention spec PODs shared by the gfx942 and gfx950 ports (see
 * rocke/instance_attention_tiled_3d_spec.h).
 */
#include "rocke/instance_attention_tiled_3d_spec.h"

rocke_unified_attention_3d_tiled_spec_t rocke_unified_attention_3d_tiled_spec_default(void)
{
    /* Required fields stay zero/NULL (the caller sets them). Defaulted fields: */
    rocke_unified_attention_3d_tiled_spec_t s{};
    s.use_alibi = false;
    s.use_qq_bias = false;
    s.num_seqs = 0;
    s.has_waves_per_eu = false;
    s.waves_per_eu = 0;
    s.kv_storage_dtype = nullptr;
    s.has_tile_size_override = false;
    s.tile_size_override = 0;
    s.use_invariant_hoist = false;
    s.use_wide_kv_load = false;
    s.use_i64_kv_addr = false;
    s.use_decode_grid = false;
    s.kv_cache_policy = "all";
    s.use_kv_lds_swizzle = false;
    s.use_wave_page_id_load = false;
    s.emit_lse = false;
    return s;
}

rocke_unified_attention_reduce_tiled_spec_t rocke_unified_attention_reduce_tiled_spec_default(void)
{
    rocke_unified_attention_reduce_tiled_spec_t s{};
    s.has_waves_per_eu = false;
    s.waves_per_eu = 0;
    s.emit_lse = false;
    return s;
}
