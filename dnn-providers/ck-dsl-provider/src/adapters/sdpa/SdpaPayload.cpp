// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaPayload.hpp"

namespace py = pybind11;

namespace ck_dsl_provider {

py::dict sdpaSpecToPayload(const SdpaSpec& spec) {
    py::dict d;
    d["batch"] = spec.problem.B;

    py::dict shape;
    shape["head_size"] = spec.problem.D;
    shape["num_query_heads"] = spec.problem.Hq;
    shape["num_kv_heads"] = spec.problem.Hkv;
    d["shape"] = shape;

    d["dtype"] = spec.dtype;
    d["mask_mode"] = spec.mask_mode;
    d["seqlen_q"] = spec.problem.Sq;
    d["seqlen_k"] = spec.problem.Skv;

    // Unified paged/varlen problem lanes. These select the marshalling
    // path and the KV layout the kernel sees; all are codegen-relevant
    // (folded into the cache key). ``block_size`` was finalised on the
    // plan-builder path (a concrete value in {16,32,64}) before this
    // emit, so the Python side always receives a usable block size.
    d["is_paged"] = spec.is_paged;
    d["block_size"] = spec.block_size;
    d["is_varlen"] = spec.is_varlen;
    d["sliding_window"] = spec.sliding_window;
    d["use_sinks"] = spec.use_sinks;

    // Chosen perf config for the unified tiled-2D kernel. The nine fields
    // map 1:1 onto the Python ``_unified_tiled_spec_from_problem``
    // keyword arguments and, downstream, onto the
    // ``UnifiedAttention2DTiledSpec`` knobs. Key names MUST match the
    // ``_SDPA_FWD_UNIFIED_KNOB_KEYS`` whitelist exactly or the strict
    // whitelist rejects the payload. ``tile_size`` / ``waves_per_eu`` of
    // 0 mean "unset" -- the Python side maps those to ``None``.
    py::dict knobs;
    knobs["num_warps"] = spec.knobs.num_warps;
    knobs["block_m_per_warp"] = spec.knobs.block_m_per_warp;
    knobs["tile_size"] = spec.knobs.tile_size;
    knobs["waves_per_eu"] = spec.knobs.waves_per_eu;
    knobs["use_mfma_32x32"] = spec.knobs.use_mfma_32x32;
    knobs["use_transposed_qk_32x32"] = spec.knobs.use_transposed_qk_32x32;
    knobs["use_register_pv"] = spec.knobs.use_register_pv;
    knobs["use_early_v_schedule"] = spec.knobs.use_early_v_schedule;
    knobs["use_fast_paged_kv_desc"] = spec.knobs.use_fast_paged_kv_desc;
    d["knobs"] = knobs;

    // Deliberately NOT emitted: the eight stride_* scalars, scale_log2,
    // and the k/v/out scale + softcap floats. They are launch-time kernel
    // arguments (the 18-slot arg buffer), not codegen inputs -- the
    // compiled kernel and its grid are identical regardless of their
    // values. ``generate_stats`` is also absent: the unified paged kernel
    // emits no LSE (the adapter gate rejects any stats request), so its
    // 18-arg ABI has no LSE_out slot.
    return d;
}

}  // namespace ck_dsl_provider
