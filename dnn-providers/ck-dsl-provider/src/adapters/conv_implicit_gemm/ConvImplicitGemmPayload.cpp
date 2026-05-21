// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ConvImplicitGemmPayload.hpp"

namespace py = pybind11;

namespace ck_dsl_provider {

namespace {

py::dict convProblemToPayload(const ConvProblem& p) {
    py::dict d;
    d["N"] = p.N;
    d["Hi"] = p.Hi;
    d["Wi"] = p.Wi;
    d["C"] = p.C;
    d["K"] = p.K;
    d["R"] = p.R;
    d["S"] = p.S;
    d["sH"] = p.sH;
    d["sW"] = p.sW;
    d["pH"] = p.pH;
    d["pW"] = p.pW;
    d["dH"] = p.dH;
    d["dW"] = p.dW;
    return d;
}

py::object optionalI32ToPayload(const std::optional<std::int32_t>& v) {
    return v.has_value() ? py::cast(*v) : py::none();
}

}  // namespace

py::dict convImplicitGemmSpecToPayload(const ConvImplicitGemmSpec& spec) {
    py::dict d;
    d["problem"] = convProblemToPayload(spec.problem);
    d["name"] = spec.name;

    d["tile_m"] = spec.tile_m;
    d["tile_n"] = spec.tile_n;
    d["tile_k"] = spec.tile_k;

    d["warp_m"] = spec.warp_m;
    d["warp_n"] = spec.warp_n;

    d["warp_tile_m"] = spec.warp_tile_m;
    d["warp_tile_n"] = spec.warp_tile_n;
    d["warp_tile_k"] = spec.warp_tile_k;

    d["wave_size"] = spec.wave_size;

    d["pipeline"] = spec.pipeline;
    d["epilogue"] = spec.epilogue;
    d["async_dma"] = spec.async_dma;
    d["unroll_k"] = spec.unroll_k;

    d["lds_k_pad"] = optionalI32ToPayload(spec.lds_k_pad);

    d["chiplet_swizzle"] = spec.chiplet_swizzle;
    d["chiplet_wgm"] = spec.chiplet_wgm;
    d["chiplet_num_xcds"] = spec.chiplet_num_xcds;
    d["chiplet_chunk_size"] = spec.chiplet_chunk_size;

    d["waves_per_eu"] = optionalI32ToPayload(spec.waves_per_eu);

    // Deliberately NOT emitted: ``lds_layout``. The dataclass default
    // is None and the Python ``effective_lds_layout()`` re-derives it
    // from async_dma / lds_k_pad / tile_k -- letting the dataclass
    // own that logic keeps the C++ side free of LdsLayout knowledge
    // until we have a reason to expose it through the adapter.
    return d;
}

}  // namespace ck_dsl_provider
