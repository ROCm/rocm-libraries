// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ConvImplicitGemmPayload.hpp"

namespace ck_dsl_provider {

namespace {

PayloadValue convProblemToPayload(const ConvProblem& p) {
    PayloadDict d;
    d.emplace_back("N", PayloadValue::ofInt(p.N));
    d.emplace_back("Hi", PayloadValue::ofInt(p.Hi));
    d.emplace_back("Wi", PayloadValue::ofInt(p.Wi));
    d.emplace_back("C", PayloadValue::ofInt(p.C));
    d.emplace_back("K", PayloadValue::ofInt(p.K));
    d.emplace_back("R", PayloadValue::ofInt(p.R));
    d.emplace_back("S", PayloadValue::ofInt(p.S));
    d.emplace_back("sH", PayloadValue::ofInt(p.sH));
    d.emplace_back("sW", PayloadValue::ofInt(p.sW));
    d.emplace_back("pH", PayloadValue::ofInt(p.pH));
    d.emplace_back("pW", PayloadValue::ofInt(p.pW));
    d.emplace_back("dH", PayloadValue::ofInt(p.dH));
    d.emplace_back("dW", PayloadValue::ofInt(p.dW));
    return PayloadValue::ofDict(std::move(d));
}

PayloadValue optionalI32ToPayload(const std::optional<std::int32_t>& v) {
    return v.has_value() ? PayloadValue::ofInt(*v) : PayloadValue::ofNone();
}

}  // namespace

PayloadDict convImplicitGemmSpecToPayload(const ConvImplicitGemmSpec& spec) {
    PayloadDict d;
    d.emplace_back("problem", convProblemToPayload(spec.problem));
    d.emplace_back("name", PayloadValue::ofStr(spec.name));

    d.emplace_back("tile_m", PayloadValue::ofInt(spec.tile_m));
    d.emplace_back("tile_n", PayloadValue::ofInt(spec.tile_n));
    d.emplace_back("tile_k", PayloadValue::ofInt(spec.tile_k));

    d.emplace_back("warp_m", PayloadValue::ofInt(spec.warp_m));
    d.emplace_back("warp_n", PayloadValue::ofInt(spec.warp_n));

    d.emplace_back("warp_tile_m", PayloadValue::ofInt(spec.warp_tile_m));
    d.emplace_back("warp_tile_n", PayloadValue::ofInt(spec.warp_tile_n));
    d.emplace_back("warp_tile_k", PayloadValue::ofInt(spec.warp_tile_k));

    d.emplace_back("wave_size", PayloadValue::ofInt(spec.wave_size));

    d.emplace_back("pipeline", PayloadValue::ofStr(spec.pipeline));
    d.emplace_back("epilogue", PayloadValue::ofStr(spec.epilogue));
    d.emplace_back("async_dma", PayloadValue::ofBool(spec.async_dma));
    d.emplace_back("unroll_k", PayloadValue::ofBool(spec.unroll_k));

    d.emplace_back("lds_k_pad", optionalI32ToPayload(spec.lds_k_pad));

    d.emplace_back("chiplet_swizzle", PayloadValue::ofBool(spec.chiplet_swizzle));
    d.emplace_back("chiplet_wgm", PayloadValue::ofInt(spec.chiplet_wgm));
    d.emplace_back("chiplet_num_xcds", PayloadValue::ofInt(spec.chiplet_num_xcds));
    d.emplace_back("chiplet_chunk_size", PayloadValue::ofInt(spec.chiplet_chunk_size));

    d.emplace_back("waves_per_eu", optionalI32ToPayload(spec.waves_per_eu));

    // Deliberately NOT emitted: ``lds_layout`` (dataclass re-derives it).
    return d;
}

}  // namespace ck_dsl_provider
