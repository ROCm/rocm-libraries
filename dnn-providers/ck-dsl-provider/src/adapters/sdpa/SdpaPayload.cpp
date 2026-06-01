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

    // Opt-in forward stats (LSE) output. Codegen-relevant: stats-on emits
    // the 16-arg kernel that appends the LSE_out pointer; stats-off keeps
    // the byte-identical 15-arg kernel.
    d["generate_stats"] = spec.generate_stats;

    // Deliberately NOT emitted: the eight stride_* scalars and
    // scale_log2. They are launch-time kernel arguments carried on the
    // spec for the plan builder, not codegen inputs -- the compiled
    // kernel and its grid are identical regardless of stride/scale.
    return d;
}

}  // namespace ck_dsl_provider
