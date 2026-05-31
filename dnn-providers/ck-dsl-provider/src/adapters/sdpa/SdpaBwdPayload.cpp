// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaBwdPayload.hpp"

namespace py = pybind11;

namespace ck_dsl_provider {

py::dict sdpaBwdSpecToPayload(const SdpaBwdSpec& spec) {
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

    // Deliberately NOT emitted: the stride_* scalars and the scale_*
    // values. They are launch-time kernel arguments carried on the spec
    // for the plan builder, not codegen inputs -- the compiled kernel and
    // its grid are identical regardless of stride/scale.
    return d;
}

py::dict sdpaLsePrepSpecToPayload(const SdpaBwdSpec& spec) {
    py::dict d;
    // The LSE-prep kernel only depends on batch, the query-head count,
    // and the query sequence length.
    d["batch"] = spec.problem.B;
    d["num_query_heads"] = spec.problem.Hq;
    d["seqlen_q"] = spec.problem.Sq;
    return d;
}

}  // namespace ck_dsl_provider
