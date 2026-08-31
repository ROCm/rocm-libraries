// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/VariantPack.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/OutputComparison.hpp"

namespace hipdnn_integration_tests::bundle::detail
{

/// Assembles the uid -> buffer map an executor is handed for one bundle.
///
/// Shared by both harnesses: the engine harness builds one to run the engine, the
/// reference harness builds one to run a reference. It used to live in the engine
/// harness's translation unit, which meant the golden-data binary could not link
/// without dragging the whole engine harness in for one pure function.
///
/// Inputs that are also outputs are skipped -- the output allocation owns that uid.
/// `useDevice` selects host or device pointers; a runtime-pass-by-value input is
/// always passed by value regardless.
VariantPack buildVariantPack(
    TensorMap& inputs,
    OutputTensors& outputs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    const std::vector<int64_t>& outputTensorUids,
    bool useDevice);

} // namespace hipdnn_integration_tests::bundle::detail
