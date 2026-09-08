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
/// reference harness builds one to run a reference. Its own translation unit so the
/// golden-data binary can link one pure function without dragging in the engine
/// harness.
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

/// The output buffers one bundle is run into, each prefilled with the sentinel
/// value so a tensor nobody wrote is visibly untouched rather than plausibly zero.
///
/// Shared for the same reason buildVariantPack() is: both harnesses allocate the
/// same buffers from the same attributes, and two copies of that would drift.
OutputTensors allocateSentinelOutputs(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    const std::vector<int64_t>& outputTensorUids);

/// Tell each tensor which side now holds the fresh data. Without it the comparison
/// reads the stale copy — silently, and in whichever direction is wrong.
void markOutputsModified(OutputTensors& outputs, bool device);

} // namespace hipdnn_integration_tests::bundle::detail
