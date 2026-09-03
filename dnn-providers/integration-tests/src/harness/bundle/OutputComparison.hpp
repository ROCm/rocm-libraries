// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace hipdnn_integration_tests::bundle
{

using OutputTensors
    = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

/// One output tensor that did not match, with the diff already formatted.
///
/// Returned rather than reported so the comparison owns no gtest state: the harness
/// turns each of these into one failure, and a test can call the comparison directly
/// and read the answer.
struct TensorMismatch
{
    int64_t uid = 0;
    std::string label; ///< the tensor's name, or "uid=N" when it has none
    std::string report; ///< formatted header plus per-element diff, ready to print
};

/// Where the expected values for one output uid come from — golden data on the
/// bundle, or a reference executor's own output buffers.
using ExpectedTensorLookup = std::function<hipdnn_data_sdk::utilities::ITensor&(int64_t uid)>;

/// The tolerances one tensor is compared at. Resolved per data type, and overridable
/// per test from the TOML config.
struct ComparisonTolerance
{
    float atol = 0.0f;
    float rtol = 0.0f;
};

/// Compare one tensor. Returns nullopt when it matched.
///
/// Pure: no gtest, no config lookups, no harness state. Everything it needs to
/// describe a failure is an argument.
std::optional<TensorMismatch>
    compareTensor(int64_t uid,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                  hipdnn_data_sdk::utilities::ITensor& expected,
                  hipdnn_data_sdk::utilities::ITensor& actual,
                  ComparisonTolerance tolerance,
                  const std::string& contextLine);

/// Compare every uid in `outputUids`, and keep going after the first mismatch: one
/// failing test should name every tensor that drifted, not just the lowest uid.
///
/// `toleranceFor` resolves the tolerance per data type, so the caller owns the TOML
/// override and this stays free of TestConfig.
std::vector<TensorMismatch> compareOutputs(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
    const std::vector<int64_t>& outputUids,
    OutputTensors& actual,
    const ExpectedTensorLookup& expectedFor,
    const std::function<ComparisonTolerance(hipdnn_flatbuffers_sdk::data_objects::DataType)>&
        toleranceFor,
    const std::string& contextLine);

/// The tensor's name, or "uid=N" when the graph did not give it one.
std::string tensorLabel(int64_t uid,
                        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs);

} // namespace hipdnn_integration_tests::bundle
