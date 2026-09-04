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

/// How one output tensor is graded.
///
/// ALLCLOSE is the default and stays the default: per-element
/// |ref - impl| <= atol + rtol*|ref|. RMS is MIOpen's aggregate relative-RMS check,
/// which normalises by the tensor's largest magnitude instead of each element's own.
/// It exists for reduction outputs whose elements can land arbitrarily close to zero
/// through cancellation — layernorm/RMSNorm backward dscale/dbias — where per-element
/// relative error is unbounded while the aggregate error is not. See ALMIOPEN-2561.
///
/// Nothing selects RMS on its own: only an engine's TOML config can, via a
/// [[validator_overrides]] entry naming the tensor.
enum class ValidatorKind
{
    ALLCLOSE,
    RMS,
};

/// How one tensor is compared. Resolved per output tensor, and overridable per test
/// from the TOML config.
struct ComparisonTolerance
{
    float atol = 0.0f;
    float rtol = 0.0f;
    ValidatorKind kind = ValidatorKind::ALLCLOSE;
    /// Relative-RMS threshold. Read only when `kind == RMS`; atol/rtol are ignored then.
    float rmsThreshold = 0.0f;

    static ComparisonTolerance allClose(float atolIn, float rtolIn)
    {
        return ComparisonTolerance{atolIn, rtolIn, ValidatorKind::ALLCLOSE, 0.0f};
    }

    static ComparisonTolerance rms(float threshold)
    {
        return ComparisonTolerance{0.0f, 0.0f, ValidatorKind::RMS, threshold};
    }
};

/// Resolves how one output tensor is compared.
///
/// `label` is the tensor's name, or "uid=N" when the graph did not give it one — the
/// same string the failure report uses, and what a TOML `tensors` glob matches against.
using ToleranceLookup
    = std::function<ComparisonTolerance(int64_t uid,
                                        const std::string& label,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType dataType)>;

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
/// `toleranceFor` resolves how each output is compared, so the caller owns the TOML
/// override and this stays free of TestConfig.
std::vector<TensorMismatch>
    compareOutputs(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
                   const std::vector<int64_t>& outputUids,
                   OutputTensors& actual,
                   const ExpectedTensorLookup& expectedFor,
                   const ToleranceLookup& toleranceFor,
                   const std::string& contextLine);

/// The tensor's name, or "uid=N" when the graph did not give it one.
std::string tensorLabel(int64_t uid,
                        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs);

} // namespace hipdnn_integration_tests::bundle
