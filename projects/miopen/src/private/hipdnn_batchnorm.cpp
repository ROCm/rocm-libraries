// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Forwarded batch-norm inference: read the MIOpen descriptors through the public
// _impl getters, decline what the hipDNN node cannot express, then hand the
// problem to RunCachedGraph with the function that builds its graph.

#include "hipdnn_batchnorm.hpp"
#include "hipdnn_graph.hpp"
#include "hipdnn_tensor.hpp"
#include "hipdnn_types.hpp"

#include <hipdnn_frontend.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>

namespace miopen {
namespace wrapper {
namespace hipdnn {

namespace {

namespace fe = hipdnn_frontend;

constexpr int64_t kUidX        = 1;
constexpr int64_t kUidY        = 2;
constexpr int64_t kUidScale    = 3;
constexpr int64_t kUidBias     = 4;
constexpr int64_t kUidMean     = 5;
constexpr int64_t kUidVariance = 6;
constexpr int64_t kUidEpsilon  = 7;

// The legacy entry point describes scale, bias, mean and variance with one
// descriptor, so the problem holds it once.
struct BatchnormProblem
{
    TensorInfo x;
    TensorInfo y;
    TensorInfo params;
    double epsilon = 0.0;
};

// MIOpen reads alpha and beta as float for every tensor type. Exact comparison:
// the graph has no output scaling, so only a literal 1 and 0 mean "none".
bool IsDefaultScaling(const void* alpha, const void* beta)
{
    return *static_cast<const float*>(alpha) == 1.0f && *static_cast<const float*>(beta) == 0.0f;
}

PlanKey MakePlanKey(const BatchnormProblem& problem)
{
    PlanKey key;
    std::vector<int64_t>& out = key.problem;
    out.push_back(static_cast<int64_t>(GraphKind::BatchnormInference));
    AppendTensor(out, problem.x);
    AppendTensor(out, problem.y);
    AppendTensor(out, problem.params);
    // Epsilon is baked into the graph, so a different value is a different plan. The bit
    // pattern keys it exactly, where a rounded integer would merge nearby values.
    out.push_back(std::bit_cast<int64_t>(problem.epsilon));
    return key;
}

bool PopulateGraph(const BatchnormProblem& problem, fe::graph::Graph& graph)
{
    fe::DataType ioType{};
    fe::DataType yType{};
    fe::DataType paramType{};
    fe::DataType computeType{};
    if(!ToHipdnnDataType(problem.x.dataType, ioType) ||
       !ToHipdnnDataType(problem.y.dataType, yType) ||
       !ToHipdnnDataType(problem.params.dataType, paramType) ||
       !ComputeTypeFor(problem.x.dataType, computeType))
        return false;

    graph.set_io_data_type(ioType)
        .set_compute_data_type(computeType)
        .set_intermediate_data_type(computeType);
    // As for convolution: an engine that rounds its inputs to a narrower type first would
    // not compute what MIOpen computes.
    graph.deselect_numeric_notes({fe::NumericalNote::DOWN_CONVERT_INPUTS});

    auto x        = MakeTensor(problem.x, ioType, kUidX);
    auto scale    = MakeTensor(problem.params, paramType, kUidScale);
    auto bias     = MakeTensor(problem.params, paramType, kUidBias);
    auto mean     = MakeTensor(problem.params, paramType, kUidMean);
    auto variance = MakeTensor(problem.params, paramType, kUidVariance);

    // A compile-time constant rather than a runtime scalar: a layer's epsilon does not change
    // between calls, so baking it costs one plan per value, needs no host pointer in every
    // call's variant pack, and works with every provider version. Kept as a double, the type
    // MIOpen receives, so no rounding happens on the way.
    auto epsilon = std::make_shared<fe::graph::TensorAttributes>(problem.epsilon);
    epsilon->set_uid(kUidEpsilon);

    auto y = graph.batchnorm_inference_variance_ext(
        x,
        mean,
        variance,
        scale,
        bias,
        epsilon,
        fe::graph::BatchnormInferenceAttributesVarianceExt());
    if(y == nullptr)
        return false;
    y->set_dim(problem.y.dims)
        .set_stride(problem.y.strides)
        .set_data_type(yType)
        .set_alignment(problem.y.alignment)
        .set_uid(kUidY)
        .set_output(true);
    return true;
}

} // namespace

miopenStatus_t BatchNormalizationForwardInference(miopenHandle_t handle,
                                                  miopenBatchNormMode_t bn_mode,
                                                  void* alpha,
                                                  void* beta,
                                                  miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  miopenTensorDescriptor_t yDesc,
                                                  void* y,
                                                  miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                                  void* bnScale,
                                                  void* bnBias,
                                                  void* estimatedMean,
                                                  void* estimatedVariance,
                                                  double epsilon)
{
    if(!IsAvailable())
        return RecordFailure(miopenStatusInternalError, "hipDNN forwarding is unavailable");

    // Without running statistics MIOpen normalizes by the batch's own mean and variance,
    // which is a training graph, not this one.
    if(estimatedMean == nullptr || estimatedVariance == nullptr)
        return RecordFailure(miopenStatusUnsupportedOp,
                             "batch-norm inference without a running mean and variance (MIOpen "
                             "then uses batch statistics) is not forwarded to hipDNN");
    if(x == nullptr || y == nullptr || bnScale == nullptr || bnBias == nullptr)
        return RecordFailure(miopenStatusBadParm, "x, y, bnScale and bnBias must not be null");

    // MIOpen returns BadParm for alpha other than 1 or beta other than 0. It dereferences a
    // null alpha or beta, so there is no MIOpen behaviour to match for those; they fail alike.
    if(alpha == nullptr || beta == nullptr || !IsDefaultScaling(alpha, beta))
        return RecordFailure(miopenStatusBadParm,
                             "hipDNN batch-norm inference supports only alpha=1, beta=0");

    // hipDNN's node takes per-channel parameters; per-activation mode has one per element.
    if(bn_mode != miopenBNSpatial)
        return RecordFailure(miopenStatusUnsupportedOp,
                             "per-activation batch norm (miopenBNPerActivation) is not forwarded "
                             "to hipDNN");

    BatchnormProblem problem;
    problem.epsilon = epsilon;
    if(ReadTensor(xDesc, problem.x) != miopenStatusSuccess ||
       ReadTensor(yDesc, problem.y) != miopenStatusSuccess ||
       ReadTensor(bnScaleBiasMeanVarDesc, problem.params) != miopenStatusSuccess)
        return RecordFailure(miopenStatusBadParm, "could not read the MIOpen descriptors");

    // The parameter checks MIOpen makes, with its status.
    if(problem.x.dims.size() < 3 || problem.y.dims.size() != problem.x.dims.size() ||
       problem.params.dims.size() != problem.x.dims.size())
        return RecordFailure(miopenStatusBadParm,
                             "x, y and the parameter tensor must have the same rank, at least 3");
    if(problem.x.dataType != problem.y.dataType)
        return RecordFailure(miopenStatusBadParm, "x and y must have the same data type");

    // A cast type is ignored, as MIOpen's batch norm ignores it. A vector packed into each
    // element cannot be described by dims and strides.
    for(const TensorInfo* tensor : {&problem.x, &problem.y, &problem.params})
    {
        if(tensor->vectorLength != 1)
            return RecordFailure(miopenStatusUnsupportedOp,
                                 "vectorized tensor layouts (NCHWc4, NCHWc8, CHWNc4, CHWNc8) are "
                                 "not forwarded to hipDNN");
    }

    // One TensorInfo describes the four per-channel tensors, so it declares the alignment all
    // four pointers have.
    problem.x.alignment      = PointerAlignment(x);
    problem.y.alignment      = PointerAlignment(y);
    problem.params.alignment = std::min({PointerAlignment(bnScale),
                                         PointerAlignment(bnBias),
                                         PointerAlignment(estimatedMean),
                                         PointerAlignment(estimatedVariance)});
    VariantPack variantPack{
        {kUidX, const_cast<void*>(x)},
        {kUidY, y},
        {kUidScale, bnScale},
        {kUidBias, bnBias},
        {kUidMean, estimatedMean},
        {kUidVariance, estimatedVariance},
    };
    return RunCachedGraph(
        handle,
        MakePlanKey(problem),
        [&problem](fe::graph::Graph& graph) { return PopulateGraph(problem, graph); },
        variantPack,
        nullptr,
        0);
}

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
