// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The thirteen forwarded batchnorm entry points reduce to three shared paths:
// training, inference and backward. Each reads the MIOpen descriptors through
// the public _impl getters, fails the calls MIOpen itself rejects with MIOpen's
// status and in MIOpen's order, declines the combinations hipDNN cannot express,
// then hands the problem to RunCachedGraph with the function that builds its
// graph.

#include "hipdnn_batchnorm.hpp"
#include "hipdnn_graph.hpp"
#include "hipdnn_tensor.hpp"
#include "hipdnn_types.hpp"

#include "miopen_impl.h"

#include <hipdnn_frontend.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace miopen {
namespace wrapper {
namespace hipdnn {

namespace {

namespace fe = hipdnn_frontend;

// Any distinct values would do; hipDNN only needs them to match between the
// tensor attributes and the variant pack.
constexpr int64_t kUidX           = 1;
constexpr int64_t kUidY           = 2;
constexpr int64_t kUidScale       = 3;
constexpr int64_t kUidBias        = 4;
constexpr int64_t kUidSavedMean   = 5;
constexpr int64_t kUidSavedInvVar = 6;
constexpr int64_t kUidPrevRunMean = 7;
constexpr int64_t kUidPrevRunVar  = 8;
constexpr int64_t kUidNextRunMean = 9;
constexpr int64_t kUidNextRunVar  = 10;
constexpr int64_t kUidEstMean     = 11;
constexpr int64_t kUidEstVar      = 12; // the variance or the inverse variance
constexpr int64_t kUidDy          = 13;
constexpr int64_t kUidDx          = 14;
constexpr int64_t kUidDscale      = 15;
constexpr int64_t kUidDbias       = 16;

// Positions in BnProblem::tensors. Every kind uses the same order, so the
// checks and the graph builders need not know which entry point they serve.
constexpr size_t kIdxX     = 0;
constexpr size_t kIdxY     = 1; // dy for backward
constexpr size_t kIdxScale = 2;
constexpr size_t kIdxBias  = 3;
constexpr size_t kIdxMean  = 4; // saved, running or estimated mean, by kind
constexpr size_t kIdxVar   = 5; // the matching variance or inverse variance
constexpr size_t kIdxDx    = 6; // backward only

// MIOpen and the hipDNN reference both default to this epsilon, and the MIOpen
// provider uses it whenever backward has to recompute the statistics.
constexpr double kDefaultEpsilon = 1e-5;

struct BnProblem
{
    GraphKind kind = GraphKind::BnFwdTraining;
    std::vector<TensorInfo> tensors;
    miopenActivationMode_t act   = miopenActivationPASTHRU;
    double actAlpha              = 0.0; // CLIPPEDRELU's ceiling, or CLAMP's floor
    double actBeta               = 0.0; // CLAMP's ceiling
    miopenBatchNormMode_t bnMode = miopenBNSpatial;
    double epsilon               = 0.0; // 0 for the kinds whose graph takes none
    double momentum              = 0.0;
    bool hasRunningStats         = false;
    bool hasSavedStats           = false;
    bool benchmark               = false; // tune, as miopenTuningPolicySearch does
};

// MIOpen's packed test: the elements exactly fill the span the strides reach.
// It is computed in the same unsigned 64-bit arithmetic MIOpen uses, so the
// two agree even where the products wrap.
bool IsPacked(const TensorInfo& info)
{
    const auto vector = static_cast<uint64_t>(info.vectorLength);
    uint64_t count    = 1;
    uint64_t span     = 0;
    for(size_t i = 0; i < info.dims.size(); ++i)
    {
        if(info.dims[i] <= 0)
            return false;
        count *= static_cast<uint64_t>(info.dims[i]);
        span += static_cast<uint64_t>(info.dims[i] - 1) * static_cast<uint64_t>(info.strides[i]);
    }
    return vector * count == vector + span;
}

// MIOpen's own shape check: each statistic needs more than one value. Every dim
// is at least 1 once the first test passes, so a product exceeds 1 exactly when
// one of its factors does, and nothing needs multiplying.
bool ValidateShape(const TensorInfo& x, miopenBatchNormMode_t mode)
{
    if(std::any_of(x.dims.begin(), x.dims.end(), [](int64_t dim) { return dim <= 0; }))
        return false;
    if(x.dims[0] > 1)
        return true;
    if(mode == miopenBNPerActivation)
        return false;
    return std::any_of(x.dims.begin() + 2, x.dims.end(), [](int64_t dim) { return dim > 1; });
}

// MIOpen rank-checks y (or dy), scale, bias and the two statistics against x.
// It does not rank-check dx, so neither does this.
bool SameRank(const std::vector<TensorInfo>& tensors)
{
    const size_t rank = tensors[kIdxX].dims.size();
    return std::all_of(tensors.begin(),
                       tensors.begin() + kIdxVar + 1,
                       [rank](const TensorInfo& tensor) { return tensor.dims.size() == rank; });
}

// A null descriptor is how the entry points without an activation reach the
// shared paths, and it means no activation. The coefficients are kept only for
// the modes that use them, so that a stale value never splits the cache.
miopenStatus_t ReadActivation(miopenActivationDescriptor_t desc, BnProblem& problem)
{
    problem.act = miopenActivationPASTHRU;
    if(desc == nullptr)
        return miopenStatusSuccess;
    double alpha = 0.0;
    double beta  = 0.0;
    double gamma = 0.0;
    const miopenStatus_t status =
        miopenGetActivationDescriptor_impl(desc, &problem.act, &alpha, &beta, &gamma);
    if(problem.act == miopenActivationCLIPPEDRELU || problem.act == miopenActivationCLAMP)
        problem.actAlpha = alpha;
    if(problem.act == miopenActivationCLAMP)
        problem.actBeta = beta;
    return status;
}

// Reads the descriptors into problem.tensors, in the kIdx order, and then the
// activation.
const char* ReadProblem(BnProblem& problem,
                        std::initializer_list<miopenTensorDescriptor_t> descs,
                        miopenActivationDescriptor_t activDesc)
{
    problem.tensors.resize(descs.size());
    auto tensor = problem.tensors.begin();
    for(miopenTensorDescriptor_t desc : descs)
    {
        if(ReadTensor(desc, *tensor++) != miopenStatusSuccess)
            return "could not read the MIOpen descriptors";
    }
    if(ReadActivation(activDesc, problem) != miopenStatusSuccess)
        return "could not read the activation descriptor";
    return nullptr;
}

// MIOpen's BadParm checks for the forward entry points, in its order, so a
// forwarded call fails the way the native one does. A truncated x has no
// reliable dims, so the checks that read them are left to CheckSupported.
const char* CheckForwardParams(const BnProblem& problem,
                               bool training,
                               const void* x,
                               const void* y,
                               const void* scale,
                               const void* bias)
{
    const std::vector<TensorInfo>& tensors = problem.tensors;
    const TensorInfo& xInfo                = tensors[kIdxX];
    const bool checkShape                  = !xInfo.truncated;

    if(x == nullptr || y == nullptr || scale == nullptr || bias == nullptr)
        return "x, y, scale and bias must not be null";
    if(!SameRank(tensors))
        return "y, scale, bias and the statistics must have as many dimensions as x";
    if(xInfo.dataType != tensors[kIdxY].dataType)
        return "x and y must have the same data type";
    // MIOpen reshapes a 5-D x to a packed 4-D descriptor before it tests, so it
    // never rejects a 5-D x here. CheckSupported declines a non-packed one.
    if(training && checkShape && xInfo.dims.size() < 5 && !IsPacked(xInfo))
        return "x must be packed";
    if(xInfo.dims.size() < 3)
        return "batchnorm needs at least 3 dimensions";
    if(training && checkShape && !ValidateShape(xInfo, problem.bnMode))
        return "batchnorm needs more than one value per statistic";
    return nullptr;
}

// The backward counterpart of CheckForwardParams.
const char* CheckBackwardParams(
    const BnProblem& problem, const void* x, const void* dy, const void* scale, const void* dx)
{
    const std::vector<TensorInfo>& tensors = problem.tensors;
    const TensorInfo& xInfo                = tensors[kIdxX];

    if(x == nullptr || dy == nullptr || scale == nullptr || dx == nullptr)
        return "x, dy, scale and dx must not be null";
    if(!SameRank(tensors))
        return "dy, scale, bias and the statistics must have as many dimensions as x";
    if(tensors[kIdxDx].dataType != tensors[kIdxY].dataType)
        return "dx and dy must have the same data type";
    if(xInfo.dims.size() < 3)
        return "batchnorm needs at least 3 dimensions";
    if(!xInfo.truncated && !ValidateShape(xInfo, problem.bnMode))
        return "batchnorm needs more than one value per statistic";
    return nullptr;
}

// MIOpen reads each scale factor as a float and compares it exactly, so an
// exact float compare rejects the same values. MIOpen would dereference a null
// one, so there is no status to match, and BadParm is the honest answer.
miopenStatus_t CheckScaleFactors(const void* alpha, const void* beta, miopenStatus_t mismatchStatus)
{
    if(alpha == nullptr || beta == nullptr)
        return RecordFailure(miopenStatusBadParm, "batchnorm alpha and beta must not be null");
    if(!ScalarEquals(alpha, miopenFloat, 1.0) || !ScalarEquals(beta, miopenFloat, 0.0))
        return RecordFailure(mismatchStatus, "batchnorm supports only alpha=1 and beta=0");
    return miopenStatusSuccess;
}

// The provider runs MIOpen on a handle of its own, so a tuning policy set on the
// caller's handle never reaches it. The spatial training and backward solvers
// read that policy (Search tunes, DbClean drops the perf-db entry), so the
// forwarded call has to carry it over. The provider sets None, or Search when
// the plan has its benchmarking knob on, so Search is carried over as that knob.
// It cannot express the other policies, which are declined until hipDNN can
// carry the whole policy to the provider (ROCm/rocm-libraries#14372);
// MIOPEN_DISABLE_HIPDNN_FOR sends these calls back to MIOpen meanwhile. The
// inference and per-activation solvers are not tunable and ignore the policy,
// so inference skips this check and per-activation runs as under None.
// MIOPEN_FIND_ENFORCE needs no check: it is process-wide, so the provider's
// handle sees it too.
miopenStatus_t CheckTuningPolicy(miopenHandle_t handle, BnProblem& problem)
{
    miopenTuningPolicy_t policy = miopenTuningPolicyNone;
    if(const miopenStatus_t status = miopenGetTuningPolicy_impl(handle, &policy);
       status != miopenStatusSuccess)
        return RecordFailure(status, "could not read the handle's tuning policy");
    if(problem.bnMode == miopenBNPerActivation)
        policy = miopenTuningPolicyNone;
    if(policy != miopenTuningPolicyNone && policy != miopenTuningPolicySearch)
        return RecordFailure(miopenStatusUnsupportedOp,
                             "hipDNN batchnorm cannot apply the handle's tuning policy; "
                             "only miopenTuningPolicyNone and miopenTuningPolicySearch "
                             "are forwarded");
    problem.benchmark = policy == miopenTuningPolicySearch;
    return miopenStatusSuccess;
}

// MIOpen's problem description knows only 2-D (rank 4) and 3-D (rank 5)
// batchnorm and throws for any other rank, which reaches the caller as
// UnknownError. MIOpen builds it right after the scale-factor checks, before
// it looks for a solver, so this check runs at the same point.
miopenStatus_t CheckSpatialRank(const BnProblem& problem)
{
    const size_t rank = problem.tensors[kIdxX].dims.size();
    if(rank == 4 || rank == 5)
        return miopenStatusSuccess;
    return RecordFailure(miopenStatusUnknownError,
                         "batchnorm supports only 4-D and 5-D tensors (Unknown spatial dim!)");
}

bool IsSolverIoType(miopenDataType_t type)
{
    return type == miopenFloat || type == miopenHalf || type == miopenBFloat16;
}

// The data types every MIOpen batchnorm solver needs: x, y (or dy) and dx all
// FLOAT, all HALF or all BFLOAT16, and FLOAT scale, bias and statistics. The
// BadParm checks have already matched y to x on forward and dx to dy on
// backward, but not x to dy.
bool HasSolverDataTypes(const std::vector<TensorInfo>& tensors)
{
    const miopenDataType_t type = tensors[kIdxX].dataType;
    if(!IsSolverIoType(type))
        return false;
    for(size_t index : {kIdxY, kIdxDx})
    {
        if(index < tensors.size() && tensors[index].dataType != type)
            return false;
    }
    return std::all_of(tensors.begin() + kIdxScale,
                       tensors.begin() + kIdxVar + 1,
                       [](const TensorInfo& tensor) { return tensor.dataType == miopenFloat; });
}

// The activations MIOpen's spatial training and inference solvers accept.
bool IsSolverActivation(miopenActivationMode_t act)
{
    return act == miopenActivationPASTHRU || act == miopenActivationRELU ||
           act == miopenActivationCLIPPEDRELU || act == miopenActivationCLAMP;
}

// When no MIOpen solver applies, MIOpen fails with NotImplemented ("No solver
// found"), so the forwarded call does too. Training has a spatial and a
// per-activation solver, and only the spatial one limits the activation.
// Inference has one solver for both modes, which limits it. Backward has a
// spatial and a per-activation solver, and the per-activation one accepts
// spatial problems with any activation, so backward never fails on one.
miopenStatus_t CheckSolverExists(const BnProblem& problem)
{
    if(!HasSolverDataTypes(problem.tensors))
        return RecordFailure(miopenStatusNotImplemented,
                             "MIOpen has no batchnorm solver for these data types");
    bool limitsActivation = false;
    switch(problem.kind)
    {
    case GraphKind::BnFwdTraining: limitsActivation = problem.bnMode == miopenBNSpatial; break;
    case GraphKind::BnFwdInference:
    case GraphKind::BnFwdInferenceInvVar: limitsActivation = true; break;
    case GraphKind::BnBackward:
    case GraphKind::BnBackwardActivation:
    case GraphKind::ConvFprop:
    case GraphKind::ConvDgrad:
    case GraphKind::ConvWgrad:
    case GraphKind::ConvBiasActivation: break;
    }
    if(limitsActivation && !IsSolverActivation(problem.act))
        return RecordFailure(miopenStatusNotImplemented,
                             std::string("MIOpen has no batchnorm solver for the ") +
                                 ActivationModeName(problem.act) + " activation");
    return miopenStatusSuccess;
}

// The checks between MIOpen's scale-factor checks and its solver: the rank,
// then whether any solver applies.
miopenStatus_t CheckProblem(const BnProblem& problem)
{
    if(const miopenStatus_t status = CheckSpatialRank(problem); status != miopenStatusSuccess)
        return status;
    return CheckSolverExists(problem);
}

// The pointwise node for the activation, or nothing for one hipDNN cannot
// express. hipDNN's relu takes clip values, and the MIOpen provider turns an
// upper clip alone back into CLIPPEDRELU and both clips into CLAMP. MIOpen's
// kernels take the coefficients as float, so narrowing them here loses nothing.
std::optional<fe::graph::PointwiseAttributes> ActivationAttributes(const BnProblem& problem,
                                                                   bool backward)
{
    fe::graph::PointwiseAttributes attributes;
    attributes.set_mode(backward ? fe::PointwiseMode::RELU_BWD : fe::PointwiseMode::RELU_FWD)
        .set_compute_data_type(fe::DataType::FLOAT);
    if(problem.act == miopenActivationRELU)
        return attributes;
    if(problem.act == miopenActivationCLIPPEDRELU)
    {
        attributes.set_relu_upper_clip(static_cast<float>(problem.actAlpha));
        return attributes;
    }
    if(problem.act == miopenActivationCLAMP)
    {
        attributes.set_relu_lower_clip(static_cast<float>(problem.actAlpha))
            .set_relu_upper_clip(static_cast<float>(problem.actBeta));
        return attributes;
    }
    return std::nullopt;
}

// What the MIOpen provider can run, whatever the kind. Empty when it can.
// CheckProblem has already limited the rank to 4 or 5.
std::string CheckSupported(const BnProblem& problem)
{
    const std::vector<TensorInfo>& tensors = problem.tensors;
    const TensorInfo& x                    = tensors[kIdxX];
    const bool perActivation               = problem.bnMode == miopenBNPerActivation;

    if(problem.bnMode != miopenBNSpatial && !perActivation)
        return "hipDNN batchnorm supports only spatial and per-activation modes";
    if(problem.act != miopenActivationPASTHRU && !ActivationAttributes(problem, false))
        return std::string("hipDNN batchnorm does not support the ") +
               ActivationModeName(problem.act) + " activation";
    // Before the packed test, so a truncated tensor is declined for the reason
    // that applies to it.
    if(const char* reason = CheckTensorLayouts(tensors))
        return reason;
    if(!std::all_of(tensors.begin(), tensors.end(), IsPacked))
        return "hipDNN batchnorm supports only packed tensors";
    // CheckProblem has limited every type to one a MIOpen solver accepts.
    for(size_t index : {kIdxY, kIdxDx})
    {
        if(index >= tensors.size())
            continue;
        if(tensors[index].dataType != x.dataType)
            return "hipDNN batchnorm needs every data tensor in the same type";
        if(tensors[index].dims != x.dims || tensors[index].strides != x.strides)
            return "hipDNN batchnorm needs every data tensor in the same shape and layout";
    }
    // Spatial mode keeps one value per channel: {1, C, 1, ...} at the rank of x.
    // Per-activation mode keeps one per element of a sample: {1, C, H, W, ...}.
    std::vector<int64_t> paramDims(x.dims.size(), 1);
    paramDims[1] = x.dims[1];
    if(perActivation)
        std::copy(x.dims.begin() + 1, x.dims.end(), paramDims.begin() + 1);
    for(size_t index = kIdxScale; index <= kIdxVar; ++index)
    {
        if(tensors[index].dims != paramDims)
            return perActivation ? "hipDNN per-activation batchnorm needs scale, bias and "
                                   "statistics shaped {1, C, H, W, ...}"
                                 : "hipDNN batchnorm needs scale, bias and statistics shaped "
                                   "{1, C, 1, ...}";
    }
    return {};
}

// Per-activation batchnorm is spatial batchnorm with every element of a sample
// as its own channel, so it runs as spatial on {N, K, 1, 1}, K being the
// elements per sample. Even the running variance agrees: both modes scale it
// by N / (N - 1). Only the descriptions change, and they ignore the strides:
// MIOpen's per-activation kernels index every packed tensor as if it were NCHW,
// whatever its layout, pairing the element at offset j of a sample with the
// parameters at offset j. CheckSupported has made every tensor packed, so this
// does the same. That is also the per-element answer when the parameters are laid
// out like x, as with NHWC x and NHWC parameters. The plan key keeps the mode, so
// the two never share a plan by accident of shape.
void FoldPerActivation(BnProblem& problem)
{
    if(problem.bnMode != miopenBNPerActivation)
        return;
    const std::vector<int64_t> dims = problem.tensors[kIdxX].dims;
    int64_t elements                = 1;
    for(size_t i = 1; i < dims.size(); ++i)
        elements *= dims[i];
    for(size_t index = 0; index < problem.tensors.size(); ++index)
    {
        TensorInfo& tensor = problem.tensors[index];
        const bool isParam = index >= kIdxScale && index <= kIdxVar;
        tensor.dims        = {isParam ? 1 : dims[0], elements, 1, 1};
        tensor.strides     = {elements, 1, 1, 1};
    }
}

int64_t DoubleBits(double value)
{
    int64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

PlanKey MakePlanKey(miopenHandle_t handle, const BnProblem& problem)
{
    PlanKey key;
    key.handle                = handle;
    key.benchmark             = problem.benchmark;
    std::vector<int64_t>& out = key.problem;
    out.push_back(static_cast<int64_t>(problem.kind));
    out.push_back(static_cast<int64_t>(problem.act));
    // The clip values are constants in the plan, and zero for the modes
    // without them.
    out.push_back(DoubleBits(problem.actAlpha));
    out.push_back(DoubleBits(problem.actBeta));
    out.push_back(static_cast<int64_t>(problem.tensors.size()));
    for(const TensorInfo& tensor : problem.tensors)
        AppendTensor(out, tensor);
    out.push_back(static_cast<int64_t>(problem.bnMode));
    // Epsilon and momentum are built into the plan as constants, so a different
    // value needs a different plan.
    out.push_back(DoubleBits(problem.epsilon));
    out.push_back(problem.hasRunningStats ? 1 : 0);
    out.push_back(problem.hasSavedStats ? 1 : 0);
    // Momentum only reaches the graph with the running statistics, and leaving
    // it out otherwise keeps it from splitting the cache.
    if(problem.hasRunningStats)
        out.push_back(DoubleBits(problem.momentum));
    return key;
}

std::shared_ptr<fe::graph::TensorAttributes> MakeScalar(fe::graph::Graph& graph, double value)
{
    return graph.tensor(fe::graph::TensorAttributes(value));
}

bool MarkOutput(const std::shared_ptr<fe::graph::TensorAttributes>& tensor,
                const TensorInfo& info,
                fe::DataType dataType,
                int64_t uid)
{
    if(tensor == nullptr)
        return false;
    tensor->set_dim(info.dims)
        .set_stride(info.strides)
        .set_data_type(dataType)
        .set_uid(uid)
        .set_output(true);
    return true;
}

// Marks the batchnorm result as y. With an activation the result becomes an
// intermediate, and the activation's result takes y's place.
bool MarkResult(const BnProblem& problem,
                fe::graph::Graph& graph,
                std::shared_ptr<fe::graph::TensorAttributes> out,
                fe::DataType ioType)
{
    const TensorInfo& yInfo = problem.tensors[kIdxY];
    if(out == nullptr)
        return false;
    if(problem.act != miopenActivationPASTHRU)
    {
        const std::optional<fe::graph::PointwiseAttributes> activation =
            ActivationAttributes(problem, false);
        if(!activation)
            return false;
        out->set_dim(yInfo.dims).set_stride(yInfo.strides);
        out = graph.pointwise(out, *activation);
    }
    return MarkOutput(out, yInfo, ioType, kUidY);
}

bool PopulateTraining(const BnProblem& problem, fe::DataType ioType, fe::graph::Graph& graph)
{
    const std::vector<TensorInfo>& tensors = problem.tensors;
    auto x                                 = MakeTensor(tensors[kIdxX], ioType, kUidX);
    auto scale = MakeTensor(tensors[kIdxScale], fe::DataType::FLOAT, kUidScale);
    auto bias  = MakeTensor(tensors[kIdxBias], fe::DataType::FLOAT, kUidBias);

    fe::graph::BatchnormAttributes attributes;
    attributes.set_epsilon(MakeScalar(graph, problem.epsilon));
    if(problem.hasRunningStats)
        attributes.set_previous_running_stats(
            MakeTensor(tensors[kIdxMean], fe::DataType::FLOAT, kUidPrevRunMean),
            MakeTensor(tensors[kIdxVar], fe::DataType::FLOAT, kUidPrevRunVar),
            MakeScalar(graph, problem.momentum));

    auto [y, mean, invVariance, nextMean, nextVariance] =
        graph.batchnorm(x, scale, bias, attributes);
    if(!MarkOutput(mean, tensors[kIdxMean], fe::DataType::FLOAT, kUidSavedMean) ||
       !MarkOutput(invVariance, tensors[kIdxVar], fe::DataType::FLOAT, kUidSavedInvVar))
        return false;
    if(problem.hasRunningStats &&
       (!MarkOutput(nextMean, tensors[kIdxMean], fe::DataType::FLOAT, kUidNextRunMean) ||
        !MarkOutput(nextVariance, tensors[kIdxVar], fe::DataType::FLOAT, kUidNextRunVar)))
        return false;
    return MarkResult(problem, graph, y, ioType);
}

bool PopulateInference(const BnProblem& problem, fe::DataType ioType, fe::graph::Graph& graph)
{
    const std::vector<TensorInfo>& tensors = problem.tensors;
    auto x                                 = MakeTensor(tensors[kIdxX], ioType, kUidX);
    auto scale    = MakeTensor(tensors[kIdxScale], fe::DataType::FLOAT, kUidScale);
    auto bias     = MakeTensor(tensors[kIdxBias], fe::DataType::FLOAT, kUidBias);
    auto mean     = MakeTensor(tensors[kIdxMean], fe::DataType::FLOAT, kUidEstMean);
    auto variance = MakeTensor(tensors[kIdxVar], fe::DataType::FLOAT, kUidEstVar);

    std::shared_ptr<fe::graph::TensorAttributes> y;
    if(problem.kind == GraphKind::BnFwdInference)
    {
        fe::graph::BatchnormInferenceAttributesVarianceExt attributes;
        y = graph.batchnorm_inference_variance_ext(
            x, mean, variance, scale, bias, MakeScalar(graph, problem.epsilon), attributes);
    }
    else
    {
        fe::graph::BatchnormInferenceAttributes attributes;
        y = graph.batchnorm_inference(x, mean, variance, scale, bias, attributes);
    }
    return MarkResult(problem, graph, y, ioType);
}

bool PopulateBackward(const BnProblem& problem, fe::DataType ioType, fe::graph::Graph& graph)
{
    const std::vector<TensorInfo>& tensors = problem.tensors;
    const bool fusedActivation             = problem.kind == GraphKind::BnBackwardActivation;
    // The fused graph recomputes y from the saved statistics, so without them
    // it would drop the activation. Backward computes them first instead.
    if(fusedActivation && !problem.hasSavedStats)
        return false;
    const std::optional<fe::graph::PointwiseAttributes> activation =
        ActivationAttributes(problem, true);
    if(fusedActivation && !activation)
        return false;

    auto x     = MakeTensor(tensors[kIdxX], ioType, kUidX);
    auto dy    = MakeTensor(tensors[kIdxY], ioType, kUidDy);
    auto scale = MakeTensor(tensors[kIdxScale], fe::DataType::FLOAT, kUidScale);

    fe::graph::BatchnormBackwardAttributes attributes;
    if(problem.hasSavedStats)
    {
        auto mean        = MakeTensor(tensors[kIdxMean], fe::DataType::FLOAT, kUidSavedMean);
        auto invVariance = MakeTensor(tensors[kIdxVar], fe::DataType::FLOAT, kUidSavedInvVar);
        attributes.set_saved_mean_and_inv_variance(mean, invVariance);

        if(fusedActivation)
        {
            // The MIOpen provider recognizes exactly this shape: recompute y,
            // take the activation's gradient against it, then run backward on
            // that, with the same statistics objects feeding both batchnorm
            // nodes.
            auto bias = MakeTensor(tensors[kIdxBias], fe::DataType::FLOAT, kUidBias);
            fe::graph::BatchnormInferenceAttributes inferenceAttributes;
            auto y =
                graph.batchnorm_inference(x, mean, invVariance, scale, bias, inferenceAttributes);
            if(y == nullptr)
                return false;
            y->set_dim(tensors[kIdxX].dims).set_stride(tensors[kIdxX].strides);

            dy = graph.pointwise(dy, y, *activation);
            if(dy == nullptr)
                return false;
            dy->set_dim(tensors[kIdxY].dims).set_stride(tensors[kIdxY].strides);
        }
    }

    auto [dx, dscale, dbias] = graph.batchnorm_backward(dy, x, scale, attributes);
    return MarkOutput(dx, tensors[kIdxDx], ioType, kUidDx) &&
           MarkOutput(dscale, tensors[kIdxScale], fe::DataType::FLOAT, kUidDscale) &&
           MarkOutput(dbias, tensors[kIdxBias], fe::DataType::FLOAT, kUidDbias);
}

bool PopulateGraph(const BnProblem& problem, fe::graph::Graph& graph)
{
    fe::DataType ioType{};
    if(!ToHipdnnDataType(problem.tensors[kIdxX].dataType, ioType))
        return false;

    // The MIOpen provider computes batchnorm in FLOAT for every data type.
    graph.set_io_data_type(ioType)
        .set_compute_data_type(fe::DataType::FLOAT)
        .set_intermediate_data_type(fe::DataType::FLOAT);

    switch(problem.kind)
    {
    case GraphKind::BnFwdTraining: return PopulateTraining(problem, ioType, graph);
    case GraphKind::BnFwdInference:
    case GraphKind::BnFwdInferenceInvVar: return PopulateInference(problem, ioType, graph);
    case GraphKind::BnBackward:
    case GraphKind::BnBackwardActivation: return PopulateBackward(problem, ioType, graph);
    case GraphKind::ConvFprop:
    case GraphKind::ConvDgrad:
    case GraphKind::ConvWgrad:
    case GraphKind::ConvBiasActivation: break;
    }
    return false;
}

miopenStatus_t RunProblem(miopenHandle_t handle,
                          const BnProblem& problem,
                          VariantPack& variantPack,
                          const GraphScratch& scratch = {})
{
    return RunCachedGraph(
        handle,
        MakePlanKey(handle, problem),
        [&problem](fe::graph::Graph& graph) { return PopulateGraph(problem, graph); },
        variantPack,
        nullptr,
        0,
        scratch);
}

size_t ElementCount(const TensorInfo& info)
{
    size_t count = 1;
    for(int64_t dim : info.dims)
        count *= static_cast<size_t>(dim);
    return count;
}

// The graphs keep every statistic in FLOAT.
size_t FloatBytes(const TensorInfo& info) { return ElementCount(info) * sizeof(float); }

// The bytes of a packed data tensor. CheckProblem has limited its type to
// FLOAT, HALF or BFLOAT16.
size_t DataBytes(const TensorInfo& info)
{
    const bool twoBytes = info.dataType == miopenHalf || info.dataType == miopenBFloat16;
    return ElementCount(info) * (twoBytes ? 2 : 4);
}

// Where the next scratch buffer starts after `bytes`, keeping each one aligned
// as the handle's buffer is.
size_t AlignScratch(size_t bytes) { return (bytes + 255) / 256 * 256; }

// The training graph always produces the saved statistics. When the caller
// wants none, they go to scratch instead and are dropped.
GraphScratch SavedStatsScratch(const BnProblem& problem)
{
    const size_t varOffset = AlignScratch(FloatBytes(problem.tensors[kIdxMean]));
    return {varOffset + FloatBytes(problem.tensors[kIdxVar]),
            [varOffset](VariantPack& variantPack, void* scratch) {
                variantPack[kUidSavedMean]   = scratch;
                variantPack[kUidSavedInvVar] = static_cast<char*>(scratch) + varOffset;
            }};
}

// Shared by the five training entry points. The ones that update the running
// statistics in place pass the same pointers as prev and next.
miopenStatus_t ForwardTraining(miopenHandle_t handle,
                               miopenBatchNormMode_t bnMode,
                               const void* alpha,
                               const void* beta,
                               miopenTensorDescriptor_t xDesc,
                               const void* x,
                               miopenTensorDescriptor_t yDesc,
                               void* y,
                               miopenTensorDescriptor_t scaleDesc,
                               miopenTensorDescriptor_t biasDesc,
                               miopenTensorDescriptor_t meanDesc,
                               miopenTensorDescriptor_t varianceDesc,
                               void* scale,
                               void* bias,
                               double momentum,
                               const void* prevRunningMean,
                               const void* prevRunningVariance,
                               void* nextRunningMean,
                               void* nextRunningVariance,
                               double epsilon,
                               void* saveMean,
                               void* saveInvVariance,
                               miopenActivationDescriptor_t activDesc)
{
    if(!IsAvailable())
        return RecordFailure(miopenStatusInternalError, "hipDNN forwarding is unavailable");

    BnProblem problem;
    problem.kind     = GraphKind::BnFwdTraining;
    problem.bnMode   = bnMode;
    problem.epsilon  = epsilon;
    problem.momentum = momentum;
    if(const char* reason = ReadProblem(
           problem, {xDesc, yDesc, scaleDesc, biasDesc, meanDesc, varianceDesc}, activDesc))
        return RecordFailure(miopenStatusBadParm, reason);

    if(const char* reason = CheckForwardParams(problem, true, x, y, scale, bias))
        return RecordFailure(miopenStatusBadParm, reason);
    // MIOpen throws without a status here, which reaches the caller as
    // UnknownError.
    if(const miopenStatus_t status = CheckScaleFactors(alpha, beta, miopenStatusUnknownError);
       status != miopenStatusSuccess)
        return status;
    if(const miopenStatus_t status = CheckProblem(problem); status != miopenStatusSuccess)
        return status;

    if(const std::string reason = CheckSupported(problem); !reason.empty())
        return RecordFailure(miopenStatusUnsupportedOp, reason);
    if(const miopenStatus_t status = CheckTuningPolicy(handle, problem);
       status != miopenStatusSuccess)
        return status;
    FoldPerActivation(problem);
    // Like MIOpen, write the saved statistics only when both are given, and
    // update the running statistics only when all four are.
    const bool saveStats    = saveMean != nullptr && saveInvVariance != nullptr;
    problem.hasRunningStats = prevRunningMean != nullptr && prevRunningVariance != nullptr &&
                              nextRunningMean != nullptr && nextRunningVariance != nullptr;

    VariantPack variantPack{
        {kUidX, const_cast<void*>(x)},
        {kUidY, y},
        {kUidScale, scale},
        {kUidBias, bias},
    };
    if(saveStats)
    {
        variantPack[kUidSavedMean]   = saveMean;
        variantPack[kUidSavedInvVar] = saveInvVariance;
    }
    // An in-place update binds one buffer as both input and output. That holds
    // because the graph runs once. Under Search, MIOpen's own search re-runs the
    // kernel on these buffers, as it does when MIOpen serves the call.
    if(problem.hasRunningStats)
    {
        variantPack[kUidPrevRunMean] = const_cast<void*>(prevRunningMean);
        variantPack[kUidPrevRunVar]  = const_cast<void*>(prevRunningVariance);
        variantPack[kUidNextRunMean] = nextRunningMean;
        variantPack[kUidNextRunVar]  = nextRunningVariance;
    }
    return RunProblem(
        handle, problem, variantPack, saveStats ? GraphScratch{} : SavedStatsScratch(problem));
}

// Shared by the five inference entry points. With an epsilon the statistics
// are a mean and a variance; without one they are a mean and an inverse
// variance.
miopenStatus_t ForwardInference(miopenHandle_t handle,
                                miopenBatchNormMode_t bnMode,
                                const void* alpha,
                                const void* beta,
                                miopenTensorDescriptor_t xDesc,
                                const void* x,
                                miopenTensorDescriptor_t yDesc,
                                void* y,
                                miopenTensorDescriptor_t scaleDesc,
                                miopenTensorDescriptor_t biasDesc,
                                miopenTensorDescriptor_t meanDesc,
                                miopenTensorDescriptor_t varianceDesc,
                                void* scale,
                                void* bias,
                                void* mean,
                                void* variance,
                                std::optional<double> epsilon,
                                miopenActivationDescriptor_t activDesc)
{
    // Without the statistics MIOpen runs training instead, with no running
    // update and no saved statistics, and with the statistics' descriptors as
    // the saved ones. It makes this choice before any inference check.
    if(mean == nullptr || variance == nullptr)
        return ForwardTraining(handle,
                               bnMode,
                               alpha,
                               beta,
                               xDesc,
                               x,
                               yDesc,
                               y,
                               scaleDesc,
                               biasDesc,
                               meanDesc,
                               varianceDesc,
                               scale,
                               bias,
                               0.0,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               epsilon.value_or(kDefaultEpsilon),
                               nullptr,
                               nullptr,
                               activDesc);

    if(!IsAvailable())
        return RecordFailure(miopenStatusInternalError, "hipDNN forwarding is unavailable");

    BnProblem problem;
    problem.kind    = epsilon ? GraphKind::BnFwdInference : GraphKind::BnFwdInferenceInvVar;
    problem.bnMode  = bnMode;
    problem.epsilon = epsilon.value_or(0.0);
    if(const char* reason = ReadProblem(
           problem, {xDesc, yDesc, scaleDesc, biasDesc, meanDesc, varianceDesc}, activDesc))
        return RecordFailure(miopenStatusBadParm, reason);

    if(const char* reason = CheckForwardParams(problem, false, x, y, scale, bias))
        return RecordFailure(miopenStatusBadParm, reason);
    if(const miopenStatus_t status = CheckScaleFactors(alpha, beta, miopenStatusBadParm);
       status != miopenStatusSuccess)
        return status;
    if(const miopenStatus_t status = CheckProblem(problem); status != miopenStatusSuccess)
        return status;

    if(const std::string reason = CheckSupported(problem); !reason.empty())
        return RecordFailure(miopenStatusUnsupportedOp, reason);
    FoldPerActivation(problem);

    VariantPack variantPack{
        {kUidX, const_cast<void*>(x)},
        {kUidY, y},
        {kUidScale, scale},
        {kUidBias, bias},
        {kUidEstMean, mean},
        {kUidEstVar, variance},
    };
    return RunProblem(handle, problem, variantPack);
}

// Backward without saved statistics, when hipDNN cannot recompute them itself:
// at an epsilon other than 1e-5, or with an activation, whose graph needs them
// to recompute y. A training graph first computes them into scratch, at the
// caller's epsilon, and backward then reads them from there. Its y goes to
// scratch too and is dropped.
miopenStatus_t BackwardFromComputedStats(miopenHandle_t handle,
                                         const BnProblem& backward,
                                         double epsilon,
                                         const void* x,
                                         const void* scale,
                                         const void* bias,
                                         const VariantPack& backwardPack)
{
    const std::vector<TensorInfo>& tensors = backward.tensors;

    // Not benchmarked: MIOpen's backward computes these statistics itself, so
    // Search tunes no training solver for it.
    BnProblem training;
    training.kind    = GraphKind::BnFwdTraining;
    training.bnMode  = backward.bnMode;
    training.epsilon = epsilon;
    training.tensors = {tensors[kIdxX],
                        tensors[kIdxX],
                        tensors[kIdxScale],
                        tensors[kIdxBias],
                        tensors[kIdxMean],
                        tensors[kIdxVar]};

    BnProblem withStats     = backward;
    withStats.hasSavedStats = true;

    const size_t meanOffset   = AlignScratch(DataBytes(tensors[kIdxX]));
    const size_t varOffset    = meanOffset + AlignScratch(FloatBytes(tensors[kIdxMean]));
    const size_t scratchBytes = varOffset + FloatBytes(tensors[kIdxVar]);
    auto bindStats            = [meanOffset, varOffset](VariantPack& variantPack, void* scratch) {
        variantPack[kUidSavedMean]   = static_cast<char*>(scratch) + meanOffset;
        variantPack[kUidSavedInvVar] = static_cast<char*>(scratch) + varOffset;
    };

    std::vector<GraphStep> steps;
    // Plain backward has no bias, and its value cannot change the statistics.
    steps.push_back({MakePlanKey(handle, training),
                     [training](fe::graph::Graph& graph) { return PopulateGraph(training, graph); },
                     VariantPack{
                         {kUidX, const_cast<void*>(x)},
                         {kUidScale, const_cast<void*>(scale)},
                         {kUidBias, const_cast<void*>(bias != nullptr ? bias : scale)},
                     },
                     [bindStats](VariantPack& variantPack, void* scratch) {
                         variantPack[kUidY] = scratch;
                         bindStats(variantPack, scratch);
                     }});
    steps.push_back(
        {MakePlanKey(handle, withStats),
         [withStats](fe::graph::Graph& graph) { return PopulateGraph(withStats, graph); },
         backwardPack,
         bindStats});
    return RunCachedGraphs(handle, steps, scratchBytes);
}

// Shared by the three backward entry points. Only the fused one has a bias
// and an activation.
miopenStatus_t Backward(miopenHandle_t handle,
                        miopenBatchNormMode_t bnMode,
                        const void* alphaDataDiff,
                        const void* betaDataDiff,
                        const void* alphaParamDiff,
                        const void* betaParamDiff,
                        miopenTensorDescriptor_t xDesc,
                        const void* x,
                        miopenTensorDescriptor_t dyDesc,
                        const void* dy,
                        miopenTensorDescriptor_t dxDesc,
                        void* dx,
                        miopenTensorDescriptor_t scaleDesc,
                        miopenTensorDescriptor_t biasDesc,
                        miopenTensorDescriptor_t meanDesc,
                        miopenTensorDescriptor_t varianceDesc,
                        const void* scale,
                        const void* bias,
                        void* dscale,
                        void* dbias,
                        double epsilon,
                        const void* savedMean,
                        const void* savedInvVariance,
                        miopenActivationDescriptor_t activDesc)
{
    if(!IsAvailable())
        return RecordFailure(miopenStatusInternalError, "hipDNN forwarding is unavailable");

    BnProblem problem;
    problem.bnMode = bnMode;
    if(const char* reason =
           ReadProblem(problem,
                       {xDesc, dyDesc, scaleDesc, biasDesc, meanDesc, varianceDesc, dxDesc},
                       activDesc))
        return RecordFailure(miopenStatusBadParm, reason);
    // Without an activation the fused entry point is plain backward, and sharing
    // its kind lets the two share a plan.
    problem.kind = problem.act == miopenActivationPASTHRU ? GraphKind::BnBackward
                                                          : GraphKind::BnBackwardActivation;
    // Like MIOpen, recompute the statistics unless both saved ones are given.
    problem.hasSavedStats = savedMean != nullptr && savedInvVariance != nullptr;

    if(const char* reason = CheckBackwardParams(problem, x, dy, scale, dx))
        return RecordFailure(miopenStatusBadParm, reason);
    if(const miopenStatus_t status =
           CheckScaleFactors(alphaDataDiff, betaDataDiff, miopenStatusBadParm);
       status != miopenStatusSuccess)
        return status;
    if(const miopenStatus_t status =
           CheckScaleFactors(alphaParamDiff, betaParamDiff, miopenStatusBadParm);
       status != miopenStatusSuccess)
        return status;
    if(const miopenStatus_t status = CheckProblem(problem); status != miopenStatusSuccess)
        return status;
    // MIOpen would write through these, so a null one has no behavior to match.
    if(dscale == nullptr || dbias == nullptr)
        return RecordFailure(miopenStatusBadParm, "dscale and dbias must not be null");
    if(problem.act != miopenActivationPASTHRU && bias == nullptr)
        return RecordFailure(miopenStatusBadParm,
                             "batchnorm backward with an activation needs the bias");

    if(const std::string reason = CheckSupported(problem); !reason.empty())
        return RecordFailure(miopenStatusUnsupportedOp, reason);
    if(const miopenStatus_t status = CheckTuningPolicy(handle, problem);
       status != miopenStatusSuccess)
        return status;
    FoldPerActivation(problem);

    VariantPack variantPack{
        {kUidX, const_cast<void*>(x)},
        {kUidDy, const_cast<void*>(dy)},
        {kUidScale, const_cast<void*>(scale)},
        {kUidDx, dx},
        {kUidDscale, dscale},
        {kUidDbias, dbias},
    };
    if(problem.kind == GraphKind::BnBackwardActivation)
        variantPack[kUidBias] = const_cast<void*>(bias);
    if(problem.hasSavedStats)
    {
        variantPack[kUidSavedMean]   = const_cast<void*>(savedMean);
        variantPack[kUidSavedInvVar] = const_cast<void*>(savedInvVariance);
        return RunProblem(handle, problem, variantPack);
    }
    // hipDNN's backward recomputes the statistics with epsilon 1e-5, which is
    // all plain backward needs at that epsilon.
    if(problem.kind == GraphKind::BnBackward && epsilon == kDefaultEpsilon)
        return RunProblem(handle, problem, variantPack);
    return BackwardFromComputedStats(handle, problem, epsilon, x, scale, bias, variantPack);
}

} // namespace

// The V1 entry points describe scale, bias and every statistic with one
// descriptor, and V1 and V2 training update the running statistics in place.

miopenStatus_t
BatchNormalizationForwardTraining(miopenHandle_t handle,
                                  miopenBatchNormMode_t bn_mode,
                                  void* alpha,
                                  void* beta,
                                  const miopenTensorDescriptor_t xDesc,
                                  const void* x,
                                  const miopenTensorDescriptor_t yDesc,
                                  void* y,
                                  const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                  void* bnScale,
                                  void* bnBias,
                                  double expAvgFactor,
                                  void* resultRunningMean,
                                  void* resultRunningVariance,
                                  double epsilon,
                                  void* resultSaveMean,
                                  void* resultSaveInvVariance)
{
    return ForwardTraining(handle,
                           bn_mode,
                           alpha,
                           beta,
                           xDesc,
                           x,
                           yDesc,
                           y,
                           bnScaleBiasMeanVarDesc,
                           bnScaleBiasMeanVarDesc,
                           bnScaleBiasMeanVarDesc,
                           bnScaleBiasMeanVarDesc,
                           bnScale,
                           bnBias,
                           expAvgFactor,
                           resultRunningMean,
                           resultRunningVariance,
                           resultRunningMean,
                           resultRunningVariance,
                           epsilon,
                           resultSaveMean,
                           resultSaveInvVariance,
                           nullptr);
}

miopenStatus_t BatchNormalizationForwardTraining_V2(miopenHandle_t handle,
                                                    miopenBatchNormMode_t bn_mode,
                                                    void* alpha,
                                                    void* beta,
                                                    const miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t yDesc,
                                                    void* y,
                                                    const miopenTensorDescriptor_t scaleDesc,
                                                    const miopenTensorDescriptor_t biasVarDesc,
                                                    const miopenTensorDescriptor_t savedMeanDesc,
                                                    const miopenTensorDescriptor_t savedVarDesc,
                                                    void* bnScale,
                                                    void* bnBias,
                                                    double expAvgFactor,
                                                    void* resultRunningMean,
                                                    void* resultRunningVariance,
                                                    double epsilon,
                                                    void* resultSaveMean,
                                                    void* resultSaveInvVariance)
{
    return ForwardTraining(handle,
                           bn_mode,
                           alpha,
                           beta,
                           xDesc,
                           x,
                           yDesc,
                           y,
                           scaleDesc,
                           biasVarDesc,
                           savedMeanDesc,
                           savedVarDesc,
                           bnScale,
                           bnBias,
                           expAvgFactor,
                           resultRunningMean,
                           resultRunningVariance,
                           resultRunningMean,
                           resultRunningVariance,
                           epsilon,
                           resultSaveMean,
                           resultSaveInvVariance,
                           nullptr);
}

miopenStatus_t BatchNormalizationForwardTraining_V3(miopenHandle_t handle,
                                                    miopenBatchNormMode_t bn_mode,
                                                    void* alpha,
                                                    void* beta,
                                                    const miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t yDesc,
                                                    void* y,
                                                    const miopenTensorDescriptor_t scaleDesc,
                                                    const miopenTensorDescriptor_t biasVarDesc,
                                                    const miopenTensorDescriptor_t savedMeanDesc,
                                                    const miopenTensorDescriptor_t savedVarDesc,
                                                    void* bnScale,
                                                    void* bnBias,
                                                    double expAvgFactor,
                                                    const void* prevResultRunningMean,
                                                    const void* prevResultRunningVariance,
                                                    void* nextResultRunningMean,
                                                    void* nextResultRunningVariance,
                                                    double epsilon,
                                                    void* resultSaveMean,
                                                    void* resultSaveInvVariance)
{
    return ForwardTraining(handle,
                           bn_mode,
                           alpha,
                           beta,
                           xDesc,
                           x,
                           yDesc,
                           y,
                           scaleDesc,
                           biasVarDesc,
                           savedMeanDesc,
                           savedVarDesc,
                           bnScale,
                           bnBias,
                           expAvgFactor,
                           prevResultRunningMean,
                           prevResultRunningVariance,
                           nextResultRunningMean,
                           nextResultRunningVariance,
                           epsilon,
                           resultSaveMean,
                           resultSaveInvVariance,
                           nullptr);
}

miopenStatus_t BatchNormForwardTrainingActivation(miopenHandle_t handle,
                                                  miopenBatchNormMode_t bn_mode,
                                                  void* alpha,
                                                  void* beta,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  void* y,
                                                  const miopenTensorDescriptor_t scaleDesc,
                                                  const miopenTensorDescriptor_t biasVarDesc,
                                                  const miopenTensorDescriptor_t savedMeanDesc,
                                                  const miopenTensorDescriptor_t savedVarDesc,
                                                  void* bnScale,
                                                  void* bnBias,
                                                  double expAvgFactor,
                                                  void* resultRunningMean,
                                                  void* resultRunningVariance,
                                                  double epsilon,
                                                  void* resultSaveMean,
                                                  void* resultSaveInvVariance,
                                                  const miopenActivationDescriptor_t activDesc)
{
    return ForwardTraining(handle,
                           bn_mode,
                           alpha,
                           beta,
                           xDesc,
                           x,
                           yDesc,
                           y,
                           scaleDesc,
                           biasVarDesc,
                           savedMeanDesc,
                           savedVarDesc,
                           bnScale,
                           bnBias,
                           expAvgFactor,
                           resultRunningMean,
                           resultRunningVariance,
                           resultRunningMean,
                           resultRunningVariance,
                           epsilon,
                           resultSaveMean,
                           resultSaveInvVariance,
                           activDesc);
}

miopenStatus_t BatchNormForwardTrainingActivation_V2(miopenHandle_t handle,
                                                     miopenBatchNormMode_t bn_mode,
                                                     void* alpha,
                                                     void* beta,
                                                     const miopenTensorDescriptor_t xDesc,
                                                     const void* x,
                                                     const miopenTensorDescriptor_t yDesc,
                                                     void* y,
                                                     const miopenTensorDescriptor_t scaleDesc,
                                                     const miopenTensorDescriptor_t biasVarDesc,
                                                     const miopenTensorDescriptor_t savedMeanDesc,
                                                     const miopenTensorDescriptor_t savedVarDesc,
                                                     void* bnScale,
                                                     void* bnBias,
                                                     double expAvgFactor,
                                                     const void* prevResultRunningMean,
                                                     const void* prevResultRunningVariance,
                                                     void* nextResultRunningMean,
                                                     void* nextResultRunningVariance,
                                                     double epsilon,
                                                     void* resultSaveMean,
                                                     void* resultSaveInvVariance,
                                                     const miopenActivationDescriptor_t activDesc)
{
    return ForwardTraining(handle,
                           bn_mode,
                           alpha,
                           beta,
                           xDesc,
                           x,
                           yDesc,
                           y,
                           scaleDesc,
                           biasVarDesc,
                           savedMeanDesc,
                           savedVarDesc,
                           bnScale,
                           bnBias,
                           expAvgFactor,
                           prevResultRunningMean,
                           prevResultRunningVariance,
                           nextResultRunningMean,
                           nextResultRunningVariance,
                           epsilon,
                           resultSaveMean,
                           resultSaveInvVariance,
                           activDesc);
}

miopenStatus_t
BatchNormalizationForwardInference(miopenHandle_t handle,
                                   miopenBatchNormMode_t bn_mode,
                                   void* alpha,
                                   void* beta,
                                   const miopenTensorDescriptor_t xDesc,
                                   const void* x,
                                   const miopenTensorDescriptor_t yDesc,
                                   void* y,
                                   const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                   void* bnScale,
                                   void* bnBias,
                                   void* estimatedMean,
                                   void* estimatedVariance,
                                   double epsilon)
{
    return ForwardInference(handle,
                            bn_mode,
                            alpha,
                            beta,
                            xDesc,
                            x,
                            yDesc,
                            y,
                            bnScaleBiasMeanVarDesc,
                            bnScaleBiasMeanVarDesc,
                            bnScaleBiasMeanVarDesc,
                            bnScaleBiasMeanVarDesc,
                            bnScale,
                            bnBias,
                            estimatedMean,
                            estimatedVariance,
                            epsilon,
                            nullptr);
}

miopenStatus_t BatchNormalizationForwardInference_V2(miopenHandle_t handle,
                                                     miopenBatchNormMode_t bn_mode,
                                                     void* alpha,
                                                     void* beta,
                                                     const miopenTensorDescriptor_t xDesc,
                                                     const void* x,
                                                     const miopenTensorDescriptor_t yDesc,
                                                     void* y,
                                                     const miopenTensorDescriptor_t scaleDesc,
                                                     const miopenTensorDescriptor_t biasDesc,
                                                     const miopenTensorDescriptor_t estMeanDesc,
                                                     const miopenTensorDescriptor_t estVarianceDesc,
                                                     void* bnScale,
                                                     void* bnBias,
                                                     void* estimatedMean,
                                                     void* estimatedVariance,
                                                     double epsilon)
{
    return ForwardInference(handle,
                            bn_mode,
                            alpha,
                            beta,
                            xDesc,
                            x,
                            yDesc,
                            y,
                            scaleDesc,
                            biasDesc,
                            estMeanDesc,
                            estVarianceDesc,
                            bnScale,
                            bnBias,
                            estimatedMean,
                            estimatedVariance,
                            epsilon,
                            nullptr);
}

miopenStatus_t
BatchNormalizationForwardInferenceInvVariance(miopenHandle_t handle,
                                              miopenBatchNormMode_t bn_mode,
                                              void* alpha,
                                              void* beta,
                                              const miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              const miopenTensorDescriptor_t yDesc,
                                              void* y,
                                              const miopenTensorDescriptor_t scaleDesc,
                                              const miopenTensorDescriptor_t biasDesc,
                                              const miopenTensorDescriptor_t estMeanDesc,
                                              const miopenTensorDescriptor_t estInvVarianceDesc,
                                              void* bnScale,
                                              void* bnBias,
                                              void* estimatedMean,
                                              void* estimatedInvVariance)
{
    return ForwardInference(handle,
                            bn_mode,
                            alpha,
                            beta,
                            xDesc,
                            x,
                            yDesc,
                            y,
                            scaleDesc,
                            biasDesc,
                            estMeanDesc,
                            estInvVarianceDesc,
                            bnScale,
                            bnBias,
                            estimatedMean,
                            estimatedInvVariance,
                            std::nullopt,
                            nullptr);
}

miopenStatus_t
BatchNormForwardInferenceActivationInvVariance(miopenHandle_t handle,
                                               miopenBatchNormMode_t bn_mode,
                                               void* alpha,
                                               void* beta,
                                               const miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               const miopenTensorDescriptor_t yDesc,
                                               void* y,
                                               const miopenTensorDescriptor_t scaleDesc,
                                               const miopenTensorDescriptor_t biasDesc,
                                               const miopenTensorDescriptor_t estMeanDesc,
                                               const miopenTensorDescriptor_t estInvVarianceDesc,
                                               void* bnScale,
                                               void* bnBias,
                                               void* estimatedMean,
                                               void* estimatedInvVariance,
                                               const miopenActivationDescriptor_t activDesc)
{
    return ForwardInference(handle,
                            bn_mode,
                            alpha,
                            beta,
                            xDesc,
                            x,
                            yDesc,
                            y,
                            scaleDesc,
                            biasDesc,
                            estMeanDesc,
                            estInvVarianceDesc,
                            bnScale,
                            bnBias,
                            estimatedMean,
                            estimatedInvVariance,
                            std::nullopt,
                            activDesc);
}

miopenStatus_t BatchNormForwardInferenceActivation(miopenHandle_t handle,
                                                   miopenBatchNormMode_t bn_mode,
                                                   void* alpha,
                                                   void* beta,
                                                   const miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   void* y,
                                                   const miopenTensorDescriptor_t scaleDesc,
                                                   const miopenTensorDescriptor_t biasDesc,
                                                   const miopenTensorDescriptor_t estMeanDesc,
                                                   const miopenTensorDescriptor_t estVarianceDesc,
                                                   void* bnScale,
                                                   void* bnBias,
                                                   void* estimatedMean,
                                                   void* estimatedVariance,
                                                   double epsilon,
                                                   const miopenActivationDescriptor_t activDesc)
{
    return ForwardInference(handle,
                            bn_mode,
                            alpha,
                            beta,
                            xDesc,
                            x,
                            yDesc,
                            y,
                            scaleDesc,
                            biasDesc,
                            estMeanDesc,
                            estVarianceDesc,
                            bnScale,
                            bnBias,
                            estimatedMean,
                            estimatedVariance,
                            epsilon,
                            activDesc);
}

miopenStatus_t BatchNormalizationBackward(miopenHandle_t handle,
                                          miopenBatchNormMode_t bn_mode,
                                          const void* alphaDataDiff,
                                          const void* betaDataDiff,
                                          const void* alphaParamDiff,
                                          const void* betaParamDiff,
                                          const miopenTensorDescriptor_t xDesc,
                                          const void* x,
                                          const miopenTensorDescriptor_t dyDesc,
                                          const void* dy,
                                          const miopenTensorDescriptor_t dxDesc,
                                          void* dx,
                                          const miopenTensorDescriptor_t bnScaleBiasDiffDesc,
                                          const void* bnScale,
                                          void* resultBnScaleDiff,
                                          void* resultBnBiasDiff,
                                          double epsilon,
                                          const void* savedMean,
                                          const void* savedInvVariance)
{
    return Backward(handle,
                    bn_mode,
                    alphaDataDiff,
                    betaDataDiff,
                    alphaParamDiff,
                    betaParamDiff,
                    xDesc,
                    x,
                    dyDesc,
                    dy,
                    dxDesc,
                    dx,
                    bnScaleBiasDiffDesc,
                    bnScaleBiasDiffDesc,
                    bnScaleBiasDiffDesc,
                    bnScaleBiasDiffDesc,
                    bnScale,
                    nullptr,
                    resultBnScaleDiff,
                    resultBnBiasDiff,
                    epsilon,
                    savedMean,
                    savedInvVariance,
                    nullptr);
}

miopenStatus_t BatchNormalizationBackward_V2(miopenHandle_t handle,
                                             miopenBatchNormMode_t bn_mode,
                                             const void* alphaDataDiff,
                                             const void* betaDataDiff,
                                             const void* alphaParamDiff,
                                             const void* betaParamDiff,
                                             const miopenTensorDescriptor_t xDesc,
                                             const void* x,
                                             const miopenTensorDescriptor_t dyDesc,
                                             const void* dy,
                                             const miopenTensorDescriptor_t dxDesc,
                                             void* dx,
                                             const miopenTensorDescriptor_t scaleDesc,
                                             const miopenTensorDescriptor_t biasDesc,
                                             const miopenTensorDescriptor_t savedMeanDesc,
                                             const miopenTensorDescriptor_t savedVarDesc,
                                             const void* bnScale,
                                             void* resultBnScaleDiff,
                                             void* resultBnBiasDiff,
                                             double epsilon,
                                             const void* savedMean,
                                             const void* savedInvVariance)
{
    return Backward(handle,
                    bn_mode,
                    alphaDataDiff,
                    betaDataDiff,
                    alphaParamDiff,
                    betaParamDiff,
                    xDesc,
                    x,
                    dyDesc,
                    dy,
                    dxDesc,
                    dx,
                    scaleDesc,
                    biasDesc,
                    savedMeanDesc,
                    savedVarDesc,
                    bnScale,
                    nullptr,
                    resultBnScaleDiff,
                    resultBnBiasDiff,
                    epsilon,
                    savedMean,
                    savedInvVariance,
                    nullptr);
}

miopenStatus_t BatchNormBackwardActivation(miopenHandle_t handle,
                                           miopenBatchNormMode_t bn_mode,
                                           const void* alphaDataDiff,
                                           const void* betaDataDiff,
                                           const void* alphaParamDiff,
                                           const void* betaParamDiff,
                                           const miopenTensorDescriptor_t xDesc,
                                           const void* x,
                                           const miopenTensorDescriptor_t dyDesc,
                                           const void* dy,
                                           const miopenTensorDescriptor_t dxDesc,
                                           void* dx,
                                           const miopenTensorDescriptor_t scaleDesc,
                                           const miopenTensorDescriptor_t biasDesc,
                                           const miopenTensorDescriptor_t savedMeanDesc,
                                           const miopenTensorDescriptor_t savedVarianceDesc,
                                           const void* bnScale,
                                           const void* bnBias,
                                           void* resultBnScaleDiff,
                                           void* resultBnBiasDiff,
                                           double epsilon,
                                           const void* savedMean,
                                           const void* savedInvVariance,
                                           const miopenActivationDescriptor_t activDesc)
{
    return Backward(handle,
                    bn_mode,
                    alphaDataDiff,
                    betaDataDiff,
                    alphaParamDiff,
                    betaParamDiff,
                    xDesc,
                    x,
                    dyDesc,
                    dy,
                    dxDesc,
                    dx,
                    scaleDesc,
                    biasDesc,
                    savedMeanDesc,
                    savedVarianceDesc,
                    bnScale,
                    bnBias,
                    resultBnScaleDiff,
                    resultBnBiasDiff,
                    epsilon,
                    savedMean,
                    savedInvVariance,
                    activDesc);
}

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
