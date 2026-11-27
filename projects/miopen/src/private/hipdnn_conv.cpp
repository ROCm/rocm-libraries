// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The four forwarded convolution entry points share one shape: read the MIOpen
// descriptors through the public _impl getters, decline the combinations hipDNN
// cannot express, then hand the problem to RunCachedGraph with the function that
// builds its graph.

#include "hipdnn_conv.hpp"
#include "hipdnn_graph.hpp"
#include "hipdnn_tensor.hpp"
#include "hipdnn_types.hpp"

#include "miopen_impl.h"

#include <hipdnn_frontend.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace miopen {
namespace wrapper {
namespace hipdnn {

namespace {

namespace fe = hipdnn_frontend;

// Any distinct values would do; hipDNN only needs them to match between the
// tensor attributes and the variant pack.
constexpr int64_t kUidA    = 1; // x for fprop and wgrad, dy for dgrad
constexpr int64_t kUidB    = 2; // w for fprop and dgrad, x for wgrad
constexpr int64_t kUidOut  = 3;
constexpr int64_t kUidBias = 4;

struct ConvInfo
{
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    miopenConvolutionMode_t mode{};
    int groupCount      = 1;
    int fp16AltImpl     = -1;
    int deterministic   = 0;
    int fp8RoundingMode = miopenF8RoundingModeStochastic;
    int mathType        = miopenMathPedantic;
};

// Tensors in node order: the two convolution inputs, the result, then the bias
// for the fused kind.
struct ConvProblem
{
    GraphKind kind = GraphKind::ConvFprop;
    std::vector<TensorInfo> tensors;
    ConvInfo conv;
    miopenActivationMode_t act = miopenActivationPASTHRU;
};

miopenStatus_t ReadConvolution(miopenConvolutionDescriptor_t desc, ConvInfo& out)
{
    int spatialDim = 0;
    if(miopenGetConvolutionSpatialDim_impl(desc, &spatialDim) != miopenStatusSuccess ||
       spatialDim <= 0)
        return miopenStatusInvalidValue;

    std::vector<int> pads(static_cast<size_t>(spatialDim));
    std::vector<int> strides(static_cast<size_t>(spatialDim));
    std::vector<int> dilations(static_cast<size_t>(spatialDim));
    int returnedSpatialDim = 0;
    miopenConvolutionMode_t mode{};
    if(miopenGetConvolutionNdDescriptor_impl(desc,
                                             spatialDim,
                                             &returnedSpatialDim,
                                             pads.data(),
                                             strides.data(),
                                             dilations.data(),
                                             &mode) != miopenStatusSuccess)
        return miopenStatusInvalidValue;

    int groupCount = 1;
    if(miopenGetConvolutionGroupCount_impl(desc, &groupCount) != miopenStatusSuccess)
        return miopenStatusInvalidValue;

    const auto readAttribute = [desc](miopenConvolutionAttrib_t attribute, int& value) {
        return miopenGetConvolutionAttribute_impl(desc, attribute, &value) == miopenStatusSuccess;
    };
    if(!readAttribute(MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL, out.fp16AltImpl) ||
       !readAttribute(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, out.deterministic) ||
       !readAttribute(MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, out.mathType) ||
       !readAttribute(MIOPEN_CONVOLUTION_ATTRIB_FP8_ROUNDING_MODE, out.fp8RoundingMode))
        return miopenStatusInvalidValue;

    out.pads.assign(pads.begin(), pads.end());
    out.strides.assign(strides.begin(), strides.end());
    out.dilations.assign(dilations.begin(), dilations.end());
    out.mode       = mode;
    out.groupCount = groupCount;
    return miopenStatusSuccess;
}

// Exact comparison is the point: the only scaling hipDNN's convolution op can
// express is none at all, so anything but a literal 1.0 or 0.0 has to be
// declined rather than rounded into.
bool ScalarEquals(const void* value, miopenDataType_t type, double expected)
{
    if(value == nullptr)
        return false;
    if(type == miopenDouble)
        return *static_cast<const double*>(value) == expected;
    return static_cast<double>(*static_cast<const float*>(value)) == expected;
}

// MIOpen itself serves only 2-D and 3-D convolutions.
constexpr size_t kMinSpatialDims = 2;
constexpr size_t kMaxSpatialDims = 3;

// Returns why hipDNN cannot take this problem, or null when it can.
const char* CheckSupported(const std::vector<TensorInfo>& tensors, const ConvInfo& conv)
{
    fe::DataType unused{};
    if(!ComputeTypeFor(tensors.front().dataType, unused))
        return "this data type has no hipDNN convolution";
    for(const TensorInfo& tensor : tensors)
    {
        if(!ToHipdnnDataType(tensor.dataType, unused))
            return "this data type has no hipDNN convolution";
        // A cast type asks MIOpen to convert the elements (to FP8) before the
        // convolution; a hipDNN graph has no per-tensor conversion.
        if(tensor.castType != tensor.dataType)
            return "a tensor cast type (miopenSetTensorCastType) is not forwarded to hipDNN";
        // hipDNN describes a tensor only by dims and strides, which cannot
        // express a vector packed into each element.
        if(tensor.vectorLength != 1)
            return "vectorized tensor layouts (NCHWc4, NCHWc8, CHWNc4, CHWNc8) are not "
                   "forwarded to hipDNN";
    }
    // Grouped convolution is forwarded: MIOpen's weight descriptor is (K, C/g, ...), and hipDNN's
    // conv nodes derive the group count from the input and weight channel counts. The group count
    // is part of the plan key.
    if(conv.pads.size() < kMinSpatialDims || conv.pads.size() > kMaxSpatialDims)
        return "only 2-D and 3-D convolutions are forwarded to hipDNN";
    // Transposed convolution is accepted: ToOrdinaryRoles maps it onto the
    // ordinary convolution that computes it. DETERMINISTIC and MATH_TYPE are honoured by the
    // engines the graph deselects (ExcludedNotes). The two attributes below have no
    // hipDNN setting, so a forwarded call would ignore them; any non-default
    // value is declined, because whether it changes the result depends on the
    // data type and device, and copying MIOpen's rules for that here would drift.
    if(conv.fp16AltImpl != -1)
        return "a non-default MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL is not forwarded to hipDNN";
    if(conv.fp8RoundingMode != miopenF8RoundingModeStochastic)
        return "a non-default MIOPEN_CONVOLUTION_ATTRIB_FP8_ROUNDING_MODE is not forwarded to "
               "hipDNN";
    return nullptr;
}

// TF32 applies to fp32 convolutions only, and only when the caller sets
// MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE to miopenMathDefault; the default math type,
// miopenMathPedantic, asks for full fp32.
bool AllowsTf32(const ConvProblem& problem)
{
    return problem.tensors.front().dataType == miopenFloat &&
           problem.conv.mathType == miopenMathDefault;
}

// The engines a forwarded call must not run, by the numerical notes they report, as PyTorch
// filters cuDNN's engines: never one that down-converts its inputs, no TF32 (TENSOR_CORE) on
// fp32 unless TF32 is allowed, and no non-deterministic one when DETERMINISTIC is set.
std::vector<fe::NumericalNote> ExcludedNotes(const ConvProblem& problem)
{
    std::vector<fe::NumericalNote> notes{fe::NumericalNote::DOWN_CONVERT_INPUTS};
    if(problem.tensors.front().dataType == miopenFloat && !AllowsTf32(problem))
        notes.push_back(fe::NumericalNote::TENSOR_CORE);
    if(problem.conv.deterministic != 0)
        notes.push_back(fe::NumericalNote::NONDETERMINISTIC);
    return notes;
}

PlanKey MakePlanKey(const ConvProblem& problem)
{
    PlanKey key;
    std::vector<int64_t>& out = key.problem;
    out.push_back(static_cast<int64_t>(problem.kind));
    out.push_back(static_cast<int64_t>(problem.act));
    out.push_back(static_cast<int64_t>(problem.tensors.size()));
    for(const TensorInfo& tensor : problem.tensors)
        AppendTensor(out, tensor);
    AppendList(out, problem.conv.pads);
    AppendList(out, problem.conv.strides);
    AppendList(out, problem.conv.dilations);
    out.push_back(static_cast<int64_t>(problem.conv.mode));
    out.push_back(problem.conv.groupCount);
    out.push_back(problem.conv.deterministic != 0 ? 1 : 0);
    out.push_back(AllowsTf32(problem) ? 1 : 0);
    return key;
}

// Builds the graph for one problem. `a` and `b` are the two convolution inputs
// in node order, `out` is the node's result; bias is present only for the fused
// kind.
bool PopulateGraph(const ConvProblem& problem, fe::graph::Graph& graph)
{
    // Each tensor keeps its own type: MIOpen accepts mixed-type problems such as
    // int8 x and w with an int32 or float y.
    std::vector<fe::DataType> types(problem.tensors.size());
    for(size_t i = 0; i < problem.tensors.size(); ++i)
    {
        if(!ToHipdnnDataType(problem.tensors[i].dataType, types[i]))
            return false;
    }
    fe::DataType computeType{};
    if(!ComputeTypeFor(problem.tensors.front().dataType, computeType))
        return false;

    // Without the intermediate type, the virtual tensors between the fused
    // graph's nodes are left with no data type at all and the build fails.
    graph.set_io_data_type(types.front())
        .set_compute_data_type(computeType)
        .set_intermediate_data_type(computeType);
    graph.deselect_numeric_notes(ExcludedNotes(problem));

    auto a                    = MakeTensor(problem.tensors[0], types[0], kUidA);
    auto b                    = MakeTensor(problem.tensors[1], types[1], kUidB);
    const TensorInfo& outInfo = problem.tensors[2];

    std::shared_ptr<fe::graph::TensorAttributes> out;
    switch(problem.kind)
    {
    case GraphKind::ConvFprop:
    case GraphKind::ConvBiasActivation: {
        fe::graph::ConvFpropAttributes conv;
        conv.set_padding(problem.conv.pads)
            .set_stride(problem.conv.strides)
            .set_dilation(problem.conv.dilations);
        out = graph.conv_fprop(a, b, conv);
        break;
    }
    case GraphKind::ConvDgrad: {
        // MIOpen's padding is symmetric -- one array -- so pre and post both get
        // the same values.
        fe::graph::ConvDgradAttributes conv;
        conv.set_pre_padding(problem.conv.pads)
            .set_post_padding(problem.conv.pads)
            .set_stride(problem.conv.strides)
            .set_dilation(problem.conv.dilations);
        out = graph.conv_dgrad(a, b, conv);
        break;
    }
    case GraphKind::ConvWgrad: {
        fe::graph::ConvWgradAttributes conv;
        conv.set_pre_padding(problem.conv.pads)
            .set_post_padding(problem.conv.pads)
            .set_stride(problem.conv.strides)
            .set_dilation(problem.conv.dilations);
        out = graph.conv_wgrad(a, b, conv);
        break;
    }
    case GraphKind::BatchnormInference: return false;
    }

    if(out == nullptr)
        return false;

    if(problem.kind == GraphKind::ConvBiasActivation)
    {
        // The convolution result is an intermediate now, so it needs the shape
        // the bias add will broadcast against.
        out->set_dim(outInfo.dims).set_stride(outInfo.strides);

        auto bias = MakeTensor(problem.tensors[3], types[3], kUidBias);

        // The MIOpen provider requires the bias add to compute in the bias type
        // and the activation in FLOAT.
        fe::graph::PointwiseAttributes add;
        add.set_mode(fe::PointwiseMode::ADD).set_compute_data_type(types[3]);
        out = graph.pointwise(out, bias, add);

        fe::PointwiseMode activation{};
        if(!ToPointwiseActivation(problem.act, activation))
            return false;
        out->set_dim(outInfo.dims).set_stride(outInfo.strides);

        fe::graph::PointwiseAttributes activationAttributes;
        activationAttributes.set_mode(activation).set_compute_data_type(computeType);
        out = graph.pointwise(out, activationAttributes);
    }

    out->set_dim(outInfo.dims)
        .set_stride(outInfo.strides)
        .set_data_type(types[2])
        .set_alignment(outInfo.alignment)
        .set_uid(kUidOut)
        .set_output(true);
    return true;
}

miopenStatus_t RunProblem(miopenHandle_t handle,
                          const ConvProblem& problem,
                          VariantPack& variantPack,
                          void* workspace,
                          size_t workspaceSize)
{
    return RunCachedGraph(
        handle,
        MakePlanKey(problem),
        [&problem](fe::graph::Graph& graph) { return PopulateGraph(problem, graph); },
        variantPack,
        workspace,
        workspaceSize);
}

// MIOpen's transposed convolution is the ordinary convolution with forward and
// backward data exchanged, as convolution_api.cpp maps it: the transposed forward
// is the ordinary backward data with x as dy and y as dx, the transposed backward
// data is the ordinary forward with dy as x and dx as y, and the transposed
// backward weights is the ordinary backward weights with x and dy exchanged. The
// weight tensor, (C, K/g, ...), is already the ordinary one. Output padding only
// lengthens the transposed output, which the ordinary node's shape rule, a floor
// division by the stride, accepts. Returns whether the two inputs were exchanged,
// so the caller exchanges their data pointers too.
bool ToOrdinaryRoles(ConvProblem& problem)
{
    if(problem.conv.mode != miopenTranspose)
        return false;
    switch(problem.kind)
    {
    case GraphKind::ConvFprop: problem.kind = GraphKind::ConvDgrad; return false;
    case GraphKind::ConvDgrad: problem.kind = GraphKind::ConvFprop; return false;
    case GraphKind::ConvWgrad: std::swap(problem.tensors[0], problem.tensors[1]); return true;
    case GraphKind::ConvBiasActivation:
    case GraphKind::BatchnormInference: return false;
    }
    return false;
}

// Reads the descriptors of one plain call. `a`, `b` and `out` are the call's
// inputs and result in the order of its own direction; for a transposed call the
// problem comes back in the ordinary convolution's roles, and `swapInputs` says
// whether `a` and `b` were exchanged. Returns the status to fail with, or success.
miopenStatus_t ReadPlainProblem(GraphKind kind,
                                miopenTensorDescriptor_t aDesc,
                                miopenTensorDescriptor_t bDesc,
                                miopenConvolutionDescriptor_t convDesc,
                                miopenTensorDescriptor_t outDesc,
                                ConvProblem& problem,
                                bool& swapInputs)
{
    if(!IsAvailable())
        return RecordFailure(miopenStatusInternalError, "hipDNN forwarding is unavailable");

    problem.kind = kind;
    problem.tensors.resize(3);
    if(ReadTensor(aDesc, problem.tensors[0]) != miopenStatusSuccess ||
       ReadTensor(bDesc, problem.tensors[1]) != miopenStatusSuccess ||
       ReadTensor(outDesc, problem.tensors[2]) != miopenStatusSuccess ||
       ReadConvolution(convDesc, problem.conv) != miopenStatusSuccess)
        return RecordFailure(miopenStatusBadParm, "could not read the MIOpen descriptors");

    if(const char* reason = CheckSupported(problem.tensors, problem.conv))
        return RecordFailure(miopenStatusUnsupportedOp, reason);
    swapInputs = ToOrdinaryRoles(problem);
    return miopenStatusSuccess;
}

// Declares each tensor's pointer alignment in `problem`, so a plan that needs more than this
// call's pointers have is never chosen for it, then runs it.
miopenStatus_t RunPlainProblem(miopenHandle_t handle,
                               ConvProblem& problem,
                               const void* aData,
                               const void* bData,
                               void* outData,
                               void* workspace,
                               size_t workspaceSize)
{
    problem.tensors[0].alignment = PointerAlignment(aData);
    problem.tensors[1].alignment = PointerAlignment(bData);
    problem.tensors[2].alignment = PointerAlignment(outData);
    VariantPack variantPack{
        {kUidA, const_cast<void*>(aData)},
        {kUidB, const_cast<void*>(bData)},
        {kUidOut, outData},
    };
    return RunProblem(handle, problem, variantPack, workspace, workspaceSize);
}

// Shared tail of the three plain convolution entry points.
miopenStatus_t ForwardConvolution(miopenHandle_t handle,
                                  GraphKind kind,
                                  const void* alpha,
                                  const void* beta,
                                  miopenTensorDescriptor_t aDesc,
                                  const void* aData,
                                  miopenTensorDescriptor_t bDesc,
                                  const void* bData,
                                  miopenConvolutionDescriptor_t convDesc,
                                  miopenTensorDescriptor_t outDesc,
                                  void* outData,
                                  void* workspace,
                                  size_t workspaceSize)
{
    ConvProblem problem;
    bool swapInputs = false;
    if(const auto status =
           ReadPlainProblem(kind, aDesc, bDesc, convDesc, outDesc, problem, swapInputs);
       status != miopenStatusSuccess)
        return status;
    if(swapInputs)
        std::swap(aData, bData);

    // MIOpen reads alpha and beta as the output tensor's type. Its own 2-D
    // convolution returns the same status.
    const miopenDataType_t scalarType = problem.tensors[2].dataType;
    if(!ScalarEquals(alpha, scalarType, 1.0) || !ScalarEquals(beta, scalarType, 0.0))
        return RecordFailure(miopenStatusNotImplemented,
                             "hipDNN convolution supports only alpha=1, beta=0");

    return RunPlainProblem(handle, problem, aData, bData, outData, workspace, workspaceSize);
}

// The workspace of the plan the forwarded execution will run when given `limit`
// bytes (no limit: hipDNN's best plan). Builds and caches the graph, so a problem
// hipDNN cannot serve fails here, at query time, as it would on execution.
miopenStatus_t PlainWorkspaceSize(miopenHandle_t handle,
                                  GraphKind kind,
                                  miopenTensorDescriptor_t aDesc,
                                  miopenTensorDescriptor_t bDesc,
                                  miopenConvolutionDescriptor_t convDesc,
                                  miopenTensorDescriptor_t outDesc,
                                  std::optional<size_t> limit,
                                  size_t& workspaceSize)
{
    ConvProblem problem;
    bool swapInputs = false;
    if(const auto status =
           ReadPlainProblem(kind, aDesc, bDesc, convDesc, outDesc, problem, swapInputs);
       status != miopenStatusSuccess)
        return status;
    return CachedGraphWorkspaceSize(
        handle,
        MakePlanKey(problem),
        [&problem](fe::graph::Graph& graph) { return PopulateGraph(problem, graph); },
        limit,
        workspaceSize);
}

// The smallest workspace any engine runs the problem in, and the best plan's.
miopenStatus_t PlainWorkspaceRange(miopenHandle_t handle,
                                   GraphKind kind,
                                   miopenTensorDescriptor_t aDesc,
                                   miopenTensorDescriptor_t bDesc,
                                   miopenConvolutionDescriptor_t convDesc,
                                   miopenTensorDescriptor_t outDesc,
                                   size_t& minWorkspaceSize,
                                   size_t& maxWorkspaceSize)
{
    ConvProblem problem;
    bool swapInputs = false;
    if(const auto status =
           ReadPlainProblem(kind, aDesc, bDesc, convDesc, outDesc, problem, swapInputs);
       status != miopenStatusSuccess)
        return status;
    return CachedGraphWorkspaceRange(
        handle,
        MakePlanKey(problem),
        [&problem](fe::graph::Graph& graph) { return PopulateGraph(problem, graph); },
        minWorkspaceSize,
        maxWorkspaceSize);
}

} // namespace

miopenStatus_t ConvolutionForward(miopenHandle_t handle,
                                  const void* alpha,
                                  const miopenTensorDescriptor_t xDesc,
                                  const void* x,
                                  const miopenTensorDescriptor_t wDesc,
                                  const void* w,
                                  const miopenConvolutionDescriptor_t convDesc,
                                  miopenConvFwdAlgorithm_t /*algo*/,
                                  const void* beta,
                                  const miopenTensorDescriptor_t yDesc,
                                  void* y,
                                  void* workSpace,
                                  size_t workSpaceSize)
{
    return ForwardConvolution(handle,
                              GraphKind::ConvFprop,
                              alpha,
                              beta,
                              xDesc,
                              x,
                              wDesc,
                              w,
                              convDesc,
                              yDesc,
                              y,
                              workSpace,
                              workSpaceSize);
}

miopenStatus_t ConvolutionBackwardData(miopenHandle_t handle,
                                       const void* alpha,
                                       const miopenTensorDescriptor_t dyDesc,
                                       const void* dy,
                                       const miopenTensorDescriptor_t wDesc,
                                       const void* w,
                                       const miopenConvolutionDescriptor_t convDesc,
                                       miopenConvBwdDataAlgorithm_t /*algo*/,
                                       const void* beta,
                                       const miopenTensorDescriptor_t dxDesc,
                                       void* dx,
                                       void* workSpace,
                                       size_t workSpaceSize)
{
    return ForwardConvolution(handle,
                              GraphKind::ConvDgrad,
                              alpha,
                              beta,
                              dyDesc,
                              dy,
                              wDesc,
                              w,
                              convDesc,
                              dxDesc,
                              dx,
                              workSpace,
                              workSpaceSize);
}

miopenStatus_t ConvolutionBackwardWeights(miopenHandle_t handle,
                                          const void* alpha,
                                          const miopenTensorDescriptor_t dyDesc,
                                          const void* dy,
                                          const miopenTensorDescriptor_t xDesc,
                                          const void* x,
                                          const miopenConvolutionDescriptor_t convDesc,
                                          miopenConvBwdWeightsAlgorithm_t /*algo*/,
                                          const void* beta,
                                          const miopenTensorDescriptor_t dwDesc,
                                          void* dw,
                                          void* workSpace,
                                          size_t workSpaceSize)
{
    return ForwardConvolution(handle,
                              GraphKind::ConvWgrad,
                              alpha,
                              beta,
                              dyDesc,
                              dy,
                              xDesc,
                              x,
                              convDesc,
                              dwDesc,
                              dw,
                              workSpace,
                              workSpaceSize);
}

miopenStatus_t ConvolutionImmediate(miopenHandle_t handle,
                                    GraphKind kind,
                                    miopenTensorDescriptor_t aDesc,
                                    const void* a,
                                    miopenTensorDescriptor_t bDesc,
                                    const void* b,
                                    miopenConvolutionDescriptor_t convDesc,
                                    miopenTensorDescriptor_t outDesc,
                                    void* out,
                                    void* workSpace,
                                    size_t workSpaceSize)
{
    ConvProblem problem;
    bool swapInputs = false;
    if(const auto status =
           ReadPlainProblem(kind, aDesc, bDesc, convDesc, outDesc, problem, swapInputs);
       status != miopenStatusSuccess)
        return status;
    if(swapInputs)
        std::swap(a, b);
    return RunPlainProblem(handle, problem, a, b, out, workSpace, workSpaceSize);
}

miopenStatus_t ConvolutionWorkspaceSize(miopenHandle_t handle,
                                        GraphKind kind,
                                        miopenTensorDescriptor_t aDesc,
                                        miopenTensorDescriptor_t bDesc,
                                        miopenConvolutionDescriptor_t convDesc,
                                        miopenTensorDescriptor_t outDesc,
                                        size_t* workSpaceSize)
{
    if(workSpaceSize == nullptr)
        return RecordFailure(miopenStatusBadParm, "workSpaceSize is null");
    size_t size = 0;
    if(const auto status =
           PlainWorkspaceSize(handle, kind, aDesc, bDesc, convDesc, outDesc, std::nullopt, size);
       status != miopenStatusSuccess)
        return status;
    *workSpaceSize = size;
    return miopenStatusSuccess;
}

miopenStatus_t ConvolutionWorkspaceSizeRange(miopenHandle_t handle,
                                             GraphKind kind,
                                             miopenTensorDescriptor_t aDesc,
                                             miopenTensorDescriptor_t bDesc,
                                             miopenConvolutionDescriptor_t convDesc,
                                             miopenTensorDescriptor_t outDesc,
                                             size_t* minWorkspaceSize,
                                             size_t* maxWorkspaceSize)
{
    if(minWorkspaceSize == nullptr || maxWorkspaceSize == nullptr)
        return miopenStatusBadParm;
    return PlainWorkspaceRange(
        handle, kind, aDesc, bDesc, convDesc, outDesc, *minWorkspaceSize, *maxWorkspaceSize);
}

miopenStatus_t FindConvolution(miopenHandle_t handle,
                               GraphKind kind,
                               miopenTensorDescriptor_t aDesc,
                               miopenTensorDescriptor_t bDesc,
                               miopenConvolutionDescriptor_t convDesc,
                               miopenTensorDescriptor_t outDesc,
                               int requestAlgoCount,
                               int* returnedAlgoCount,
                               miopenConvAlgoPerf_t* perfResults,
                               void* workSpace,
                               size_t workSpaceSize)
{
    if(requestAlgoCount < 1 || returnedAlgoCount == nullptr || perfResults == nullptr)
        return RecordFailure(miopenStatusBadParm,
                             "Find needs requestAlgoCount >= 1 and non-null result pointers");
    // As MIOpen's Find returns only what fits the workspace it is given, the result is the
    // plan the forwarded execution will run in that workspace.
    const size_t given = workSpace == nullptr ? 0 : workSpaceSize;
    size_t size        = 0;
    if(const auto status =
           PlainWorkspaceSize(handle, kind, aDesc, bDesc, convDesc, outDesc, given, size);
       status != miopenStatusSuccess)
        return status;

    // One result: hipDNN's choice. Its algorithm value is never read back, since
    // the forwarded execution ignores the algorithm argument. No time is measured.
    miopenConvAlgoPerf_t result{};
    switch(kind)
    {
    case GraphKind::ConvFprop: result.fwd_algo = miopenConvolutionFwdAlgoDirect; break;
    case GraphKind::ConvDgrad: result.bwd_data_algo = miopenConvolutionBwdDataAlgoDirect; break;
    case GraphKind::ConvWgrad:
        result.bwd_weights_algo = miopenConvolutionBwdWeightsAlgoDirect;
        break;
    case GraphKind::ConvBiasActivation:
    case GraphKind::BatchnormInference:
        return RecordFailure(miopenStatusBadParm,
                             "Find serves only the three plain convolution directions");
    }
    result.time        = 0.0f;
    result.memory      = size;
    perfResults[0]     = result;
    *returnedAlgoCount = 1;
    return miopenStatusSuccess;
}

namespace {

// The solutions the solution queries report: hipDNN's best plan, and, when some engine
// runs in less workspace, the plan for the smallest workspace, so a caller choosing a
// solution by the workspace it can afford finds one, as with MIOpen's solution list. The
// immediate call picks the plan by the workspace it is given, not by the id.
struct SolutionSizes
{
    size_t best    = 0;
    size_t minimum = 0;
    size_t Count() const { return minimum < best ? 2 : 1; }
};

miopenStatus_t ReadSolutionSizes(miopenHandle_t handle,
                                 GraphKind kind,
                                 miopenTensorDescriptor_t aDesc,
                                 miopenTensorDescriptor_t bDesc,
                                 miopenConvolutionDescriptor_t convDesc,
                                 miopenTensorDescriptor_t outDesc,
                                 SolutionSizes& sizes)
{
    return PlainWorkspaceRange(
        handle, kind, aDesc, bDesc, convDesc, outDesc, sizes.minimum, sizes.best);
}

} // namespace

miopenStatus_t ConvolutionSolutionCount(miopenHandle_t handle,
                                        GraphKind kind,
                                        miopenTensorDescriptor_t aDesc,
                                        miopenTensorDescriptor_t bDesc,
                                        miopenConvolutionDescriptor_t convDesc,
                                        miopenTensorDescriptor_t outDesc,
                                        size_t* solutionCount)
{
    if(solutionCount == nullptr)
        return RecordFailure(miopenStatusBadParm, "solutionCount is null");
    SolutionSizes sizes;
    if(const auto status = ReadSolutionSizes(handle, kind, aDesc, bDesc, convDesc, outDesc, sizes);
       status != miopenStatusSuccess)
        return status;
    *solutionCount = sizes.Count();
    return miopenStatusSuccess;
}

miopenStatus_t ConvolutionSolutions(miopenHandle_t handle,
                                    GraphKind kind,
                                    miopenTensorDescriptor_t aDesc,
                                    miopenTensorDescriptor_t bDesc,
                                    miopenConvolutionDescriptor_t convDesc,
                                    miopenTensorDescriptor_t outDesc,
                                    size_t maxSolutionCount,
                                    size_t* solutionCount,
                                    miopenConvSolution_t* solutions)
{
    SolutionSizes sizes;
    if(const auto status = ReadSolutionSizes(handle, kind, aDesc, bDesc, convDesc, outDesc, sizes);
       status != miopenStatusSuccess)
        return status;

    // As MIOpen's own query: at most maxSolutionCount solutions, best first, and a null
    // pointer leaves that output unwritten.
    const size_t returned = std::min(maxSolutionCount, sizes.Count());
    if(solutions != nullptr)
    {
        for(size_t i = 0; i < returned; ++i)
        {
            miopenConvSolution_t solution{};
            solution.time           = 0.0f;
            solution.workspace_size = i == 0 ? sizes.best : sizes.minimum;
            solution.solution_id    = kHipdnnSolutionId + i;
            solution.algorithm      = miopenConvolutionAlgoDirect;
            solutions[i]            = solution;
        }
    }
    if(solutionCount != nullptr)
        *solutionCount = returned;
    return miopenStatusSuccess;
}

miopenStatus_t ConvolutionSolutionWorkspaceSize(miopenHandle_t handle,
                                                GraphKind kind,
                                                miopenTensorDescriptor_t aDesc,
                                                miopenTensorDescriptor_t bDesc,
                                                miopenConvolutionDescriptor_t convDesc,
                                                miopenTensorDescriptor_t outDesc,
                                                uint64_t solutionId,
                                                size_t* workSpaceSize)
{
    if(workSpaceSize == nullptr)
        return RecordFailure(miopenStatusBadParm, "workSpaceSize is null");
    SolutionSizes sizes;
    if(const auto status = ReadSolutionSizes(handle, kind, aDesc, bDesc, convDesc, outDesc, sizes);
       status != miopenStatusSuccess)
        return status;
    *workSpaceSize = solutionId == kHipdnnSolutionId + 1 ? sizes.minimum : sizes.best;
    return miopenStatusSuccess;
}

miopenStatus_t CompileConvolutionSolution(miopenHandle_t handle,
                                          GraphKind kind,
                                          miopenTensorDescriptor_t aDesc,
                                          miopenTensorDescriptor_t bDesc,
                                          miopenConvolutionDescriptor_t convDesc,
                                          miopenTensorDescriptor_t outDesc,
                                          uint64_t solutionId)
{
    // Builds the plan the solution's workspace selects.
    SolutionSizes sizes;
    if(const auto status = ReadSolutionSizes(handle, kind, aDesc, bDesc, convDesc, outDesc, sizes);
       status != miopenStatusSuccess)
        return status;
    size_t unused = 0;
    return PlainWorkspaceSize(handle,
                              kind,
                              aDesc,
                              bDesc,
                              convDesc,
                              outDesc,
                              solutionId == kHipdnnSolutionId + 1 ? sizes.minimum : sizes.best,
                              unused);
}

miopenStatus_t ConvolutionBiasActivationForward(miopenHandle_t handle,
                                                const void* alpha1,
                                                const miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                const miopenTensorDescriptor_t wDesc,
                                                const void* w,
                                                const miopenConvolutionDescriptor_t convDesc,
                                                miopenConvFwdAlgorithm_t /*algo*/,
                                                void* workspace,
                                                size_t workspaceSizeInBytes,
                                                const void* alpha2,
                                                const miopenTensorDescriptor_t /*zDesc*/,
                                                const void* /*z*/,
                                                const miopenTensorDescriptor_t biasDesc,
                                                const void* bias,
                                                const miopenActivationDescriptor_t activationDesc,
                                                const miopenTensorDescriptor_t yDesc,
                                                void* y)
{
    if(!IsAvailable())
        return RecordFailure(miopenStatusInternalError, "hipDNN forwarding is unavailable");

    ConvProblem problem;
    problem.kind = GraphKind::ConvBiasActivation;
    problem.tensors.resize(4);
    if(ReadTensor(xDesc, problem.tensors[0]) != miopenStatusSuccess ||
       ReadTensor(wDesc, problem.tensors[1]) != miopenStatusSuccess ||
       ReadTensor(yDesc, problem.tensors[2]) != miopenStatusSuccess ||
       ReadTensor(biasDesc, problem.tensors[3]) != miopenStatusSuccess ||
       ReadConvolution(convDesc, problem.conv) != miopenStatusSuccess)
        return RecordFailure(miopenStatusBadParm, "could not read the MIOpen descriptors");

    // MIOpen reads alpha1 and alpha2 as float for every tensor type, and a null
    // one as 1. alpha2 scales z, which the graph lacks, so only alpha2 = 0 can be
    // forwarded.
    if(alpha1 != nullptr && !ScalarEquals(alpha1, miopenFloat, 1.0))
        return RecordFailure(miopenStatusUnsupportedOp,
                             "hipDNN fused convolution supports only alpha1=1");
    if(alpha2 == nullptr)
        return RecordFailure(miopenStatusUnsupportedOp,
                             "alpha2 is null, which MIOpen reads as 1, and hipDNN fused "
                             "convolution does not support adding z");
    if(!ScalarEquals(alpha2, miopenFloat, 0.0))
        return RecordFailure(miopenStatusUnsupportedOp,
                             "hipDNN fused convolution supports only alpha2=0");

    if(const char* reason = CheckSupported(problem.tensors, problem.conv))
        return RecordFailure(miopenStatusUnsupportedOp, reason);
    // The fused graph is a forward convolution; a transposed one would be a
    // backward-data node followed by the bias and activation.
    if(problem.conv.mode == miopenTranspose)
        return RecordFailure(miopenStatusUnsupportedOp,
                             "transposed fused convolution is not forwarded to hipDNN");

    // RELU, the only mode accepted, ignores these. The getter just needs
    // somewhere to write them.
    double activAlpha = 0.0;
    double activBeta  = 0.0;
    double activGamma = 0.0;
    if(miopenGetActivationDescriptor_impl(
           activationDesc, &problem.act, &activAlpha, &activBeta, &activGamma) !=
       miopenStatusSuccess)
        return RecordFailure(miopenStatusBadParm, "could not read the activation descriptor");

    // Matches the status MIOpen's own fused path returns.
    if(problem.act != miopenActivationRELU)
        return RecordFailure(miopenStatusNotImplemented,
                             "only Activation Mode == miopenActivationRELU is supported");

    problem.tensors[0].alignment = PointerAlignment(x);
    problem.tensors[1].alignment = PointerAlignment(w);
    problem.tensors[2].alignment = PointerAlignment(y);
    problem.tensors[3].alignment = PointerAlignment(bias);
    VariantPack variantPack{
        {kUidA, const_cast<void*>(x)},
        {kUidB, const_cast<void*>(w)},
        {kUidBias, const_cast<void*>(bias)},
        {kUidOut, y},
    };
    return RunProblem(handle, problem, variantPack, workspace, workspaceSizeInBytes);
}

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
