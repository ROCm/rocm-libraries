// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The four forwarded convolution entry points share one shape: read the MIOpen
// descriptors through the public _impl getters, decline the combinations hipDNN
// cannot express, then hand the problem to RunCachedGraph with the function that
// builds its graph.

#include "hipdnn_conv.hpp"
#include "hipdnn_graph.hpp"
#include "hipdnn_types.hpp"

#include "miopen_impl.h"

#include <hipdnn_frontend.hpp>

#include <cstdint>
#include <memory>
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

struct TensorInfo
{
    miopenDataType_t dataType{};
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

struct ConvInfo
{
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    miopenConvolutionMode_t mode{};
    int groupCount = 1;
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

// miopenGetTensorDescriptor fills caller-provided int arrays, so the size query
// has to come first.
miopenStatus_t ReadTensor(miopenTensorDescriptor_t desc, TensorInfo& out)
{
    int size = 0;
    if(miopenGetTensorDescriptorSize_impl(desc, &size) != miopenStatusSuccess || size <= 0)
        return miopenStatusInvalidValue;

    std::vector<int> dims(static_cast<size_t>(size));
    std::vector<int> strides(static_cast<size_t>(size));
    miopenDataType_t dataType{};
    if(miopenGetTensorDescriptor_impl(desc, &dataType, dims.data(), strides.data()) !=
       miopenStatusSuccess)
        return miopenStatusInvalidValue;

    out.dataType = dataType;
    out.dims.assign(dims.begin(), dims.end());
    out.strides.assign(strides.begin(), strides.end());
    return miopenStatusSuccess;
}

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

// Boundary of what this translation has been exercised against.
constexpr size_t kMaxSpatialDims = 5;

// The vectorized layouts (NCHWc4, NCHWc8, CHWNc4, CHWNc8) have no dims-plus-
// strides spelling, which is the only way hipDNN takes a layout. There is no
// public getter for the layout, but MIOpen sets the innermost stride of a
// vectorized descriptor to the vector length, so such a tensor is exactly the
// one with no unit-stride dimension.
bool IsPlainLayout(const TensorInfo& tensor)
{
    for(const int64_t stride : tensor.strides)
    {
        if(stride == 1)
            return true;
    }
    return false;
}

miopenStatus_t CheckSupported(const std::vector<TensorInfo>& tensors, const ConvInfo& conv)
{
    fe::DataType unused{};
    if(!ComputeTypeFor(tensors.front().dataType, unused))
        return miopenStatusUnsupportedOp;
    for(const TensorInfo& tensor : tensors)
    {
        if(!ToHipdnnDataType(tensor.dataType, unused) || !IsPlainLayout(tensor))
            return miopenStatusUnsupportedOp;
    }
    if(conv.groupCount != 1)
        return miopenStatusUnsupportedOp;
    if(conv.pads.size() > kMaxSpatialDims)
        return miopenStatusUnsupportedOp;
    // A transposed convolution is not a forward-convolution op: it maps to the
    // backward-data node and is a different graph.
    if(conv.mode == miopenTranspose)
        return miopenStatusUnsupportedOp;
    return miopenStatusSuccess;
}

void AppendList(std::vector<int64_t>& out, const std::vector<int64_t>& values)
{
    out.push_back(static_cast<int64_t>(values.size()));
    out.insert(out.end(), values.begin(), values.end());
}

PlanKey MakePlanKey(miopenHandle_t handle, const ConvProblem& problem)
{
    PlanKey key;
    key.handle = handle;

    std::vector<int64_t>& out = key.problem;
    out.push_back(static_cast<int64_t>(problem.kind));
    out.push_back(static_cast<int64_t>(problem.act));
    out.push_back(static_cast<int64_t>(problem.tensors.size()));
    for(const TensorInfo& tensor : problem.tensors)
    {
        out.push_back(static_cast<int64_t>(tensor.dataType));
        AppendList(out, tensor.dims);
        AppendList(out, tensor.strides);
    }
    AppendList(out, problem.conv.pads);
    AppendList(out, problem.conv.strides);
    AppendList(out, problem.conv.dilations);
    out.push_back(static_cast<int64_t>(problem.conv.mode));
    out.push_back(problem.conv.groupCount);
    return key;
}

std::shared_ptr<fe::graph::TensorAttributes>
MakeTensor(const TensorInfo& info, fe::DataType dataType, int64_t uid)
{
    return fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                        .set_dim(info.dims)
                                        .set_stride(info.strides)
                                        .set_data_type(dataType)
                                        .set_uid(uid));
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
    }

    if(out == nullptr)
        return false;

    if(problem.kind == GraphKind::ConvBiasActivation)
    {
        // The convolution result is an intermediate now, so it needs the shape
        // the bias add will broadcast against.
        out->set_dim(outInfo.dims).set_stride(outInfo.strides);

        auto bias = MakeTensor(problem.tensors[3], types[3], kUidBias);

        fe::graph::PointwiseAttributes add;
        add.set_mode(fe::PointwiseMode::ADD).set_compute_data_type(computeType);
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
        .set_uid(kUidOut)
        .set_output(true);
    return true;
}

miopenStatus_t
RunProblem(miopenHandle_t handle, const ConvProblem& problem, VariantPack& variantPack)
{
    return RunCachedGraph(
        handle,
        MakePlanKey(handle, problem),
        [&problem](fe::graph::Graph& graph) { return PopulateGraph(problem, graph); },
        variantPack);
}

// Shared tail of the three plain convolution entry points. `a`, `b` and `out`
// are the node's inputs and result in node order.
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
                                  void* outData)
{
    if(!IsAvailable())
        return RecordFailure(miopenStatusInternalError, "hipDNN forwarding is unavailable");

    ConvProblem problem;
    problem.kind = kind;
    problem.tensors.resize(3);
    if(ReadTensor(aDesc, problem.tensors[0]) != miopenStatusSuccess ||
       ReadTensor(bDesc, problem.tensors[1]) != miopenStatusSuccess ||
       ReadTensor(outDesc, problem.tensors[2]) != miopenStatusSuccess ||
       ReadConvolution(convDesc, problem.conv) != miopenStatusSuccess)
        return RecordFailure(miopenStatusBadParm, "could not read the MIOpen descriptors");

    const miopenDataType_t dataType = problem.tensors[0].dataType;
    if(!ScalarEquals(alpha, dataType, 1.0) || !ScalarEquals(beta, dataType, 0.0))
        return RecordFailure(miopenStatusUnsupportedOp,
                             "hipDNN convolution supports only alpha=1, beta=0");

    if(const miopenStatus_t status = CheckSupported(problem.tensors, problem.conv);
       status != miopenStatusSuccess)
        return RecordFailure(status, "this convolution is not expressible as a hipDNN graph");

    VariantPack variantPack{
        {kUidA, const_cast<void*>(aData)},
        {kUidB, const_cast<void*>(bData)},
        {kUidOut, outData},
    };
    return RunProblem(handle, problem, variantPack);
}

} // namespace

// The caller's `algo` is ignored and the caller's workSpace/workSpaceSize are
// left untouched: hipDNN picks its own engine through its heuristics and
// computes its own workspace requirement, which this wrapper allocates and owns.
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
                                  void* /*workSpace*/,
                                  size_t /*workSpaceSize*/)
{
    return ForwardConvolution(
        handle, GraphKind::ConvFprop, alpha, beta, xDesc, x, wDesc, w, convDesc, yDesc, y);
}

// As with ConvolutionForward, `algo` and the caller's workspace are unused:
// hipDNN owns engine selection and its own workspace.
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
                                       void* /*workSpace*/,
                                       size_t /*workSpaceSize*/)
{
    return ForwardConvolution(
        handle, GraphKind::ConvDgrad, alpha, beta, dyDesc, dy, wDesc, w, convDesc, dxDesc, dx);
}

// As with ConvolutionForward, `algo` and the caller's workspace are unused:
// hipDNN owns engine selection and its own workspace.
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
                                          void* /*workSpace*/,
                                          size_t /*workSpaceSize*/)
{
    return ForwardConvolution(
        handle, GraphKind::ConvWgrad, alpha, beta, dyDesc, dy, xDesc, x, convDesc, dwDesc, dw);
}

// As with ConvolutionForward, `algo` and the caller's workspace are unused:
// hipDNN owns engine selection and its own workspace.
miopenStatus_t ConvolutionBiasActivationForward(miopenHandle_t handle,
                                                const void* alpha1,
                                                const miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                const miopenTensorDescriptor_t wDesc,
                                                const void* w,
                                                const miopenConvolutionDescriptor_t convDesc,
                                                miopenConvFwdAlgorithm_t /*algo*/,
                                                void* /*workspace*/,
                                                size_t /*workspaceSizeInBytes*/,
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

    const miopenDataType_t dataType = problem.tensors[0].dataType;
    // MIOpen reads a null alpha1 or alpha2 as 1. alpha2 scales z, and the graph
    // has no z, so only alpha2 = 0 can be forwarded.
    if(alpha1 != nullptr && !ScalarEquals(alpha1, dataType, 1.0))
        return RecordFailure(miopenStatusUnsupportedOp,
                             "hipDNN fused convolution supports only alpha1=1");
    if(alpha2 == nullptr)
        return RecordFailure(miopenStatusUnsupportedOp,
                             "alpha2 is null, which MIOpen reads as 1, and hipDNN fused "
                             "convolution does not support adding z");
    if(!ScalarEquals(alpha2, dataType, 0.0))
        return RecordFailure(miopenStatusUnsupportedOp,
                             "hipDNN fused convolution supports only alpha2=0");

    if(const miopenStatus_t status = CheckSupported(problem.tensors, problem.conv);
       status != miopenStatusSuccess)
        return RecordFailure(status, "this convolution is not expressible as a hipDNN graph");

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

    VariantPack variantPack{
        {kUidA, const_cast<void*>(x)},
        {kUidB, const_cast<void*>(w)},
        {kUidBias, const_cast<void*>(bias)},
        {kUidOut, y},
    };
    return RunProblem(handle, problem, variantPack);
}

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
