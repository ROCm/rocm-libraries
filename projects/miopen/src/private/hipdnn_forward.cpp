// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Implementation of the hipDNN forwarding surface declared in
// src/private/hipdnn_forward.hpp.
//
// The four entry points share one shape: read the MIOpen descriptors through the
// public _impl getters, decline the combinations hipDNN cannot express, look up
// (or build and cache) a hipDNN graph for the problem, then execute it.

#include "hipdnn_forward.hpp"

#include "miopen_impl.h"

#include <hipdnn_frontend.hpp>

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
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

    bool operator==(const TensorInfo& other) const
    {
        return dataType == other.dataType && dims == other.dims && strides == other.strides;
    }
};

struct ConvInfo
{
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    miopenConvolutionMode_t mode{};
    int groupCount = 1;

    bool operator==(const ConvInfo& other) const
    {
        return pads == other.pads && strides == other.strides && dilations == other.dilations &&
               mode == other.mode && groupCount == other.groupCount;
    }
};

enum class NodeKind
{
    Fprop,
    Dgrad,
    Wgrad,
    FusedBiasActivation,
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

bool ToHipdnnDataType(miopenDataType_t type, fe::DataType& out)
{
    switch(type)
    {
    case miopenHalf: out = fe::DataType::HALF; return true;
    case miopenFloat: out = fe::DataType::FLOAT; return true;
    case miopenBFloat16: out = fe::DataType::BFLOAT16; return true;
    case miopenDouble: out = fe::DataType::DOUBLE; return true;
    case miopenInt8: out = fe::DataType::INT8; return true;
    case miopenInt32: out = fe::DataType::INT32; return true;
    case miopenInt64: out = fe::DataType::INT64; return true;
    case miopenFloat8_fnuz: out = fe::DataType::FP8_E4M3_FNUZ; return true;
    case miopenBFloat8_fnuz: out = fe::DataType::FP8_E5M2_FNUZ; return true;
    }
    return false;
}

// Accumulate wider than you store: half and bfloat16 both compute in fp32.
bool ComputeTypeFor(miopenDataType_t type, fe::DataType& out)
{
    switch(type)
    {
    case miopenHalf:
    case miopenFloat:
    case miopenBFloat16:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz: out = fe::DataType::FLOAT; return true;
    case miopenDouble: out = fe::DataType::DOUBLE; return true;
    case miopenInt8:
    case miopenInt32:
    case miopenInt64: out = fe::DataType::INT32; return true;
    }
    return false;
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
    const TensorInfo& reference = tensors.front();
    fe::DataType unused{};
    if(!ToHipdnnDataType(reference.dataType, unused) || !ComputeTypeFor(reference.dataType, unused))
        return miopenStatusUnsupportedOp;
    for(const TensorInfo& tensor : tensors)
    {
        if(!IsPlainLayout(tensor))
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

struct PlanKey
{
    miopenHandle_t handle = nullptr;
    std::vector<TensorInfo> tensors;
    ConvInfo conv;
    NodeKind nodeKind          = NodeKind::Fprop;
    miopenActivationMode_t act = miopenActivationPASTHRU;

    bool operator==(const PlanKey& other) const
    {
        return handle == other.handle && tensors == other.tensors && conv == other.conv &&
               nodeKind == other.nodeKind && act == other.act;
    }
};

struct PlanKeyHash
{
    static void Mix(size_t& seed, size_t value)
    {
        seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    static void MixAll(size_t& seed, const std::vector<int64_t>& values)
    {
        for(const int64_t value : values)
            Mix(seed, static_cast<size_t>(value));
    }

    size_t operator()(const PlanKey& key) const
    {
        size_t seed = std::hash<const void*>{}(key.handle);
        for(const auto& tensor : key.tensors)
        {
            Mix(seed, static_cast<size_t>(tensor.dataType));
            MixAll(seed, tensor.dims);
            MixAll(seed, tensor.strides);
        }
        MixAll(seed, key.conv.pads);
        MixAll(seed, key.conv.strides);
        MixAll(seed, key.conv.dilations);
        Mix(seed, static_cast<size_t>(key.conv.mode));
        Mix(seed, static_cast<size_t>(key.conv.groupCount));
        Mix(seed, static_cast<size_t>(key.nodeKind));
        Mix(seed, static_cast<size_t>(key.act));
        return seed;
    }
};

// Everything hipDNN needs that is tied to one MIOpen handle. The workspace is
// reused across calls rather than reallocated: the hipDNN handle runs on the
// MIOpen handle's stream, so successive forwarded calls on that handle are
// already ordered against each other.
struct HandleState
{
    fe::HipdnnHandlePtr hipdnnHandle;
    hipStream_t stream   = nullptr;
    void* workspace      = nullptr;
    size_t workspaceSize = 0;
    std::mutex mutex;

    ~HandleState()
    {
        if(workspace != nullptr)
            static_cast<void>(hipFree(workspace));
    }

    bool EnsureWorkspace(size_t bytes)
    {
        if(bytes <= workspaceSize)
            return true;
        void* grown = nullptr;
        if(hipMalloc(&grown, bytes) != hipSuccess)
            return false;
        if(workspace != nullptr)
            static_cast<void>(hipFree(workspace));
        workspace     = grown;
        workspaceSize = bytes;
        return true;
    }
};

std::mutex& HandleMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<miopenHandle_t, std::unique_ptr<HandleState>>& HandleMap()
{
    static std::unordered_map<miopenHandle_t, std::unique_ptr<HandleState>> handles;
    return handles;
}

// Created on first forwarded call rather than in miopenCreate, so a process that
// never forwards never pays for hipdnnCreate.
HandleState* AcquireHandleState(miopenHandle_t handle)
{
    hipStream_t stream = nullptr;
    if(miopenGetStream_impl(handle, &stream) != miopenStatusSuccess)
        return nullptr;

    const std::lock_guard<std::mutex> lock(HandleMutex());
    auto& slot = HandleMap()[handle];
    if(slot == nullptr)
    {
        auto [created, error] = fe::createHipdnnHandle(stream);
        if(!error.is_good() || created == nullptr)
        {
            HandleMap().erase(handle);
            return nullptr;
        }
        slot               = std::make_unique<HandleState>();
        slot->hipdnnHandle = std::move(created);
        slot->stream       = stream;
    }
    else if(slot->stream != stream)
    {
        // miopenSetStream can move a handle to a different stream at any point.
        if(!fe::setHipdnnHandleStream(slot->hipdnnHandle, stream).is_good())
            return nullptr;
        slot->stream = stream;
    }
    return slot.get();
}

using GraphPtr = std::shared_ptr<fe::graph::Graph>;

std::mutex& PlanMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<PlanKey, GraphPtr, PlanKeyHash>& PlanCache()
{
    static std::unordered_map<PlanKey, GraphPtr, PlanKeyHash> plans;
    return plans;
}

miopenStatus_t TranslateHipdnnError(const fe::Error& error)
{
    const auto code = error.get_code();

    // Not a switch: hipDNN owns this enum and MIOpen compiles with -Wswitch-enum
    // -Werror, so any value it gains later would break the build here instead of
    // landing on the catch-all, which is already the right answer for it.
    if(code == fe::ErrorCode::GRAPH_NOT_SUPPORTED ||
       code == fe::ErrorCode::UNSUPPORTED_GRAPH_FORMAT)
        return miopenStatusUnsupportedOp;

    if(code == fe::ErrorCode::INVALID_VALUE || code == fe::ErrorCode::ATTRIBUTE_NOT_SET ||
       code == fe::ErrorCode::SHAPE_DEDUCTION_FAILED ||
       code == fe::ErrorCode::INVALID_TENSOR_NAME || code == fe::ErrorCode::INVALID_VARIANT_PACK)
        return miopenStatusBadParm;

    return miopenStatusInternalError;
}

struct LastForwardedError
{
    bool failed = false;
    miopenStatus_t status{};
    std::string message;
};

// Per-thread so that one thread's forwarded failure cannot be attributed to
// another thread's miopenGetErrorString call.
LastForwardedError& LastError()
{
    static thread_local LastForwardedError last;
    return last;
}

miopenStatus_t RecordFailure(miopenStatus_t status, std::string message)
{
    LastError() = LastForwardedError{true, status, std::move(message)};
    return status;
}

miopenStatus_t RecordHipdnnFailure(const fe::Error& error)
{
    return RecordFailure(TranslateHipdnnError(error), error.get_message());
}

miopenStatus_t RecordSuccess()
{
    LastError().failed = false;
    return miopenStatusSuccess;
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

bool ToPointwiseActivation(miopenActivationMode_t mode, fe::PointwiseMode& out)
{
    // Deliberately narrow. The remaining MIOpen activation modes carry alpha,
    // beta or gamma coefficients that the plain pointwise node has nowhere to
    // put, so they are declined rather than silently approximated.
    if(mode == miopenActivationRELU)
    {
        out = fe::PointwiseMode::RELU_FWD;
        return true;
    }
    return false;
}

// Builds the graph for one problem. `a` and `b` are the two convolution inputs
// in node order, `out` is the node's result; bias is present only for the fused
// kind.
bool PopulateGraph(const PlanKey& key, fe::graph::Graph& graph)
{
    fe::DataType ioType{};
    fe::DataType computeType{};
    if(!ToHipdnnDataType(key.tensors.front().dataType, ioType) ||
       !ComputeTypeFor(key.tensors.front().dataType, computeType))
        return false;

    // Without the intermediate type, the virtual tensors between the fused
    // graph's nodes are left with no data type at all and the build fails.
    graph.set_io_data_type(ioType)
        .set_compute_data_type(computeType)
        .set_intermediate_data_type(computeType);

    auto a                    = MakeTensor(key.tensors[0], ioType, kUidA);
    auto b                    = MakeTensor(key.tensors[1], ioType, kUidB);
    const TensorInfo& outInfo = key.tensors[2];

    std::shared_ptr<fe::graph::TensorAttributes> out;
    switch(key.nodeKind)
    {
    case NodeKind::Fprop:
    case NodeKind::FusedBiasActivation: {
        fe::graph::ConvFpropAttributes conv;
        conv.set_padding(key.conv.pads)
            .set_stride(key.conv.strides)
            .set_dilation(key.conv.dilations);
        out = graph.conv_fprop(a, b, conv);
        break;
    }
    case NodeKind::Dgrad: {
        // MIOpen's padding is symmetric -- one array -- so pre and post both get
        // the same values.
        fe::graph::ConvDgradAttributes conv;
        conv.set_pre_padding(key.conv.pads)
            .set_post_padding(key.conv.pads)
            .set_stride(key.conv.strides)
            .set_dilation(key.conv.dilations);
        out = graph.conv_dgrad(a, b, conv);
        break;
    }
    case NodeKind::Wgrad: {
        fe::graph::ConvWgradAttributes conv;
        conv.set_pre_padding(key.conv.pads)
            .set_post_padding(key.conv.pads)
            .set_stride(key.conv.strides)
            .set_dilation(key.conv.dilations);
        out = graph.conv_wgrad(a, b, conv);
        break;
    }
    }

    if(out == nullptr)
        return false;

    if(key.nodeKind == NodeKind::FusedBiasActivation)
    {
        // The convolution result is an intermediate now, so it needs the shape
        // the bias add will broadcast against.
        out->set_dim(outInfo.dims).set_stride(outInfo.strides);

        auto bias = MakeTensor(key.tensors[3], ioType, kUidBias);

        fe::graph::PointwiseAttributes add;
        add.set_mode(fe::PointwiseMode::ADD).set_compute_data_type(computeType);
        out = graph.pointwise(out, bias, add);

        if(key.act != miopenActivationPASTHRU)
        {
            fe::PointwiseMode activation{};
            if(!ToPointwiseActivation(key.act, activation))
                return false;
            out->set_dim(outInfo.dims).set_stride(outInfo.strides);

            fe::graph::PointwiseAttributes activationAttributes;
            activationAttributes.set_mode(activation).set_compute_data_type(computeType);
            out = graph.pointwise(out, activationAttributes);
        }
    }

    out->set_dim(outInfo.dims).set_stride(outInfo.strides).set_uid(kUidOut).set_output(true);
    return true;
}

// Held across build() on purpose. A build can take seconds, but it happens once
// per distinct problem and serializing it is far simpler than letting two
// threads race to build the same graph.
std::pair<GraphPtr, miopenStatus_t> AcquireGraph(const PlanKey& key, hipdnnHandle_t hipdnnHandle)
{
    const std::lock_guard<std::mutex> lock(PlanMutex());

    auto found = PlanCache().find(key);
    if(found != PlanCache().end())
        return {found->second, miopenStatusSuccess};

    GraphPtr graph = std::make_shared<fe::graph::Graph>();
    if(!PopulateGraph(key, *graph))
        return {nullptr, miopenStatusUnsupportedOp};

    const fe::Error error = graph->build(hipdnnHandle);
    if(!error.is_good())
        return {nullptr, RecordHipdnnFailure(error)};

    PlanCache().emplace(key, graph);
    return {graph, miopenStatusSuccess};
}

miopenStatus_t
RunGraph(HandleState& state, const GraphPtr& graph, std::unordered_map<int64_t, void*>& variantPack)
{
    const std::lock_guard<std::mutex> lock(state.mutex);

    int64_t workspaceSize = 0;
    if(const fe::Error error = graph->get_workspace_size(workspaceSize); !error.is_good())
        return RecordHipdnnFailure(error);

    if(!state.EnsureWorkspace(static_cast<size_t>(workspaceSize)))
        return RecordFailure(miopenStatusAllocFailed, "hipDNN workspace allocation failed");

    const fe::Error error = graph->execute(*state.hipdnnHandle, variantPack, state.workspace);
    if(!error.is_good())
        return RecordHipdnnFailure(error);

    return RecordSuccess();
}

// Shared tail of the three plain convolution entry points. `a`, `b` and `out`
// are the node's inputs and result in node order.
miopenStatus_t ForwardConvolution(miopenHandle_t handle,
                                  NodeKind nodeKind,
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

    PlanKey key;
    key.handle   = handle;
    key.nodeKind = nodeKind;
    key.tensors.resize(3);
    if(ReadTensor(aDesc, key.tensors[0]) != miopenStatusSuccess ||
       ReadTensor(bDesc, key.tensors[1]) != miopenStatusSuccess ||
       ReadTensor(outDesc, key.tensors[2]) != miopenStatusSuccess ||
       ReadConvolution(convDesc, key.conv) != miopenStatusSuccess)
        return RecordFailure(miopenStatusBadParm, "could not read the MIOpen descriptors");

    const miopenDataType_t dataType = key.tensors[0].dataType;
    if(!ScalarEquals(alpha, dataType, 1.0) || !ScalarEquals(beta, dataType, 0.0))
        return RecordFailure(miopenStatusUnsupportedOp,
                             "hipDNN convolution supports only alpha=1, beta=0");

    if(const miopenStatus_t status = CheckSupported(key.tensors, key.conv);
       status != miopenStatusSuccess)
        return RecordFailure(status, "this convolution is not expressible as a hipDNN graph");

    HandleState* state = AcquireHandleState(handle);
    if(state == nullptr)
        return RecordFailure(miopenStatusInternalError, "could not create a hipDNN handle");

    auto [graph, status] = AcquireGraph(key, *state->hipdnnHandle);
    if(graph == nullptr)
        return status;

    std::unordered_map<int64_t, void*> variantPack{
        {kUidA, const_cast<void*>(aData)},
        {kUidB, const_cast<void*>(bData)},
        {kUidOut, outData},
    };
    return RunGraph(*state, graph, variantPack);
}

} // namespace

// Creating a handle is the cheapest thing that exercises the whole chain: the
// dynamic frontend dlopens the backend here, and a backend reporting a version
// this build cannot talk to fails here rather than at the first convolution.
bool IsAvailable()
{
    static const bool available = [] {
        auto [handle, error] = fe::createHipdnnHandle();
        const bool probed    = error.is_good() && handle != nullptr;
        if(!probed)
        {
            std::cerr << "[MIOpen] hipDNN forwarding is unavailable: libhipdnn_backend.so could "
                         "not be loaded, or reports a version this MIOpen was not built "
                         "against.\n";
        }
        return probed;
    }();
    return available;
}

void ReleaseHandle(miopenHandle_t handle)
{
    {
        const std::lock_guard<std::mutex> lock(PlanMutex());
        for(auto it = PlanCache().begin(); it != PlanCache().end();)
            it = it->first.handle == handle ? PlanCache().erase(it) : std::next(it);
    }

    const std::lock_guard<std::mutex> lock(HandleMutex());
    HandleMap().erase(handle);
}

const char* PrefixedErrorString(miopenStatus_t status, const char* nativeMessage)
{
    const LastForwardedError& last = LastError();
    if(!last.failed || last.status != status || nativeMessage == nullptr)
        return nullptr;

    // miopenGetErrorString returns a bare const char* the caller does not own,
    // so the prefixed text has to outlive this call without being leaked.
    static thread_local std::string prefixed;
    prefixed = "[hipDNN-forwarded] " + std::string(nativeMessage);
    if(!last.message.empty())
        prefixed += ": " + last.message;
    return prefixed.c_str();
}

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
        handle, NodeKind::Fprop, alpha, beta, xDesc, x, wDesc, w, convDesc, yDesc, y);
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
        handle, NodeKind::Dgrad, alpha, beta, dyDesc, dy, wDesc, w, convDesc, dxDesc, dx);
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
        handle, NodeKind::Wgrad, alpha, beta, dyDesc, dy, xDesc, x, convDesc, dwDesc, dw);
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

    PlanKey key;
    key.handle   = handle;
    key.nodeKind = NodeKind::FusedBiasActivation;
    key.tensors.resize(4);
    if(ReadTensor(xDesc, key.tensors[0]) != miopenStatusSuccess ||
       ReadTensor(wDesc, key.tensors[1]) != miopenStatusSuccess ||
       ReadTensor(yDesc, key.tensors[2]) != miopenStatusSuccess ||
       ReadTensor(biasDesc, key.tensors[3]) != miopenStatusSuccess ||
       ReadConvolution(convDesc, key.conv) != miopenStatusSuccess)
        return RecordFailure(miopenStatusBadParm, "could not read the MIOpen descriptors");

    const miopenDataType_t dataType = key.tensors[0].dataType;
    // alpha2 scales the z tensor; with no scaling attribute to carry it, the
    // only expressible case is the one where z drops out of the graph entirely.
    if(!ScalarEquals(alpha1, dataType, 1.0) || !ScalarEquals(alpha2, dataType, 0.0))
        return RecordFailure(miopenStatusUnsupportedOp,
                             "hipDNN fused convolution supports only alpha1=1, alpha2=0");

    if(const miopenStatus_t status = CheckSupported(key.tensors, key.conv);
       status != miopenStatusSuccess)
        return RecordFailure(status, "this convolution is not expressible as a hipDNN graph");

    double activAlpha = 0.0;
    double activBeta  = 0.0;
    double activGamma = 0.0;
    if(miopenGetActivationDescriptor_impl(
           activationDesc, &key.act, &activAlpha, &activBeta, &activGamma) != miopenStatusSuccess)
        return RecordFailure(miopenStatusBadParm, "could not read the activation descriptor");

    fe::PointwiseMode unused{};
    if(key.act != miopenActivationPASTHRU && !ToPointwiseActivation(key.act, unused))
        return RecordFailure(miopenStatusUnsupportedOp,
                             "this activation mode has no hipDNN pointwise equivalent");

    HandleState* state = AcquireHandleState(handle);
    if(state == nullptr)
        return RecordFailure(miopenStatusInternalError, "could not create a hipDNN handle");

    auto [graph, status] = AcquireGraph(key, *state->hipdnnHandle);
    if(graph == nullptr)
        return status;

    std::unordered_map<int64_t, void*> variantPack{
        {kUidA, const_cast<void*>(x)},
        {kUidB, const_cast<void*>(w)},
        {kUidBias, const_cast<void*>(bias)},
        {kUidOut, y},
    };
    return RunGraph(*state, graph, variantPack);
}

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
