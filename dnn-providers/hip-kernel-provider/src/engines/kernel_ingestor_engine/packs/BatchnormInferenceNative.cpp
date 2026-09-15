// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "core/Utils.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file BatchnormInferenceNative.cpp
 * @brief The batchnorm-inference engine's native half: matching, scoring, dispatch, and
 *        the one function that registers them.
 *
 * One pack, one operation, so a single graph matcher both admits the node type and
 * validates it -- the ConvNative shape rather than the Pointwise one, which splits an
 * applicability matcher across three packs.
 *
 * The kernel (kernels/BatchnormInference.cpp) is layout-agnostic: it walks the logical
 * extents of x and addresses memory through each operand's own strides, so packed NCHW,
 * packed NHWC and padded inputs are the same code with different stride arguments. What
 * it is NOT agnostic about is rank -- it indexes n/c/h/w explicitly -- so the matcher
 * refuses anything but rank 4 rather than letting a 3-D or 5-D graph in and
 * miscomputing. The frontend admits both.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.batchnorm_inference.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.batchnorm_inference.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.batchnorm_inference.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.batchnorm_inference.dispatch";

// KMD fields this pack varies along, and the tokens matching binds for dispatch.
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view IO_DTYPE_FIELD = "io_dtype";
constexpr std::string_view PARAM_DTYPE_FIELD = "param_dtype";
constexpr std::string_view X_TOKEN = "batchnorm_inference.x.uid";
constexpr std::string_view MEAN_TOKEN = "batchnorm_inference.mean.uid";
constexpr std::string_view INV_VARIANCE_TOKEN = "batchnorm_inference.inv_variance.uid";
constexpr std::string_view SCALE_TOKEN = "batchnorm_inference.scale.uid";
constexpr std::string_view BIAS_TOKEN = "batchnorm_inference.bias.uid";
constexpr std::string_view Y_TOKEN = "batchnorm_inference.y.uid";

/// x and y are N,C,H,W and the per-channel operands are [1,C,1,1]; the kernel unravels
/// a flat index into exactly four coordinates, so rank 4 is not a simplification here.
constexpr uint32_t SUPPORTED_RANK = 4;
/// Channel is logical dim 1, as the frontend's node hard-codes
/// (BatchnormInferenceNode.hpp:110) and as the kernel's `c * stride1` addressing
/// assumes for the per-channel operands.
constexpr flatbuffers::uoffset_t CHANNEL_AXIS = 1;

/// The kernel computes in float regardless of storage width -- BnCompute is `float`
/// unconditionally (BatchnormInferenceTypes.h) -- so the node's compute precision must
/// be float whatever the operands are stored as.
constexpr data_objects::DataType SUPPORTED_COMPUTE_DATA_TYPE = data_objects::DataType::FLOAT;

/// BnIoElement is the storage type of x and y. All three tags the kernel's header maps
/// are admitted: the CPU reference registers a plan for each of them paired with float
/// parameters and float compute -- (io, FLOAT, FLOAT, io, FLOAT) in
/// BatchnormFwdInferenceSignatureKey.hpp -- so each is numerically provable rather than
/// merely compilable.
constexpr std::array<data_objects::DataType, 3> SUPPORTED_IO_DATA_TYPES = {
    data_objects::DataType::FLOAT,
    data_objects::DataType::HALF,
    data_objects::DataType::BFLOAT16,
};

/// BnParamElement is the storage type of the four per-channel operands, and it is pinned
/// to float. The header would accept a 16-bit tag here and the pairing is legal, but no
/// graph in any inventoried corpus carries 16-bit parameters -- every one pairs a 16-bit
/// or float io dtype with float mean/inv_variance/scale/bias. A candidate compiled for a
/// 16-bit parameter type could therefore never be selected, so none ships.
constexpr data_objects::DataType SUPPORTED_PARAM_DATA_TYPE = data_objects::DataType::FLOAT;

/// True when @p dataType is one of the io storage types a shipped candidate is compiled
/// for.
bool isSupportedIoDataType(data_objects::DataType dataType)
{
    return std::find(SUPPORTED_IO_DATA_TYPES.begin(), SUPPORTED_IO_DATA_TYPES.end(), dataType)
           != SUPPORTED_IO_DATA_TYPES.end();
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched batchnorm-inference graph binds, in kernel argument order.
struct BatchnormInferenceBinding
{
    int64_t x = 0;
    int64_t mean = 0;
    int64_t invVariance = 0;
    int64_t scale = 0;
    int64_t bias = 0;
    int64_t y = 0;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// True when @p tensor is rank 4 real device data stored as @p dataType, with a stride
/// per dim that the kernel can multiply a coordinate by.
///
/// The dtype is a parameter rather than a constant because the kernel's two storage
/// types are independent: callers pass the graph's io dtype for x and y, and
/// SUPPORTED_PARAM_DATA_TYPE for the per-channel operands.
///
/// Strides are otherwise unconstrained -- that is the point of this kernel, and it is
/// why there is no packed-layout check here as ConvNative has. A stride of 0 is the one
/// refusal: on an axis of extent > 1 it is a broadcast, which for x would read one
/// element repeatedly and for y would have every thread on that axis race to write the
/// same address. An extent-1 axis may carry any stride, because its coordinate is
/// always 0 and the term drops out.
bool isSupportedOperand(const data_objects::TensorAttributes& tensor,
                        data_objects::DataType dataType)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || strides->size() != dims->size()
       || dims->size() != SUPPORTED_RANK)
    {
        return false;
    }

    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        if(dims->Get(axis) < 1)
        {
            return false;
        }
        if(dims->Get(axis) > 1 && strides->Get(axis) < 1)
        {
            return false;
        }
    }

    if(tensor.virtual_())
    {
        return false;
    }

    // A rank-4 tensor is also the shape a pass-by-value scalar can take; that variant-
    // pack slot holds a host pointer, not a device one.
    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor))
    {
        return false;
    }

    return tensor.data_type() == dataType;
}

/// True when @p tensor is a per-channel operand for @p channels: [1, C, 1, 1].
///
/// The frontend already enforces this via validateChannelOnlyTensorShape, but a matcher
/// runs on an unvalidated graph and the kernel reads `operand[c * strides[1]]` for
/// c < C -- a shorter operand would be read past its end with nothing raising an error.
bool isChannelOperand(const data_objects::TensorAttributes& tensor, int64_t channels)
{
    if(!isSupportedOperand(tensor, SUPPORTED_PARAM_DATA_TYPE))
    {
        return false;
    }

    const auto* dims = tensor.dims();
    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        const int64_t expected = axis == CHANNEL_AXIS ? channels : 1;
        if(dims->Get(axis) != expected)
        {
            return false;
        }
    }
    return true;
}

std::string dataTypeName(data_objects::DataType dataType)
{
    return data_objects::EnumNameDataType(dataType);
}

/// The node this engine's matchers read, or nullptr if the graph is not a single
/// batchnorm-inference node.
const data_objects::BatchnormInferenceAttributes*
    batchnormInferenceNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::BatchnormInferenceAttributes)
    {
        return nullptr;
    }

    return &node.attributesAs<data_objects::BatchnormInferenceAttributes>();
}

/// x's logical extents, for the candidate matcher and the scorer -- the kernel's
/// iteration space is exactly this product.
std::optional<int64_t> graphElementCount(const MatchContext& context)
{
    const auto* attributes = batchnormInferenceNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, attributes->x_tensor_uid());
    if(x == nullptr || x->dims() == nullptr || x->dims()->size() != SUPPORTED_RANK)
    {
        return std::nullopt;
    }

    int64_t total = 1;
    for(const auto dim : *x->dims())
    {
        total *= dim;
    }
    return total;
}

/**
 * @brief Graph-scoped applicability: is this the one batchnorm-inference shape this
 *        engine's kernel can launch?
 *
 * The attributes table has exactly six uid fields and nothing else -- no epsilon, no
 * scalar attributes, no optional uids, no layout enum
 * (flatbuffers_sdk/schemas/batchnorm_inference_attributes.fbs:10-17) -- so every field of
 * it is consumed below. Epsilon in particular is not omitted here: the operation has
 * none, it is pre-baked into inv_variance, and adding one anywhere would be a silent
 * numeric defect. The one field outside that table this matcher must still answer for is
 * the node's own `compute_data_type` (graph.fbs:79), gated below.
 */
std::optional<BoundTokens> batchnormInferenceGraphMatches(const MatchContext& context)
{
    const auto* attributesPtr = batchnormInferenceNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // BnCompute is float unconditionally (BatchnormInferenceTypes.h), so a node asking
    // for any other compute precision would be served at a precision it did not ask for
    // and compared against a reference that honoured the request. Not inert, so refused
    // rather than ignored -- it is the one node field outside the attributes table.
    if(context.graph.getNode(0).compute_data_type() != SUPPORTED_COMPUTE_DATA_TYPE)
    {
        return std::nullopt;
    }

    const auto xUid = attributes.x_tensor_uid();
    const auto meanUid = attributes.mean_tensor_uid();
    const auto invVarianceUid = attributes.inv_variance_tensor_uid();
    const auto scaleUid = attributes.scale_tensor_uid();
    const auto biasUid = attributes.bias_tensor_uid();
    const auto yUid = attributes.y_tensor_uid();

    // The aliasing gate, and the reason the kernel keeps `__restrict__` on x and y.
    //
    // y is the only pointer written through, so it is the only one whose aliasing is
    // undefined behaviour: two const inputs naming one tensor is a read/read overlap,
    // which `restrict` permits because nothing modifies the object. y naming any input
    // is not, and the frontend does not forbid it. Refusing it here is the deliberate
    // half of that trade -- an in-place graph is declined rather than miscompiled, and
    // every graph that is not in-place keeps the alias information. Nothing proved the
    // in-place numerics either, so admitting it would be claiming a shape no run
    // covered.
    if(yUid == xUid || yUid == meanUid || yUid == invVarianceUid || yUid == scaleUid
       || yUid == biasUid)
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, xUid);
    const auto* mean = findTensor(context, meanUid);
    const auto* invVariance = findTensor(context, invVarianceUid);
    const auto* scale = findTensor(context, scaleUid);
    const auto* bias = findTensor(context, biasUid);
    const auto* y = findTensor(context, yUid);
    if(x == nullptr || mean == nullptr || invVariance == nullptr || scale == nullptr
       || bias == nullptr || y == nullptr)
    {
        return std::nullopt;
    }

    // x's dtype is the graph's io dtype, and y is checked against that same value rather
    // than against a constant. The kernel has one BnIoElement covering both operands, so
    // a graph storing x and y at different widths has no candidate that can serve it:
    // whichever tag were baked, one of the two pointers would be read or written at the
    // wrong width. That pairing is real, not hypothetical -- the CPU reference registers
    // (HALF, FLOAT, FLOAT, FLOAT, FLOAT) and its bfloat16 twin, so mixed-io graphs exist
    // and are representable. They are declined here.
    const auto ioDataType = x->data_type();
    if(!isSupportedIoDataType(ioDataType))
    {
        return std::nullopt;
    }
    if(!isSupportedOperand(*x, ioDataType) || !isSupportedOperand(*y, ioDataType))
    {
        return std::nullopt;
    }

    // y's extents, not merely its element count: the kernel unravels one flat index
    // against x's dims and applies it to both operands, so a y shaped differently would
    // be written at coordinates that mean something else.
    const auto* xDims = x->dims();
    const auto* yDims = y->dims();
    for(flatbuffers::uoffset_t axis = 0; axis < xDims->size(); ++axis)
    {
        if(xDims->Get(axis) != yDims->Get(axis))
        {
            return std::nullopt;
        }
    }

    const auto channels = xDims->Get(CHANNEL_AXIS);
    if(!isChannelOperand(*mean, channels) || !isChannelOperand(*invVariance, channels)
       || !isChannelOperand(*scale, channels) || !isChannelOperand(*bias, channels))
    {
        return std::nullopt;
    }

    // Binds operand uids for the dispatch handler to read back rather than re-deriving
    // them from the graph.
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = xUid;
    bound[std::string(MEAN_TOKEN)] = meanUid;
    bound[std::string(INV_VARIANCE_TOKEN)] = invVarianceUid;
    bound[std::string(SCALE_TOKEN)] = scaleUid;
    bound[std::string(BIAS_TOKEN)] = biasUid;
    bound[std::string(Y_TOKEN)] = yUid;
    return bound;
}

/**
 * @brief Kernel-scoped applicability: are this candidate's two baked dtypes the graph's?
 *
 * Two fields rather than one, because the kernel's storage types are independent:
 * BnIoElement covers x and y, BnParamElement covers the four per-channel operands, and
 * the suite's own batchnorm graphs routinely pair a 16-bit io dtype with float
 * parameters. Comparing only one of them would let a candidate compiled for the wrong
 * parameter width read those buffers at the wrong stride.
 */
bool batchnormInferenceKernelMatches(const MatchContext& context,
                                     const BoundTokens& /*bound*/,
                                     const KernelDefinition& kernel)
{
    const auto* attributes = batchnormInferenceNode(context);
    if(attributes == nullptr)
    {
        return false;
    }

    const auto* x = findTensor(context, attributes->x_tensor_uid());
    const auto* mean = findTensor(context, attributes->mean_tensor_uid());
    if(x == nullptr || mean == nullptr)
    {
        return false;
    }

    return kernel.getStringMetadata(std::string(IO_DTYPE_FIELD)) == dataTypeName(x->data_type())
           && kernel.getStringMetadata(std::string(PARAM_DTYPE_FIELD))
                  == dataTypeName(mean->data_type());
}

/**
 * @brief Ranks the block-size variants by how little of the launched grid is idle.
 *
 * The free axis is block size, and the kernel guards its own bounds, so every candidate
 * is correct for every admitted shape and the ranking is purely about the rounded-up
 * final block. occupancy = total / (gridDim.x * blockDim.x) in (0, 1]: at 1 element a
 * 64-thread block wastes 63 lanes and a 1024-thread block wastes 1023, and this says so.
 * The `+ block` term is a tie-break, not a preference: it only separates candidates whose
 * occupancy is equal, where the larger block launches fewer blocks. Scaled small enough
 * that it can never outweigh an occupancy difference, since the smallest non-zero such
 * difference over admitted shapes is far above 1024 * 1e-9.
 */
double batchnormInferenceScore(const MatchContext& context,
                               const BoundTokens& /*bound*/,
                               const KernelDefinition& kernel)
{
    const auto blockSize = kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD));
    const auto total = graphElementCount(context);
    if(!total.has_value() || blockSize <= 0)
    {
        return 0.0;
    }

    const int64_t blocks = (*total + blockSize - 1) / blockSize;
    const double launched = static_cast<double>(blocks) * static_cast<double>(blockSize);
    return static_cast<double>(*total) / launched + static_cast<double>(blockSize) * 1e-9;
}

/**
 * @brief Re-reads the operand bindings a match established.
 *
 * @throws HipdnnPluginException if the graph is not one this matcher accepts.
 */
BatchnormInferenceBinding batchnormInferenceBinding(const BoundTokens& bound)
{
    // Every token was written by the engine's graph match, which admitted this graph; a
    // missing one means the catalog was built by an engine other than ours.
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "batchnorm_inference dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };

    return {read(X_TOKEN),
            read(MEAN_TOKEN),
            read(INV_VARIANCE_TOKEN),
            read(SCALE_TOKEN),
            read(BIAS_TOKEN),
            read(Y_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The 16 integers the kernel takes after its six pointers: x's logical extents, then
/// x's and y's own per-dim strides, then each per-channel operand's own channel stride.
struct BatchnormInferenceGeometry
{
    int64_t dims[SUPPORTED_RANK] = {0, 0, 0, 0};
    int64_t xStrides[SUPPORTED_RANK] = {0, 0, 0, 0};
    int64_t yStrides[SUPPORTED_RANK] = {0, 0, 0, 0};
    int64_t meanStrideC = 0;
    int64_t invVarianceStrideC = 0;
    int64_t scaleStrideC = 0;
    int64_t biasStrideC = 0;
};

/// The compiled kernel plus the geometry its launch needs, owning nothing that points
/// back into the MatchContext it was built from.
class PreparedBatchnormInference : public PreparedDispatch
{
public:
    PreparedBatchnormInference(std::unique_ptr<compilation::ICompiledProgram> program,
                               std::unique_ptr<compilation::IRunnableKernel> kernel,
                               BatchnormInferenceBinding binding,
                               BatchnormInferenceGeometry geometry)
        : _program(std::move(program))
        , _kernel(std::move(kernel))
        , _binding(binding)
        , _geometry(geometry)
    {
    }

    const compilation::IRunnableKernel& kernel() const
    {
        return *_kernel;
    }

    const BatchnormInferenceBinding& binding() const
    {
        return _binding;
    }

    const BatchnormInferenceGeometry& geometry() const
    {
        return _geometry;
    }

private:
    // The runnable kernel is a view into its program's module, so the program must
    // outlive it; both are held here for the plan's lifetime.
    std::unique_ptr<compilation::ICompiledProgram> _program;
    std::unique_ptr<compilation::IRunnableKernel> _kernel;
    BatchnormInferenceBinding _binding;
    BatchnormInferenceGeometry _geometry;
};

/// The tag BatchnormInferenceTypes.h maps to a device type, from a dtype metadata field.
///
/// The tag vocabulary is the header's, not FlatBuffers' -- they coincide for the three
/// the header defines, which is why the descriptor can carry one string for both the
/// candidate match and the define.
std::string dtypeTagFor(const KernelDefinition& kernel, std::string_view field)
{
    const auto& dtype = kernel.getStringMetadata(std::string(field));
    if(dtype == "FLOAT" || dtype == "HALF" || dtype == "BFLOAT16")
    {
        return dtype;
    }

    // Unreachable via matching, which admits only dtypes this pack declares.
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "kernel '" + toString(kernel.kernelId)
                                                       + "' declares unsupported "
                                                       + std::string(field) + " '" + dtype + "'");
}

const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    if(it == tensors.end() || it->second == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched batchnorm_inference graph has no tensor for uid " + std::to_string(uid));
    }
    return *it->second;
}

/// That operand's OWN channel stride, never x's.
///
/// The reference reads the per-channel operands as getHostValue(0, c), a 2-index inner
/// product against the operand's own rank-4 stride vector
/// (data_sdk/include/hipdnn_data_sdk/utilities/Tensor.hpp:562-565). Taking this from x
/// instead produces correct numbers on a packed template graph and wrong ones on every
/// other layout, which is exactly the defect that would survive a single-shape test.
int64_t channelStrideOf(const data_objects::TensorAttributes& tensor)
{
    return tensor.strides()->Get(CHANNEL_AXIS);
}

/**
 * @brief The native dispatch behind this pack's UDD: sizes and launches the batchnorm
 *        inference kernel. Everything graph- and kernel-derived resolves once at
 *        prepare(); execute() only resolves buffers and launches, so nothing mutates
 *        once prepared and concurrent execution is safe.
 */
class BatchnormInferenceDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    /// @param kpackLoader Likewise. Unused by the descriptors this pack ships, which are
    ///        all embedded_source, but buildIngestorKernelCode takes it for the kinds it
    ///        also serves.
    BatchnormInferenceDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                      const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// No scratch: one launch, no virtual tensors, every thread accumulates in registers
    /// and writes its own output element once.
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        // Reads the operand uids the graph match bound rather than re-deriving them.
        const auto binding = batchnormInferenceBinding(bound);

        const auto& xTensor = requireTensor(context, binding.x);
        const auto& yTensor = requireTensor(context, binding.y);

        BatchnormInferenceGeometry geometry;
        // Rank 4, dims and strides present and equally sized, validated by the matcher.
        for(flatbuffers::uoffset_t axis = 0; axis < SUPPORTED_RANK; ++axis)
        {
            geometry.dims[axis] = xTensor.dims()->Get(axis);
            geometry.xStrides[axis] = xTensor.strides()->Get(axis);
            geometry.yStrides[axis] = yTensor.strides()->Get(axis);
        }
        geometry.meanStrideC = channelStrideOf(requireTensor(context, binding.mean));
        geometry.invVarianceStrideC = channelStrideOf(requireTensor(context, binding.invVariance));
        geometry.scaleStrideC = channelStrideOf(requireTensor(context, binding.scale));
        geometry.biasStrideC = channelStrideOf(requireTensor(context, binding.bias));

        const auto blockSize
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));

        compilation::KernelCompileOptions options(&xTensor, context.deviceProperties.gcnArchName);
        // The three macros BatchnormInferenceTypes.h guards with #error. All handler-
        // supplied rather than descriptor-bound: they are read out of this candidate's
        // metadata and validated, and the descriptor substituter does literal
        // replacement only.
        options.add("HIPDNN_BN_IO_DTYPE", dtypeTagFor(kernel, IO_DTYPE_FIELD));
        options.add("HIPDNN_BN_PARAM_DTYPE", dtypeTagFor(kernel, PARAM_DTYPE_FIELD));
        options.add("HIPDNN_BN_BLOCK", blockSize);

        auto code
            = buildIngestorKernelCode(_kernelCompiler, _kpackLoader, context, kernel, options);

        // int64_t: the matcher admits shapes whose element count exceeds 2^31, and a
        // 32-bit product here would wrap both the grid size and the comparison the
        // kernel's own bounds guard makes against it.
        const int64_t total
            = geometry.dims[0] * geometry.dims[1] * geometry.dims[2] * geometry.dims[3];
        const auto gridSize = static_cast<unsigned int>(
            (total + static_cast<int64_t>(blockSize) - 1) / static_cast<int64_t>(blockSize));

        // Rounded up on purpose: the kernel returns early for index >= total, so a
        // partially populated final block is correct and an exact-multiple grid would
        // leave the tail unwritten.
        code.kernel->setBlockSize(blockSize, 1, 1);
        code.kernel->setGridSize(gridSize, 1, 1);

        return std::make_unique<PreparedBatchnormInference>(
            std::move(code.program), std::move(code.kernel), binding, geometry);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedBatchnorm = dynamic_cast<const PreparedBatchnormInference&>(prepared);
        const auto& binding = preparedBatchnorm.binding();
        const auto& geometry = preparedBatchnorm.geometry();

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto mean
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.mean, deviceBuffers, numDeviceBuffers);
        const auto invVariance = hipdnn_plugin_sdk::findDeviceBuffer(
            binding.invVariance, deviceBuffers, numDeviceBuffers);
        const auto scale
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.scale, deviceBuffers, numDeviceBuffers);
        const auto bias
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.bias, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        // 22 arguments, in the order kernels/BatchnormInference.cpp declares them.
        preparedBatchnorm.kernel().launch(handle.getStream(),
                                          x.ptr,
                                          mean.ptr,
                                          invVariance.ptr,
                                          scale.ptr,
                                          bias.ptr,
                                          y.ptr,
                                          geometry.dims[0],
                                          geometry.dims[1],
                                          geometry.dims[2],
                                          geometry.dims[3],
                                          geometry.xStrides[0],
                                          geometry.xStrides[1],
                                          geometry.xStrides[2],
                                          geometry.xStrides[3],
                                          geometry.yStrides[0],
                                          geometry.yStrides[1],
                                          geometry.yStrides[2],
                                          geometry.yStrides[3],
                                          geometry.meanStrideC,
                                          geometry.invVarianceStrideC,
                                          geometry.scaleStrideC,
                                          geometry.biasStrideC);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& batchnormInferenceKpackModuleCache()
{
    // Process-lifetime, as the pointwise pack's is: a loaded module outlives the plan
    // that loaded it, and the cache is what makes that one module rather than one per
    // dispatch.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetBatchnormInferenceModuleCache()
{
    batchnormInferenceKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning
/// pointer to it, but a provider's Container is created and destroyed per handle, so it
/// (and the compiler and loader it holds) must outlive every Container.
const BatchnormInferenceDispatchHandler& batchnormInferenceDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(batchnormInferenceKpackModuleCache());
    static const BatchnormInferenceDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                     s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerBatchnormInferenceSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &batchnormInferenceGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &batchnormInferenceKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &batchnormInferenceScore);
    scope.add(std::string(DISPATCH_SYMBOL), &batchnormInferenceDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
