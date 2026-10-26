// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/BindingPublication.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvNative.cpp
 * @brief The conv-forward engine's native half: matching, scoring, dispatch, and the
 *        one function that registers them.
 *
 * A second engine beside Pointwise, split on graph node type (`ConvolutionFwdAttributes`
 * vs `PointwiseAttributes`) rather than operation, so one pack needs no operation
 * matcher of its own. Deliberately narrow -- stride 1, dilation 1, no padding,
 * cross-correlation, packed NCHW/KCRS/NKPQ, FLOAT/HALF -- and the matcher must refuse
 * everything outside that shape rather than let the naive kernel compute wrong answers.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.conv_fwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_fwd.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.conv_fwd.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_fwd.dispatch";

// KMD fields this pack varies along, and the tokens matching binds for dispatch.
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view X_TOKEN = "x";
constexpr std::string_view W_TOKEN = "w";
constexpr std::string_view Y_TOKEN = "y";

// RFC 0020 §6.1 shape fields (`dims[i]`, `dtype`); a tensor uid alone is a poor model input.
// Roots are op-qualified (three levels, not §6.1's `tvar.field`) because a native matcher
// has no UED pattern naming its operands; reconcile with RFC 0020 when that path lands.
constexpr std::string_view X_ROOT = "conv_fwd.x";
constexpr std::string_view W_ROOT = "conv_fwd.w";
constexpr std::string_view Y_ROOT = "conv_fwd.y";

// RFC 0019 §13.6 op-intrinsic cost fields. They are the same for every engine of this op,
// so the binding publishes them rather than any one UHD.
constexpr std::string_view FLOPS_TOKEN = "conv_fwd.flops";
constexpr std::string_view BYTES_TOKEN = "conv_fwd.bytes";

/// x and y are 4-D NCHW/NKPQ; w is 4-D KCRS. All three are rank 4, which is the only
/// fact this constant states -- the per-tensor role is fixed by which uid is read.
constexpr uint32_t SUPPORTED_RANK = 4;
/// x/w/y are 2-D-spatial (H,W / R,S), so stride/dilation/padding are each length 2.
constexpr size_t SUPPORTED_SPATIAL_RANK = 2;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// True when @p values has exactly @p expectedSize elements, all equal to @p expected.
/// A null or wrongly-sized vector is a refusal, not a vacuous pass: a 0-length vector
/// would otherwise satisfy "every element equals" for a conv this pack cannot serve.
bool allEqual(const flatbuffers::Vector<int64_t>* values, size_t expectedSize, int64_t expected)
{
    if(values == nullptr || values->size() != expectedSize)
    {
        return false;
    }
    for(const auto value : *values)
    {
        if(value != expected)
        {
            return false;
        }
    }
    return true;
}

/// The kernel takes no stride arguments, so require exact row-major strides. A packed
/// check is not enough because it ignores strides on size-one axes.
bool isSupportedOperand(const BoundTokens& bound, std::string_view root)
{
    if(tryGetBoundInt(bound, tensorField(root, "rank")) != SUPPORTED_RANK)
    {
        return false;
    }
    int64_t expectedStride = 1;
    for(size_t i = SUPPORTED_RANK; i-- > 0;)
    {
        const auto dim = tryGetBoundInt(bound, tensorElement(root, "dims", i));
        if(!dim || *dim <= 0 || *dim > std::numeric_limits<int>::max()
           || tryGetBoundInt(bound, tensorElement(root, "strides", i)) != expectedStride
           || expectedStride > std::numeric_limits<int64_t>::max() / *dim)
        {
            return false;
        }
        expectedStride *= *dim;
    }
    const auto* isVirtual = tryGetBoundValue<bool>(bound, tensorField(root, "virtual"));
    const auto* runtime
        = tryGetBoundValue<bool>(bound, tensorField(root, "is_runtime_pass_by_value"));
    const auto* dtype = tryGetBoundValue<std::string>(bound, tensorField(root, "dtype"));
    return isVirtual != nullptr && !*isVirtual && runtime != nullptr && !*runtime
           && tryGetBoundValue<double>(bound, tensorField(root, "value_f32")) == nullptr
           && dtype != nullptr && (*dtype == "FLOAT" || *dtype == "HALF");
}

/**
 * @brief Reads an integer token the dispatch relies on.
 *
 * @throws HipdnnPluginException if @p bound has no integer under @p token.
 */
int64_t requireBoundInt(const BoundTokens& bound, std::string_view token)
{
    const auto value = tryGetBoundInt(bound, token);
    if(!value.has_value())
    {
        std::string message = "conv_fwd dispatch is missing bound token '";
        message += token;
        message += "', or it does not hold an integer";
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       std::move(message));
    }
    return *value;
}

/// @throws HipdnnPluginException if @p root has no published extent at @p axis.
int64_t dimension(const BoundTokens& bound, std::string_view root, size_t axis)
{
    return requireBoundInt(bound, tensorElement(root, "dims", axis));
}

using pack_features::bindTensorFields;
using pack_features::checkedProduct;
using pack_features::movedBytes;

/// The node this engine's matchers read, or nullptr if the graph is not a single
/// conv-forward node.
const data_objects::ConvolutionFwdAttributes* convFwdNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes
       || node.attributes() == nullptr)
    {
        return nullptr;
    }

    return &node.attributesAs<data_objects::ConvolutionFwdAttributes>();
}

/// The in-kernel formulas p = h - r + 1 and q = width - s + 1 hold only for stride 1,
/// dilation 1, and no padding. The binding does not carry these vectors, so read them
/// from the node.
bool isSupportedGeometry(const data_objects::ConvolutionFwdAttributes& attributes)
{
    return allEqual(attributes.stride(), SUPPORTED_SPATIAL_RANK, 1)
           && allEqual(attributes.dilation(), SUPPORTED_SPATIAL_RANK, 1)
           && allEqual(attributes.pre_padding(), SUPPORTED_SPATIAL_RANK, 0)
           && allEqual(attributes.post_padding(), SUPPORTED_SPATIAL_RANK, 0);
}

/**
 * @brief Checks every binding fact the dispatch relies on.
 *
 * Requires one cross-correlation node and three row-major operands of one supported
 * dtype, with extents that fit an int and a y shape the kernel can compute. It reads
 * tokens instead of the graph because a declarative `graph_match` can bind them, and
 * that binding skips the native graph match's checks.
 */
bool isServableBinding(const BoundTokens& bound)
{
    const auto* mode = tryGetBoundValue<std::string>(bound, "conv.conv_mode");
    if(tryGetBoundInt(bound, "graph.node_count") != 1 || mode == nullptr
       || *mode != "CROSS_CORRELATION" || !isSupportedOperand(bound, X_TOKEN)
       || !isSupportedOperand(bound, W_TOKEN) || !isSupportedOperand(bound, Y_TOKEN))
    {
        return false;
    }

    // Uniform dtype across operands; mixed precision is a different kernel.
    const auto* dtype = tryGetBoundValue<std::string>(bound, tensorField(X_TOKEN, DTYPE_FIELD));
    const auto* wDtype = tryGetBoundValue<std::string>(bound, tensorField(W_TOKEN, DTYPE_FIELD));
    const auto* yDtype = tryGetBoundValue<std::string>(bound, tensorField(Y_TOKEN, DTYPE_FIELD));
    if(dtype == nullptr || wDtype == nullptr || yDtype == nullptr || *dtype != *wDtype
       || *dtype != *yDtype)
    {
        return false;
    }

    // isSupportedOperand admitted all four extents of each operand, so these never throw.
    const auto xC = dimension(bound, X_TOKEN, 1);
    const auto xH = dimension(bound, X_TOKEN, 2);
    const auto xW = dimension(bound, X_TOKEN, 3);
    const auto wK = dimension(bound, W_TOKEN, 0);
    const auto wR = dimension(bound, W_TOKEN, 2);
    const auto wS = dimension(bound, W_TOKEN, 3);

    // The kernel has no groups and indexes w with c from x. A grouped conv has fewer
    // w channels, so it would read past the end of w.
    //
    // r > h or s > width makes p or q non-positive, which the kernel's flat-index
    // unravel (ConvFwd.cpp) does not handle.
    //
    // The kernel writes all n*k*p*q outputs, computed from x and w, so y must have
    // exactly that shape.
    return dimension(bound, W_TOKEN, 1) == xC && wR <= xH && wS <= xW
           && dimension(bound, Y_TOKEN, 0) == dimension(bound, X_TOKEN, 0)
           && dimension(bound, Y_TOKEN, 1) == wK && dimension(bound, Y_TOKEN, 2) == xH - wR + 1
           && dimension(bound, Y_TOKEN, 3) == xW - wS + 1;
}

/**
 * @brief Graph-scoped applicability: is this the one conv shape this engine's kernel
 *        can launch? One pack, one operation, so this matcher (unlike Pointwise's)
 *        both admits the node type and validates it in one pass.
 */
std::optional<BoundTokens> convFwdGraphMatches(const MatchContext& context)
{
    const auto* attributesPtr = convFwdNode(context);
    if(attributesPtr == nullptr || !isSupportedGeometry(*attributesPtr))
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    const auto* x = findTensor(context, attributes.x_tensor_uid());
    const auto* w = findTensor(context, attributes.w_tensor_uid());
    const auto* y = findTensor(context, attributes.y_tensor_uid());
    if(x == nullptr || w == nullptr || y == nullptr)
    {
        return std::nullopt;
    }

    BoundTokens bound;
    publishGraph(bound, context.graph);
    if(!publishTensor(bound, X_TOKEN, x) || !publishTensor(bound, W_TOKEN, w)
       || !publishTensor(bound, Y_TOKEN, y))
    {
        return std::nullopt;
    }
    const auto* mode = data_objects::EnumNameConvMode(attributes.conv_mode());
    const auto* compute
        = data_objects::EnumNameDataType(context.graph.getNodeWrapper(0).computeDataType());
    if(mode == nullptr || *mode == '\0' || compute == nullptr || *compute == '\0')
    {
        return std::nullopt;
    }
    bound.emplace("conv.conv_mode", std::string(mode));
    bound.emplace("conv.compute_data_type", std::string(compute));
    if(!isServableBinding(bound))
    {
        return std::nullopt;
    }

    // The validated dims and dtypes are the features a UHD ranks a conv on.
    bindTensorFields(bound, X_ROOT, *x);
    bindTensorFields(bound, W_ROOT, *w);
    bindTensorFields(bound, Y_ROOT, *y);

    // 2 x N x K x P x Q x C x R x S: two flops per MAC over every output element and filter
    // tap. No masking, so the effective and dense flop conventions agree. isServableBinding
    // admitted every extent read here.
    const auto flops = checkedProduct({2,
                                       dimension(bound, X_TOKEN, 0),
                                       dimension(bound, W_TOKEN, 0),
                                       dimension(bound, Y_TOKEN, 2),
                                       dimension(bound, Y_TOKEN, 3),
                                       dimension(bound, X_TOKEN, 1),
                                       dimension(bound, W_TOKEN, 2),
                                       dimension(bound, W_TOKEN, 3)});
    if(flops.has_value())
    {
        bound[std::string(FLOPS_TOKEN)] = *flops;
    }

    const auto bytes = movedBytes({x, w, y});
    if(bytes.has_value())
    {
        bound[std::string(BYTES_TOKEN)] = *bytes;
    }
    return bound;
}

/**
 * @brief Kernel-scoped applicability: does this kernel's dtype match the graph's?
 *        Evaluated once per candidate kernel; without it an f32 graph could reach an
 *        f16 binary and return wrong numbers rather than failing.
 *
 * This is the pack's only criterion, so it also re-checks the dispatch preconditions.
 * A declarative `graph_match` can admit graphs the native one refuses. It also checks
 * operand roles: a pattern that binds `$x` to the node's `w` edge can pass every shape
 * check (n == k, h == r, width == s) and launch with x and w swapped.
 */
bool convFwdKernelMatches(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel)
{
    const auto* dataType = tryGetBoundValue<std::string>(bound, tensorField(X_TOKEN, DTYPE_FIELD));
    if(dataType == nullptr || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dataType
       || !isServableBinding(bound))
    {
        return false;
    }
    const auto* attributes = convFwdNode(context);
    return attributes != nullptr && isSupportedGeometry(*attributes)
           && tryGetBoundInt(bound, X_TOKEN) == attributes->x_tensor_uid()
           && tryGetBoundInt(bound, W_TOKEN) == attributes->w_tensor_uid()
           && tryGetBoundInt(bound, Y_TOKEN) == attributes->y_tensor_uid();
}

double convFwdScore(const MatchContext& /*context*/,
                    const BoundTokens& /*bound*/,
                    const KernelDefinition& kernel)
{
    // A stand-in for a trained model: prefers the larger block size.
    return static_cast<double>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));
}

/**
 * @brief Re-reads the operand bindings a match established.
 *
 * @throws HipdnnPluginException if the graph is not one this matcher accepts.
 */
ConvFwdBinding convFwdBinding(const BoundTokens& bound)
{
    return {requireBoundInt(bound, X_TOKEN),
            requireBoundInt(bound, W_TOKEN),
            requireBoundInt(bound, Y_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The compiled kernel plus everything its launch is computed from: the operand uids, the
/// seven dims read from the graph at prepare() time and the block size, owning nothing
/// that points back into it.
class PreparedConvFwd : public IngestorPreparedDispatch
{
public:
    PreparedConvFwd(IngestorKernelCode code,
                    ConvFwdBinding binding,
                    ConvFwdExtents extents,
                    int64_t blockSize)
        : _code(std::move(code))
        , _binding(binding)
        , _extents(extents)
        , _blockSize(blockSize)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _code;
    }

    /// The kernel for the device this dispatch is running on. Resolved here rather than
    /// at prepare() because a plan outlives the handle it was built from.
    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const ConvFwdBinding& binding() const
    {
        return _binding;
    }

    const ConvFwdExtents& extents() const
    {
        return _extents;
    }

    int64_t blockSize() const
    {
        return _blockSize;
    }

private:
    // Owns each device's program alongside the kernel viewing into it, so a module
    // outlives every function resolved from it for the plan's lifetime.
    IngestorKernelCode _code;
    ConvFwdBinding _binding;
    ConvFwdExtents _extents;
    int64_t _blockSize;
};

// Applies the launch geometry to `code` and builds the prepared dispatch. A plan built
// from a graph and a plan restored from a saved payload both come through here.
std::unique_ptr<PreparedDispatch> makePreparedConvFwd(IngestorKernelCode code,
                                                      const ConvFwdLaunchInputs& launch)
{
    const ConvFwdLaunchGeometry geometry = convFwdLaunchGeometry(launch);
    code.setBlockSize(geometry.block, 1, 1);
    code.setGridSize(geometry.gridX, 1, 1);

    return std::make_unique<PreparedConvFwd>(
        std::move(code), launch.binding, launch.extents, launch.blockSize);
}

/// The C++ type the kernel is compiled for, from the kernel's dtype metadata.
std::string elementTypeFor(const KernelDefinition& kernel)
{
    const auto& dtype = kernel.getStringMetadata(std::string(DTYPE_FIELD));
    if(dtype == "FLOAT")
    {
        return "float";
    }
    if(dtype == "HALF")
    {
        return "_Float16";
    }

    // Unreachable via matching, which admits only dtypes this pack declares.
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "kernel '" + toString(kernel.kernelId) + "' declares unsupported dtype '" + dtype + "'");
}

const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    if(it == tensors.end() || it->second == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched conv_fwd graph has no tensor for uid " + std::to_string(uid));
    }
    return *it->second;
}

/// The argument list this pack launches ConvFwd with, mirroring the launch() below one
/// for one. It sits here rather than in the adapter that consumes it so that it is edited
/// alongside that launch -- a stale copy rejects the correct kernel rather than the
/// drifted one.
///
/// Names are empty and offsets zero because neither is compared for a HIP-produced kernel;
/// see requireSignatureMatch.
const std::vector<KernelArgument>& convFwdKernelSignature()
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    // The seven trailing extents (n, c, h, width, k, r, s), each an int.
    static const KernelArgument s_extent{"by_value", static_cast<uint32_t>(sizeof(int)), 0, ""};
    static const std::vector<KernelArgument> s_signature{s_buffer,
                                                         s_buffer,
                                                         s_buffer,
                                                         s_extent,
                                                         s_extent,
                                                         s_extent,
                                                         s_extent,
                                                         s_extent,
                                                         s_extent,
                                                         s_extent};
    return s_signature;
}

/**
 * @brief The native dispatch behind this pack's UDD: sizes and launches the conv
 *        kernel. Everything graph/kernel-derived resolves once at prepare(); execute()
 *        only resolves buffers and launches, so nothing mutates once prepared and
 *        concurrent execution is safe.
 */
class ConvFwdDispatchHandler : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    /// @param kpackLoader Same must-outlive contract. Which of the two is consulted
    /// depends on the selected kernel's source kind, decided in buildIngestorKernelCode.
    ConvFwdDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                           const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// This reference kernel needs no scratch: every output element is accumulated in
    /// a register and written once.
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
        const auto binding = convFwdBinding(bound);

        const auto& xTensor = requireTensor(context, binding.x);
        // Dimensions come from the binding. xTensor supplies the compile options.
        const auto n = static_cast<int>(dimension(bound, X_TOKEN, 0));
        const auto c = static_cast<int>(dimension(bound, X_TOKEN, 1));
        const auto h = static_cast<int>(dimension(bound, X_TOKEN, 2));
        const auto width = static_cast<int>(dimension(bound, X_TOKEN, 3));
        const auto k = static_cast<int>(dimension(bound, W_TOKEN, 0));
        const auto r = static_cast<int>(dimension(bound, W_TOKEN, 2));
        const auto s = static_cast<int>(dimension(bound, W_TOKEN, 3));

        const int64_t blockSizeValue = kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD));
        const auto blockSize = static_cast<unsigned int>(blockSizeValue);

        compilation::KernelCompileOptions options(&xTensor, context.deviceProperties.gcnArchName);
        options.add("HIP_PLUGIN_CONV_TYPE", elementTypeFor(kernel));
        options.add("HIP_PLUGIN_CONV_BLOCK_SIZE", blockSize);

        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, convFwdKernelSignature());

        const auto p = h - r + 1;
        const auto q = width - s + 1;
        // int64_t: n*k*p*q can exceed 2^31 for shapes this matcher admits. A grid that
        // does not fit the launch range would be truncated by convFwdLaunchGeometry.
        const int64_t total = static_cast<int64_t>(n) * k * p * q;
        const auto blocks = total / blockSize + (total % blockSize != 0 ? 1 : 0);
        if(blocks > std::numeric_limits<unsigned int>::max())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "conv_fwd grid size exceeds the launch range");
        }

        return makePreparedConvFwd(
            std::move(code),
            ConvFwdLaunchInputs{binding, ConvFwdExtents{n, c, h, width, k, r, s}, blockSizeValue});
    }

    std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& inputs,
                                                    SavedKernelCode code,
                                                    int deviceOrdinal) const override
    {
        auto restored = restoreIngestorLaunch<ConvFwdLaunchInputs>(inputs,
                                                                   std::move(code),
                                                                   convFwdKernelSignature(),
                                                                   &readConvFwdLaunchInputs,
                                                                   deviceOrdinal);
        return makePreparedConvFwd(std::move(restored.code), restored.launch);
    }

    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& prepared) const override
    {
        const auto& preparedConvFwd = dynamic_cast<const PreparedConvFwd&>(prepared);
        return SavedLaunchInputs{std::string(CONV_FWD_DISPATCH_SYMBOL_V1),
                                 convFwdLaunchValues(preparedConvFwd.binding(),
                                                     preparedConvFwd.extents(),
                                                     preparedConvFwd.blockSize())};
    }

    bool supportsSaving() const override
    {
        return true;
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedConvFwd = dynamic_cast<const PreparedConvFwd&>(prepared);
        const auto& binding = preparedConvFwd.binding();

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        const auto& extents = preparedConvFwd.extents();
        preparedConvFwd.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    x.ptr,
                    w.ptr,
                    y.ptr,
                    extents.n,
                    extents.c,
                    extents.h,
                    extents.width,
                    extents.k,
                    extents.r,
                    extents.s);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

MetadataValues convFwdLaunchValues(const ConvFwdBinding& binding,
                                   const ConvFwdExtents& extents,
                                   int64_t blockSize)
{
    return {{std::string(CONV_FWD_X_UID_VALUE), binding.x},
            {std::string(CONV_FWD_W_UID_VALUE), binding.w},
            {std::string(CONV_FWD_Y_UID_VALUE), binding.y},
            {std::string(CONV_FWD_N_VALUE), int64_t{extents.n}},
            {std::string(CONV_FWD_C_VALUE), int64_t{extents.c}},
            {std::string(CONV_FWD_H_VALUE), int64_t{extents.h}},
            {std::string(CONV_FWD_WIDTH_VALUE), int64_t{extents.width}},
            {std::string(CONV_FWD_K_VALUE), int64_t{extents.k}},
            {std::string(CONV_FWD_R_VALUE), int64_t{extents.r}},
            {std::string(CONV_FWD_S_VALUE), int64_t{extents.s}},
            {std::string(CONV_FWD_BLOCK_SIZE_VALUE), blockSize}};
}

ConvFwdLaunchInputs readConvFwdLaunchInputs(const SavedLaunchInputs& inputs)
{
    requireLaunchContract(inputs, CONV_FWD_DISPATCH_SYMBOL_V1);
    requireOnlyLaunchValues(inputs,
                            {CONV_FWD_X_UID_VALUE,
                             CONV_FWD_W_UID_VALUE,
                             CONV_FWD_Y_UID_VALUE,
                             CONV_FWD_N_VALUE,
                             CONV_FWD_C_VALUE,
                             CONV_FWD_H_VALUE,
                             CONV_FWD_WIDTH_VALUE,
                             CONV_FWD_K_VALUE,
                             CONV_FWD_R_VALUE,
                             CONV_FWD_S_VALUE,
                             CONV_FWD_BLOCK_SIZE_VALUE});

    // The kernel takes each extent as an int.
    const auto extent = [&inputs](std::string_view name) {
        return static_cast<int>(requireLaunchInt(inputs, name, 1, std::numeric_limits<int>::max()));
    };

    ConvFwdLaunchInputs launch;
    launch.binding.x = requireLaunchInt(inputs, CONV_FWD_X_UID_VALUE);
    launch.binding.w = requireLaunchInt(inputs, CONV_FWD_W_UID_VALUE);
    launch.binding.y = requireLaunchInt(inputs, CONV_FWD_Y_UID_VALUE);
    launch.extents.n = extent(CONV_FWD_N_VALUE);
    launch.extents.c = extent(CONV_FWD_C_VALUE);
    launch.extents.h = extent(CONV_FWD_H_VALUE);
    launch.extents.width = extent(CONV_FWD_WIDTH_VALUE);
    launch.extents.k = extent(CONV_FWD_K_VALUE);
    launch.extents.r = extent(CONV_FWD_R_VALUE);
    launch.extents.s = extent(CONV_FWD_S_VALUE);
    launch.blockSize = requireLaunchInt(
        inputs, CONV_FWD_BLOCK_SIZE_VALUE, 1, std::numeric_limits<uint32_t>::max());

    if(launch.extents.r > launch.extents.h || launch.extents.s > launch.extents.width)
    {
        serialization::refuseIngestorPlan(
            serialization::IngestorPlanRefusal::INCOMPATIBLE,
            "launch values 'r' and 's' of dispatch '" + inputs.dispatchSymbol
                + "' must not exceed 'h' and 'width'; the plan holds r "
                + std::to_string(launch.extents.r) + ", h " + std::to_string(launch.extents.h)
                + ", s " + std::to_string(launch.extents.s) + " and width "
                + std::to_string(launch.extents.width));
    }

    // Each factor is in [1, INT_MAX], so a product overflows int64_t only past the second
    // factor. The grid computation adds block_size to the product, so that sum must fit
    // too. The block count must fit the unsigned int grid dimension.
    const int64_t outputHeight = int64_t{launch.extents.h} - launch.extents.r + 1;
    const int64_t outputWidth = int64_t{launch.extents.width} - launch.extents.s + 1;
    int64_t total = int64_t{launch.extents.n} * launch.extents.k;
    bool fits = total <= std::numeric_limits<int64_t>::max() / outputHeight;
    if(fits)
    {
        total *= outputHeight;
        fits = total <= std::numeric_limits<int64_t>::max() / outputWidth;
    }
    if(fits)
    {
        total *= outputWidth;
        const int64_t blocks = total / launch.blockSize + (total % launch.blockSize != 0 ? 1 : 0);
        fits = total <= std::numeric_limits<int64_t>::max() - launch.blockSize
               && blocks <= int64_t{std::numeric_limits<unsigned int>::max()};
    }
    if(!fits)
    {
        serialization::refuseIngestorPlan(
            serialization::IngestorPlanRefusal::INCOMPATIBLE,
            "launch values 'n', 'k', 'h', 'width', 'r', 's' and 'block_size' of dispatch '"
                + inputs.dispatchSymbol + "' give a grid larger than "
                + std::to_string(std::numeric_limits<unsigned int>::max()) + " blocks");
    }
    return launch;
}

compilation::KpackModuleCache& convFwdKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetConvFwdModuleCache()
{
    convFwdKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning
/// pointer to it, but a provider's Container is created and destroyed per handle, so
/// it (and the compiler and loader it holds) must outlive every Container.
const ConvFwdDispatchHandler& convFwdDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convFwdKpackModuleCache());
    static const ConvFwdDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvFwdSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &convFwdGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &convFwdKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &convFwdScore);
    scope.add(std::string(DISPATCH_SYMBOL), &convFwdDispatchHandler());
    scope.add(std::string(CONV_FWD_DISPATCH_SYMBOL_V1), &convFwdDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
