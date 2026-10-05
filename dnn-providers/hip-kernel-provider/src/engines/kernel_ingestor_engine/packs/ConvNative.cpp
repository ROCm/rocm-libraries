// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
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
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

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

/// x and y are 4-D NCHW/NKPQ; w is 4-D KCRS. All three are rank 4, which is the only
/// fact this constant states -- the per-tensor role is fixed by which uid is read.
constexpr uint32_t SUPPORTED_RANK = 4;
/// x/w/y are 2-D-spatial (H,W / R,S), so stride/dilation/padding are each length 2.
constexpr size_t SUPPORTED_SPATIAL_RANK = 2;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched conv graph binds, in kernel argument order.
struct ConvFwdBinding
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t y = 0;
};

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

/// The kernel has no stride arguments, so require exact row-major strides,
/// even on size-one axes. Packedness alone is not enough.
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

/// Requires a published tensor and an in-range axis.
int64_t dimension(const BoundTokens& bound, std::string_view root, size_t axis)
{
    return *tryGetBoundInt(bound, tensorElement(root, "dims", axis));
}

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

/**
 * @brief Graph-scoped applicability: is this the one conv shape this engine's kernel
 *        can launch? One pack, one operation, so this matcher (unlike Pointwise's)
 *        both admits the node type and validates it in one pass.
 */
std::optional<BoundTokens> convFwdGraphMatches(const MatchContext& context)
{

    const auto* attributesPtr = convFwdNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // Deliberately narrow: stride 1, dilation 1, no padding is the only shape the
    // in-kernel p = h - r + 1 / q = width - s + 1 formula is correct for.
    if(!allEqual(attributes.stride(), SUPPORTED_SPATIAL_RANK, 1)
       || !allEqual(attributes.dilation(), SUPPORTED_SPATIAL_RANK, 1)
       || !allEqual(attributes.pre_padding(), SUPPORTED_SPATIAL_RANK, 0)
       || !allEqual(attributes.post_padding(), SUPPORTED_SPATIAL_RANK, 0))
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, attributes.x_tensor_uid());
    const auto* w = findTensor(context, attributes.w_tensor_uid());
    const auto* y = findTensor(context, attributes.y_tensor_uid());
    if(x == nullptr || w == nullptr || y == nullptr)
    {
        return std::nullopt;
    }

    BoundTokens bound;
    if(!publishGraph(bound, context.graph) || !publishTensor(bound, X_TOKEN, x)
       || !publishTensor(bound, W_TOKEN, w) || !publishTensor(bound, Y_TOKEN, y))
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
    if(*tryGetBoundValue<std::string>(bound, "conv.conv_mode") != "CROSS_CORRELATION"
       || !isSupportedOperand(bound, X_TOKEN) || !isSupportedOperand(bound, W_TOKEN)
       || !isSupportedOperand(bound, Y_TOKEN))
    {
        return std::nullopt;
    }

    // Uniform dtype across operands; mixed precision is a different kernel.
    if(*tryGetBoundValue<std::string>(bound, "x.dtype")
           != *tryGetBoundValue<std::string>(bound, "w.dtype")
       || *tryGetBoundValue<std::string>(bound, "x.dtype")
              != *tryGetBoundValue<std::string>(bound, "y.dtype"))
    {
        return std::nullopt;
    }

    const auto xC = dimension(bound, X_TOKEN, 1);
    const auto xH = dimension(bound, X_TOKEN, 2);
    const auto xW = dimension(bound, X_TOKEN, 3);
    const auto wK = dimension(bound, W_TOKEN, 0);
    const auto wR = dimension(bound, W_TOKEN, 2);
    const auto wS = dimension(bound, W_TOKEN, 3);

    // Filter channels vs. input channels -- this also refuses grouped conv, which is
    // encoded purely as a smaller w channel count; the kernel has no notion of groups
    // and indexes w using c from x alone, so a filter with fewer channels would read
    // past the end.
    if(dimension(bound, W_TOKEN, 1) != xC)
    {
        return std::nullopt;
    }

    // r <= h and s <= width, or p = h - r + 1 / q = width - s + 1 go non-positive,
    // which the kernel's flat-index unravel (ConvFwd.cpp) never expects.
    if(wR > xH || wS > xW)
    {
        return std::nullopt;
    }

    // y must be exactly the shape the kernel computes: total = n*k*p*q comes from x
    // and w alone, and the kernel writes every index < total into y, so a smaller y
    // overflows.
    if(dimension(bound, Y_TOKEN, 0) != dimension(bound, X_TOKEN, 0)
       || dimension(bound, Y_TOKEN, 1) != wK || dimension(bound, Y_TOKEN, 2) != xH - wR + 1
       || dimension(bound, Y_TOKEN, 3) != xW - wS + 1)
    {
        return std::nullopt;
    }

    return bound;
}

/**
 * @brief Kernel-scoped applicability: does this kernel's dtype match the graph's?
 *        Evaluated once per candidate kernel; without it an f32 graph could reach an
 *        f16 binary and return wrong numbers rather than failing.
 */
bool convFwdKernelMatches(const MatchContext& /*context*/,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel)
{
    const auto* dataType = tryGetBoundValue<std::string>(bound, "x.dtype");
    return dataType != nullptr && kernel.getStringMetadata(std::string(DTYPE_FIELD)) == *dataType;
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
    // Every token was written by the engine's graph match, which admitted this graph; a
    // missing one means the catalog was built by an engine other than ours.
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_fwd dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };

    return {read(X_TOKEN), read(W_TOKEN), read(Y_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The compiled kernel plus everything its launch geometry needs: the operand uids and
/// the seven dims read from the graph at prepare() time, owning nothing that points
/// back into it.
class PreparedConvFwd : public PreparedDispatch
{
public:
    PreparedConvFwd(IngestorKernelCode code,
                    ConvFwdBinding binding,
                    int n,
                    int c,
                    int h,
                    int width,
                    int k,
                    int r,
                    int s)
        : _code(std::move(code))
        , _binding(binding)
        , _n(n)
        , _c(c)
        , _h(h)
        , _width(width)
        , _k(k)
        , _r(r)
        , _s(s)
    {
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

    int n() const
    {
        return _n;
    }
    int c() const
    {
        return _c;
    }
    int h() const
    {
        return _h;
    }
    int width() const
    {
        return _width;
    }
    int k() const
    {
        return _k;
    }
    int r() const
    {
        return _r;
    }
    int s() const
    {
        return _s;
    }

private:
    // Owns each device's program alongside the kernel viewing into it, so a module
    // outlives every function resolved from it for the plan's lifetime.
    IngestorKernelCode _code;
    ConvFwdBinding _binding;
    int _n;
    int _c;
    int _h;
    int _width;
    int _k;
    int _r;
    int _s;
};

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
        // Read dimensions from the cached binding; keep xTensor for compile options.
        const auto n = static_cast<int>(dimension(bound, X_TOKEN, 0));
        const auto c = static_cast<int>(dimension(bound, X_TOKEN, 1));
        const auto h = static_cast<int>(dimension(bound, X_TOKEN, 2));
        const auto width = static_cast<int>(dimension(bound, X_TOKEN, 3));
        const auto k = static_cast<int>(dimension(bound, W_TOKEN, 0));
        const auto r = static_cast<int>(dimension(bound, W_TOKEN, 2));
        const auto s = static_cast<int>(dimension(bound, W_TOKEN, 3));

        const auto blockSize
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));

        compilation::KernelCompileOptions options(&xTensor, context.deviceProperties.gcnArchName);
        options.add("HIP_PLUGIN_CONV_TYPE", elementTypeFor(kernel));
        options.add("HIP_PLUGIN_CONV_BLOCK_SIZE", blockSize);

        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, convFwdKernelSignature());

        const auto p = h - r + 1;
        const auto q = width - s + 1;
        // int64_t: n*k*p*q can exceed 2^31 for shapes this matcher admits. A 32-bit
        // product would wrap, corrupting both the grid size and the kernel's own bounds
        // guard (ConvFwd.cpp).
        const int64_t total = static_cast<int64_t>(n) * k * p * q;
        const auto blocks = total / blockSize + (total % blockSize != 0 ? 1 : 0);
        if(blocks > std::numeric_limits<unsigned int>::max())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "conv_fwd grid size exceeds the launch range");
        }
        const auto gridSize = static_cast<unsigned int>(blocks);

        code.setBlockSize(blockSize, 1, 1);
        code.setGridSize(gridSize, 1, 1);

        return std::make_unique<PreparedConvFwd>(std::move(code), binding, n, c, h, width, k, r, s);
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

        preparedConvFwd.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    x.ptr,
                    w.ptr,
                    y.ptr,
                    preparedConvFwd.n(),
                    preparedConvFwd.c(),
                    preparedConvFwd.h(),
                    preparedConvFwd.width(),
                    preparedConvFwd.k(),
                    preparedConvFwd.r(),
                    preparedConvFwd.s());
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

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
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
