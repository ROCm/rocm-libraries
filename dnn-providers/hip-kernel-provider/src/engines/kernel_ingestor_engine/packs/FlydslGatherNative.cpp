// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a pack that
// names a kpack archive nothing staged.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/gather_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
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
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormCommon.hpp"

/**
 * @file FlydslGatherNative.cpp
 * @brief The flyDSL gather pack's native half (`hipkernel:flydsl_gather`): matching,
 *        scoring, dispatch, and the function that registers them.
 *
 * The kernel (`flydsl/kernels_src/kernels/gather/gather_kernel.py`) computes
 * `y[a, i, b] = x[a, indices[i], b]` over packed operands viewed as
 * x `[outer, axis_extent, inner]`, indices `[num_indices]` and
 * y `[outer, num_indices, inner]`: the op's gather along any one axis, with an embedding
 * lookup (`axis` 0, x the `[V, H]` table) the case it is built for. Every extent is a
 * kernel argument; an object bakes only the element size (by dtype), the index dtype and
 * the access width (`vec_bytes`: a 16-byte vector per lane, or one element). Row
 * addresses are 64-bit, so a table past 4 GiB is served; only one row's bytes must fit the
 * kernel's 32-bit buffer offsets.
 *
 * An index outside `[0, axis_extent)` is outside the op's contract (hipDNN's CPU reference
 * throws). The kernel cannot report it: such an index reads nothing and its output row is
 * written as zeros. No graph is declined for it, since the values are not known until
 * execute -- as cuDNN leaves out-of-range indices undefined.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using flydsl_norm::elementBytes;
using flydsl_norm::findTensor;
using flydsl_norm::hasUsableShape;
using flydsl_norm::isDeviceOperand;
using flydsl_norm::isPackedRowMajor;
using flydsl_norm::isSupportedDataType;
using flydsl_norm::kernelDataTypeName;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_gather.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_gather.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_gather.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_gather.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
/// `i32` or `i64`: the indices' element type, a load width baked into the object.
constexpr std::string_view INDEX_DTYPE_FIELD = "index_dtype";
/// Bytes one lane moves: 16 (a vector), or the element size (one element).
constexpr std::string_view VEC_BYTES_FIELD = "vec_bytes";
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";

constexpr std::string_view X_TOKEN = "flydsl_gather.x.uid";
constexpr std::string_view INDICES_TOKEN = "flydsl_gather.indices.uid";
constexpr std::string_view Y_TOKEN = "flydsl_gather.y.uid";
constexpr std::string_view NUM_INDICES_TOKEN = "flydsl_gather.num_indices";
constexpr std::string_view AXIS_EXTENT_TOKEN = "flydsl_gather.axis_extent";
constexpr std::string_view INNER_TOKEN = "flydsl_gather.inner";
constexpr std::string_view OUTER_TOKEN = "flydsl_gather.outer";

/// The vector objects' access width.
constexpr int64_t VECTOR_BYTES = 16;

/// The kernel's int32 extents, its 32-bit buffer offsets (one row of x or y, all of the
/// indices) and the grid's x extent.
constexpr int64_t MAX_INT32 = std::numeric_limits<int32_t>::max();
/// The grid's y and z extents: the row's chunks and the outer extent.
constexpr int64_t MAX_GRID_YZ = 65535;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The node this pack's matchers read, or nullptr if the graph isn't a single gather node.
const data_objects::GatherAttributes* gatherNode(const MatchContext& context)
{
    // Extents are bound from the graph's shapes at prepare, so override-shape graphs are
    // declined.
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::GatherAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::GatherAttributes>();
}

/// Bytes per index element, or nullopt for a dtype no object reads.
std::optional<int64_t> indexBytes(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::INT32:
        return 4;
    case data_objects::DataType::INT64:
        return 8;
    default:
        return std::nullopt;
    }
}

/// The kernel metadata spelling of an index dtype.
std::optional<std::string> indexDataTypeName(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::INT32:
        return std::string("i32");
    case data_objects::DataType::INT64:
        return std::string("i64");
    default:
        return std::nullopt;
    }
}

/// Whether @p tensor declares a base alignment of at least, and a multiple of, @p bytes.
bool declaresAlignment(const data_objects::TensorAttributes& tensor, int64_t bytes)
{
    return tensor.alignment() >= bytes && tensor.alignment() % bytes == 0;
}

/// Product of @p dims over [begin, end), or -1 past MAX_INT32 or at a non-positive extent.
int64_t boundedProduct(const flatbuffers::Vector<int64_t>& dims,
                       flatbuffers::uoffset_t begin,
                       flatbuffers::uoffset_t end)
{
    int64_t product = 1;
    for(auto axis = begin; axis < end; ++axis)
    {
        const int64_t extent = dims.Get(axis);
        if(extent <= 0 || product > MAX_INT32 / extent)
        {
            return -1;
        }
        product *= extent;
    }
    return product;
}

/// The packed operands as the kernel views them.
struct GatherGeometry
{
    int64_t outer = 0;
    int64_t axisExtent = 0;
    int64_t inner = 0;
    int64_t numIndices = 0;
};

/// y's dims are x.dims[:axis] ++ indices.dims ++ x.dims[axis + 1:].
bool hasGatherShape(const data_objects::TensorAttributes& x,
                    const data_objects::TensorAttributes& indices,
                    const data_objects::TensorAttributes& y,
                    flatbuffers::uoffset_t axis)
{
    const auto& xDims = *x.dims();
    const auto& indexDims = *indices.dims();
    const auto& yDims = *y.dims();
    if(yDims.size() != xDims.size() - 1 + indexDims.size())
    {
        return false;
    }
    flatbuffers::uoffset_t out = 0;
    for(flatbuffers::uoffset_t d = 0; d < axis; ++d)
    {
        if(yDims.Get(out++) != xDims.Get(d))
        {
            return false;
        }
    }
    for(flatbuffers::uoffset_t d = 0; d < indexDims.size(); ++d)
    {
        if(yDims.Get(out++) != indexDims.Get(d))
        {
            return false;
        }
    }
    for(auto d = axis + 1; d < xDims.size(); ++d)
    {
        if(yDims.Get(out++) != xDims.Get(d))
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Graph-scoped applicability: a single gather node over packed operands of a dtype,
 *        an index dtype and extents the shipped objects compute exactly.
 *
 * Declines rather than approximates: non-packed operands (the kernel addresses rows by
 * their packed offsets), y in a dtype other than x's, indices other than INT32/INT64, an
 * axis outside x's rank, a y shape other than the op's, and extents past the kernel's
 * int32 arguments, a row past its 32-bit offsets or a grid past the device's limits.
 */
std::optional<BoundTokens> flydslGatherGraphMatches(const MatchContext& context)
{
    const auto* attributes = gatherNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, attributes->x_tensor_uid());
    const auto* indices = findTensor(context, attributes->indices_tensor_uid());
    const auto* y = findTensor(context, attributes->y_tensor_uid());
    if(x == nullptr || indices == nullptr || y == nullptr)
    {
        return std::nullopt;
    }
    if(!isDeviceOperand(*x) || !isDeviceOperand(*indices) || !isDeviceOperand(*y))
    {
        return std::nullopt;
    }

    // Values are copied as raw bits, so y is in x's dtype.
    const auto indexElementBytes = indexBytes(indices->data_type());
    if(!isSupportedDataType(x->data_type()) || y->data_type() != x->data_type()
       || !indexElementBytes.has_value())
    {
        return std::nullopt;
    }

    if(!hasUsableShape(*x, 1) || !hasUsableShape(*indices, 1) || !hasUsableShape(*y, 1))
    {
        return std::nullopt;
    }
    const auto rank = static_cast<int64_t>(x->dims()->size());
    if(attributes->axis() < 0 || attributes->axis() >= rank)
    {
        return std::nullopt;
    }
    const auto axis = static_cast<flatbuffers::uoffset_t>(attributes->axis());
    if(!hasGatherShape(*x, *indices, *y, axis) || !isPackedRowMajor(*x)
       || !isPackedRowMajor(*indices) || !isPackedRowMajor(*y))
    {
        return std::nullopt;
    }

    // Each operand's base must hold one element's alignment; a vector object asks for more,
    // which the kernel match checks.
    const auto bytes = elementBytes(x->data_type());
    if(!declaresAlignment(*x, bytes) || !declaresAlignment(*y, bytes)
       || !declaresAlignment(*indices, *indexElementBytes))
    {
        return std::nullopt;
    }

    const auto& xDims = *x->dims();
    GatherGeometry geometry;
    geometry.outer = boundedProduct(xDims, 0, axis);
    geometry.axisExtent = boundedProduct(xDims, axis, axis + 1);
    geometry.inner = boundedProduct(xDims, axis + 1, xDims.size());
    geometry.numIndices = boundedProduct(*indices->dims(), 0, indices->dims()->size());
    if(geometry.outer < 0 || geometry.axisExtent < 0 || geometry.inner < 0
       || geometry.numIndices < 0)
    {
        return std::nullopt;
    }
    // One row of x or y within a 32-bit buffer range, all of the indices too (their
    // descriptor and scalar offsets are 32-bit), and the outer extent within the grid's z
    // limit. The row's chunks, the grid's y, depend on the object: the kernel match checks
    // them.
    if(geometry.inner > MAX_INT32 / bytes || geometry.numIndices > MAX_INT32 / *indexElementBytes
       || geometry.outer > MAX_GRID_YZ)
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(X_TOKEN)] = attributes->x_tensor_uid();
    bound[std::string(INDICES_TOKEN)] = attributes->indices_tensor_uid();
    bound[std::string(Y_TOKEN)] = attributes->y_tensor_uid();
    bound[std::string(NUM_INDICES_TOKEN)] = geometry.numIndices;
    bound[std::string(AXIS_EXTENT_TOKEN)] = geometry.axisExtent;
    bound[std::string(INNER_TOKEN)] = geometry.inner;
    bound[std::string(OUTER_TOKEN)] = geometry.outer;
    return bound;
}

/**
 * @brief Kernel-scoped applicability: the object's dtype and index dtype, and an access
 *        width the graph admits.
 *
 * A per-element object (`vec_bytes` equal to the element size) serves every graph the
 * graph match admits. A vector object (16) moves 16 bytes per lane from each row's base, so
 * it needs a row of a multiple of 16 bytes and 16-byte aligned bases for x and y; every
 * row then starts 16-byte aligned and no lane's vector crosses the row's end. Either way
 * the row's chunks of `block_threads` lanes must fit the grid's y extent.
 */
bool flydslGatherKernelMatches(const MatchContext& context,
                               const BoundTokens& bound,
                               const KernelDefinition& kernel)
{
    const auto* attributes = gatherNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const auto* x = findTensor(context, attributes->x_tensor_uid());
    const auto* indices = findTensor(context, attributes->indices_tensor_uid());
    const auto* y = findTensor(context, attributes->y_tensor_uid());
    if(x == nullptr || indices == nullptr || y == nullptr)
    {
        return false;
    }

    const auto dtype = kernelDataTypeName(x->data_type());
    const auto indexDtype = indexDataTypeName(indices->data_type());
    if(!dtype.has_value() || !indexDtype.has_value()
       || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dtype
       || kernel.getStringMetadata(std::string(INDEX_DTYPE_FIELD)) != *indexDtype)
    {
        return false;
    }

    const auto bytes = elementBytes(x->data_type());
    const auto vecBytes = kernel.getIntMetadata(std::string(VEC_BYTES_FIELD));
    const auto blockThreads = kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD));
    const auto inner = tryGetBoundInt(bound, INNER_TOKEN);
    if((vecBytes != bytes && vecBytes != VECTOR_BYTES) || blockThreads <= 0 || !inner.has_value())
    {
        return false;
    }
    const auto rowElems = blockThreads * (vecBytes / bytes);
    if((*inner + rowElems - 1) / rowElems > MAX_GRID_YZ)
    {
        return false;
    }
    return vecBytes == bytes
           || ((*inner * bytes) % VECTOR_BYTES == 0 && declaresAlignment(*x, VECTOR_BYTES)
               && declaresAlignment(*y, VECTOR_BYTES));
}

/// The descriptors' priority: a vector object (100) above a per-element one (10).
double flydslGatherScore(const MatchContext& /*context*/,
                         const BoundTokens& /*bound*/,
                         const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

int64_t requireBound(const BoundTokens& bound, std::string_view token)
{
    const auto value = tryGetBoundInt(bound, token);
    if(!value.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "flydsl_gather dispatch is missing bound token '" + std::string(token)
                + "', or it does not hold an integer");
    }
    return *value;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The tensor uids a matched gather graph binds, and its extents.
struct FlydslGatherBinding
{
    int64_t x = 0;
    int64_t indices = 0;
    int64_t y = 0;
    int32_t numIndices = 0;
    int32_t axisExtent = 0;
    int32_t inner = 0;
};

/// The compiled kernel plus everything read from the graph, resolved once.
class PreparedFlydslGather : public PreparedDispatch
{
public:
    PreparedFlydslGather(IngestorKernelCode code, FlydslGatherBinding binding)
        : _code(std::move(code))
        , _binding(binding)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslGatherBinding& binding() const
    {
        return _binding;
    }

private:
    IngestorKernelCode _code;
    FlydslGatherBinding _binding;
};

/**
 * @brief The argument list this pack launches with, mirroring launch() below one for one:
 *        three pointers (x, indices, y), then three int32s (num_indices, axis_extent,
 *        inner). Names are empty and offsets zero because neither is compared for this
 *        producer; see requireSignatureMatch.
 */
const std::vector<KernelArgument>& flydslGatherKernelSignature()
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_int{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const std::vector<KernelArgument> s_signature{
        s_buffer, s_buffer, s_buffer, s_int, s_int, s_int};
    return s_signature;
}

/**
 * @brief Sizes and launches a flyDSL gather kernel. Everything graph/kernel-derived
 *        resolves at prepare(); launch() only resolves buffers and launches.
 */
class FlydslGatherDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    FlydslGatherDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// A copy: no scratch.
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
        FlydslGatherBinding binding;
        binding.x = requireBound(bound, X_TOKEN);
        binding.indices = requireBound(bound, INDICES_TOKEN);
        binding.y = requireBound(bound, Y_TOKEN);
        // Bounded by the graph match to int32 and to the grid's limits.
        binding.numIndices = static_cast<int32_t>(requireBound(bound, NUM_INDICES_TOKEN));
        binding.axisExtent = static_cast<int32_t>(requireBound(bound, AXIS_EXTENT_TOKEN));
        binding.inner = static_cast<int32_t>(requireBound(bound, INNER_TOKEN));
        const auto outer = requireBound(bound, OUTER_TOKEN);

        const auto* x = findTensor(context, binding.x);
        if(x == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_gather: prepare() given a graph its matcher would decline");
        }
        const auto blockThreads = kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD));
        const auto vecElems
            = kernel.getIntMetadata(std::string(VEC_BYTES_FIELD)) / elementBytes(x->data_type());
        if(blockThreads <= 0 || vecElems <= 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_gather: kernel '" + kernel.name
                    + "' declares no usable block_threads or vec_bytes");
        }
        const auto rowElems = blockThreads * vecElems;
        const auto chunks = (static_cast<int64_t>(binding.inner) + rowElems - 1) / rowElems;

        // Build defines are ignored on the kpack path; built from the arch alone, as the
        // other flyDSL packs do (the tensor overload classifies a layout these rows lack).
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, flydslGatherKernelSignature());

        // `gather_rows`: one workgroup per (index, chunk of the row, outer position).
        code.setBlockSize(static_cast<unsigned int>(blockThreads), 1, 1);
        code.setGridSize(static_cast<unsigned int>(binding.numIndices),
                         static_cast<unsigned int>(chunks),
                         static_cast<unsigned int>(outer));

        return std::make_unique<PreparedFlydslGather>(std::move(code), binding);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedGather = dynamic_cast<const PreparedFlydslGather&>(prepared);
        const auto& binding = preparedGather.binding();

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto indices
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.indices, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        // Changing this argument list means changing flydslGatherKernelSignature() with it.
        preparedGather.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    x.ptr,
                    indices.ptr,
                    y.ptr,
                    binding.numIndices,
                    binding.axisExtent,
                    binding.inner);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& flydslGatherKpackModuleCache()
{
    // Process-lifetime, and this pack's own: a cache key is (archive, toc_key, arch), so a
    // cache shared with another pack would answer that pack's lookups too.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

} // namespace

void resetFlydslGatherModuleCache()
{
    flydslGatherKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer.
const FlydslGatherDispatchHandler& flydslGatherDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslGatherKpackModuleCache());
    static const FlydslGatherDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslGatherSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslGatherGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslGatherKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslGatherScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslGatherDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
