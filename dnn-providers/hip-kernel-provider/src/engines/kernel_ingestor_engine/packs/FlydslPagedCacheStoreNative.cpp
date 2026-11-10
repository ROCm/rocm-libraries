// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslRmsNormNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/paged_cache_store_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
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
 * @file FlydslPagedCacheStoreNative.cpp
 * @brief The flyDSL paged KV-cache store pack: matching, scoring, dispatch, registration.
 *
 * The kernel (`flydsl/kernels_src/kernels/paged_cache/paged_cache_store_kernel.py`) writes
 * `container[slot / bs, h, slot % bs, :] = cast(x[token, h, :] * scale)` for every token
 * whose slot is in range, one workgroup per token, and touches no other container element.
 * Everything but two element types is a kernel argument: H, D, block_size, the slot count,
 * every stride of x, slot_mapping and the container, the slot dtype and the scale. So one
 * object per (x dtype, container dtype) serves every graph of that pair, and the matcher's
 * obligations are the dtypes, the 32-bit bounds the kernel's in-row offsets and counts
 * assume, and a container that no two (block, head, offset, d) coordinates alias.
 *
 * The container is x's dtype or FP8. Which FP8 encodings exist is a property of the target:
 * gfx942's conversion produces the FNUZ formats and gfx950's the OCP ones, and gfx90a has
 * none. That lives in which objects each arch's shard carries, so the matcher accepts every
 * FP8 container and the kernel match finds an object only where the hardware converts to it.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using flydsl_norm::findTensor;
using flydsl_norm::hasUsableShape;
using flydsl_norm::isDeviceOperand;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_paged_cache_store.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.flydsl_paged_cache_store.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_paged_cache_store.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_paged_cache_store.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view CACHE_DTYPE_FIELD = "cache_dtype";
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";

constexpr std::string_view X_TOKEN = "flydsl_paged_cache_store.x.uid";
constexpr std::string_view SLOT_MAPPING_TOKEN = "flydsl_paged_cache_store.slot_mapping.uid";
constexpr std::string_view CONTAINER_TOKEN = "flydsl_paged_cache_store.container.uid";
/// Bound only when the graph names a scale.
constexpr std::string_view SCALE_TOKEN = "flydsl_paged_cache_store.scale.uid";

/// The vector path's width in bytes of x; the kernel's VECTOR_BYTES.
constexpr int64_t VECTOR_BYTES = 16;

constexpr int64_t INT32_LIMIT = std::numeric_limits<int32_t>::max();
/// The largest grid y the launch may ask for.
constexpr int64_t GRID_Y_LIMIT = 65535;
/// Room above H * D for the kernel's last workgroup step (`tid + base` in int32).
constexpr int64_t MAX_BLOCK_THREADS = 1024;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched graph binds.
struct PagedCacheStoreBinding
{
    int64_t x = 0;
    int64_t slotMapping = 0;
    int64_t container = 0;
    std::optional<int64_t> scale;
};

/// The graph as the kernel addresses it, every stride in elements. Token-packed x [T, H, D]
/// is one batch of T tokens with a zero batch stride.
struct StoreGeometry
{
    int64_t batch = 0;
    int64_t seqLen = 0;
    int64_t heads = 0;
    int64_t headDim = 0;
    int64_t numBlocks = 0;
    int64_t blockSize = 0;
    int64_t xStrideB = 0;
    int64_t xStrideS = 0;
    int64_t xStrideH = 0;
    int64_t xStrideD = 0;
    int64_t slotStrideB = 0;
    int64_t slotStrideS = 0;
    int64_t cStrideBlock = 0;
    int64_t cStrideSlot = 0;
    int64_t cStrideH = 0;
    int64_t cStrideD = 0;
};

const data_objects::PagedCacheStoreAttributes* pagedCacheStoreNode(const MatchContext& context)
{
    // Shapes and strides are read at prepare, so override-shape graphs are declined.
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::PagedCacheStoreAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::PagedCacheStoreAttributes>();
}

/// The kernel metadata spelling of an x dtype; nullopt for one no object reads.
std::optional<std::string> xDataTypeName(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::BFLOAT16:
        return std::string("bf16");
    case data_objects::DataType::HALF:
        return std::string("f16");
    case data_objects::DataType::FLOAT:
        return std::string("f32");
    default:
        return std::nullopt;
    }
}

/// The kernel metadata spelling of a container dtype: x's own names, or an FP8 format.
std::optional<std::string> cacheDataTypeName(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::FP8_E4M3:
        return std::string("fp8_e4m3");
    case data_objects::DataType::FP8_E5M2:
        return std::string("fp8_e5m2");
    case data_objects::DataType::FP8_E4M3_FNUZ:
        return std::string("fp8_e4m3_fnuz");
    case data_objects::DataType::FP8_E5M2_FNUZ:
        return std::string("fp8_e5m2_fnuz");
    default:
        return xDataTypeName(dataType);
    }
}

int64_t elementBytes(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::FLOAT:
    case data_objects::DataType::INT32:
        return 4;
    case data_objects::DataType::BFLOAT16:
    case data_objects::DataType::HALF:
        return 2;
    case data_objects::DataType::INT64:
        return 8;
    default:
        return 1;
    }
}

/// Bytes from an operand's first addressed element to one past its last, or nullopt when an
/// extent is non-positive, a stride negative, or the span does not fit 63 bits.
std::optional<int64_t> spanBytes(const std::vector<std::pair<int64_t, int64_t>>& axes,
                                 int64_t bytes)
{
    constexpr int64_t LIMIT = std::numeric_limits<int64_t>::max() / 2;
    int64_t last = 0;
    for(const auto& [extent, stride] : axes)
    {
        if(extent <= 0 || stride < 0)
        {
            return std::nullopt;
        }
        if(stride != 0 && extent - 1 > (LIMIT - last) / stride)
        {
            return std::nullopt;
        }
        last += (extent - 1) * stride;
    }
    if(last + 1 > LIMIT / bytes)
    {
        return std::nullopt;
    }
    return (last + 1) * bytes;
}

/// Whether no two coordinates of a layout address the same element: with its moving axes
/// ordered by stride, each stride steps past everything the axes inside it can reach. That
/// is sufficient, not necessary, but covers every packed, padded and permuted layout.
bool isNonOverlapping(std::vector<std::pair<int64_t, int64_t>> axes)
{
    axes.erase(
        std::remove_if(axes.begin(), axes.end(), [](const auto& axis) { return axis.first == 1; }),
        axes.end());
    std::sort(axes.begin(), axes.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second < rhs.second;
    });
    int64_t reach = 0;
    for(const auto& [extent, stride] : axes)
    {
        if(stride <= reach)
        {
            return false;
        }
        reach += (extent - 1) * stride;
    }
    return true;
}

std::vector<int64_t> asVector(const flatbuffers::Vector<int64_t>& values)
{
    return {values.begin(), values.end()};
}

/// The geometry of a graph the kernel can serve, or nullopt. Total: the graph is unvalidated.
std::optional<StoreGeometry> storeGeometry(const data_objects::TensorAttributes& x,
                                           const data_objects::TensorAttributes& slotMapping,
                                           const data_objects::TensorAttributes& container)
{
    if(!hasUsableShape(x, 3) || !hasUsableShape(slotMapping, 1) || !hasUsableShape(container, 4))
    {
        return std::nullopt;
    }
    const auto xDims = asVector(*x.dims());
    const auto xStrides = asVector(*x.strides());
    const auto sDims = asVector(*slotMapping.dims());
    const auto sStrides = asVector(*slotMapping.strides());
    const auto cDims = asVector(*container.dims());
    const auto cStrides = asVector(*container.strides());

    StoreGeometry geometry;
    if(xDims.size() == 3 && sDims.size() == 1 && sDims[0] == xDims[0])
    {
        geometry.batch = 1;
        geometry.seqLen = xDims[0];
        geometry.heads = xDims[1];
        geometry.xStrideS = xStrides[0];
        geometry.xStrideH = xStrides[1];
        geometry.xStrideD = xStrides[2];
        geometry.slotStrideS = sStrides[0];
    }
    else if(xDims.size() == 4 && sDims.size() == 2 && sDims[0] == xDims[0] && sDims[1] == xDims[2])
    {
        geometry.batch = xDims[0];
        geometry.seqLen = xDims[2];
        geometry.heads = xDims[1];
        geometry.xStrideB = xStrides[0];
        geometry.xStrideH = xStrides[1];
        geometry.xStrideS = xStrides[2];
        geometry.xStrideD = xStrides[3];
        geometry.slotStrideB = sStrides[0];
        geometry.slotStrideS = sStrides[1];
    }
    else
    {
        return std::nullopt;
    }
    geometry.headDim = xDims.back();
    if(cDims.size() != 4 || cDims[1] != geometry.heads || cDims[3] != geometry.headDim)
    {
        return std::nullopt;
    }
    geometry.numBlocks = cDims[0];
    geometry.blockSize = cDims[2];
    geometry.cStrideBlock = cStrides[0];
    geometry.cStrideH = cStrides[1];
    geometry.cStrideSlot = cStrides[2];
    geometry.cStrideD = cStrides[3];

    // Every operand addressable (positive extents, non-negative strides, a 63-bit span).
    const auto xBytes = elementBytes(x.data_type());
    const auto cBytes = elementBytes(container.data_type());
    const std::vector<std::pair<int64_t, int64_t>> xAxes{{geometry.batch, geometry.xStrideB},
                                                         {geometry.seqLen, geometry.xStrideS},
                                                         {geometry.heads, geometry.xStrideH},
                                                         {geometry.headDim, geometry.xStrideD}};
    const std::vector<std::pair<int64_t, int64_t>> cAxes{
        {geometry.numBlocks, geometry.cStrideBlock},
        {geometry.heads, geometry.cStrideH},
        {geometry.blockSize, geometry.cStrideSlot},
        {geometry.headDim, geometry.cStrideD}};
    if(!spanBytes(xAxes, xBytes).has_value() || !spanBytes(cAxes, cBytes).has_value()
       || !spanBytes(
               {{geometry.batch, geometry.slotStrideB}, {geometry.seqLen, geometry.slotStrideS}},
               elementBytes(slotMapping.data_type()))
               .has_value())
    {
        return std::nullopt;
    }

    // One token's row of x, and one slot's row of the container, are addressed with 32-bit
    // byte offsets from a 64-bit base; the slot count, S and H * D are int32 kernel
    // arguments, the last stepped through a workgroup at a time; B is the grid's y.
    const auto xRow = spanBytes(
        {{geometry.heads, geometry.xStrideH}, {geometry.headDim, geometry.xStrideD}}, xBytes);
    const auto cRow = spanBytes(
        {{geometry.heads, geometry.cStrideH}, {geometry.headDim, geometry.cStrideD}}, cBytes);
    if(!xRow.has_value() || *xRow > INT32_LIMIT || !cRow.has_value() || *cRow > INT32_LIMIT
       || geometry.seqLen > INT32_LIMIT || geometry.batch > GRID_Y_LIMIT
       || geometry.heads > (INT32_LIMIT - MAX_BLOCK_THREADS) / geometry.headDim
       || geometry.numBlocks > INT32_LIMIT / geometry.blockSize)
    {
        return std::nullopt;
    }

    // Every addressed slot row is written by exactly one token; aliasing coordinates would
    // make two of them, or two elements of one, race on one address.
    if(!isNonOverlapping(cAxes))
    {
        return std::nullopt;
    }
    return geometry;
}

/// Whether the graph's scale is one the kernel reads: one FLOAT element, either a device
/// tensor or a pass-by-value scalar.
bool isUsableScale(const MatchContext& context, int64_t uid)
{
    const auto* scale = findTensor(context, uid);
    if(scale == nullptr || scale->dims() == nullptr
       || scale->data_type() != data_objects::DataType::FLOAT)
    {
        return false;
    }
    for(const auto extent : *scale->dims())
    {
        if(extent != 1)
        {
            return false;
        }
    }
    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(scale))
    {
        try
        {
            static_cast<void>(
                hipdnn_plugin_sdk::makeScalarOperand(context.graph.getTensorMap(), uid, "scale"));
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
        {
            return false;
        }
        return true;
    }
    return isDeviceOperand(*scale);
}

/**
 * @brief Graph-scoped applicability: a single paged-cache-store node the shipped kernels
 *        compute exactly.
 */
std::optional<BoundTokens> flydslPagedCacheStoreGraphMatches(const MatchContext& context)
{
    const auto* attributes = pagedCacheStoreNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }

    // The product is formed in f32 and cast once, which is the reference's FLOAT compute.
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    if(computeType != data_objects::DataType::UNSET && computeType != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, attributes->x_tensor_uid());
    const auto* slotMapping = findTensor(context, attributes->slot_mapping_tensor_uid());
    const auto* container = findTensor(context, attributes->container_tensor_uid());
    if(x == nullptr || slotMapping == nullptr || container == nullptr || !isDeviceOperand(*x)
       || !isDeviceOperand(*slotMapping) || !isDeviceOperand(*container))
    {
        return std::nullopt;
    }

    const auto xName = xDataTypeName(x->data_type());
    const auto cacheName = cacheDataTypeName(container->data_type());
    const bool fp8Cache = elementBytes(container->data_type()) == 1;
    if(!xName.has_value() || !cacheName.has_value()
       || (!fp8Cache && container->data_type() != x->data_type()))
    {
        return std::nullopt;
    }
    if(slotMapping->data_type() != data_objects::DataType::INT32
       && slotMapping->data_type() != data_objects::DataType::INT64)
    {
        return std::nullopt;
    }

    if(!storeGeometry(*x, *slotMapping, *container).has_value())
    {
        return std::nullopt;
    }

    const auto scaleUid = attributes->scale_tensor_uid();
    if(scaleUid.has_value() && !isUsableScale(context, *scaleUid))
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(X_TOKEN)] = attributes->x_tensor_uid();
    bound[std::string(SLOT_MAPPING_TOKEN)] = attributes->slot_mapping_tensor_uid();
    bound[std::string(CONTAINER_TOKEN)] = attributes->container_tensor_uid();
    if(scaleUid.has_value())
    {
        bound[std::string(SCALE_TOKEN)] = *scaleUid;
    }
    return bound;
}

/// Kernel-scoped applicability: the object was built for x's dtype and the container's.
bool flydslPagedCacheStoreKernelMatches(const MatchContext& context,
                                        const BoundTokens& /*bound*/,
                                        const KernelDefinition& kernel)
{
    const auto* attributes = pagedCacheStoreNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const auto* x = findTensor(context, attributes->x_tensor_uid());
    const auto* container = findTensor(context, attributes->container_tensor_uid());
    if(x == nullptr || container == nullptr)
    {
        return false;
    }
    const auto xName = xDataTypeName(x->data_type());
    const auto cacheName = cacheDataTypeName(container->data_type());
    return xName.has_value() && cacheName.has_value()
           && kernel.getStringMetadata(std::string(DTYPE_FIELD)) == *xName
           && kernel.getStringMetadata(std::string(CACHE_DTYPE_FIELD)) == *cacheName;
}

/// One object per dtype pair, so at most one survives; its priority is the whole ranking.
double flydslPagedCacheStoreScore(const MatchContext& /*context*/,
                                  const BoundTokens& /*bound*/,
                                  const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

PagedCacheStoreBinding flydslPagedCacheStoreBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_paged_cache_store dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };
    return {read(X_TOKEN),
            read(SLOT_MAPPING_TOKEN),
            read(CONTAINER_TOKEN),
            hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, SCALE_TOKEN)};
}

const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    const auto* tensor = findTensor(context, uid);
    if(tensor == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched flydsl paged_cache_store graph has no tensor for uid " + std::to_string(uid));
    }
    return *tensor;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The kernel's by-value arguments after its four pointers, in order. The 64-bit strides
/// come first so the segment has no padding; flydslPagedCacheStoreKernelSignature() and
/// the generator's _EXPECTED_SIGNATURE state the same layout.
struct StoreArguments
{
    int64_t xStrideB = 0;
    int64_t xStrideS = 0;
    int64_t slotStrideB = 0;
    int64_t slotStrideS = 0;
    int64_t cStrideBlock = 0;
    int64_t cStrideSlot = 0;
    int32_t heads = 0;
    int32_t headDim = 0;
    int32_t xStrideH = 0;
    int32_t xStrideD = 0;
    int32_t cStrideH = 0;
    int32_t cStrideD = 0;
    int32_t blockSize = 0;
    int32_t numSlots = 0;
    int32_t slotIs64 = 0;
    /// Whether strides and D admit the vector path; the bases' alignment is checked per
    /// launch, since only the pointers say it.
    bool vectorStrides = false;
    /// The container bytes one vector stores: its alignment requirement.
    int64_t containerVectorBytes = 0;
};

StoreArguments storeArguments(const StoreGeometry& g,
                              const data_objects::TensorAttributes& x,
                              const data_objects::TensorAttributes& slotMapping,
                              const data_objects::TensorAttributes& container)
{
    StoreArguments a;
    a.xStrideB = g.xStrideB;
    a.xStrideS = g.xStrideS;
    a.slotStrideB = g.slotStrideB;
    a.slotStrideS = g.slotStrideS;
    a.cStrideBlock = g.cStrideBlock;
    a.cStrideSlot = g.cStrideSlot;
    a.heads = static_cast<int32_t>(g.heads);
    a.headDim = static_cast<int32_t>(g.headDim);
    a.xStrideH = static_cast<int32_t>(g.xStrideH);
    a.xStrideD = static_cast<int32_t>(g.xStrideD);
    a.cStrideH = static_cast<int32_t>(g.cStrideH);
    a.cStrideD = static_cast<int32_t>(g.cStrideD);
    a.blockSize = static_cast<int32_t>(g.blockSize);
    a.numSlots = static_cast<int32_t>(g.numBlocks * g.blockSize);
    a.slotIs64 = slotMapping.data_type() == data_objects::DataType::INT64 ? 1 : 0;

    // A vector of `v` elements moves 16 bytes of x and v container elements per step. It
    // needs contiguous rows, D a multiple of v, and every stride an axis can step along a
    // multiple of v, so that each vector's x and container addresses stay aligned.
    const auto v = VECTOR_BYTES / elementBytes(x.data_type());
    const auto steps
        = [v](int64_t extent, int64_t stride) { return extent == 1 || stride % v == 0; };
    a.containerVectorBytes = v * elementBytes(container.data_type());
    a.vectorStrides = g.headDim % v == 0 && g.xStrideD == 1 && g.cStrideD == 1
                      && steps(g.batch, g.xStrideB) && steps(g.seqLen, g.xStrideS)
                      && steps(g.heads, g.xStrideH) && steps(g.numBlocks, g.cStrideBlock)
                      && steps(g.blockSize, g.cStrideSlot) && steps(g.heads, g.cStrideH);
    return a;
}

class PreparedFlydslPagedCacheStore : public PreparedDispatch
{
public:
    PreparedFlydslPagedCacheStore(IngestorKernelCode code,
                                  PagedCacheStoreBinding binding,
                                  StoreArguments arguments,
                                  std::optional<hipdnn_plugin_sdk::ScalarOperand> scalarScale)
        : _code(std::move(code))
        , _binding(binding)
        , _arguments(arguments)
        , _scalarScale(scalarScale)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const PagedCacheStoreBinding& binding() const
    {
        return _binding;
    }

    const StoreArguments& arguments() const
    {
        return _arguments;
    }

    /// The scale when it is a pass-by-value scalar; nullopt when absent or a device tensor.
    const std::optional<hipdnn_plugin_sdk::ScalarOperand>& scalarScale() const
    {
        return _scalarScale;
    }

private:
    IngestorKernelCode _code;
    PagedCacheStoreBinding _binding;
    StoreArguments _arguments;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _scalarScale;
};

/**
 * @brief The argument list this pack launches with, mirroring launch() one for one: four
 *        global pointers (x, slot_mapping, container, scale), six 64-bit strides, eleven
 *        int32 and the f32 scale value.
 */
const std::vector<KernelArgument>& flydslPagedCacheStoreKernelSignature()
{
    static const std::vector<KernelArgument> s_signature = [] {
        std::vector<KernelArgument> signature;
        signature.reserve(4 + 6 + 11 + 1);
        for(int i = 0; i < 4; ++i)
        {
            signature.push_back({"global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""});
        }
        for(int i = 0; i < 6; ++i)
        {
            signature.push_back({"by_value", static_cast<uint32_t>(sizeof(int64_t)), 0, ""});
        }
        for(int i = 0; i < 11; ++i)
        {
            signature.push_back({"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""});
        }
        signature.push_back({"by_value", static_cast<uint32_t>(sizeof(float)), 0, ""});
        return signature;
    }();
    return s_signature;
}

bool isAligned(const void* pointer, int64_t bytes)
{
    return reinterpret_cast<uintptr_t>(pointer) % static_cast<uintptr_t>(bytes) == 0;
}

class FlydslPagedCacheStoreDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    FlydslPagedCacheStoreDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                         const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// Each element is written once, straight from x: no scratch.
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
        const auto binding = flydslPagedCacheStoreBinding(bound);
        const auto& x = requireTensor(context, binding.x);
        const auto& slotMapping = requireTensor(context, binding.slotMapping);
        const auto& container = requireTensor(context, binding.container);
        const auto geometry = storeGeometry(x, slotMapping, container);
        if(!geometry.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_paged_cache_store: prepare() given a graph its matcher would decline");
        }

        std::optional<hipdnn_plugin_sdk::ScalarOperand> scalarScale;
        if(binding.scale.has_value()
           && hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(
               &requireTensor(context, *binding.scale)))
        {
            scalarScale = hipdnn_plugin_sdk::makeScalarOperand(
                context.graph.getTensorMap(), *binding.scale, "scale");
        }

        const auto blockThreads
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD)));
        // Built from the arch alone: no layout define applies to a pre-built object.
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code = buildIngestorKernelCode(_kernelCompiler,
                                            _kpackLoader,
                                            context,
                                            kernel,
                                            options,
                                            flydslPagedCacheStoreKernelSignature());
        // `one_block_per_token`: grid (S, B), the block the kernel's own width.
        code.setBlockSize(blockThreads, 1, 1);
        code.setGridSize(static_cast<unsigned int>(geometry->seqLen),
                         static_cast<unsigned int>(geometry->batch),
                         1);

        return std::make_unique<PreparedFlydslPagedCacheStore>(
            std::move(code),
            binding,
            storeArguments(*geometry, x, slotMapping, container),
            scalarScale);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& store = dynamic_cast<const PreparedFlydslPagedCacheStore&>(prepared);
        const auto& binding = store.binding();
        const auto& a = store.arguments();

        void* x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers).ptr;
        void* slots = hipdnn_plugin_sdk::findDeviceBuffer(
                          binding.slotMapping, deviceBuffers, numDeviceBuffers)
                          .ptr;
        void* container = hipdnn_plugin_sdk::findDeviceBuffer(
                              binding.container, deviceBuffers, numDeviceBuffers)
                              .ptr;

        // A device scale is read through its pointer; otherwise the kernel takes the value,
        // and its scale pointer is never dereferenced (a zero-byte buffer resource).
        void* scalePointer = x;
        int32_t scaleFromPointer = 0;
        float scaleValue = 1.0F;
        if(store.scalarScale().has_value())
        {
            scaleValue = static_cast<float>(
                hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
                    *store.scalarScale(), deviceBuffers, numDeviceBuffers)));
        }
        else if(binding.scale.has_value())
        {
            scalePointer = hipdnn_plugin_sdk::findDeviceBuffer(
                               *binding.scale, deviceBuffers, numDeviceBuffers)
                               .ptr;
            scaleFromPointer = 1;
        }

        const int32_t vectorOk = a.vectorStrides && isAligned(x, VECTOR_BYTES)
                                         && isAligned(container, a.containerVectorBytes)
                                     ? 1
                                     : 0;

        // Changing this list means changing flydslPagedCacheStoreKernelSignature() with it.
        store.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    x,
                    slots,
                    container,
                    scalePointer,
                    a.xStrideB,
                    a.xStrideS,
                    a.slotStrideB,
                    a.slotStrideS,
                    a.cStrideBlock,
                    a.cStrideSlot,
                    a.heads,
                    a.headDim,
                    a.xStrideH,
                    a.xStrideD,
                    a.cStrideH,
                    a.cStrideD,
                    a.blockSize,
                    a.numSlots,
                    a.slotIs64,
                    scaleFromPointer,
                    vectorOk,
                    scaleValue);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& flydslPagedCacheStoreKpackModuleCache()
{
    // This pack's own: a cache key is (archive, toc_key, arch), so a shared one would
    // answer another pack's lookups too.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

} // namespace

void resetFlydslPagedCacheStoreModuleCache()
{
    flydslPagedCacheStoreKpackModuleCache().clear();
}

namespace
{

/// Process-lifetime: the registry holds a non-owning pointer to it.
const FlydslPagedCacheStoreDispatchHandler& flydslPagedCacheStoreDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        flydslPagedCacheStoreKpackModuleCache());
    static const FlydslPagedCacheStoreDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                        s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslPagedCacheStoreSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslPagedCacheStoreGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslPagedCacheStoreKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslPagedCacheStoreScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslPagedCacheStoreDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
