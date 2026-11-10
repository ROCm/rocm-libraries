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
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/topk_attributes_generated.h>
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
 * @file FlydslTopkNative.cpp
 * @brief The flyDSL top-k pack's native half (`hipkernel:flydsl_topk`): matching, scoring,
 *        dispatch, and the function that registers them.
 *
 * The kernel (`flydsl/kernels_src/kernels/topk/topk_kernel.py`) selects, for every row of
 * x viewed as `[rows, n]`, the k best entries along the last dim, sorted best-first, with
 * equal values in position order and NaN above every number -- the hipDNN CPU reference's
 * order exactly, since every element is ordered by an integer key that ties precisely when
 * the reference's float comparison does (bf16 and f16 widen to f32 exactly). Values are
 * copied from x bit for bit. Rows, n, k, `largest`, whether values are written, and whether
 * the rows admit 128-bit loads are kernel arguments; an object bakes x's dtype, the
 * indices' dtype and its body (`k_max`): 1 for the k = 1 reduction, or the radix select for
 * every k up to its bound.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using flydsl_norm::Desc2D;
using flydsl_norm::elementBytes;
using flydsl_norm::findTensor;
using flydsl_norm::hasUsableShape;
using flydsl_norm::isDeviceOperand;
using flydsl_norm::isPackedRowMajor;
using flydsl_norm::isSupportedDataType;
using flydsl_norm::kernelDataTypeName;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_topk.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_topk.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_topk.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_topk.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
/// `i32` or `i64`: the indices' element type, a store width baked into the object.
constexpr std::string_view INDEX_DTYPE_FIELD = "index_dtype";
/// The largest k the object serves: 1 for the reduction body, else the radix select's bound.
constexpr std::string_view K_MAX_FIELD = "k_max";
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";

constexpr std::string_view X_TOKEN = "flydsl_topk.x.uid";
constexpr std::string_view INDICES_TOKEN = "flydsl_topk.indices.uid";
/// Bound only when the graph names a values output.
constexpr std::string_view VALUES_TOKEN = "flydsl_topk.values.uid";
constexpr std::string_view ROWS_TOKEN = "flydsl_topk.rows";
constexpr std::string_view COLUMNS_TOKEN = "flydsl_topk.columns";
constexpr std::string_view K_TOKEN = "flydsl_topk.k";
constexpr std::string_view LARGEST_TOKEN = "flydsl_topk.largest";

/// The kernel's int32 extents and each row's 32-bit buffer offsets.
constexpr int64_t MAX_INT32 = std::numeric_limits<int32_t>::max();
/// The base alignment and row bytes at which the kernel reads x with 128-bit loads.
constexpr uintptr_t VECTOR_BYTES = 16;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The node this pack's matchers read, or nullptr if the graph isn't a single top-k node.
const data_objects::TopkAttributes* topkNode(const MatchContext& context)
{
    // Extents are bound from the graph's shapes at prepare, so override-shape graphs are
    // declined.
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::TopkAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::TopkAttributes>();
}

/// Bytes per index element, or nullopt for a dtype no object writes.
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

/// @p output has x's dims with the last one k.
bool isTopkOutputShape(const data_objects::TensorAttributes& output,
                       const data_objects::TensorAttributes& x,
                       int64_t k)
{
    const auto& outDims = *output.dims();
    const auto& xDims = *x.dims();
    if(outDims.size() != xDims.size())
    {
        return false;
    }
    const auto last = xDims.size() - 1;
    for(flatbuffers::uoffset_t axis = 0; axis < last; ++axis)
    {
        if(outDims.Get(axis) != xDims.Get(axis))
        {
            return false;
        }
    }
    return outDims.Get(last) == k;
}

/// Whether the comparison the graph asks for is the kernel's: the CPU reference compares in
/// f32 whatever the node says, and every type x widens into exactly (f32, f64, x's own)
/// orders and ties x's values as f32 does. A narrower compute type would round before
/// comparing, which no object does.
bool isExactComputeType(data_objects::DataType computeType, data_objects::DataType xType)
{
    return computeType == data_objects::DataType::UNSET
           || computeType == data_objects::DataType::FLOAT
           || computeType == data_objects::DataType::DOUBLE || computeType == xType;
}

/**
 * @brief Graph-scoped applicability: a single top-k node over packed operands of a dtype,
 *        an index dtype and extents the shipped objects compute exactly.
 *
 * Declines rather than approximates: non-packed operands (the kernel addresses rows by the
 * packed row stride and writes each output row contiguously), values in a dtype other than
 * x's, indices other than INT32/INT64, k outside [1, n], output shapes other than the
 * op's, a compute type that would round before comparing, and extents past the kernel's
 * int32 arguments or a row past its 32-bit buffer offsets. Every k up to the general
 * objects' bound is served; the kernel match declines a larger k.
 */
std::optional<BoundTokens> flydslTopkGraphMatches(const MatchContext& context)
{
    const auto* attributes = topkNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, attributes->x_tensor_uid());
    const auto* indices = findTensor(context, attributes->indices_tensor_uid());
    if(x == nullptr || indices == nullptr || !isDeviceOperand(*x) || !isDeviceOperand(*indices))
    {
        return std::nullopt;
    }
    const data_objects::TensorAttributes* values = nullptr;
    if(attributes->values_tensor_uid().has_value())
    {
        values = findTensor(context, *attributes->values_tensor_uid());
        if(values == nullptr || !isDeviceOperand(*values))
        {
            return std::nullopt;
        }
    }

    const auto indexElementBytes = indexBytes(indices->data_type());
    if(!isSupportedDataType(x->data_type()) || !indexElementBytes.has_value()
       || (values != nullptr && values->data_type() != x->data_type()))
    {
        return std::nullopt;
    }
    if(!isExactComputeType(context.graph.getNodeWrapper(0).computeDataType(), x->data_type()))
    {
        return std::nullopt;
    }

    if(!hasUsableShape(*x, 1) || !hasUsableShape(*indices, 1)
       || (values != nullptr && !hasUsableShape(*values, 1)))
    {
        return std::nullopt;
    }
    const auto& xDims = *x->dims();
    const auto last = xDims.size() - 1;
    const int64_t columns = xDims.Get(last);
    const int64_t k = attributes->k();
    if(columns <= 0 || k < 1 || k > columns)
    {
        return std::nullopt;
    }
    if(!isTopkOutputShape(*indices, *x, k)
       || (values != nullptr && !isTopkOutputShape(*values, *x, k)))
    {
        return std::nullopt;
    }
    if(!isPackedRowMajor(*x) || !isPackedRowMajor(*indices)
       || (values != nullptr && !isPackedRowMajor(*values)))
    {
        return std::nullopt;
    }

    // Every base holds one element's alignment: the kernel's scalar accesses need it. The
    // 128-bit row loads are chosen at launch from the actual base.
    const auto bytes = elementBytes(x->data_type());
    if(!declaresAlignment(*x, bytes) || !declaresAlignment(*indices, *indexElementBytes)
       || (values != nullptr && !declaresAlignment(*values, bytes)))
    {
        return std::nullopt;
    }

    int64_t rows = 1;
    for(flatbuffers::uoffset_t axis = 0; axis < last; ++axis)
    {
        const int64_t extent = xDims.Get(axis);
        if(extent <= 0 || rows > MAX_INT32 / extent)
        {
            return std::nullopt;
        }
        rows *= extent;
    }
    // One row of x within the kernel's 32-bit buffer offsets (which also bounds n and k to
    // int32); the row count is the grid's x extent and an int32 argument.
    if(columns > MAX_INT32 / bytes)
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(X_TOKEN)] = attributes->x_tensor_uid();
    bound[std::string(INDICES_TOKEN)] = attributes->indices_tensor_uid();
    if(values != nullptr)
    {
        bound[std::string(VALUES_TOKEN)] = *attributes->values_tensor_uid();
    }
    bound[std::string(ROWS_TOKEN)] = rows;
    bound[std::string(COLUMNS_TOKEN)] = columns;
    bound[std::string(K_TOKEN)] = k;
    bound[std::string(LARGEST_TOKEN)] = attributes->largest() ? 1 : 0;
    return bound;
}

/**
 * @brief Kernel-scoped applicability: the object's dtype and index dtype, and a body that
 *        serves the graph's k (`k <= k_max`; the reduction objects' k_max is 1).
 */
bool flydslTopkKernelMatches(const MatchContext& context,
                             const BoundTokens& bound,
                             const KernelDefinition& kernel)
{
    const auto* attributes = topkNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const auto* x = findTensor(context, attributes->x_tensor_uid());
    const auto* indices = findTensor(context, attributes->indices_tensor_uid());
    if(x == nullptr || indices == nullptr)
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
    const auto k = tryGetBoundInt(bound, K_TOKEN);
    return k.has_value() && *k >= 1 && *k <= kernel.getIntMetadata(std::string(K_MAX_FIELD));
}

/// The descriptors' priority: the k = 1 reduction (100) above the radix select (10).
double flydslTopkScore(const MatchContext& /*context*/,
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
            "flydsl_topk dispatch is missing bound token '" + std::string(token)
                + "', or it does not hold an integer");
    }
    return *value;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The tensor uids a matched top-k graph binds, and its extents.
struct FlydslTopkBinding
{
    int64_t x = 0;
    int64_t indices = 0;
    std::optional<int64_t> values;
    int32_t rows = 0;
    int32_t columns = 0;
    int32_t k = 0;
    int32_t largest = 1;
    /// x's row bytes, for the 128-bit load test at launch.
    int64_t rowBytes = 0;
};

/// The compiled kernel plus everything read from the graph, resolved once.
class PreparedFlydslTopk : public PreparedDispatch
{
public:
    PreparedFlydslTopk(IngestorKernelCode code, FlydslTopkBinding binding)
        : _code(std::move(code))
        , _binding(binding)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslTopkBinding& binding() const
    {
        return _binding;
    }

private:
    IngestorKernelCode _code;
    FlydslTopkBinding _binding;
};

/**
 * @brief The argument list this pack launches with, mirroring launch() below one for one:
 *        x, indices and values, each a pointer and a 2-D descriptor, then four int32s (k,
 *        largest, write_values, vec_ok). Names are empty and offsets zero because neither
 *        is compared for this producer; see requireSignatureMatch.
 */
const std::vector<KernelArgument>& flydslTopkKernelSignature()
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_desc2d{"by_value", static_cast<uint32_t>(sizeof(Desc2D)), 0, ""};
    static const KernelArgument s_int{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const std::vector<KernelArgument> s_signature{
        s_buffer, s_desc2d, s_buffer, s_desc2d, s_buffer, s_desc2d, s_int, s_int, s_int, s_int};
    return s_signature;
}

/**
 * @brief Sizes and launches a flyDSL top-k kernel. Everything graph/kernel-derived resolves
 *        at prepare(); launch() only resolves buffers, tests x's base for 128-bit loads, and
 *        launches.
 */
class FlydslTopkDispatchHandler : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    FlydslTopkDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                              const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// One workgroup per row, selecting in registers and LDS: no scratch.
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
        FlydslTopkBinding binding;
        binding.x = requireBound(bound, X_TOKEN);
        binding.indices = requireBound(bound, INDICES_TOKEN);
        binding.values = tryGetBoundInt(bound, VALUES_TOKEN);
        // Bounded by the graph match to int32.
        binding.rows = static_cast<int32_t>(requireBound(bound, ROWS_TOKEN));
        binding.columns = static_cast<int32_t>(requireBound(bound, COLUMNS_TOKEN));
        binding.k = static_cast<int32_t>(requireBound(bound, K_TOKEN));
        binding.largest = static_cast<int32_t>(requireBound(bound, LARGEST_TOKEN));

        const auto* x = findTensor(context, binding.x);
        const auto blockThreads = kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD));
        if(x == nullptr || blockThreads <= 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_topk: prepare() given a graph its matcher would decline, or kernel '"
                    + kernel.name + "' declares no usable block_threads");
        }
        binding.rowBytes = static_cast<int64_t>(binding.columns) * elementBytes(x->data_type());

        // Build defines are ignored on the kpack path; built from the arch alone, as the
        // other flyDSL packs do (the tensor overload classifies a layout these rows lack).
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, flydslTopkKernelSignature());

        // `one_block_per_row`.
        code.setBlockSize(static_cast<unsigned int>(blockThreads), 1, 1);
        code.setGridSize(static_cast<unsigned int>(binding.rows), 1, 1);

        return std::make_unique<PreparedFlydslTopk>(std::move(code), binding);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedTopk = dynamic_cast<const PreparedFlydslTopk&>(prepared);
        const auto& binding = preparedTopk.binding();

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto indices
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.indices, deviceBuffers, numDeviceBuffers);

        // Packed operands: x's row stride is its width, each output's is k.
        const Desc2D xDesc{binding.rows, binding.columns, static_cast<int64_t>(binding.columns)};
        const Desc2D outDesc{binding.rows, binding.k, static_cast<int64_t>(binding.k)};

        // Without a values output the kernel never stores through the values slots, which
        // are given x's pointer and descriptor so the kernarg segment holds a valid mapping.
        void* valuesPointer = x.ptr;
        Desc2D valuesDesc = xDesc;
        int32_t writeValues = 0;
        if(binding.values.has_value())
        {
            valuesPointer = hipdnn_plugin_sdk::findDeviceBuffer(
                                *binding.values, deviceBuffers, numDeviceBuffers)
                                .ptr;
            valuesDesc = outDesc;
            writeValues = 1;
        }

        // Every row starts 16-byte aligned when the base is and the row bytes are a
        // multiple of 16; the kernel then reads whole vectors and the tail element by
        // element. The base is the actual pointer, so a graph that declares less alignment
        // than its buffer has still gets the vector path.
        const bool vecOk = reinterpret_cast<uintptr_t>(x.ptr) % VECTOR_BYTES == 0
                           && binding.rowBytes % static_cast<int64_t>(VECTOR_BYTES) == 0;

        // Changing this argument list means changing flydslTopkKernelSignature() with it.
        preparedTopk.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    x.ptr,
                    xDesc,
                    indices.ptr,
                    outDesc,
                    valuesPointer,
                    valuesDesc,
                    binding.k,
                    binding.largest,
                    writeValues,
                    static_cast<int32_t>(vecOk ? 1 : 0));
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& flydslTopkKpackModuleCache()
{
    // Process-lifetime, and this pack's own: a cache key is (archive, toc_key, arch), so a
    // cache shared with another pack would answer that pack's lookups too.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

} // namespace

void resetFlydslTopkModuleCache()
{
    flydslTopkKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer.
const FlydslTopkDispatchHandler& flydslTopkDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslTopkKpackModuleCache());
    static const FlydslTopkDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslTopkSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslTopkGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslTopkKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslTopkScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslTopkDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
