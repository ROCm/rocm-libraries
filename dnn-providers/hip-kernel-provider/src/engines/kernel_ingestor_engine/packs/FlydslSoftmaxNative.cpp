// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslRmsNormNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/softmax_attributes_generated.h>
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
#include "engines/kernel_ingestor_engine/packs/FlydslSoftmaxCommon.hpp"

/**
 * @file FlydslSoftmaxNative.cpp
 * @brief The flyDSL softmax pack's native half: matching, scoring, dispatch, and the one
 *        function that registers them.
 *
 * Every kernel this pack selects is a pre-built HSACO (flydsl/kernels_src/kernels/softmax,
 * FlyDSL v0.5.0's softmax ported to 0.3.4). Baked at pack time, and so matcher obligations
 * here: p's and s's element types; the launch geometry; the vector width an object copies
 * with (`n_align`), which the row width and both row strides must be multiples of; and
 * either the row width (`N`, the register-buffered tier) or nothing (`N` 0: the width is
 * read from p's descriptor). The row count, whether stats are written and their units are
 * kernel arguments, so one object serves every graph of its class.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using flydsl_softmax::SoftmaxProblem;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_softmax.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_softmax.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_softmax.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_softmax.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view OUT_DTYPE_FIELD = "out_dtype";
constexpr std::string_view N_FIELD = "N";
constexpr std::string_view N_MAX_FIELD = "n_max";
constexpr std::string_view N_ALIGN_FIELD = "n_align";
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";
constexpr std::string_view THREADS_PER_ROW_FIELD = "threads_per_row";

constexpr std::string_view P_TOKEN = "flydsl_softmax.p.uid";
constexpr std::string_view S_TOKEN = "flydsl_softmax.s.uid";
/// Bound only when the graph asks for stats.
constexpr std::string_view STATS_TOKEN = "flydsl_softmax.stats.uid";
constexpr std::string_view COLUMNS_TOKEN = "flydsl_softmax.columns";
constexpr std::string_view P_ROW_STRIDE_TOKEN = "flydsl_softmax.p_row_stride";
constexpr std::string_view S_ROW_STRIDE_TOKEN = "flydsl_softmax.s_row_stride";

/// The base alignment a vector object's 128-bit copies need.
constexpr int64_t VECTOR_ALIGNMENT = 16;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched softmax graph binds, in argument order.
struct FlydslSoftmaxBinding
{
    int64_t p = 0;
    int64_t s = 0;
    std::optional<int64_t> stats;
};

/// The node this pack's matchers read, or nullptr if the graph isn't a single softmax
/// node. Override-shape graphs are declined: the problem is resolved from the graph's
/// shapes at prepare.
const data_objects::SoftmaxAttributes* softmaxNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::SoftmaxAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::SoftmaxAttributes>();
}

/// The problem for the graph's single softmax node, or nullopt if the kernels do not
/// serve it.
std::optional<SoftmaxProblem> softmaxProblem(const MatchContext& context,
                                             const data_objects::SoftmaxAttributes& attributes)
{
    const auto* p = flydsl_norm::findTensor(context, attributes.p_tensor_uid());
    const auto* s = flydsl_norm::findTensor(context, attributes.s_tensor_uid());
    if(p == nullptr || s == nullptr || !flydsl_norm::isDeviceOperand(*p)
       || !flydsl_norm::isDeviceOperand(*s))
    {
        return std::nullopt;
    }
    const data_objects::TensorAttributes* stats = nullptr;
    if(attributes.stats_tensor_uid().has_value())
    {
        stats = flydsl_norm::findTensor(context, *attributes.stats_tensor_uid());
        if(stats == nullptr || !flydsl_norm::isDeviceOperand(*stats))
        {
            return std::nullopt;
        }
    }
    return flydsl_softmax::resolveSoftmaxProblem(attributes, *p, *s, stats);
}

/**
 * @brief Graph-scoped applicability: a single softmax node whose problem the pre-built
 *        kernels compute exactly.
 *
 * The kernels compute in f32: at least the precision an UNSET, FLOAT, HALF or BFLOAT16
 * compute type asks for; any other compute type declines.
 */
std::optional<BoundTokens> flydslSoftmaxGraphMatches(const MatchContext& context)
{
    const auto* attributes = softmaxNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    if(computeType != data_objects::DataType::UNSET && computeType != data_objects::DataType::FLOAT
       && computeType != data_objects::DataType::HALF
       && computeType != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }
    const auto problem = softmaxProblem(context, *attributes);
    if(!problem.has_value())
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(P_TOKEN)] = attributes->p_tensor_uid();
    bound[std::string(S_TOKEN)] = attributes->s_tensor_uid();
    if(attributes->stats_tensor_uid().has_value())
    {
        bound[std::string(STATS_TOKEN)] = *attributes->stats_tensor_uid();
    }
    bound[std::string(COLUMNS_TOKEN)] = problem->columns;
    bound[std::string(P_ROW_STRIDE_TOKEN)] = problem->pRowStride;
    bound[std::string(S_ROW_STRIDE_TOKEN)] = problem->sRowStride;
    return bound;
}

/**
 * @brief Kernel-scoped applicability: the object built for p's and s's dtypes, whose
 *        copies the row width, the row strides and the bases admit, and whose width rule
 *        serves the graph's.
 *
 * A baked object (`N` > 0) serves exactly its width; a runtime one (`N` 0) any width up to
 * its `n_max` (0: any). Every candidate that passes stays in the set, and the descriptor
 * priority orders them (flydslSoftmaxScore).
 */
bool flydslSoftmaxKernelMatches(const MatchContext& context,
                                const BoundTokens& bound,
                                const KernelDefinition& kernel)
{
    const auto* attributes = softmaxNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const auto* p = flydsl_norm::findTensor(context, attributes->p_tensor_uid());
    const auto* s = flydsl_norm::findTensor(context, attributes->s_tensor_uid());
    if(p == nullptr || s == nullptr)
    {
        return false;
    }
    const auto pType = flydsl_norm::kernelDataTypeName(p->data_type());
    const auto sType = flydsl_norm::kernelDataTypeName(s->data_type());
    if(!pType.has_value() || !sType.has_value()
       || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *pType
       || kernel.getStringMetadata(std::string(OUT_DTYPE_FIELD)) != *sType)
    {
        return false;
    }

    const auto columns = tryGetBoundInt(bound, COLUMNS_TOKEN);
    const auto pRowStride = tryGetBoundInt(bound, P_ROW_STRIDE_TOKEN);
    const auto sRowStride = tryGetBoundInt(bound, S_ROW_STRIDE_TOKEN);
    if(!columns.has_value() || !pRowStride.has_value() || !sRowStride.has_value())
    {
        return false;
    }

    // A vector object copies n_align elements at a time from each row's start: the width
    // and both row pitches must be multiples of it, and the bases 16-byte aligned.
    const auto nAlign = kernel.getIntMetadata(std::string(N_ALIGN_FIELD));
    if(nAlign <= 0 || *columns % nAlign != 0 || *pRowStride % nAlign != 0
       || *sRowStride % nAlign != 0)
    {
        return false;
    }
    if(nAlign > 1
       && (!flydsl_softmax::declaresAlignment(*p, VECTOR_ALIGNMENT)
           || !flydsl_softmax::declaresAlignment(*s, VECTOR_ALIGNMENT)))
    {
        return false;
    }

    const auto bakedN = kernel.getIntMetadata(std::string(N_FIELD));
    if(bakedN != flydsl_softmax::GENERIC_N_SENTINEL)
    {
        return *columns == bakedN;
    }
    const auto nMax = kernel.getIntMetadata(std::string(N_MAX_FIELD));
    return nMax == flydsl_softmax::UNBOUNDED_N_MAX || *columns <= nMax;
}

/**
 * @brief Ranks the survivors by descriptor priority: the baked width first, then the
 *        runtime tiers narrowest first (each bounded by `n_max`), the per-element objects
 *        last. A measured model replaces it when there is data to fit one.
 */
double flydslSoftmaxScore(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

/**
 * @brief Re-reads the operand bindings a match established.
 *
 * @throws HipdnnPluginException if the graph is not one this matcher accepts.
 */
FlydslSoftmaxBinding flydslSoftmaxBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_softmax dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };
    return {read(P_TOKEN), read(S_TOKEN), tryGetBoundInt(bound, STATS_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The compiled kernel plus everything read from the graph, resolved once, owning nothing
/// that points back into it.
class PreparedFlydslSoftmax : public PreparedDispatch
{
public:
    PreparedFlydslSoftmax(IngestorKernelCode code,
                          FlydslSoftmaxBinding binding,
                          SoftmaxProblem problem,
                          int64_t rowsPerLaunch)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _rowsPerLaunch(rowsPerLaunch)
    {
    }

    /// The kernel for the device this dispatch is running on; see FlydslRmsNormNative.cpp.
    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslSoftmaxBinding& binding() const
    {
        return _binding;
    }

    const SoftmaxProblem& problem() const
    {
        return _problem;
    }

    int64_t rowsPerLaunch() const
    {
        return _rowsPerLaunch;
    }

private:
    IngestorKernelCode _code;
    FlydslSoftmaxBinding _binding;
    SoftmaxProblem _problem;
    int64_t _rowsPerLaunch;
};

/**
 * @brief The argument list this pack launches with, mirroring launch() below one for one
 *        and the `signature` every one of this pack's UKDs carries: p and s, each a pointer
 *        and a 2-D descriptor (rows, width, row stride); stats, a pointer and a 1-D
 *        descriptor (rows); the row count (int32); whether to store stats (int32); the
 *        stats scale (f32).
 *
 * Names are empty and offsets zero because neither is compared for this producer; see
 * requireSignatureMatch.
 */
const std::vector<KernelArgument>& flydslSoftmaxKernelSignature()
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_desc2d{
        "by_value", static_cast<uint32_t>(sizeof(flydsl_norm::Desc2D)), 0, ""};
    static const KernelArgument s_desc1d{
        "by_value", static_cast<uint32_t>(sizeof(flydsl_norm::Desc1D)), 0, ""};
    static const KernelArgument s_i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const KernelArgument s_f32{"by_value", static_cast<uint32_t>(sizeof(float)), 0, ""};
    static const std::vector<KernelArgument> s_signature{
        s_buffer, s_desc2d, s_buffer, s_desc2d, s_buffer, s_desc1d, s_i32, s_i32, s_f32};
    return s_signature;
}

/// The bytes [begin, end) an operand of @p rows rows spans.
struct ByteRange
{
    uintptr_t begin = 0;
    uintptr_t end = 0;
};

ByteRange
    rowsSpan(const void* base, int64_t rows, int64_t rowStride, int64_t columns, int64_t bytes)
{
    const auto begin = reinterpret_cast<uintptr_t>(base);
    return {begin, begin + static_cast<uintptr_t>(((rows - 1) * rowStride + columns) * bytes)};
}

bool overlaps(const ByteRange& lhs, const ByteRange& rhs)
{
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

/**
 * @brief The native dispatch behind this pack's UDD. Everything graph/kernel-derived
 *        resolves once at prepare(); launch() only resolves buffers and launches, so
 *        nothing mutates once prepared and concurrent execution is safe.
 */
class FlydslSoftmaxDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler, @param kpackLoader Must outlive this handler; see
    /// FlydslRmsNormNative.cpp.
    FlydslSoftmaxDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                 const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// Rows reduce in registers, shuffles and LDS: no global scratch.
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
        const auto binding = flydslSoftmaxBinding(bound);
        const auto* attributes = softmaxNode(context);
        const auto problem
            = attributes == nullptr ? std::nullopt : softmaxProblem(context, *attributes);
        if(!problem.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_softmax: prepare() given a graph its matcher would decline");
        }

        const auto blockThreads = kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD));
        const auto threadsPerRow = kernel.getIntMetadata(std::string(THREADS_PER_ROW_FIELD));
        if(blockThreads <= 0 || threadsPerRow <= 0 || blockThreads % threadsPerRow != 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_softmax: kernel " + kernel.name + " declares block_threads "
                    + std::to_string(blockThreads) + " and threads_per_row "
                    + std::to_string(threadsPerRow));
        }
        const auto rowsPerBlock = blockThreads / threadsPerRow;
        const auto rowsPerLaunch = flydsl_softmax::rowsPerLaunch(*problem, rowsPerBlock);

        // Built from the arch alone; see FlydslRmsNormNative.cpp for why not from p.
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code = buildIngestorKernelCode(_kernelCompiler,
                                            _kpackLoader,
                                            context,
                                            kernel,
                                            options,
                                            flydslSoftmaxKernelSignature());

        // `rows_per_block`: each block serves block_threads / threads_per_row rows. Every
        // launch gets the grid of a full slice; the kernel skips the rows past its row
        // count, so the last, shorter slice needs no other geometry.
        code.setBlockSize(static_cast<unsigned int>(blockThreads), 1, 1);
        code.setGridSize(
            static_cast<unsigned int>((rowsPerLaunch + rowsPerBlock - 1) / rowsPerBlock), 1, 1);

        return std::make_unique<PreparedFlydslSoftmax>(
            std::move(code), binding, *problem, rowsPerLaunch);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedSoftmax = dynamic_cast<const PreparedFlydslSoftmax&>(prepared);
        const auto& binding = preparedSoftmax.binding();
        const auto& problem = preparedSoftmax.problem();

        auto* p = static_cast<char*>(
            hipdnn_plugin_sdk::findDeviceBuffer(binding.p, deviceBuffers, numDeviceBuffers).ptr);
        auto* s = static_cast<char*>(
            hipdnn_plugin_sdk::findDeviceBuffer(binding.s, deviceBuffers, numDeviceBuffers).ptr);
        char* stats = nullptr;
        if(binding.stats.has_value())
        {
            stats = static_cast<char*>(
                hipdnn_plugin_sdk::findDeviceBuffer(*binding.stats, deviceBuffers, numDeviceBuffers)
                    .ptr);
        }

        const auto pBytes = flydsl_norm::elementBytes(problem.pType);
        const auto sBytes = flydsl_norm::elementBytes(problem.sType);
        // s may be p itself, as the CPU reference allows: each element is read before the
        // thread that reads it writes it, and a row is never written before every read of
        // it is done. Any other overlap would have one row's writes clobber another's reads.
        const auto pSpan = rowsSpan(p, problem.rows, problem.pRowStride, problem.columns, pBytes);
        const auto sSpan = rowsSpan(s, problem.rows, problem.sRowStride, problem.columns, sBytes);
        const bool inPlace = p == s && pBytes == sBytes && problem.pRowStride == problem.sRowStride;
        bool aliased = !inPlace && overlaps(pSpan, sSpan);
        if(stats != nullptr)
        {
            const auto statsSpan = rowsSpan(stats, problem.rows, 1, 1, 4);
            aliased = aliased || overlaps(statsSpan, pSpan) || overlaps(statsSpan, sSpan);
        }
        if(aliased)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "flydsl_softmax: s may only alias p exactly (same address, dtype and row "
                "stride), and stats may alias neither");
        }

        const int32_t storeStats = stats != nullptr ? 1 : 0;
        const float statsScale = problem.statsLog2 ? static_cast<float>(1.0 / std::log(2.0)) : 1.0F;

        // Several launches only when one would address more than 2^31 - 1 bytes of p or s
        // (flydsl_softmax::rowsPerLaunch()); each covers a slice of whole rows.
        auto& kernel = preparedSoftmax.kernelForStream(handle.getStream());
        for(int64_t first = 0; first < problem.rows; first += preparedSoftmax.rowsPerLaunch())
        {
            const auto rows = static_cast<int32_t>(
                std::min(preparedSoftmax.rowsPerLaunch(), problem.rows - first));
            const flydsl_norm::Desc2D pDesc{
                rows, static_cast<int32_t>(problem.columns), problem.pRowStride};
            const flydsl_norm::Desc2D sDesc{
                rows, static_cast<int32_t>(problem.columns), problem.sRowStride};
            const flydsl_norm::Desc1D statsDesc{rows};
            void* pSlice = p + first * problem.pRowStride * pBytes;
            void* sSlice = s + first * problem.sRowStride * sBytes;
            // Without stats the slot is never stored through; s fills it with a valid
            // mapping rather than a null. Changing this argument list means changing
            // flydslSoftmaxKernelSignature() with it.
            void* statsSlice = stats != nullptr ? stats + first * 4 : sSlice;
            kernel.launch(handle.getStream(),
                          pSlice,
                          pDesc,
                          sSlice,
                          sDesc,
                          statsSlice,
                          statsDesc,
                          rows,
                          storeStats,
                          statsScale);
        }
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& flydslSoftmaxKpackModuleCache()
{
    // Process-lifetime and this pack's own; see FlydslRmsNormNative.cpp.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

} // namespace

void resetFlydslSoftmaxModuleCache()
{
    flydslSoftmaxKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime; see FlydslRmsNormNative.cpp.
const FlydslSoftmaxDispatchHandler& flydslSoftmaxDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslSoftmaxKpackModuleCache());
    static const FlydslSoftmaxDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslSoftmaxSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslSoftmaxGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslSoftmaxKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslSoftmaxScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslSoftmaxDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
