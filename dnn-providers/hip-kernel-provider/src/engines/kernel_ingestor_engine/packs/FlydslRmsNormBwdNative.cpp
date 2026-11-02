// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslRmsNormNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
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
 * @file FlydslRmsNormBwdNative.cpp
 * @brief The flyDSL RMS-norm backward pack's native half: matching, scoring, dispatch, and
 *        the one function that registers them.
 *
 * Every kernel is a pre-built object from the vendored two-stage builder
 * (`flydsl/kernels_src/kernels/norm/rmsnorm_bwd_kernel.py`), holding two kernels:
 *
 *   - the partial kernel, `programs` blocks of 512 threads walking the rows, writes dx and
 *     leaves one f32 dgamma row per program in the workspace;
 *   - the reduce kernel, one block of 256 threads per 64 columns, sums those rows into
 *     dgamma in gamma's dtype.
 *
 * dgamma is therefore deterministic: no atomics, and the same program count and row
 * order on every launch of a plan. The program count is chosen here, from the row count
 * and the device's compute units, and read by both kernels from the workspace's
 * descriptor (vendored modification 2), so it is not baked into the objects.
 *
 * As for the forward pack, the kernels cannot be adapted, only chosen: the dtypes, the
 * width (baked, or bounded by a generic tier's `n_max`) and the packed layout are matcher
 * obligations. There is no epsilon (the forward folded it into inv_rms) and no dbias.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

using namespace flydsl_norm;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_rmsnorm_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_rmsnorm_bwd.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_rmsnorm_bwd.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_rmsnorm_bwd.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view N_FIELD = "N";
/// A generic object's width bound; a baked object's equals its N.
constexpr std::string_view N_MAX_FIELD = "n_max";
constexpr std::string_view SCALE_F32_FIELD = "scale_f32";
/// The reduce kernel's symbol in the same object.
constexpr std::string_view REDUCE_SYMBOL_FIELD = "reduce_symbol";

constexpr std::string_view DY_TOKEN = "flydsl_rmsnorm_bwd.dy.uid";
constexpr std::string_view X_TOKEN = "flydsl_rmsnorm_bwd.x.uid";
constexpr std::string_view SCALE_TOKEN = "flydsl_rmsnorm_bwd.scale.uid";
constexpr std::string_view INV_RMS_TOKEN = "flydsl_rmsnorm_bwd.inv_rms.uid";
constexpr std::string_view DX_TOKEN = "flydsl_rmsnorm_bwd.dx.uid";
constexpr std::string_view DSCALE_TOKEN = "flydsl_rmsnorm_bwd.dscale.uid";
constexpr std::string_view ROWS_TOKEN = "flydsl_rmsnorm_bwd.rows";
constexpr std::string_view COLUMNS_TOKEN = "flydsl_rmsnorm_bwd.columns";

// The builder's launch geometry, asserted against the kernel source by
// `generators/gen_rmsnorm_bwd.py` before any object is written.
constexpr unsigned int PARTIAL_THREADS = 512;
constexpr unsigned int REDUCE_THREADS = 256;
constexpr int64_t REDUCE_COLUMNS = 64;

/// Used when the device reports no CU count.
constexpr int64_t FALLBACK_COMPUTE_UNITS = 64;
/// The partial kernel's vec8 path: a 16-bit dtype at a baked N that fills full 128-bit
/// tiles across all 512 threads (`is_rmsnorm_bwd_two_stage_vec_config`).
constexpr int64_t VEC_WIDTH = 8;

/// The tensor uids a matched backward graph binds.
struct FlydslRmsNormBwdBinding
{
    int64_t dy = 0;
    int64_t x = 0;
    int64_t scale = 0;
    int64_t invRms = 0;
    int64_t dx = 0;
    int64_t dscale = 0;
};

const data_objects::RMSNormBackwardAttributes* rmsNormBackwardNode(const MatchContext& context)
{
    // Rows and width are bound from the graph's shapes at prepare, which an override-shape
    // graph may replace at execute.
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::RMSNormBackwardAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::RMSNormBackwardAttributes>();
}

bool usesVecPath(const KernelDefinition& kernel)
{
    const auto bakedN = kernel.getIntMetadata(std::string(N_FIELD));
    const auto& dtype = kernel.getStringMetadata(std::string(DTYPE_FIELD));
    return bakedN != GENERIC_N_SENTINEL && dtype != "f32"
           && bakedN >= static_cast<int64_t>(PARTIAL_THREADS) * VEC_WIDTH
           && bakedN % VEC_WIDTH == 0;
}

/// The partial-row count both kernels use: upstream's selector (one program per CU for
/// fewer rows; 1.5 per CU on the vec8 path and 2 per CU otherwise above 2048 / 1024 rows),
/// never more programs than rows -- a program with no row would only write zeros -- and
/// never fewer than one. Measured on gfx90a, gfx942 and gfx950 against a flat 2 per CU:
/// within 2% almost everywhere and up to 10% faster at 4096 x 8192.
int64_t partialPrograms(int64_t rows, int computeUnits, const KernelDefinition& kernel)
{
    const int64_t units = computeUnits > 0 ? computeUnits : FALLBACK_COMPUTE_UNITS;
    int64_t programs = units;
    if(usesVecPath(kernel))
    {
        programs = rows < 2048 ? units : (3 * units) / 2;
    }
    else
    {
        programs = rows < 1024 ? units : 2 * units;
    }
    return std::max<int64_t>(1, std::min(rows, programs));
}

/**
 * @brief Graph-scoped applicability: one RMS-norm backward node over packed operands the
 *        pre-built objects compute.
 */
std::optional<BoundTokens> flydslRmsNormBwdGraphMatches(const MatchContext& context)
{
    const auto* attributesPtr = rmsNormBackwardNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // No dbias: no object sums dy into one. The kernels accumulate in f32: at least the
    // precision a FLOAT, HALF or BFLOAT16 compute type asks for.
    if(attributes.dbias_tensor_uid().has_value())
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

    const auto* dy = findTensor(context, attributes.dy_tensor_uid());
    const auto* x = findTensor(context, attributes.x_tensor_uid());
    const auto* scale = findTensor(context, attributes.scale_tensor_uid());
    const auto* invRms = findTensor(context, attributes.inv_rms_tensor_uid());
    const auto* dx = findTensor(context, attributes.dx_tensor_uid());
    const auto* dscale = findTensor(context, attributes.dscale_tensor_uid());
    if(dy == nullptr || x == nullptr || scale == nullptr || invRms == nullptr || dx == nullptr
       || dscale == nullptr)
    {
        return std::nullopt;
    }
    for(const auto* tensor : {dy, x, scale, invRms, dx, dscale})
    {
        if(!isDeviceOperand(*tensor))
        {
            return std::nullopt;
        }
    }
    // The 16-bit objects at a baked width copy 128-bit vectors from every row operand and
    // from gamma; inv_rms is read one f32 at a time and has its own rule below.
    for(const auto* tensor : {dy, x, scale, dx, dscale})
    {
        if(!declaresVectorAlignment(*tensor))
        {
            return std::nullopt;
        }
    }

    // x, dy and dx in one dtype; gamma and dgamma in one dtype, x's or f32 beside 16-bit x.
    const auto rowType = x->data_type();
    const auto scaleType = scale->data_type();
    const bool scaleMatchesX = scaleType == rowType
                               || (scaleType == data_objects::DataType::FLOAT
                                   && rowType != data_objects::DataType::FLOAT);
    if(!isSupportedDataType(rowType) || dy->data_type() != rowType || dx->data_type() != rowType
       || !scaleMatchesX || dscale->data_type() != scaleType)
    {
        return std::nullopt;
    }

    if(!hasUsableShape(*x, 2) || !hasUsableShape(*dy, 2) || !hasUsableShape(*dx, 2)
       || !hasUsableShape(*scale, 1) || !hasUsableShape(*dscale, 1) || !hasUsableShape(*invRms, 2))
    {
        return std::nullopt;
    }
    if(!sameShape(*x, *dy) || !sameShape(*x, *dx) || !sameShape(*scale, *dscale))
    {
        return std::nullopt;
    }
    for(const auto* tensor : {dy, x, scale, dx, dscale})
    {
        if(!isPackedRowMajor(*tensor))
        {
            return std::nullopt;
        }
    }

    // Gamma covers exactly the normalised axes; inv_rms is one f32 per row.
    const auto split = normalisedSplit(*x, *scale);
    if(!split.has_value() || !isPerRowStatistic(*invRms, *x, *split))
    {
        return std::nullopt;
    }
    const auto rows = split->rows;
    const auto columns = split->columns;
    // Every element of each row operand within one buffer view's 32-bit byte offsets, which
    // also keeps M and N within the kernels' int32 arguments.
    const auto bytes = elementBytes(rowType);
    if(columns > MAX_OPERAND_BYTES / bytes || rows > MAX_OPERAND_BYTES / (bytes * columns))
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(DY_TOKEN)] = attributes.dy_tensor_uid();
    bound[std::string(X_TOKEN)] = attributes.x_tensor_uid();
    bound[std::string(SCALE_TOKEN)] = attributes.scale_tensor_uid();
    bound[std::string(INV_RMS_TOKEN)] = attributes.inv_rms_tensor_uid();
    bound[std::string(DX_TOKEN)] = attributes.dx_tensor_uid();
    bound[std::string(DSCALE_TOKEN)] = attributes.dscale_tensor_uid();
    bound[std::string(ROWS_TOKEN)] = rows;
    bound[std::string(COLUMNS_TOKEN)] = columns;
    return bound;
}

/**
 * @brief Kernel-scoped applicability: dtype, gamma form, and width. A baked object serves
 *        its own N; a generic one (`N == 0`) every N up to its `n_max`.
 */
bool flydslRmsNormBwdKernelMatches(const MatchContext& context,
                                   const BoundTokens& bound,
                                   const KernelDefinition& kernel)
{
    const auto* attributes = rmsNormBackwardNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const auto* x = findTensor(context, attributes->x_tensor_uid());
    const auto* scale = findTensor(context, attributes->scale_tensor_uid());
    if(x == nullptr || scale == nullptr)
    {
        return false;
    }

    const auto dtype = kernelDataTypeName(x->data_type());
    if(!dtype.has_value() || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dtype)
    {
        return false;
    }
    const int64_t scaleF32 = scale->data_type() != x->data_type() ? 1 : 0;
    if(kernel.getIntMetadata(std::string(SCALE_F32_FIELD)) != scaleF32)
    {
        return false;
    }

    const auto columns = tryGetBoundInt(bound, COLUMNS_TOKEN);
    if(!columns.has_value())
    {
        return false;
    }
    const auto bakedN = kernel.getIntMetadata(std::string(N_FIELD));
    if(bakedN == GENERIC_N_SENTINEL)
    {
        return *columns <= kernel.getIntMetadata(std::string(N_MAX_FIELD));
    }
    return *columns == bakedN;
}

/// The descriptors already order the candidates: baked widths first, then the narrower
/// generic tier before the wider.
double flydslRmsNormBwdScore(const MatchContext& /*context*/,
                             const BoundTokens& /*bound*/,
                             const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

FlydslRmsNormBwdBinding flydslRmsNormBwdBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_rmsnorm_bwd dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };
    return {read(DY_TOKEN),
            read(X_TOKEN),
            read(SCALE_TOKEN),
            read(INV_RMS_TOKEN),
            read(DX_TOKEN),
            read(DSCALE_TOKEN)};
}

/// Rows and columns as the matcher bound them.
std::pair<int64_t, int64_t> boundRowsAndColumns(const BoundTokens& bound)
{
    const auto rows = tryGetBoundInt(bound, ROWS_TOKEN);
    const auto columns = tryGetBoundInt(bound, COLUMNS_TOKEN);
    if(!rows.has_value() || !columns.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "flydsl_rmsnorm_bwd dispatch is missing its bound rows or columns");
    }
    return {*rows, *columns};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Both kernels plus everything read from the graph, resolved once.
class PreparedFlydslRmsNormBwd : public PreparedDispatch
{
public:
    PreparedFlydslRmsNormBwd(IngestorKernelCode partial,
                             IngestorKernelCode reduce,
                             FlydslRmsNormBwdBinding binding,
                             int32_t rows,
                             int32_t columns,
                             int32_t programs)
        : _partial(std::move(partial))
        , _reduce(std::move(reduce))
        , _binding(binding)
        , _rows(rows)
        , _columns(columns)
        , _programs(programs)
    {
    }

    compilation::IRunnableKernel& partialForStream(hipStream_t stream) const
    {
        return _partial.kernelForStream(stream);
    }

    compilation::IRunnableKernel& reduceForStream(hipStream_t stream) const
    {
        return _reduce.kernelForStream(stream);
    }

    const FlydslRmsNormBwdBinding& binding() const
    {
        return _binding;
    }

    int32_t rows() const
    {
        return _rows;
    }

    int32_t columns() const
    {
        return _columns;
    }

    int32_t programs() const
    {
        return _programs;
    }

private:
    IngestorKernelCode _partial;
    IngestorKernelCode _reduce;
    FlydslRmsNormBwdBinding _binding;
    int32_t _rows;
    int32_t _columns;
    int32_t _programs;
};

/**
 * @brief The partial kernel's arguments: Source (x), Gamma, DY, DResidualOut (dy again: the
 *        plain variant fills the fused-add slot with it), Rstd (inv_rms), DX and
 *        DWeightPartial, each a pointer then its descriptor, then the row count.
 */
const std::vector<KernelArgument>& flydslRmsNormBwdPartialSignature()
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_desc2d{"by_value", static_cast<uint32_t>(sizeof(Desc2D)), 0, ""};
    static const KernelArgument s_desc1d{"by_value", static_cast<uint32_t>(sizeof(Desc1D)), 0, ""};
    static const KernelArgument s_rows{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const std::vector<KernelArgument> s_signature{s_buffer,
                                                         s_desc2d,
                                                         s_buffer,
                                                         s_desc1d,
                                                         s_buffer,
                                                         s_desc2d,
                                                         s_buffer,
                                                         s_desc2d,
                                                         s_buffer,
                                                         s_desc1d,
                                                         s_buffer,
                                                         s_desc2d,
                                                         s_buffer,
                                                         s_desc2d,
                                                         s_rows};
    return s_signature;
}

/// The reduce kernel's arguments: DWeightPartial (2-D), then DWeight (dgamma, 1-D).
const std::vector<KernelArgument>& flydslRmsNormBwdReduceSignature()
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_desc2d{"by_value", static_cast<uint32_t>(sizeof(Desc2D)), 0, ""};
    static const KernelArgument s_desc1d{"by_value", static_cast<uint32_t>(sizeof(Desc1D)), 0, ""};
    static const std::vector<KernelArgument> s_signature{s_buffer, s_desc2d, s_buffer, s_desc1d};
    return s_signature;
}

/**
 * @brief The native dispatch behind this pack's UDD: sizes the dgamma workspace and launches
 *        the partial and reduce kernels.
 */
class FlydslRmsNormBwdDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    FlydslRmsNormBwdDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                    const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// One f32 dgamma row per partial program.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        const auto [rows, columns] = boundRowsAndColumns(bound);
        const auto programs
            = partialPrograms(rows, context.deviceProperties.multiProcessorCount, kernel);
        return static_cast<size_t>(programs) * static_cast<size_t>(columns) * sizeof(float);
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = flydslRmsNormBwdBinding(bound);
        const auto [rows, columns] = boundRowsAndColumns(bound);
        const auto programs
            = partialPrograms(rows, context.deviceProperties.multiProcessorCount, kernel);

        const auto reduceSymbol = kernel.getStringMetadata(std::string(REDUCE_SYMBOL_FIELD));
        if(reduceSymbol.empty())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_rmsnorm_bwd: kernel '" + kernel.name + "' names no reduce kernel");
        }

        // Built from the arch alone; see FlydslRmsNormNative.cpp's prepare().
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto partial = buildIngestorKernelCode(_kernelCompiler,
                                               _kpackLoader,
                                               context,
                                               kernel,
                                               options,
                                               flydslRmsNormBwdPartialSignature());
        // The reduce kernel is the same code object under another symbol: the same archive
        // entry, with that symbol and its own argument list.
        KernelDefinition reduceKernel = kernel;
        reduceKernel.source.symbol = reduceSymbol;
        reduceKernel.source.signature = flydslRmsNormBwdReduceSignature();
        auto reduce = buildIngestorKernelCode(_kernelCompiler,
                                              _kpackLoader,
                                              context,
                                              reduceKernel,
                                              options,
                                              flydslRmsNormBwdReduceSignature());

        partial.setBlockSize(PARTIAL_THREADS, 1, 1);
        partial.setGridSize(static_cast<unsigned int>(programs), 1, 1);
        reduce.setBlockSize(REDUCE_THREADS, 1, 1);
        reduce.setGridSize(
            static_cast<unsigned int>((columns + REDUCE_COLUMNS - 1) / REDUCE_COLUMNS), 1, 1);

        return std::make_unique<PreparedFlydslRmsNormBwd>(std::move(partial),
                                                          std::move(reduce),
                                                          binding,
                                                          static_cast<int32_t>(rows),
                                                          static_cast<int32_t>(columns),
                                                          static_cast<int32_t>(programs));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& preparedBwd = dynamic_cast<const PreparedFlydslRmsNormBwd&>(prepared);
        const auto& binding = preparedBwd.binding();
        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "flydsl_rmsnorm_bwd needs its dgamma workspace");
        }

        const auto buffer = [&](int64_t uid) {
            return hipdnn_plugin_sdk::findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers).ptr;
        };
        void* x = buffer(binding.x);
        void* dy = buffer(binding.dy);
        void* dx = buffer(binding.dx);
        void* scale = buffer(binding.scale);
        void* dscale = buffer(binding.dscale);
        void* invRms = buffer(binding.invRms);

        const auto rows = preparedBwd.rows();
        const auto columns = preparedBwd.columns();
        // Packed row-major is an applicability condition, so every row stride is the width.
        const Desc2D rowDesc{rows, columns, static_cast<int64_t>(columns)};
        const Desc1D scaleDesc{columns};
        const Desc1D invRmsDesc{rows};
        const Desc2D partialDesc{preparedBwd.programs(), columns, static_cast<int64_t>(columns)};

        const auto stream = handle.getStream();
        // Changing either argument list means changing its signature function with it.
        preparedBwd.partialForStream(stream).launch(stream,
                                                    x,
                                                    rowDesc,
                                                    scale,
                                                    scaleDesc,
                                                    dy,
                                                    rowDesc,
                                                    dy,
                                                    rowDesc,
                                                    invRms,
                                                    invRmsDesc,
                                                    dx,
                                                    rowDesc,
                                                    workspace,
                                                    partialDesc,
                                                    rows);
        preparedBwd.reduceForStream(stream).launch(
            stream, workspace, partialDesc, dscale, scaleDesc);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& flydslRmsNormBwdKpackModuleCache()
{
    // This pack's own; see flydslRmsNormKpackModuleCache().
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetFlydslRmsNormBwdModuleCache()
{
    flydslRmsNormBwdKpackModuleCache().clear();
}

namespace
{

/// Process-lifetime, as the forward pack's handler is.
const FlydslRmsNormBwdDispatchHandler& flydslRmsNormBwdDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslRmsNormBwdKpackModuleCache());
    static const FlydslRmsNormBwdDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslRmsNormBwdSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslRmsNormBwdGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslRmsNormBwdKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslRmsNormBwdScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslRmsNormBwdDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
