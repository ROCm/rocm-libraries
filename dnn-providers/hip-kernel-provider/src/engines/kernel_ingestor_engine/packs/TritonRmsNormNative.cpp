// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: if this file lands on an ungated source list
// it compiles to an empty translation unit instead of a pack naming an unstaged archive.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_TRITON)

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/TritonPackCommon.hpp"

/**
 * @file TritonRmsNormNative.cpp
 * @brief Native half of the Triton AOT RMS-norm pack: graph and kernel matching, scoring,
 *        dispatch, and the function that registers them.
 *
 * Adapted from dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/packs/FlydslRmsNormNative.cpp at commit 03911535bd0 (FlyDSL provider)
 *
 * Every kernel is a prebuilt code object compiled offline by `triton/`, shipped as an
 * `hsaco` UKD in the production descriptor root and packed into the per-arch
 * `hip_kernel_provider_<arch>.kpack`. The matchers decline every graph those objects were
 * not built for.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The descriptor files restate these strings: a descriptor cannot reference a C++ constant.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.triton_rmsnorm.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.triton_rmsnorm.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.triton_rmsnorm.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.triton_rmsnorm.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view BLOCK_N_FIELD = "block_n";
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";
constexpr std::string_view SHARED_MEM_BYTES_FIELD = "shared_mem_bytes";
constexpr std::string_view ALIGNMENT_FIELD = "alignment";

constexpr std::string_view X_TOKEN = "triton_rmsnorm.x.uid";
constexpr std::string_view SCALE_TOKEN = "triton_rmsnorm.scale.uid";
constexpr std::string_view EPSILON_TOKEN = "triton_rmsnorm.epsilon.uid";
constexpr std::string_view Y_TOKEN = "triton_rmsnorm.y.uid";
constexpr std::string_view ROWS_TOKEN = "triton_rmsnorm.rows";
constexpr std::string_view COLUMNS_TOKEN = "triton_rmsnorm.columns";

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// What a matched RMS-norm graph binds: the tensor uids in argument order, then the row
/// count (the grid) and the normalised width (the `n_cols` kernarg).
struct TritonRmsNormBinding
{
    int64_t x = 0;
    int64_t scale = 0;
    int64_t epsilon = 0;
    int64_t y = 0;
    int64_t rows = 0;
    int64_t columns = 0;
};

/// The node this pack's matchers read, or nullptr if the graph isn't a single RMS-norm
/// node. `RMSNormAttributes` is its own union arm, so matching it is the whole node-type
/// test: there is no mode enum to disambiguate it from layer norm.
const data_objects::RMSNormAttributes* rmsNormNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::RMSNormAttributes)
    {
        return nullptr;
    }

    return &node.attributesAs<data_objects::RMSNormAttributes>();
}

/**
 * @brief Graph-scoped applicability: a single forward-inference RMS-norm node over
 *        packed row-major operands of one dtype, with fixed shapes.
 *
 * Refuses rather than approximates. Any epsilon is accepted: it is a kernarg. Which
 * dtypes and alignments a kernel serves is its own metadata, checked by
 * tritonRmsNormKernelMatches.
 */
std::optional<BoundTokens> tritonRmsNormGraphMatches(const MatchContext& context)
{
    // The grid and n_cols come from the graph's shapes; launch() takes none at execute.
    if(context.graph.getGraph().is_override_shape_enabled())
    {
        return std::nullopt;
    }

    // Exactly one node: each kernel serves one complete graph.
    const auto* attributesPtr = rmsNormNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // The kernel computes `x * rstd * scale` only: no bias, no saved reciprocal-RMS. A
    // TRAINING graph is refused on the phase even if it names no stat tensor.
    if(attributes.bias_tensor_uid().has_value() || attributes.inv_rms_tensor_uid().has_value()
       || attributes.forward_phase() == data_objects::NormFwdPhase::TRAINING)
    {
        return std::nullopt;
    }

    const auto* x = triton::findTensor(context, attributes.x_tensor_uid());
    const auto* scale = triton::findTensor(context, attributes.scale_tensor_uid());
    const auto* y = triton::findTensor(context, attributes.y_tensor_uid());
    if(x == nullptr || scale == nullptr || y == nullptr)
    {
        return std::nullopt;
    }

    if(!triton::isDeviceOperand(*x) || !triton::isDeviceOperand(*scale)
       || !triton::isDeviceOperand(*y))
    {
        return std::nullopt;
    }

    // The kernel addresses every row at `r * N`; a ragged offset tensor cannot be honoured.
    if(x->ragged_offset_tensor_uid().has_value() || scale->ragged_offset_tensor_uid().has_value()
       || y->ragged_offset_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    // One dtype for all three: the kernel loads scale and writes y in x's element type.
    if(scale->data_type() != x->data_type() || y->data_type() != x->data_type())
    {
        return std::nullopt;
    }

    // x and y need rank >= 2 (a row to reduce over); scale is a vector along that axis.
    if(!triton::hasUsableShape(*x, 2) || !triton::hasUsableShape(*y, 2)
       || !triton::hasUsableShape(*scale, 1))
    {
        return std::nullopt;
    }

    if(!triton::sameShape(*x, *y) || !triton::isPackedRowMajor(*x) || !triton::isPackedRowMajor(*y)
       || !triton::isPackedRowMajor(*scale))
    {
        return std::nullopt;
    }

    // Both positive: every extent passed hasUsableShape.
    const auto rows = triton::rowCount(*x);
    const auto columns = triton::innermostExtent(*x);

    // Rows and columns are bounded to int32 (n_cols is a 32-bit kernarg). The grid limit
    // depends on each instance's block width, so tritonRmsNormKernelMatches checks it.
    constexpr auto INT32_MAX_VALUE = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    if(rows > INT32_MAX_VALUE || columns > INT32_MAX_VALUE)
    {
        return std::nullopt;
    }

    // Scale must span exactly the normalised axis; the kernel does no broadcasting.
    if(triton::rowCount(*scale) != 1 || triton::innermostExtent(*scale) != columns)
    {
        return std::nullopt;
    }

    if(!triton::scalarIsClassifiable(context, attributes.epsilon_tensor_uid(), "epsilon"))
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(X_TOKEN)] = attributes.x_tensor_uid();
    bound[std::string(SCALE_TOKEN)] = attributes.scale_tensor_uid();
    bound[std::string(EPSILON_TOKEN)] = attributes.epsilon_tensor_uid();
    bound[std::string(Y_TOKEN)] = attributes.y_tensor_uid();
    bound[std::string(ROWS_TOKEN)] = rows;
    bound[std::string(COLUMNS_TOKEN)] = columns;
    return bound;
}

/// The backend checks only `ptr % alignment() == 0`, so a declared alignment guarantees a
/// kernel's base-pointer divisibility iff it is a positive multiple of it; any other
/// (0, 24 for a 16-byte kernel) is declined. Launch does not re-check.
bool isAlignedFor(const data_objects::TensorAttributes& tensor, int64_t kernelAlignment)
{
    return tensor.alignment() > 0 && tensor.alignment() % kernelAlignment == 0;
}

/**
 * @brief Kernel-scoped applicability: the kernel's dtype, every operand's declared
 *        alignment a positive multiple of the kernel's, `block_n` >= the normalised axis,
 *        and a grid of one `block_threads`-wide block per row that HIP can launch.
 *
 * Each instance reduces one row with a single `block_n`-wide tile, so it must cover the
 * row. Several instances usually apply; the descriptors' priorities prefer the smallest.
 * A row wider than the largest `block_n` admits none, and the engine declines. HIP caps
 * the threads per grid dimension at UINT32_MAX, so a row count too large for a wide block
 * may still be served by a narrower one.
 */
bool tritonRmsNormKernelMatches(const MatchContext& context,
                                const BoundTokens& bound,
                                const KernelDefinition& kernel)
{
    const auto* attributes = rmsNormNode(context);
    if(attributes == nullptr)
    {
        return false;
    }

    const auto* x = triton::findTensor(context, attributes->x_tensor_uid());
    const auto* scale = triton::findTensor(context, attributes->scale_tensor_uid());
    const auto* y = triton::findTensor(context, attributes->y_tensor_uid());
    if(x == nullptr || scale == nullptr || y == nullptr)
    {
        return false;
    }

    // A graph dtype no kernel is built for has no spelling, and equals no kernel's.
    const auto dtype = triton::kernelDataTypeName(x->data_type());
    if(kernel.getStringMetadata(std::string(DTYPE_FIELD)) != dtype)
    {
        return false;
    }

    // The kernel's base-pointer divisibility hint, which every operand must honour. A
    // kernel stating none, or none positive, admits nothing.
    const auto alignmentField = kernel.tryGetMetadata(std::string(ALIGNMENT_FIELD));
    const auto* alignment
        = alignmentField.has_value() ? std::get_if<int64_t>(&alignmentField.value()) : nullptr;
    if(alignment == nullptr || *alignment <= 0)
    {
        return false;
    }
    if(!isAlignedFor(*x, *alignment) || !isAlignedFor(*scale, *alignment)
       || !isAlignedFor(*y, *alignment))
    {
        return false;
    }

    const auto columns = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, COLUMNS_TOKEN);
    if(!columns.has_value() || kernel.getIntMetadata(std::string(BLOCK_N_FIELD)) < columns.value())
    {
        return false;
    }

    const auto rows = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, ROWS_TOKEN);
    if(!rows.has_value())
    {
        return false;
    }

    // Division rather than rows * block_threads, which could overflow.
    constexpr auto UINT32_MAX_VALUE = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    const auto blockThreads = kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD));
    return blockThreads > 0 && rows.value() <= UINT32_MAX_VALUE / blockThreads;
}

/**
 * @brief Re-reads the operand bindings a match established.
 *
 * @throws HipdnnPluginException if a token is missing or does not hold an integer.
 */
TritonRmsNormBinding tritonRmsNormBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "triton_rmsnorm dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };

    return {read(X_TOKEN),
            read(SCALE_TOKEN),
            read(EPSILON_TOKEN),
            read(Y_TOKEN),
            read(ROWS_TOKEN),
            read(COLUMNS_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Everything launch() reads from the graph, resolved once at prepare().
struct TritonRmsNormLaunch
{
    TritonRmsNormBinding binding;
    hipdnn_plugin_sdk::ScalarOperand epsilon;
    int32_t columns;
};

using PreparedTritonRmsNorm = triton::PreparedTritonDispatch<TritonRmsNormLaunch>;

/**
 * @brief The explicit kernarg list, mirroring launch() below one for one.
 *
 * Slots: x, scale, y, n_cols, eps, global_scratch, profile_scratch (7 slots, 48 bytes).
 * Hidden arguments the runtime fills after slot 6 are not listed.
 *
 * Names are empty because requireSignatureMatch compares them only when both sides carry
 * one; kind, size and slot count are what is checked.
 */
const std::vector<KernelArgument>& tritonRmsNormKernelSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"global_buffer", 8, 0, ""},
        {"global_buffer", 8, 8, ""},
        {"global_buffer", 8, 16, ""},
        {"by_value", 4, 24, ""},
        {"by_value", 4, 28, ""},
        {"global_buffer", 8, 32, ""},
        {"global_buffer", 8, 40, ""},
    };
    return s_signature;
}

/**
 * @brief The native dispatch behind this pack's UDD. Everything graph/kernel-derived
 *        resolves once at prepare(); launch() only resolves buffers and launches, so
 *        concurrent execution is safe.
 */
class TritonRmsNormDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kpackLoader Must outlive this handler; process-lifetime.
    explicit TritonRmsNormDispatchHandler(const compilation::KpackKernelLoader& kpackLoader)
        : _kpackLoader(kpackLoader)
    {
    }

    /// One program per row, reducing in registers and LDS: no global scratch.
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
        const auto binding = tritonRmsNormBinding(bound);

        // The graph matcher bounded both to 32 bits; the kernel matcher bounded the grid.
        const auto rows = binding.rows;
        const auto columns = binding.columns;

        const auto blockThreads
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD)));
        const auto sharedMemBytes
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(SHARED_MEM_BYTES_FIELD)));

        auto code = buildIngestorKernelCode(
            _kpackLoader, context, kernel, tritonRmsNormKernelSignature());

        // One program per row: grid = row count, block = the instance's own width. Triton's
        // LDS is entirely dynamic, so the launch must pass the object's `metadata.shared`;
        // launching with 0 runs but silently computes wrong values.
        code.setBlockSize(blockThreads, 1, 1);
        code.setGridSize(static_cast<unsigned int>(rows), 1, 1);
        code.setSharedMemBytes(sharedMemBytes);

        return std::make_unique<PreparedTritonRmsNorm>(
            std::move(code),
            TritonRmsNormLaunch{binding,
                                hipdnn_plugin_sdk::makeScalarOperand(
                                    context.graph.getTensorMap(), binding.epsilon, "epsilon"),
                                static_cast<int32_t>(columns)});
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedRmsNorm = dynamic_cast<const PreparedTritonRmsNorm&>(prepared);
        const auto& state = preparedRmsNorm.state();
        const auto& binding = state.binding;

        // A baked graph constant returns its value; a runtime pass-by-value operand reads
        // the host scalar from the execute buffers.
        const auto epsilon = static_cast<float>(
            hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
                state.epsilon, deviceBuffers, numDeviceBuffers)));

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto scale
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.scale, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        // Lvalues, because launch() takes each argument's address. Keep in step with
        // tritonRmsNormKernelSignature().
        const int32_t nCols = state.columns;
        void* globalScratch = triton::NO_SCRATCH;
        void* profileScratch = triton::NO_SCRATCH;

        preparedRmsNorm.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    x.ptr,
                    scale.ptr,
                    y.ptr,
                    nCols,
                    epsilon,
                    globalScratch,
                    profileScratch);
    }

private:
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& tritonRmsNormKpackModuleCache()
{
    return triton::kpackModuleCache<TritonRmsNormDispatchHandler>();
}

void resetTritonRmsNormModuleCache()
{
    tritonRmsNormKpackModuleCache().clear();
}

void registerTritonRmsNormSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &tritonRmsNormGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &tritonRmsNormKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &triton::priorityScore);
    scope.add(std::string(DISPATCH_SYMBOL),
              &triton::processDispatchHandler<TritonRmsNormDispatchHandler>());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_TRITON
