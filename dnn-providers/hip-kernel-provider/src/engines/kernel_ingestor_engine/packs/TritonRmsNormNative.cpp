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
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
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

constexpr std::string_view X_TOKEN = "triton_rmsnorm.x.uid";
constexpr std::string_view SCALE_TOKEN = "triton_rmsnorm.scale.uid";
constexpr std::string_view EPSILON_TOKEN = "triton_rmsnorm.epsilon.uid";
constexpr std::string_view Y_TOKEN = "triton_rmsnorm.y.uid";
constexpr std::string_view ROWS_TOKEN = "triton_rmsnorm.rows";
constexpr std::string_view COLUMNS_TOKEN = "triton_rmsnorm.columns";

/// The kernels carry a `tt.divisibility = 16` hint on the base pointers, so every operand
/// base must be 16-byte aligned. The backend enforces only `ptr % alignment() == 0`, so a
/// declared alignment guarantees that iff it is a positive multiple of 16; any other
/// (8, 0, 24) is declined at match time. Launch does not re-check.
constexpr int64_t REQUIRED_ALIGNMENT = 16;

/// Kernarg slots 5 and 6, Triton's global and profile scratch pointers. Always null: the
/// generator rejects any object that needs scratch.
constexpr void* NO_SCRATCH = nullptr;

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

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

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

/// Runs on an unvalidated graph, so it must be total: dims/strides may be null or
/// mismatched, or an extent non-positive. Later helpers may assume positive extents.
bool hasUsableShape(const data_objects::TensorAttributes& tensor, uint32_t minimumRank)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || strides->size() != dims->size()
       || dims->size() < minimumRank)
    {
        return false;
    }

    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        if(dims->Get(axis) <= 0)
        {
            return false;
        }
    }
    return true;
}

/// Dense row-major: innermost stride 1 and every outer stride the product of the extents
/// inside it. This makes collapsing rank > 2 to (rows, N) sound, since the kernel addresses
/// row `r` at `r * N`. Also false when the element count overflows int64, so products of
/// extents after a true result (see rowCount) cannot overflow. Requires positive extents.
bool isPackedRowMajor(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    int64_t expected = 1;
    for(auto axis = static_cast<int64_t>(dims->size()) - 1; axis >= 0; --axis)
    {
        const auto index = static_cast<flatbuffers::uoffset_t>(axis);
        if(strides->Get(index) != expected)
        {
            return false;
        }
        const auto extent = dims->Get(index);
        if(expected > std::numeric_limits<int64_t>::max() / extent)
        {
            return false;
        }
        expected *= extent;
    }
    return true;
}

/// Product of every extent except the innermost: the row count the grid is sized from.
/// Requires a tensor that passed isPackedRowMajor, which bounds the product.
int64_t rowCount(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    int64_t rows = 1;
    for(flatbuffers::uoffset_t axis = 0; axis + 1 < dims->size(); ++axis)
    {
        rows *= dims->Get(axis);
    }
    return rows;
}

int64_t innermostExtent(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    return dims->Get(dims->size() - 1);
}

bool sameShape(const data_objects::TensorAttributes& lhs, const data_objects::TensorAttributes& rhs)
{
    const auto* lhsDims = lhs.dims();
    const auto* rhsDims = rhs.dims();
    if(lhsDims->size() != rhsDims->size())
    {
        return false;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < lhsDims->size(); ++axis)
    {
        if(lhsDims->Get(axis) != rhsDims->Get(axis))
        {
            return false;
        }
    }
    return true;
}

/// A regular device operand: real storage, readable by the kernel through a pointer.
bool isDeviceOperand(const data_objects::TensorAttributes& tensor)
{
    return !tensor.virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor);
}

bool isAlignedForKernel(const data_objects::TensorAttributes& tensor)
{
    return tensor.alignment() > 0 && tensor.alignment() % REQUIRED_ALIGNMENT == 0;
}

/// The dtypes the shipped instances were built for (the UKDs' `dtype` metadata).
bool isSupportedDataType(data_objects::DataType dataType)
{
    return dataType == data_objects::DataType::BFLOAT16 || dataType == data_objects::DataType::HALF;
}

/// The kernel metadata spelling of a graph dtype (`bf16`/`f16`, not the flatbuffer
/// `BFLOAT16`/`HALF`).
std::optional<std::string> kernelDataTypeName(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::BFLOAT16:
        return std::string("bf16");
    case data_objects::DataType::HALF:
        return std::string("f16");
    default:
        return std::nullopt;
    }
}

/// True when the epsilon tensor can be classified as a scalar operand (an unknown dtype
/// cannot). The value is read at execute, never compared here.
bool epsilonIsClassifiable(const MatchContext& context, int64_t uid)
{
    try
    {
        (void)hipdnn_plugin_sdk::makeScalarOperand(context.graph.getTensorMap(), uid, "epsilon");
        return true;
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
    {
        return false;
    }
}

/**
 * @brief Graph-scoped applicability: a single forward-inference RMS-norm node over
 *        packed row-major bf16/f16 operands whose declared alignment is a positive
 *        multiple of 16.
 *
 * Refuses rather than approximates. Any epsilon is accepted: it is a kernarg.
 */
std::optional<BoundTokens> tritonRmsNormGraphMatches(const MatchContext& context)
{
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

    const auto* x = findTensor(context, attributes.x_tensor_uid());
    const auto* scale = findTensor(context, attributes.scale_tensor_uid());
    const auto* y = findTensor(context, attributes.y_tensor_uid());
    if(x == nullptr || scale == nullptr || y == nullptr)
    {
        return std::nullopt;
    }

    if(!isDeviceOperand(*x) || !isDeviceOperand(*scale) || !isDeviceOperand(*y))
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
    if(!isSupportedDataType(x->data_type()) || scale->data_type() != x->data_type()
       || y->data_type() != x->data_type())
    {
        return std::nullopt;
    }

    // x and y need rank >= 2 (a row to reduce over); scale is a vector along that axis.
    if(!hasUsableShape(*x, 2) || !hasUsableShape(*y, 2) || !hasUsableShape(*scale, 1))
    {
        return std::nullopt;
    }

    if(!sameShape(*x, *y) || !isPackedRowMajor(*x) || !isPackedRowMajor(*y)
       || !isPackedRowMajor(*scale))
    {
        return std::nullopt;
    }

    if(!isAlignedForKernel(*x) || !isAlignedForKernel(*scale) || !isAlignedForKernel(*y))
    {
        return std::nullopt;
    }

    const auto rows = rowCount(*x);
    const auto columns = innermostExtent(*x);
    if(rows <= 0 || columns <= 0)
    {
        return std::nullopt;
    }

    // Rows and columns are bounded to int32 (n_cols is a 32-bit kernarg). The grid limit
    // depends on each instance's block width, so tritonRmsNormKernelMatches checks it.
    constexpr auto INT32_MAX_VALUE = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    if(rows > INT32_MAX_VALUE || columns > INT32_MAX_VALUE)
    {
        return std::nullopt;
    }

    // Scale must span exactly the normalised axis; the kernel does no broadcasting.
    if(rowCount(*scale) != 1 || innermostExtent(*scale) != columns)
    {
        return std::nullopt;
    }

    if(!epsilonIsClassifiable(context, attributes.epsilon_tensor_uid()))
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

/**
 * @brief Kernel-scoped applicability: matching dtype, `block_n` >= the normalised axis, and
 *        a grid of one `block_threads`-wide block per row that HIP can launch.
 *
 * Each instance reduces one row with a single `block_n`-wide tile, so it must cover the
 * row. Several instances usually apply; the descriptors' priorities (see
 * tritonRmsNormScore) prefer the smallest. A row wider than the largest `block_n` admits
 * none, and the engine declines. HIP caps the threads per grid dimension at UINT32_MAX, so
 * a row count too large for a wide block may still be served by a narrower one.
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

    const auto* x = findTensor(context, attributes->x_tensor_uid());
    if(x == nullptr)
    {
        return false;
    }

    const auto dtype = kernelDataTypeName(x->data_type());
    if(!dtype.has_value() || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dtype)
    {
        return false;
    }

    const auto columns = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, COLUMNS_TOKEN);
    if(!columns.has_value() || kernel.getIntMetadata(std::string(BLOCK_N_FIELD)) < *columns)
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
    return blockThreads > 0 && *rows <= UINT32_MAX_VALUE / blockThreads;
}

/**
 * @brief Ranks the survivors by descriptor `priority`; the smallest `block_n` carries the
 *        highest.
 */
double tritonRmsNormScore(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
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

/// The loaded kernel plus everything read from the graph, resolved once at prepare().
class PreparedTritonRmsNorm : public PreparedDispatch
{
public:
    PreparedTritonRmsNorm(IngestorKernelCode code,
                          TritonRmsNormBinding binding,
                          hipdnn_plugin_sdk::ScalarOperand epsilon,
                          int32_t columns)
        : _code(std::move(code))
        , _binding(binding)
        , _epsilon(epsilon)
        , _columns(columns)
    {
    }

    /// The kernel for the stream's device; resolved per launch because a plan outlives the
    /// handle it was built from.
    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const TritonRmsNormBinding& binding() const
    {
        return _binding;
    }

    const hipdnn_plugin_sdk::ScalarOperand& epsilon() const
    {
        return _epsilon;
    }

    int32_t columns() const
    {
        return _columns;
    }

private:
    // Owns each device's program alongside the kernel viewing into it, so the module
    // outlives every function resolved from it.
    IngestorKernelCode _code;
    TritonRmsNormBinding _binding;
    hipdnn_plugin_sdk::ScalarOperand _epsilon;
    int32_t _columns;
};

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
            binding,
            hipdnn_plugin_sdk::makeScalarOperand(
                context.graph.getTensorMap(), binding.epsilon, "epsilon"),
            static_cast<int32_t>(columns));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedRmsNorm = dynamic_cast<const PreparedTritonRmsNorm&>(prepared);
        const auto& binding = preparedRmsNorm.binding();

        // A baked graph constant returns its value; a runtime pass-by-value operand reads
        // the host scalar from the execute buffers.
        const auto epsilon = static_cast<float>(
            hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
                preparedRmsNorm.epsilon(), deviceBuffers, numDeviceBuffers)));

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto scale
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.scale, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        // Lvalues, because launch() takes each argument's address. Keep in step with
        // tritonRmsNormKernelSignature().
        const int32_t nCols = preparedRmsNorm.columns();
        void* globalScratch = NO_SCRATCH;
        void* profileScratch = NO_SCRATCH;

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
    // Process-lifetime and per pack: the key is (archive, toc_key, arch), so a shared cache
    // would answer other packs' lookups too.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetTritonRmsNormModuleCache()
{
    tritonRmsNormKpackModuleCache().clear();
}

namespace
{

/// Process-lifetime: the registry holds a non-owning pointer, and Containers are created
/// and destroyed per handle, so the handler and its loader must outlive every Container.
const TritonRmsNormDispatchHandler& tritonRmsNormDispatchHandler()
{
    static const compilation::KpackKernelLoader s_kpackLoader(tritonRmsNormKpackModuleCache());
    static const TritonRmsNormDispatchHandler s_dispatchHandler(s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerTritonRmsNormSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &tritonRmsNormGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &tritonRmsNormKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &tritonRmsNormScore);
    scope.add(std::string(DISPATCH_SYMBOL), &tritonRmsNormDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_TRITON
