// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a pack that
// names a kpack archive nothing staged.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cmath>
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
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
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
 * @file FlydslRmsNormNative.cpp
 * @brief The flyDSL RMS-norm pack's native half: matching, scoring, dispatch, and the one
 *        function that registers them.
 *
 * Unlike the other packs in this directory there is no embedded-source path: every kernel
 * this pack can select is a pre-built HSACO, produced ahead of time by the flyDSL
 * compiler (see `flydsl/REGEN.md`), checked in as an `hsaco` UKD of the production
 * descriptor root and packed into the per-arch `hip_kernel_provider_<arch>.kpack` with
 * every other producer's objects. The consequence is that this file cannot change what
 * the kernel does -- it can only decline graphs the kernel was not built for. Two facts
 * were baked at pack time and are therefore matcher obligations here:
 *
 *   - the element types of x/y and of gamma, so a graph is matched to an object built for
 *     exactly its dtypes;
 *   - the innermost stride is a literal 1, absent from the descriptor structs below, so
 *     only packed row-major operands may be accepted.
 *
 * Epsilon is not baked: it is the kernel's trailing argument (vendored modification 4),
 * so one object serves every epsilon a graph asks for, fixed in the graph or supplied at
 * execute.
 *
 * Symbol names are restated here rather than shared via a header, since a descriptor can't
 * reference a C++ constant; the loader pre-flights every symbol a descriptor names, so a
 * mismatched string is caught without becoming a compile error.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using namespace flydsl_norm;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_rmsnorm.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_rmsnorm.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_rmsnorm.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_rmsnorm.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view N_FIELD = "N";
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";
/// 1 when the object reads gamma as f32 beside bf16/f16 x and y, else 0 (gamma in x's type).
constexpr std::string_view SCALE_F32_FIELD = "scale_f32";
/// 1 when the object also writes inv_rms (the training forward), else 0.
constexpr std::string_view STORE_RSTD_FIELD = "store_rstd";

constexpr std::string_view X_TOKEN = "flydsl_rmsnorm.x.uid";
constexpr std::string_view SCALE_TOKEN = "flydsl_rmsnorm.scale.uid";
constexpr std::string_view EPSILON_TOKEN = "flydsl_rmsnorm.epsilon.uid";
constexpr std::string_view Y_TOKEN = "flydsl_rmsnorm.y.uid";
/// Bound only for a training graph, which saves inv_rms.
constexpr std::string_view INV_RMS_TOKEN = "flydsl_rmsnorm.inv_rms.uid";
constexpr std::string_view ROWS_TOKEN = "flydsl_rmsnorm.rows";
constexpr std::string_view COLUMNS_TOKEN = "flydsl_rmsnorm.columns";

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched RMS-norm graph binds, in argument order.
struct FlydslRmsNormBinding
{
    int64_t x = 0;
    int64_t scale = 0;
    int64_t epsilon = 0;
    int64_t y = 0;
    /// The training forward's saved statistic; absent for inference.
    std::optional<int64_t> invRms;
};

/// The node this pack's matchers read, or nullptr if the graph isn't a single RMS-norm
/// node. `RMSNormAttributes` is its own union arm, so matching it is the whole node-type
/// test: there is no mode enum to disambiguate it from layer norm.
const data_objects::RMSNormAttributes* rmsNormNode(const MatchContext& context)
{
    // The engine declares graph schema 1.2.0 so a runtime pass-by-value epsilon reaches
    // it, which also admits override-shape graphs (1.1.0). Rows and width are bound from
    // the graph's shapes at prepare, so those are declined.
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
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

/// The epsilon operand, or nullopt when it cannot be classified at all -- an unknown dtype,
/// say -- in which case the graph is simply not ours.
std::optional<hipdnn_plugin_sdk::ScalarOperand> epsilonOperand(const MatchContext& context,
                                                               int64_t uid)
{
    try
    {
        return hipdnn_plugin_sdk::makeScalarOperand(context.graph.getTensorMap(), uid, "epsilon");
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
    {
        return std::nullopt;
    }
}

/// An epsilon the kernel computes with: a finite, non-negative f32. Anything else would
/// make every output NaN or infinite rather than a norm.
bool isUsableEpsilon(double epsilon)
{
    return std::isfinite(epsilon) && epsilon >= 0.0
           && epsilon <= static_cast<double>(std::numeric_limits<float>::max());
}

/**
 * @brief Graph-scoped applicability: is this a single RMS-norm node over packed 2-D-
 *        collapsible operands that the pre-built flyDSL kernels actually compute?
 *
 * Refuses rather than approximates. Epsilon is a kernel argument, so any value is served; a
 * baked one that is not a usable f32 is declined here, and one supplied at execute is
 * checked by `launch()`.
 */
std::optional<BoundTokens> flydslRmsNormGraphMatches(const MatchContext& context)
{
    // Exactly one node: this pack's kernels each serve one complete graph.
    const auto* attributesPtr = rmsNormNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // No bias term: no instance adds anything after scaling. inv_rms goes with TRAINING
    // and only with it: a training graph without the tensor still wants the statistic, and
    // one that names it outside training asks for an output cuDNN's semantics do not
    // produce, so both decline. The phase is an allow-list, so a value added to the schema
    // later declines rather than being taken for inference. The kernels accumulate in f32:
    // at least the precision a FLOAT, HALF or BFLOAT16 compute type asks for; DOUBLE or any
    // other type declines.
    const auto phase = attributes.forward_phase();
    const bool training = phase == data_objects::NormFwdPhase::TRAINING;
    if(attributes.bias_tensor_uid().has_value()
       || attributes.inv_rms_tensor_uid().has_value() != training
       || (!training && phase != data_objects::NormFwdPhase::NOT_SET
           && phase != data_objects::NormFwdPhase::INFERENCE))
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
    // The fast path copies 128-bit vectors from each operand's base, and the backend enforces
    // only the alignment a graph declares.
    if(!declaresVectorAlignment(*x) || !declaresVectorAlignment(*scale)
       || !declaresVectorAlignment(*y))
    {
        return std::nullopt;
    }

    // x and y in one dtype; gamma in that dtype too, or f32 beside bf16/f16 x (cuDNN's and
    // PyTorch's usual mixed form), which the kernel reads with its own element type.
    const bool scaleMatchesX = scale->data_type() == x->data_type()
                               || (scale->data_type() == data_objects::DataType::FLOAT
                                   && x->data_type() != data_objects::DataType::FLOAT);
    if(!isSupportedDataType(x->data_type()) || !scaleMatchesX || y->data_type() != x->data_type())
    {
        return std::nullopt;
    }

    // x and y are the normalised operands (rank >= 2 so there is a row to reduce over);
    // gamma is one vector along the normalised axis.
    if(!hasUsableShape(*x, 2) || !hasUsableShape(*y, 2) || !hasUsableShape(*scale, 1))
    {
        return std::nullopt;
    }

    if(!sameShape(*x, *y) || !isPackedRowMajor(*x) || !isPackedRowMajor(*y)
       || !isPackedRowMajor(*scale))
    {
        return std::nullopt;
    }

    // Gamma must cover exactly the normalised axes: any other shape would be a broadcast
    // the kernel does not perform.
    const auto split = normalisedSplit(*x, *scale);
    if(!split.has_value())
    {
        return std::nullopt;
    }
    const auto rows = split->rows;
    const auto columns = split->columns;
    // Every element of x and y within the 32-bit byte offsets of one buffer view, which
    // also keeps M and N within the kernel's int32 arguments.
    const auto bytes = elementBytes(x->data_type());
    if(columns > MAX_OPERAND_BYTES / bytes || rows > MAX_OPERAND_BYTES / (bytes * columns))
    {
        return std::nullopt;
    }

    // Epsilon is one scalar: a tensor of more than one element would be read as its first.
    const auto* epsilonTensor = findTensor(context, attributes.epsilon_tensor_uid());
    if(epsilonTensor == nullptr || epsilonTensor->dims() == nullptr)
    {
        return std::nullopt;
    }
    for(const auto extent : *epsilonTensor->dims())
    {
        if(extent != 1)
        {
            return std::nullopt;
        }
    }
    const auto epsilon = epsilonOperand(context, attributes.epsilon_tensor_uid());
    if(!epsilon.has_value())
    {
        return std::nullopt;
    }
    if(!epsilon->isRuntimeUserSupplied
       && !isUsableEpsilon(hipdnn_plugin_sdk::toDouble(epsilon->bakedDefault)))
    {
        return std::nullopt;
    }

    if(training)
    {
        const auto* invRms = findTensor(context, *attributes.inv_rms_tensor_uid());
        if(invRms == nullptr || !isDeviceOperand(*invRms) || !hasUsableShape(*invRms, 2)
           || !isPerRowStatistic(*invRms, *x, *split))
        {
            return std::nullopt;
        }
    }

    BoundTokens bound;
    if(training)
    {
        bound[std::string(INV_RMS_TOKEN)] = *attributes.inv_rms_tensor_uid();
    }
    bound[std::string(X_TOKEN)] = attributes.x_tensor_uid();
    bound[std::string(SCALE_TOKEN)] = attributes.scale_tensor_uid();
    bound[std::string(EPSILON_TOKEN)] = attributes.epsilon_tensor_uid();
    bound[std::string(Y_TOKEN)] = attributes.y_tensor_uid();
    bound[std::string(ROWS_TOKEN)] = rows;
    bound[std::string(COLUMNS_TOKEN)] = columns;
    return bound;
}

/**
 * @brief Kernel-scoped applicability: dtype, and the normalised width this kernel was built
 *        for. Evaluated once per candidate kernel.
 *
 * Two tiers reach this function. A specialized kernel names its width, and is admitted only
 * for that width; a generic kernel carries the `N == 0` sentinel and takes the width as a
 * kernarg, so it is admitted for any. Both tiers stay in the candidate set when they both
 * apply, and `flydslRmsNormScore` is what prefers the specialized one.
 */
bool flydslRmsNormKernelMatches(const MatchContext& context,
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

    const auto* scale = findTensor(context, attributes->scale_tensor_uid());
    if(scale == nullptr)
    {
        return false;
    }

    const auto dtype = kernelDataTypeName(x->data_type());
    if(!dtype.has_value() || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dtype)
    {
        return false;
    }

    // An f32 gamma beside a 16-bit x needs an object built to read it as f32, and a gamma
    // in x's own type needs one that is not.
    const int64_t scaleF32 = scale->data_type() != x->data_type() ? 1 : 0;
    if(kernel.getIntMetadata(std::string(SCALE_F32_FIELD)) != scaleF32)
    {
        return false;
    }

    // A training graph needs an object that writes inv_rms; an inference one must not get
    // one, which would store through the gamma pointer launch() fills the slots with.
    const int64_t storeRstd = attributes->inv_rms_tensor_uid().has_value() ? 1 : 0;
    if(kernel.getIntMetadata(std::string(STORE_RSTD_FIELD)) != storeRstd)
    {
        return false;
    }

    const auto bakedN = kernel.getIntMetadata(std::string(N_FIELD));
    if(bakedN == GENERIC_N_SENTINEL)
    {
        return true;
    }

    const auto columns = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, COLUMNS_TOKEN);
    return columns.has_value() && *columns == bakedN;
}

/**
 * @brief Ranks the survivors. The descriptors already encode the intended order -- a
 *        specialized instance is `priority: 100`, the generic fallback `priority: 10` --
 *        so this reproduces that rather than inventing a second opinion. A measured cost
 *        model replaces it when there is data to fit one.
 */
double flydslRmsNormScore(const MatchContext& /*context*/,
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
FlydslRmsNormBinding flydslRmsNormBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_rmsnorm dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };

    return {read(X_TOKEN),
            read(SCALE_TOKEN),
            read(EPSILON_TOKEN),
            read(Y_TOKEN),
            hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, INV_RMS_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The compiled kernel plus everything read from the graph, resolved once, owning nothing
/// that points back into it.
class PreparedFlydslRmsNorm : public PreparedDispatch
{
public:
    PreparedFlydslRmsNorm(IngestorKernelCode code,
                          FlydslRmsNormBinding binding,
                          hipdnn_plugin_sdk::ScalarOperand epsilon,
                          int32_t rows,
                          int32_t columns)
        : _code(std::move(code))
        , _binding(binding)
        , _epsilon(epsilon)
        , _rows(rows)
        , _columns(columns)
    {
    }

    /// The kernel for the device this dispatch is running on. Resolved here rather than at
    /// prepare() because a plan outlives the handle it was built from.
    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslRmsNormBinding& binding() const
    {
        return _binding;
    }

    const hipdnn_plugin_sdk::ScalarOperand& epsilon() const
    {
        return _epsilon;
    }

    int32_t rows() const
    {
        return _rows;
    }

    int32_t columns() const
    {
        return _columns;
    }

private:
    // Owns each device's program alongside the kernel viewing into it, so a module outlives
    // every function resolved from it for the plan's lifetime.
    IngestorKernelCode _code;
    FlydslRmsNormBinding _binding;
    hipdnn_plugin_sdk::ScalarOperand _epsilon;
    int32_t _rows;
    int32_t _columns;
};

/**
 * @brief The argument list this pack launches with, mirroring the launch() below one for
 *        one, and the `signature` block every one of this pack's UKDs carries.
 *
 * Four operands, each a pointer followed by its descriptor struct: x (2-D), gamma (1-D),
 * rstd (1-D), y (2-D); then epsilon, an f32. A training object (`store_rstd: 1`) writes
 * inv_rms through the rstd pair; an inference one never touches it, so launch() fills it
 * with gamma rather than leaving it undefined.
 *
 * Names are empty and offsets zero because neither is compared for this producer; see
 * requireSignatureMatch.
 */
const std::vector<KernelArgument>& flydslRmsNormKernelSignature()
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_desc2d{"by_value", static_cast<uint32_t>(sizeof(Desc2D)), 0, ""};
    static const KernelArgument s_desc1d{"by_value", static_cast<uint32_t>(sizeof(Desc1D)), 0, ""};
    static const KernelArgument s_epsilon{"by_value", static_cast<uint32_t>(sizeof(float)), 0, ""};
    static const std::vector<KernelArgument> s_signature{
        s_buffer, s_desc2d, s_buffer, s_desc1d, s_buffer, s_desc1d, s_buffer, s_desc2d, s_epsilon};
    return s_signature;
}

/**
 * @brief The native dispatch behind this pack's UDD: sizes and launches a flyDSL RMS-norm
 *        kernel. Everything graph/kernel-derived resolves once at prepare(); execute() only
 *        resolves buffers and launches, so nothing mutates once prepared and concurrent
 *        execution is safe.
 */
class FlydslRmsNormDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime. Unused in
    /// practice -- every kernel this pack selects comes from a kpack archive -- but
    /// buildIngestorKernelCode decides that from the kernel's source kind, not from here.
    /// @param kpackLoader Same must-outlive contract.
    FlydslRmsNormDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                 const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// One block per row, reducing entirely in registers and LDS: no global scratch.
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
        const auto binding = flydslRmsNormBinding(bound);
        const auto& xTensor = requireTensor(context, binding.x);
        const auto& scaleTensor = requireTensor(context, binding.scale);

        // Shapes were validated packed, rank >= 2 and splittable by the graph matcher.
        const auto split = normalisedSplit(xTensor, scaleTensor);
        if(!split.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_rmsnorm: prepare() given a graph its matcher would decline");
        }
        const auto rows = split->rows;
        const auto columns = split->columns;

        const auto blockThreads
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD)));

        // Build defines are ignored on the kpack path -- they were baked when the archive
        // was produced. This calls the compile-capable overload, which takes an options
        // object; a prebuilt-only overload without a compiler or options also exists
        // (IngestorKernelCode.hpp) and would serve these kpack-only kernels equally.
        //
        // Built from the arch alone, without x. The tensor overload derives the dtype and
        // layout defines, and deriving the layout means classifying stride order as
        // channels-first or channels-last, which throws outright below rank 4. RMS-norm
        // operands are rank-2 rows with no channel axis, so there is no answer to give --
        // and no kernel here would read one.
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);

        auto code = buildIngestorKernelCode(_kernelCompiler,
                                            _kpackLoader,
                                            context,
                                            kernel,
                                            options,
                                            flydslRmsNormKernelSignature());

        // `one_block_per_row`: the grid is the row count, the block the kernel's own width.
        code.setBlockSize(blockThreads, 1, 1);
        code.setGridSize(static_cast<unsigned int>(rows), 1, 1);

        return std::make_unique<PreparedFlydslRmsNorm>(
            std::move(code),
            binding,
            hipdnn_plugin_sdk::makeScalarOperand(
                context.graph.getTensorMap(), binding.epsilon, "epsilon"),
            static_cast<int32_t>(rows),
            static_cast<int32_t>(columns));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedRmsNorm = dynamic_cast<const PreparedFlydslRmsNorm&>(prepared);
        const auto& binding = preparedRmsNorm.binding();

        // Baked or supplied now, epsilon reaches the kernel as its last argument. A baked one
        // was screened by the matcher; one supplied at execute can only be checked here.
        const auto epsilon = hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
            preparedRmsNorm.epsilon(), deviceBuffers, numDeviceBuffers));
        if(!isUsableEpsilon(epsilon))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "flydsl_rmsnorm needs a finite, non-negative f32 epsilon; this execution "
                "supplied "
                    + std::to_string(epsilon));
        }

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto scale
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.scale, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        // Packed row-major is an applicability condition, so the row stride is the width.
        const Desc2D xDesc{preparedRmsNorm.rows(),
                           preparedRmsNorm.columns(),
                           static_cast<int64_t>(preparedRmsNorm.columns())};
        const Desc2D yDesc = xDesc;
        const Desc1D scaleDesc{preparedRmsNorm.columns()};

        // A training object writes one f32 per row through the rstd slots. An inference
        // object never stores through them, so they are given gamma's pointer and descriptor
        // and the kernarg segment holds a valid mapping rather than a null. Changing this
        // argument list means changing flydslRmsNormKernelSignature() with it.
        void* rstdPointer = scale.ptr;
        Desc1D rstdDesc = scaleDesc;
        if(binding.invRms.has_value())
        {
            rstdPointer = hipdnn_plugin_sdk::findDeviceBuffer(
                              *binding.invRms, deviceBuffers, numDeviceBuffers)
                              .ptr;
            rstdDesc = Desc1D{preparedRmsNorm.rows()};
        }
        preparedRmsNorm.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    x.ptr,
                    xDesc,
                    scale.ptr,
                    scaleDesc,
                    rstdPointer,
                    rstdDesc,
                    y.ptr,
                    yDesc,
                    static_cast<float>(epsilon));
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& flydslRmsNormKpackModuleCache()
{
    // Process-lifetime, and this pack's own: a cache key is (archive, toc_key, arch), so a
    // cache shared with another pack would answer that pack's lookups too.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetFlydslRmsNormModuleCache()
{
    flydslRmsNormKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer
/// to it, but a provider's Container is created and destroyed per handle, so it (and the
/// compiler and loader it holds) must outlive every Container.
const FlydslRmsNormDispatchHandler& flydslRmsNormDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslRmsNormKpackModuleCache());
    static const FlydslRmsNormDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslRmsNormSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslRmsNormGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslRmsNormKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslRmsNormScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslRmsNormDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
