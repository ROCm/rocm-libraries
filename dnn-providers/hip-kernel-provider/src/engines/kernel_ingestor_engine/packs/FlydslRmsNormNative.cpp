// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a pack that
// names a kpack archive nothing staged.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cmath>
#include <cstddef>
#include <cstdint>
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
 *   - epsilon is a literal in the kernel body (`EPS = 1e-5`), not a kernarg, so a graph
 *     asking for a different epsilon must be refused rather than silently computed with
 *     the wrong constant;
 *   - the innermost stride is a literal 1, absent from the descriptor structs below, so
 *     only packed row-major operands may be accepted.
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

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_rmsnorm.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_rmsnorm.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_rmsnorm.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_rmsnorm.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view N_FIELD = "N";
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";

constexpr std::string_view X_TOKEN = "flydsl_rmsnorm.x.uid";
constexpr std::string_view SCALE_TOKEN = "flydsl_rmsnorm.scale.uid";
constexpr std::string_view EPSILON_TOKEN = "flydsl_rmsnorm.epsilon.uid";
constexpr std::string_view Y_TOKEN = "flydsl_rmsnorm.y.uid";
constexpr std::string_view ROWS_TOKEN = "flydsl_rmsnorm.rows";
constexpr std::string_view COLUMNS_TOKEN = "flydsl_rmsnorm.columns";

/// `N: 0` in a UKD's metadata means "this kernel takes N at runtime". A real graph never
/// has a zero-width normalised axis, so the value is free to carry the sentinel; see
/// `GENERIC_N` in `flydsl/gen_descriptors.py` for why null could not be used instead.
constexpr int64_t GENERIC_N_SENTINEL = 0;

/// `EPS` in `kernels_src/kernels/norm/rmsnorm_common.py`, baked into every instance.
constexpr double BAKED_EPSILON = 1e-5;

/// Relative slack when comparing a graph's epsilon against the baked one. Wide enough that
/// 1e-5 surviving a half-precision round-trip (~1.4e-3 relative error) still matches, and
/// far too narrow to admit a caller who genuinely wanted a different epsilon.
constexpr double EPSILON_RELATIVE_TOLERANCE = 1e-2;

/// A row-major 2-D operand as the kernel reads it: `fly.make_layout((M, N), (stride, 1))`
/// lowered to a packed LLVM struct. The innermost stride is a compile-time 1 and so has no
/// slot here -- the reason `isPackedRowMajor` below is an applicability condition and not a
/// convenience.
struct Desc2D
{
    int32_t shape0 = 0; ///< M, the number of rows.
    int32_t shape1 = 0; ///< N, the normalised axis.
    int64_t stride0 = 0; ///< Row stride in elements.
};

/// A 1-D operand (the gamma vector): length only, unit stride implied.
struct Desc1D
{
    int32_t shape0 = 0;
};

// The kernarg segment is 80 bytes with the descriptors laid out packed; natural C++
// alignment happens to reproduce that layout exactly, which is worth asserting rather
// than assuming, since a mismatch would corrupt every launch silently.
static_assert(sizeof(Desc2D) == 16, "Desc2D must occupy the kernel's 16-byte 2-D layout");
static_assert(offsetof(Desc2D, shape1) == 4, "Desc2D.shape1 must sit at +4");
static_assert(offsetof(Desc2D, stride0) == 8, "Desc2D.stride0 must sit at +8");
static_assert(sizeof(Desc1D) == 4, "Desc1D must occupy the kernel's 4-byte 1-D layout");

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

/// Runs on an unvalidated graph, so must be total: a caller can present a tensor the
/// frontend would have rejected, with null or mismatched dims/strides.
bool hasUsableShape(const data_objects::TensorAttributes& tensor, uint32_t minimumRank)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    return dims != nullptr && strides != nullptr && strides->size() == dims->size()
           && dims->size() >= minimumRank;
}

/// Dense row-major: innermost stride 1 and every outer stride the product of the extents
/// inside it. Stronger than the descriptor strictly needs -- it could carry a padded row
/// stride -- but it is what makes collapsing rank > 2 to (M, N) sound, since one row stride
/// has to describe all of the leading axes at once.
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
        expected *= dims->Get(index);
    }
    return true;
}

/// Product of every extent except the innermost: the row count the grid is sized from.
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

/// The dtypes the shipped instances were built for; see the `dtype` values in the UKDs.
bool isSupportedDataType(data_objects::DataType dataType)
{
    return dataType == data_objects::DataType::BFLOAT16 || dataType == data_objects::DataType::HALF;
}

/// The kernel metadata spelling of a graph dtype. Deliberately not `EnumNameDataType`: the
/// descriptors say `bf16`/`f16`, the flatbuffer enum says `BFLOAT16`/`HALF`, and this is the
/// one place the two vocabularies meet.
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

bool matchesBakedEpsilon(double epsilon)
{
    return std::abs(epsilon - BAKED_EPSILON) <= BAKED_EPSILON * EPSILON_RELATIVE_TOLERANCE;
}

/**
 * @brief Graph-scoped applicability: is this a single RMS-norm node over packed 2-D-
 *        collapsible operands that the pre-built flyDSL kernels actually compute?
 *
 * Refuses rather than approximates. In particular a graph whose epsilon is baked into the
 * op-graph and differs from the kernel's own `EPS` is declined here, so it falls through to
 * an engine that can honour it instead of failing the whole plan build. Epsilon supplied by
 * the caller at execute cannot be inspected yet; `launch()` re-checks it there.
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

    // No bias term and no saved reciprocal-RMS: every shipped instance was compiled with
    // `store_rstd=False` and adds nothing after scaling. A TRAINING graph wants the stat
    // even when it forgot to name the tensor, so it is refused on the phase as well.
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

    // One dtype for all three: the kernel loads gamma with the same element type it loads x
    // with, and writes y in that type too.
    if(!isSupportedDataType(x->data_type()) || scale->data_type() != x->data_type()
       || y->data_type() != x->data_type())
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

    const auto rows = rowCount(*x);
    const auto columns = innermostExtent(*x);
    if(rows <= 0 || columns <= 0)
    {
        return std::nullopt;
    }

    // Gamma must be exactly the normalised axis: any other extent would be a broadcast the
    // kernel does not perform.
    if(rowCount(*scale) != 1 || innermostExtent(*scale) != columns)
    {
        return std::nullopt;
    }

    const auto epsilon = epsilonOperand(context, attributes.epsilon_tensor_uid());
    if(!epsilon.has_value())
    {
        return std::nullopt;
    }
    if(!epsilon->isRuntimeUserSupplied
       && !matchesBakedEpsilon(hipdnn_plugin_sdk::toDouble(epsilon->bakedDefault)))
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

    const auto dtype = kernelDataTypeName(x->data_type());
    if(!dtype.has_value() || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dtype)
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

    return {read(X_TOKEN), read(SCALE_TOKEN), read(EPSILON_TOKEN), read(Y_TOKEN)};
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

const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    if(it == tensors.end() || it->second == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched flydsl_rmsnorm graph has no tensor for uid " + std::to_string(uid));
    }
    return *it->second;
}

/**
 * @brief The argument list this pack launches with, mirroring the launch() below one for
 *        one, and the `signature` block every one of this pack's UKDs carries.
 *
 * Four operands, each a pointer followed by its descriptor struct: x (2-D), gamma (1-D),
 * rstd (1-D), y (2-D). The rstd pair is vestigial -- the kernels were built with
 * `store_rstd=False` and never store through it -- but the entry point still takes the
 * slots, so launch() fills them with gamma rather than leaving them undefined.
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
    static const std::vector<KernelArgument> s_signature{
        s_buffer, s_desc2d, s_buffer, s_desc1d, s_buffer, s_desc1d, s_buffer, s_desc2d};
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

        // Shapes were validated packed and rank >= 2 by the graph matcher.
        const auto rows = rowCount(xTensor);
        const auto columns = innermostExtent(xTensor);

        const auto blockThreads
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD)));

        // Build defines are ignored on the kpack path -- they were baked when the archive
        // was produced -- but the signature still wants an options object.
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

        // The kernel's epsilon is a literal, so a caller-supplied value that disagrees with
        // it would be quietly ignored. A graph that baked its epsilon was already screened
        // out by the matcher; one that defers the value to execute can only be checked here.
        const auto epsilon = hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
            preparedRmsNorm.epsilon(), deviceBuffers, numDeviceBuffers));
        if(!matchesBakedEpsilon(epsilon))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "flydsl_rmsnorm kernels bake epsilon = " + std::to_string(BAKED_EPSILON)
                    + "; this execution supplied " + std::to_string(epsilon));
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

        // The rstd slots are never stored through; they are given gamma's pointer and
        // descriptor so the kernarg segment holds a valid mapping rather than a null.
        // Changing this argument list means changing flydslRmsNormKernelSignature() with it.
        preparedRmsNorm.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    x.ptr,
                    xDesc,
                    scale.ptr,
                    scaleDesc,
                    scale.ptr,
                    scaleDesc,
                    y.ptr,
                    yDesc);
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
