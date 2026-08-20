// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "asm/AsmKernelPath.hpp"
#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdLaunch.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdParams.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaModuleCache.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file AsmSdpaForwardNative.cpp
 * @brief The ASM SDPA forward engine's native half, against descriptors carrying what
 *        used to be `fmha_v3_fwd/fmha_fwd.csv` and its generated config map.
 *
 * The split this file exists to show. The CSV was already data, and it is now the
 * `kernels` array of two per-architecture packs: one entry per row, one metadata field
 * per column. What no descriptor field can carry stays here, and it is three things.
 *
 * - **Applicability.** Two dozen attribute and tensor predicates over a graph nothing
 *   has validated yet, ending in the uint32 byte-stride check that keeps a silently
 *   truncating kernarg from being dispatched. A declarative criteria vocabulary that
 *   could express these is RFC 0017's `nodes`/`criteria`, which nothing parses yet.
 * - **Which code object.** The descriptor names a path relative to the kernel tree.
 *   The tree's root is an environment override or a build-time definition, and on
 *   gfx942 the MI300 and MI308 copies are chosen by a PCI chip-id probe. Both are
 *   properties of the running device, not of the kernel, so both stay here.
 * - **Launch geometry and the kernarg struct.** Grid, block and a tuning selector
 *   derived from sequence length, head count, mask and architecture, then ~40 fields
 *   packed into a struct whose layout is the assembly's ABI.
 *
 * The symbol names below are restated rather than shared through a header, because a
 * descriptor file cannot export a constant to C++. The loader pre-flights every symbol
 * a descriptor names, so a typo costs the engine at load rather than at dispatch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace plan_utils = asm_sdpa_engine::plan_utils;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.asm_sdpa_fwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.asm_sdpa_fwd.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.asm_sdpa_fwd.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.asm_sdpa_fwd.dispatch";

// KMD fields, one per column of the CSV these kernels came from.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HDIM_Q_FIELD = "hdim_q";
constexpr std::string_view HDIM_V_FIELD = "hdim_v";
constexpr std::string_view MASK_FIELD = "mask";
constexpr std::string_view MODE_FIELD = "mode";
constexpr std::string_view TS_QO_FIELD = "ts_qo";

// Tokens the graph matcher binds for dispatch, so prepare() re-reads a decision rather
// than re-deriving it (RFC 0017 §8.1).
constexpr std::string_view Q_TOKEN = "asm_sdpa_fwd.q.uid";
constexpr std::string_view K_TOKEN = "asm_sdpa_fwd.k.uid";
constexpr std::string_view V_TOKEN = "asm_sdpa_fwd.v.uid";
constexpr std::string_view O_TOKEN = "asm_sdpa_fwd.o.uid";

/// Rank every operand of a servable graph has.
constexpr uint32_t SDPA_RANK = 4;
/// The forward kernarg stores byte strides as uint32, and every shipped kernel is bf16.
constexpr int64_t BF16_BYTES = 2;

/// The MI308 PCI chip ids. gfx942 ships one code object per die, and only this
/// distinguishes them; a device reports the same gcnArchName either way.
constexpr std::array<int, 4> MI308_CHIP_IDS = {0x74a2, 0x74a8, 0x74b6, 0x74bc};

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched SDPA forward graph binds.
struct SdpaBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// True when the tensor is a rank-4 operand whose byte strides fit the kernarg's uint32
/// fields. A value that does not fit truncates silently and returns wrong numbers, so
/// this is a refusal rather than a runtime check.
bool isServableOperand(const data_objects::TensorAttributes* tensor)
{
    if(tensor == nullptr)
    {
        return false;
    }

    // A matcher runs on a graph nothing has validated: a plugin-ABI caller or a
    // deserialized graph can present a tensor the frontend would have rejected, so
    // dims() and strides() are both as optional as each other.
    const auto* dims = tensor->dims();
    const auto* strides = tensor->strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return false;
    }

    // Batch, head and sequence strides are what the kernarg carries; the innermost is
    // implied.
    for(uint32_t axis = 0; axis < 3U; ++axis)
    {
        if(!plan_utils::byteStrideFitsU32("stride", strides->Get(axis), BF16_BYTES))
        {
            return false;
        }
    }
    return true;
}

/// The single SDPA node this engine serves, or nullptr when the graph is not one.
const data_objects::SdpaAttributes* sdpaNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::SdpaAttributes
       || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::SdpaAttributes>();
}

/// The mask class the graph asks for, matching the CSV's `mask` column, or nullopt when
/// the attributes contradict each other.
std::optional<int64_t> graphMask(const data_objects::SdpaAttributes& attrs)
{
    try
    {
        return static_cast<int64_t>(plan_utils::getMaskType(attrs));
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
    {
        // Contradictory mask attributes are an invalid-input condition this engine
        // declines rather than dispatches.
        return std::nullopt;
    }
}

/// The batch mode, matching the CSV's `mode` column: sequence-length tensors mean the
/// grouped (variable-length) kernels.
int64_t graphMode(const data_objects::SdpaAttributes& attrs)
{
    return (attrs.seq_len_q_tensor_uid().has_value() || attrs.seq_len_kv_tensor_uid().has_value())
               ? 1
               : 0;
}

/**
 * @brief Is this a single SDPA forward this engine's prebuilt kernels can serve at all?
 *
 * Everything a kernel's own metadata cannot answer. The shape question -- which of the
 * CSV rows fits -- is the kernel-scoped matcher below; this one is the gate that makes
 * the question worth asking, run once per (graph, device) ahead of every pack.
 */
std::optional<BoundTokens> asmSdpaForwardGraphMatches(const MatchContext& context)
{
    const auto* attrs = sdpaNode(context);
    if(attrs == nullptr)
    {
        return std::nullopt;
    }

    // Execute-time override shapes can diverge from the dims matched here, and this
    // engine serves fixed prebuilt shapes (RFC 0008 §4.6).
    if(context.graph.getGraph().is_override_shape_enabled())
    {
        return std::nullopt;
    }

    // Features with no prebuilt kernel. Each is a decline, not an error.
    if((attrs->dropout_probability().has_value() && attrs->dropout_probability().value() != 0.F)
       || attrs->alibi_mask() || attrs->padding_mask() || attrs->attn_mask_tensor_uid()
       || attrs->page_table_k_tensor_uid() || attrs->page_table_v_tensor_uid()
       || attrs->generate_stats() || attrs->mma_core_mode() != data_objects::DataType::UNSET)
    {
        return std::nullopt;
    }

    // A scale tensor is servable only pass-by-value; the kernarg carries a scalar, not a
    // pointer to read at launch (RFC 0016).
    if(attrs->scale_tensor_uid().has_value())
    {
        const auto* scale = findTensor(context, attrs->scale_tensor_uid().value());
        if(scale == nullptr || !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(scale))
        {
            return std::nullopt;
        }
    }

    const SdpaBinding binding{
        attrs->q_tensor_uid(), attrs->k_tensor_uid(), attrs->v_tensor_uid(), attrs->o_tensor_uid()};

    const auto* q = findTensor(context, binding.q);
    const auto* k = findTensor(context, binding.k);
    const auto* v = findTensor(context, binding.v);
    const auto* o = findTensor(context, binding.o);
    // Inputs are checked for shape and stride; the output is not. `o` is inferred by the
    // frontend and its dims and strides are not necessarily populated when matching runs,
    // so requiring a rank-4 shape here declines every graph that has not been through
    // shape inference yet -- which on a sweep is all of them.
    if(!isServableOperand(q) || !isServableOperand(k) || !isServableOperand(v) || o == nullptr)
    {
        return std::nullopt;
    }

    // Every shipped kernel is uniform bf16 in and out.
    if(q->data_type() != data_objects::DataType::BFLOAT16
       || k->data_type() != data_objects::DataType::BFLOAT16
       || v->data_type() != data_objects::DataType::BFLOAT16
       || o->data_type() != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }

    // K and V must agree on head count; the GQA ratio the kernarg carries is derived
    // from Q against that one number.
    if(k->dims()->Get(1) != v->dims()->Get(1) || v->dims()->Get(1) == 0)
    {
        return std::nullopt;
    }

    if(!graphMask(*attrs).has_value())
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound.emplace(std::string(Q_TOKEN), binding.q);
    bound.emplace(std::string(K_TOKEN), binding.k);
    bound.emplace(std::string(V_TOKEN), binding.v);
    bound.emplace(std::string(O_TOKEN), binding.o);
    return bound;
}

/**
 * @brief Kernel-scoped applicability: does this kernel's row fit the graph?
 *
 * The CSV lookup, one candidate at a time. What was a loop over a generated config map
 * comparing five columns is the same five comparisons against one kernel's metadata;
 * the loop is now the catalog's.
 */
bool asmSdpaForwardKernelMatches(const MatchContext& context,
                                 const BoundTokens& /*bound*/,
                                 const KernelDefinition& kernel)
{
    const auto* attrs = sdpaNode(context);
    if(attrs == nullptr)
    {
        return false;
    }

    const auto* q = findTensor(context, attrs->q_tensor_uid());
    const auto* v = findTensor(context, attrs->v_tensor_uid());
    if(q == nullptr || v == nullptr || q->dims() == nullptr || v->dims() == nullptr
       || q->dims()->size() != SDPA_RANK || v->dims()->size() != SDPA_RANK)
    {
        return false;
    }

    const auto mask = graphMask(*attrs);
    if(!mask.has_value())
    {
        return false;
    }

    return kernel.getStringMetadata(std::string(DTYPE_FIELD)) == "bf16"
           && kernel.getIntMetadata(std::string(HDIM_Q_FIELD)) == q->dims()->Get(3)
           && kernel.getIntMetadata(std::string(HDIM_V_FIELD)) == v->dims()->Get(3)
           && kernel.getIntMetadata(std::string(MASK_FIELD)) == *mask
           && kernel.getIntMetadata(std::string(MODE_FIELD)) == graphMode(*attrs);
}

/// Every surviving kernel is an exact shape match, so ranking only has to be total.
double asmSdpaForwardScore(const MatchContext& /*context*/,
                           const BoundTokens& /*bound*/,
                           const KernelDefinition& kernel)
{
    // Larger Q/O tiles are the better-occupancy variant where two rows both fit.
    return static_cast<double>(kernel.getIntMetadata(std::string(TS_QO_FIELD)));
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/**
 * @brief @p gcnArchName without its target-id feature suffix.
 *
 * A device reports `gfx942:sramecc+:xnack-`; SdpaFwdParams::archString is compared for
 * equality against a bare `gfx942` inside computeFwdLaunchParams(), which is what selects
 * the hd192x128 grid swap. Handing it the raw name silently loses that branch: the kernel
 * still matches, still loads, and launches with the wrong geometry.
 *
 * The hand-written builder took this string from getDeviceString(), which strips the
 * suffix. This is that same normalisation, at the one point the ingestor's raw
 * DeviceProperties crosses into code expecting the stripped form.
 */
std::string baseArchIdentifier(const std::string& gcnArchName)
{
    return gcnArchName.substr(0, gcnArchName.find(':'));
}

/// True when @p deviceId is an MI308 die. gfx942 ships one code object per die under a
/// directory the descriptor deliberately does not name, because two devices reporting
/// the same gcnArchName want different files.
bool isMi308Device(DeviceId deviceId)
{
    int chipId = 0;
    if(deviceId == NO_DEVICE
       || hipDeviceGetAttribute(&chipId, hipDeviceAttributePciChipId, deviceId) != hipSuccess)
    {
        return false;
    }

    for(const int candidate : MI308_CHIP_IDS)
    {
        if(chipId == candidate)
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief The absolute path of @p kernel's code object on @p deviceId.
 *
 * The descriptor carries a path relative to the kernel tree because the tree moves
 * between the build and install prefixes and can be redirected by environment. On
 * gfx942 the die-specific directory is spliced in here, for the same reason it is not
 * in the file: it is a property of the device, not of the kernel.
 */
std::string codeObjectPath(const KernelDefinition& kernel, DeviceId deviceId)
{
    std::string relative = kernel.source.codeObjectFile;
    const auto slash = relative.rfind('/');
    if(slash != std::string::npos && relative.rfind("gfx942/", 0) == 0)
    {
        relative = relative.substr(0, slash + 1) + (isMi308Device(deviceId) ? "MI308/" : "MI300/")
                   + relative.substr(slash + 1);
    }
    return asm_sdpa_engine::asm_kernels::getAsmKernelPath(relative);
}

/// The loaded code object plus everything its launch needs, resolved once and owning
/// nothing that points back into the graph.
class PreparedAsmSdpaForward : public PreparedDispatch
{
public:
    PreparedAsmSdpaForward(asm_sdpa_engine::CachedModule module,
                           asm_sdpa_engine::SdpaFwdParams params,
                           SdpaBinding binding)
        : _module(std::move(module))
        , _params(std::move(params))
        , _binding(binding)
    {
    }

    hipFunction_t function() const
    {
        return _module->function();
    }

    const asm_sdpa_engine::SdpaFwdParams& params() const
    {
        return _params;
    }

    const SdpaBinding& binding() const
    {
        return _binding;
    }

private:
    asm_sdpa_engine::CachedModule _module;
    asm_sdpa_engine::SdpaFwdParams _params;
    SdpaBinding _binding;
};

SdpaBinding readBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "asm sdpa forward dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    return {read(Q_TOKEN), read(K_TOKEN), read(V_TOKEN), read(O_TOKEN)};
}

/**
 * @brief The native dispatch behind this engine's UDD.
 *
 * Loads the selected code object, derives the launch geometry, and packs the assembly's
 * kernarg struct. Splits per RFC 0017 §8.5: everything derived from the graph resolves
 * at plan build, and launch only resolves device pointers by uid.
 */
class AsmSdpaForwardDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// These kernels are prebuilt code objects; nothing here compiles source.
    bool supportsSourceKind(KernelSourceKind kind) const override
    {
        return kind == KernelSourceKind::HSACO_FILE;
    }

    /// The forward kernels keep their working set in LDS; none asks for global scratch.
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
        const auto binding = readBinding(bound);
        const auto path = codeObjectPath(kernel, context.deviceId);

        auto module = _moduleCache.getOrLoad(path, kernel.source.codeObjectSymbol.c_str());
        if(!module)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "asm sdpa forward dispatch failed to load '" + kernel.source.codeObjectSymbol
                    + "' from " + path);
        }

        return std::make_unique<PreparedAsmSdpaForward>(
            std::move(module), buildParams(context, binding, kernel), binding);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& forward = dynamic_cast<const PreparedAsmSdpaForward&>(prepared);
        asm_sdpa_engine::launchForward(forward.function(),
                                       forward.params(),
                                       deviceBuffers,
                                       numDeviceBuffers,
                                       handle.getStream());
    }

private:
    /**
     * @brief Everything the kernarg struct is built from, read off the graph once.
     *
     * The dims and strides the assembly wants, plus the Q/O tile size, which is the one
     * launch input that comes from the chosen kernel's metadata rather than the graph:
     * it is the CSV's `ts_qo` column, and the grid is derived from it.
     */
    static asm_sdpa_engine::SdpaFwdParams buildParams(const MatchContext& context,
                                                      const SdpaBinding& binding,
                                                      const KernelDefinition& kernel)
    {
        const auto& tensors = context.graph.getTensorMap();
        const auto tensor = [&tensors](int64_t uid) -> const data_objects::TensorAttributes& {
            const auto it = tensors.find(uid);
            if(it == tensors.end() || it->second == nullptr)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "matched sdpa graph has no tensor for uid " + std::to_string(uid));
            }
            return *it->second;
        };

        const auto& q = tensor(binding.q);
        const auto& k = tensor(binding.k);
        const auto& v = tensor(binding.v);
        const auto& o = tensor(binding.o);

        // The matcher admitted this graph, so every operand is rank 4 with matching
        // strides; an out-of-range read here is not reachable from a matched graph.
        const auto dim = [](const data_objects::TensorAttributes& t, uint32_t axis) {
            return static_cast<unsigned int>(t.dims()->Get(axis));
        };
        const auto stride = [](const data_objects::TensorAttributes& t, uint32_t axis) {
            return static_cast<unsigned int>(t.strides()->Get(axis));
        };

        const auto& attrs
            = context.graph.getNodeWrapper(0).attributesAs<data_objects::SdpaAttributes>();

        asm_sdpa_engine::SdpaFwdParams params{};
        params.qUid = binding.q;
        params.kUid = binding.k;
        params.vUid = binding.v;
        params.oUid = binding.o;
        params.lseUid = -1;
        params.batchSize = dim(q, 0);
        params.numHeadsQ = dim(q, 1);
        params.numHeadsKv = dim(k, 1);
        params.seqLenQ = dim(q, 2);
        params.seqLenKv = dim(k, 2);
        params.headDimQk = dim(q, 3);
        params.headDimV = dim(v, 3);
        params.qStrideBatch = stride(q, 0);
        params.qStrideHead = stride(q, 1);
        params.qStrideSeq = stride(q, 2);
        params.qStrideRow = params.qStrideSeq;
        params.kStrideBatch = stride(k, 0);
        params.kStrideHead = stride(k, 1);
        params.kStrideSeq = stride(k, 2);
        params.vStrideBatch = stride(v, 0);
        params.vStrideHead = stride(v, 1);
        params.vStrideSeq = stride(v, 2);
        params.oStrideBatch = stride(o, 0);
        params.oStrideHead = stride(o, 1);
        params.oStrideSeq = stride(o, 2);
        params.tileSizeQo
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(TS_QO_FIELD)));
        params.archString = baseArchIdentifier(context.deviceProperties.gcnArchName);
        params.maskType = plan_utils::getMaskType(attrs);

        if(attrs.scale_tensor_uid().has_value())
        {
            params.attnScale = hipdnn_plugin_sdk::makeScalarOperand(
                tensors, attrs.scale_tensor_uid().value(), "attn_scale");
        }
        else
        {
            const float scale = attrs.attn_scale_value().value_or(
                1.0F / std::sqrt(static_cast<float>(params.headDimQk)));
            params.attnScale = hipdnn_plugin_sdk::ScalarOperand{
                0, data_objects::DataType::FLOAT, false, hipdnn_plugin_sdk::ScalarValue{scale}};
        }
        return params;
    }

    // hipModuleLoad dominates this engine's cost and the set of distinct code objects is
    // bounded by the descriptor count, so modules are loaded once per process.
    mutable asm_sdpa_engine::SdpaModuleCache _moduleCache;
};

/// This engine's dispatch handler. Process-lifetime: the registry holds a non-owning
/// pointer while a Container is created and destroyed per handle.
const AsmSdpaForwardDispatchHandler& asmSdpaForwardDispatchHandler()
{
    static const AsmSdpaForwardDispatchHandler s_dispatchHandler;
    return s_dispatchHandler;
}

} // namespace

void registerAsmSdpaForwardSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &asmSdpaForwardGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &asmSdpaForwardKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &asmSdpaForwardScore);
    scope.add(std::string(DISPATCH_SYMBOL), &asmSdpaForwardDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
