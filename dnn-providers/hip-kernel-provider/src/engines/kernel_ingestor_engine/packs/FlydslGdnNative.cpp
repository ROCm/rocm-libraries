// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a pack that
// names a kpack archive nothing staged.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
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
#include <hipdnn_flatbuffers_sdk/data_objects/gdn_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/linear_attention_common_generated.h>
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
 * @file FlydslGdnNative.cpp
 * @brief The flyDSL Gated DeltaNet pack's native half (`hipkernel:flydsl_gdn`): matching,
 *        scoring, dispatch, and the function that registers them.
 *
 * The kernel (`flydsl/kernels_src/kernels/gdn/gdn_recurrent_kernel.py`, ported from AITER's
 * `gdr_decode.py`) is the GDN forward of `gdn_attributes.fbs` as a token recurrence: one
 * workgroup per (sequence, output head, 32-row slice of V) carries its f32 state slice in
 * registers across the sequence's tokens `[cu[n], cu[n+1])`, so one object serves decode
 * (one token per sequence) and prefill alike. Token, sequence and head counts, GQA head
 * replication (output head h reads q head h / (HO / H), k head h / (HO / HK), v head
 * h / (HO / HV)), every stride, the scale, and every flag -- q/k L2 norm, the LOG or LINEAR
 * gate or the safe gate with optional a_log and dt_bias, beta as given or sigmoid'ed (and
 * doubled for allow_neg_eigval), initial and final state, state_indices -- are kernel
 * arguments, as are the dtypes (f32, f16 or bf16) of g, beta, a_log and dt_bias. An object
 * bakes the q/k/v/o dtype, K and V, and how it addresses the state: `state_strided` 0
 * objects move it in 128-bit vectors along a unit-stride K, the 1 objects take any strides.
 *
 * Declined: state checkpoints, an f16/bf16 state, int64 cu_seqlens or state_indices, and
 * graphs whose cross-field rules (gdn_attributes.fbs) do not hold. batch_invariant is
 * served: every sequence is computed alone in a fixed order. overwrite_initial_state is
 * served: each workgroup reads its slice of the incoming state before it writes the same
 * slice of the outgoing one, and no two workgroups share a slice.
 *
 * Out-of-contract values the graph cannot show -- cu_seqlens that do not run from 0 to T
 * monotonically, a state_indices row outside the pool, two sequences writing one row --
 * are undefined here (hipDNN's CPU reference throws on them).
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using flydsl_norm::findTensor;
using flydsl_norm::isDeviceOperand;
using flydsl_norm::kernelDataTypeName;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_gdn.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_gdn.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_gdn.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_gdn.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_DIM_FIELD = "head_dim";
constexpr std::string_view HEAD_DIM_V_FIELD = "head_dim_v";
/// 0: the state moves in 128-bit vectors along a unit-stride K. 1: any state strides.
constexpr std::string_view STATE_STRIDED_FIELD = "state_strided";
/// Workgroups per (sequence, head), each a 32-row slice of V.
constexpr std::string_view DV_SPLIT_FIELD = "dv_split";
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";

constexpr std::string_view Q_TOKEN = "flydsl_gdn.q.uid";
constexpr std::string_view K_TOKEN = "flydsl_gdn.k.uid";
constexpr std::string_view V_TOKEN = "flydsl_gdn.v.uid";
constexpr std::string_view G_TOKEN = "flydsl_gdn.g.uid";
constexpr std::string_view BETA_TOKEN = "flydsl_gdn.beta.uid";
constexpr std::string_view CU_SEQLENS_TOKEN = "flydsl_gdn.cu_seqlens.uid";
constexpr std::string_view O_TOKEN = "flydsl_gdn.o.uid";
constexpr std::string_view INITIAL_STATE_TOKEN = "flydsl_gdn.initial_state.uid";
constexpr std::string_view FINAL_STATE_TOKEN = "flydsl_gdn.final_state.uid";
constexpr std::string_view A_LOG_TOKEN = "flydsl_gdn.a_log.uid";
constexpr std::string_view DT_BIAS_TOKEN = "flydsl_gdn.dt_bias.uid";
constexpr std::string_view STATE_INDICES_TOKEN = "flydsl_gdn.state_indices.uid";
constexpr std::string_view TOKENS_TOKEN = "flydsl_gdn.tokens";
constexpr std::string_view SEQUENCES_TOKEN = "flydsl_gdn.sequences";
constexpr std::string_view HEADS_OUT_TOKEN = "flydsl_gdn.heads_out";
constexpr std::string_view KEY_DIM_TOKEN = "flydsl_gdn.key_dim";
constexpr std::string_view VALUE_DIM_TOKEN = "flydsl_gdn.value_dim";

// The kernel's `flags` bits (gdn_recurrent_kernel.py FLAG_*).
constexpr int32_t FLAG_L2NORM = 1 << 0;
constexpr int32_t FLAG_BETA_SIGMOID = 1 << 1;
constexpr int32_t FLAG_BETA_DOUBLE = 1 << 2;
constexpr int32_t FLAG_GATE_LINEAR = 1 << 3;
constexpr int32_t FLAG_GATE_SAFE = 1 << 4;
constexpr int32_t FLAG_STATE_INDICES = 1 << 5;
constexpr int32_t FLAG_INITIAL_STATE = 1 << 6;
constexpr int32_t FLAG_FINAL_STATE = 1 << 7;
constexpr int32_t FLAG_A_LOG = 1 << 8;
constexpr int32_t FLAG_DT_BIAS = 1 << 9;

// The kernel's two-bit operand dtype codes and their places in `aux_dtypes`.
constexpr int32_t AUX_F32 = 0;
constexpr int32_t AUX_F16 = 1;
constexpr int32_t AUX_BF16 = 2;
constexpr int32_t AUX_SHIFT_G = 0;
constexpr int32_t AUX_SHIFT_BETA = 2;
constexpr int32_t AUX_SHIFT_A_LOG = 4;
constexpr int32_t AUX_SHIFT_DT_BIAS = 6;

/// q and k move in vectors of this many elements, along a unit-stride K.
constexpr int64_t QK_VECTOR_ELEMENTS = 4;
/// The state moves in vectors of this many f32 elements in a `state_strided` 0 object.
constexpr int64_t STATE_VECTOR_ELEMENTS = 4;
/// The rows of V one workgroup covers.
constexpr int64_t V_SLICE = 32;

constexpr int64_t MAX_INT32 = std::numeric_limits<int32_t>::max();
/// The grid's y extent: the output heads.
constexpr int64_t MAX_GRID_Y = 65535;
/// The furthest byte an operand's 32-bit buffer offsets may reach. The kernel pushes a
/// suppressed load to byte 0x7FFFFFFF of a 0x7FFFFFFF-record buffer, so every real
/// element ends well short of it.
constexpr int64_t MAX_OPERAND_BYTES = 0x7FFFFFFFLL - 16;

constexpr size_t POINTER_COUNT = 12;
constexpr size_t INT_COUNT = 29;

// ---------------------------------------------------------------------------
// The plan: everything a launch needs, derived from the graph alone
// ---------------------------------------------------------------------------

struct GdnUids
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t g = 0;
    int64_t beta = 0;
    int64_t cuSeqlens = 0;
    int64_t o = 0;
    std::optional<int64_t> initialState;
    std::optional<int64_t> finalState;
    std::optional<int64_t> aLog;
    std::optional<int64_t> dtBias;
    std::optional<int64_t> stateIndices;
};

/// A matched graph as the kernel takes it.
struct GdnPlan
{
    GdnUids uids;
    std::string dtype;
    int64_t tokens = 0;
    int64_t sequences = 0;
    int64_t headsOut = 0;
    int64_t keyDim = 0;
    int64_t valueDim = 0;
    /// Whether every state present admits a `state_strided` 0 object.
    bool stateVectorizable = true;
    /// The kernel's int32 arguments, in its order (gdn_recurrent_kernel.py INT_PARAMS).
    std::array<int32_t, INT_COUNT> ints{};
    float scale = 0.0F;
};

/// The node this pack's matchers read, or nullptr if the graph isn't a single GDN node.
const data_objects::GdnAttributes* gdnNode(const MatchContext& context)
{
    // Extents and strides are bound from the graph's shapes at prepare, so override-shape
    // graphs are declined.
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::GdnAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::GdnAttributes>();
}

/// The kernel's dtype code for g, beta, a_log or dt_bias, or nullopt for one it cannot read.
std::optional<int32_t> auxCode(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::FLOAT:
        return AUX_F32;
    case data_objects::DataType::HALF:
        return AUX_F16;
    case data_objects::DataType::BFLOAT16:
        return AUX_BF16;
    default:
        return std::nullopt;
    }
}

int64_t auxBytes(int32_t code)
{
    return code == AUX_F32 ? 4 : 2;
}

bool declaresAlignment(const data_objects::TensorAttributes& tensor, int64_t bytes)
{
    return tensor.alignment() >= bytes && tensor.alignment() % bytes == 0;
}

/// @p tensor's dims, when it has exactly @p rank positive extents and as many
/// non-negative strides; nullopt otherwise. Runs on an unvalidated graph, so it is total.
std::optional<std::vector<int64_t>> dimsOfRank(const data_objects::TensorAttributes& tensor,
                                               size_t rank)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != rank || strides->size() != rank)
    {
        return std::nullopt;
    }
    std::vector<int64_t> result;
    for(flatbuffers::uoffset_t axis = 0; axis < rank; ++axis)
    {
        if(dims->Get(axis) <= 0 || dims->Get(axis) > MAX_INT32 || strides->Get(axis) < 0
           || strides->Get(axis) > MAX_INT32)
        {
            return std::nullopt;
        }
        result.push_back(dims->Get(axis));
    }
    return result;
}

int64_t strideOf(const data_objects::TensorAttributes& tensor, flatbuffers::uoffset_t axis)
{
    return tensor.strides()->Get(axis);
}

/// One past the furthest element @p tensor's axes [first, rank) reach, times @p bytes;
/// strides and extents were bounded to int32 by dimsOfRank, so this cannot overflow.
int64_t spanBytes(const data_objects::TensorAttributes& tensor,
                  flatbuffers::uoffset_t first,
                  int64_t bytes)
{
    int64_t last = 0;
    for(auto axis = first; axis < tensor.dims()->size(); ++axis)
    {
        last += (tensor.dims()->Get(axis) - 1) * strideOf(tensor, axis);
    }
    return (last + 1) * bytes;
}

/// Whether distinct indices of @p tensor reach distinct elements: the axes sorted by stride
/// each step past everything the smaller ones reach. Required of the outputs, since two
/// workgroups writing one element would race.
bool isNonOverlapping(const data_objects::TensorAttributes& tensor)
{
    std::vector<std::pair<int64_t, int64_t>> axes; // (stride, extent)
    for(flatbuffers::uoffset_t axis = 0; axis < tensor.dims()->size(); ++axis)
    {
        if(tensor.dims()->Get(axis) > 1)
        {
            axes.emplace_back(strideOf(tensor, axis), tensor.dims()->Get(axis));
        }
    }
    std::sort(axes.begin(), axes.end());
    int64_t reach = 1;
    for(const auto& [stride, extent] : axes)
    {
        if(stride < reach)
        {
            return false;
        }
        reach = stride * extent;
    }
    return true;
}

/// Whether @p state admits the vector object: unit-stride K, every other stride a whole
/// vector, and a vector-aligned base.
bool isStateVectorizable(const data_objects::TensorAttributes& state)
{
    return strideOf(state, 3) == 1 && strideOf(state, 0) % STATE_VECTOR_ELEMENTS == 0
           && strideOf(state, 1) % STATE_VECTOR_ELEMENTS == 0
           && strideOf(state, 2) % STATE_VECTOR_ELEMENTS == 0
           && declaresAlignment(state, STATE_VECTOR_ELEMENTS * 4);
}

/// Whether the node's attributes ask for a computation the kernel performs: no checkpoints,
/// and the cross-field rules of gdn_attributes.fbs hold.
bool attributesAreServed(const data_objects::GdnAttributes& attributes)
{
    const bool safe = attributes.safe_gate();
    const auto domain = attributes.gate_domain();
    if(attributes.checkpoint_every_n_tokens() != 0
       || attributes.state_checkpoints_tensor_uid().has_value())
    {
        return false;
    }
    if(attributes.output_final_state() != attributes.final_state_tensor_uid().has_value())
    {
        return false;
    }
    if(domain != data_objects::GateDomain::LOG && domain != data_objects::GateDomain::LINEAR)
    {
        return false;
    }
    if(safe && domain == data_objects::GateDomain::LINEAR)
    {
        return false;
    }
    if((attributes.a_log_tensor_uid().has_value() || attributes.dt_bias_tensor_uid().has_value())
       && !safe)
    {
        return false;
    }
    if(attributes.allow_neg_eigval() && !attributes.use_beta_sigmoid())
    {
        return false;
    }
    if(attributes.state_indices_tensor_uid().has_value()
       && !attributes.initial_state_tensor_uid().has_value())
    {
        return false;
    }
    if(attributes.overwrite_initial_state()
       && (!attributes.initial_state_tensor_uid().has_value()
           || !attributes.final_state_tensor_uid().has_value()))
    {
        return false;
    }
    const auto scale = attributes.scale();
    return !scale.has_value() || std::isfinite(*scale);
}

/// Every check the graph match makes, and the launch arguments it implies; nullopt for a
/// graph the kernel does not compute exactly.
std::optional<GdnPlan> planGdn(const MatchContext& context)
{
    const auto* attributesPtr = gdnNode(context);
    if(attributesPtr == nullptr || !attributesAreServed(*attributesPtr))
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // The kernel computes in f32: at least the precision any of these asks for.
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    if(computeType != data_objects::DataType::UNSET && computeType != data_objects::DataType::FLOAT
       && computeType != data_objects::DataType::HALF
       && computeType != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }

    GdnPlan plan;
    auto& uids = plan.uids;
    uids.q = attributes.q_tensor_uid();
    uids.k = attributes.k_tensor_uid();
    uids.v = attributes.v_tensor_uid();
    uids.g = attributes.g_tensor_uid();
    uids.beta = attributes.beta_tensor_uid();
    uids.cuSeqlens = attributes.cu_seqlens_tensor_uid();
    uids.o = attributes.o_tensor_uid();
    const auto optionalUid
        = [](const flatbuffers::Optional<int64_t>& uid) { return std::optional<int64_t>(uid); };
    uids.initialState = optionalUid(attributes.initial_state_tensor_uid());
    uids.finalState = optionalUid(attributes.final_state_tensor_uid());
    uids.aLog = optionalUid(attributes.a_log_tensor_uid());
    uids.dtBias = optionalUid(attributes.dt_bias_tensor_uid());
    uids.stateIndices = optionalUid(attributes.state_indices_tensor_uid());

    const auto* q = findTensor(context, uids.q);
    const auto* k = findTensor(context, uids.k);
    const auto* v = findTensor(context, uids.v);
    const auto* g = findTensor(context, uids.g);
    const auto* beta = findTensor(context, uids.beta);
    const auto* cu = findTensor(context, uids.cuSeqlens);
    const auto* o = findTensor(context, uids.o);
    const auto optionalTensor
        = [&](const std::optional<int64_t>& uid) -> const data_objects::TensorAttributes* {
        return uid.has_value() ? findTensor(context, *uid) : nullptr;
    };
    const auto* h0 = optionalTensor(uids.initialState);
    const auto* ht = optionalTensor(uids.finalState);
    const auto* aLog = optionalTensor(uids.aLog);
    const auto* dtBias = optionalTensor(uids.dtBias);
    const auto* indices = optionalTensor(uids.stateIndices);
    if(q == nullptr || k == nullptr || v == nullptr || g == nullptr || beta == nullptr
       || cu == nullptr || o == nullptr || (uids.initialState && h0 == nullptr)
       || (uids.finalState && ht == nullptr) || (uids.aLog && aLog == nullptr)
       || (uids.dtBias && dtBias == nullptr) || (uids.stateIndices && indices == nullptr))
    {
        return std::nullopt;
    }
    for(const auto* tensor : {q, k, v, g, beta, cu, o, h0, ht, aLog, dtBias, indices})
    {
        if(tensor != nullptr && !isDeviceOperand(*tensor))
        {
            return std::nullopt;
        }
    }

    // q, k, v and o in one bf16 or f16 type; the state f32; the per-token and per-head
    // scalars each f32, f16 or bf16; the offsets and indices INT32.
    const auto ioType = q->data_type();
    const auto dtype = kernelDataTypeName(ioType);
    if((ioType != data_objects::DataType::BFLOAT16 && ioType != data_objects::DataType::HALF)
       || !dtype.has_value() || k->data_type() != ioType || v->data_type() != ioType
       || o->data_type() != ioType)
    {
        return std::nullopt;
    }
    plan.dtype = *dtype;
    for(const auto* state : {h0, ht})
    {
        if(state != nullptr && state->data_type() != data_objects::DataType::FLOAT)
        {
            return std::nullopt;
        }
    }
    if(cu->data_type() != data_objects::DataType::INT32
       || (indices != nullptr && indices->data_type() != data_objects::DataType::INT32))
    {
        return std::nullopt;
    }
    const auto gCode = auxCode(g->data_type());
    const auto betaCode = auxCode(beta->data_type());
    const auto aLogCode = aLog != nullptr ? auxCode(aLog->data_type()) : std::optional(AUX_F32);
    const auto dtBiasCode
        = dtBias != nullptr ? auxCode(dtBias->data_type()) : std::optional(AUX_F32);
    if(!gCode || !betaCode || !aLogCode || !dtBiasCode)
    {
        return std::nullopt;
    }

    // Shapes: q [T, H, K], k [T, HK, K], v [T, HV, V], g and beta [T, HO], o [T, HO, V],
    // cu_seqlens [N + 1], states [N or P, HO, V, K], a_log and dt_bias [HO], indices [N].
    const auto qDims = dimsOfRank(*q, 3);
    const auto kDims = dimsOfRank(*k, 3);
    const auto vDims = dimsOfRank(*v, 3);
    const auto gDims = dimsOfRank(*g, 2);
    const auto betaDims = dimsOfRank(*beta, 2);
    const auto oDims = dimsOfRank(*o, 3);
    const auto cuDims = dimsOfRank(*cu, 1);
    if(!qDims || !kDims || !vDims || !gDims || !betaDims || !oDims || !cuDims)
    {
        return std::nullopt;
    }
    const int64_t tokens = (*qDims)[0];
    const int64_t headsQ = (*qDims)[1];
    const int64_t keyDim = (*qDims)[2];
    const int64_t headsK = (*kDims)[1];
    const int64_t headsV = (*vDims)[1];
    const int64_t valueDim = (*vDims)[2];
    const int64_t headsOut = std::max(headsQ, headsV);
    const int64_t sequences = (*cuDims)[0] - 1;
    if((*kDims)[0] != tokens || (*vDims)[0] != tokens || (*kDims)[2] != keyDim
       || (headsK != headsQ && headsK != headsV) || headsOut % headsQ != 0 || headsOut % headsV != 0
       || *gDims != std::vector<int64_t>{tokens, headsOut}
       || *betaDims != std::vector<int64_t>{tokens, headsOut}
       || *oDims != std::vector<int64_t>{tokens, headsOut, valueDim} || sequences < 1)
    {
        return std::nullopt;
    }
    for(const auto* perHead : {aLog, dtBias})
    {
        if(perHead != nullptr && dimsOfRank(*perHead, 1) != std::vector<int64_t>{headsOut})
        {
            return std::nullopt;
        }
    }
    if(indices != nullptr && dimsOfRank(*indices, 1) != std::vector<int64_t>{sequences})
    {
        return std::nullopt;
    }
    for(const auto* state : {h0, ht})
    {
        if(state == nullptr)
        {
            continue;
        }
        const auto dims = dimsOfRank(*state, 4);
        if(!dims || (*dims)[1] != headsOut || (*dims)[2] != valueDim || (*dims)[3] != keyDim
           || (indices == nullptr && (*dims)[0] != sequences))
        {
            return std::nullopt;
        }
    }
    if(h0 != nullptr && ht != nullptr && h0->dims()->Get(0) != ht->dims()->Get(0))
    {
        return std::nullopt;
    }

    // q and k move in vectors along a unit-stride K: whole vectors between rows and heads,
    // from a vector-aligned base. v and o move one element at a time along a unit-stride V.
    const auto ioBytes = int64_t{2};
    for(const auto* qk : {q, k})
    {
        if(strideOf(*qk, 2) != 1 || strideOf(*qk, 0) % QK_VECTOR_ELEMENTS != 0
           || strideOf(*qk, 1) % QK_VECTOR_ELEMENTS != 0
           || !declaresAlignment(*qk, QK_VECTOR_ELEMENTS * ioBytes))
        {
            return std::nullopt;
        }
    }
    if(strideOf(*v, 2) != 1 || strideOf(*o, 2) != 1 || !declaresAlignment(*v, ioBytes)
       || !declaresAlignment(*o, ioBytes) || !isNonOverlapping(*o))
    {
        return std::nullopt;
    }
    if(!declaresAlignment(*g, auxBytes(*gCode)) || !declaresAlignment(*beta, auxBytes(*betaCode))
       || (aLog != nullptr && !declaresAlignment(*aLog, auxBytes(*aLogCode)))
       || (dtBias != nullptr && !declaresAlignment(*dtBias, auxBytes(*dtBiasCode)))
       || !declaresAlignment(*cu, 4) || (indices != nullptr && !declaresAlignment(*indices, 4)))
    {
        return std::nullopt;
    }
    for(const auto* state : {h0, ht})
    {
        if(state != nullptr && !declaresAlignment(*state, 4))
        {
            return std::nullopt;
        }
    }
    if(ht != nullptr && !isNonOverlapping(*ht))
    {
        return std::nullopt;
    }

    // Each operand within its buffer's 32-bit offsets; a state's (row, head) tile only,
    // since each tile gets its own 64-bit base.
    if(spanBytes(*q, 0, ioBytes) > MAX_OPERAND_BYTES
       || spanBytes(*k, 0, ioBytes) > MAX_OPERAND_BYTES
       || spanBytes(*v, 0, ioBytes) > MAX_OPERAND_BYTES
       || spanBytes(*o, 0, ioBytes) > MAX_OPERAND_BYTES
       || spanBytes(*g, 0, auxBytes(*gCode)) > MAX_OPERAND_BYTES
       || spanBytes(*beta, 0, auxBytes(*betaCode)) > MAX_OPERAND_BYTES
       || spanBytes(*cu, 0, 4) > MAX_OPERAND_BYTES
       || (indices != nullptr && spanBytes(*indices, 0, 4) > MAX_OPERAND_BYTES))
    {
        return std::nullopt;
    }
    for(const auto* state : {h0, ht})
    {
        if(state != nullptr && spanBytes(*state, 2, 4) > MAX_OPERAND_BYTES)
        {
            return std::nullopt;
        }
    }
    // The grid: (N * V / 32, HO, 1).
    if(headsOut > MAX_GRID_Y || sequences > MAX_INT32 / std::max<int64_t>(1, valueDim / V_SLICE))
    {
        return std::nullopt;
    }

    plan.tokens = tokens;
    plan.sequences = sequences;
    plan.headsOut = headsOut;
    plan.keyDim = keyDim;
    plan.valueDim = valueDim;
    plan.stateVectorizable = (h0 == nullptr || isStateVectorizable(*h0))
                             && (ht == nullptr || isStateVectorizable(*ht));

    int32_t flags = 0;
    flags |= attributes.use_qk_l2norm() ? FLAG_L2NORM : 0;
    flags |= attributes.use_beta_sigmoid() ? FLAG_BETA_SIGMOID : 0;
    flags |= attributes.allow_neg_eigval() ? FLAG_BETA_DOUBLE : 0;
    flags |= attributes.safe_gate() ? FLAG_GATE_SAFE : 0;
    flags |= attributes.gate_domain() == data_objects::GateDomain::LINEAR ? FLAG_GATE_LINEAR : 0;
    flags |= indices != nullptr ? FLAG_STATE_INDICES : 0;
    flags |= h0 != nullptr ? FLAG_INITIAL_STATE : 0;
    flags |= ht != nullptr ? FLAG_FINAL_STATE : 0;
    flags |= aLog != nullptr ? FLAG_A_LOG : 0;
    flags |= dtBias != nullptr ? FLAG_DT_BIAS : 0;
    const int32_t auxDtypes = (*gCode << AUX_SHIFT_G) | (*betaCode << AUX_SHIFT_BETA)
                              | (*aLogCode << AUX_SHIFT_A_LOG) | (*dtBiasCode << AUX_SHIFT_DT_BIAS);

    // Every value was bounded to int32 above.
    const auto i32 = [](int64_t value) { return static_cast<int32_t>(value); };
    const auto stateStride
        = [&](const data_objects::TensorAttributes* state, flatbuffers::uoffset_t axis) {
              return state == nullptr ? 0 : i32(strideOf(*state, axis));
          };
    plan.ints = {i32(sequences),
                 i32(headsOut),
                 i32(headsOut / headsQ),
                 i32(headsOut / headsK),
                 i32(headsOut / headsV),
                 i32(strideOf(*q, 0)),
                 i32(strideOf(*q, 1)),
                 i32(strideOf(*k, 0)),
                 i32(strideOf(*k, 1)),
                 i32(strideOf(*v, 0)),
                 i32(strideOf(*v, 1)),
                 i32(strideOf(*o, 0)),
                 i32(strideOf(*o, 1)),
                 i32(strideOf(*g, 0)),
                 i32(strideOf(*g, 1)),
                 i32(strideOf(*beta, 0)),
                 i32(strideOf(*beta, 1)),
                 aLog == nullptr ? 0 : i32(strideOf(*aLog, 0)),
                 dtBias == nullptr ? 0 : i32(strideOf(*dtBias, 0)),
                 stateStride(h0, 0),
                 stateStride(h0, 1),
                 stateStride(h0, 2),
                 stateStride(h0, 3),
                 stateStride(ht, 0),
                 stateStride(ht, 1),
                 stateStride(ht, 2),
                 stateStride(ht, 3),
                 flags,
                 auxDtypes};
    const auto scale = attributes.scale();
    plan.scale = scale.has_value()
                     ? *scale
                     : static_cast<float>(1.0 / std::sqrt(static_cast<double>(keyDim)));
    return plan;
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/**
 * @brief Graph-scoped applicability: a single GDN node whose attributes, dtypes, shapes,
 *        strides and extents the shipped kernel computes exactly. See planGdn().
 */
std::optional<BoundTokens> flydslGdnGraphMatches(const MatchContext& context)
{
    const auto plan = planGdn(context);
    if(!plan.has_value())
    {
        return std::nullopt;
    }
    const auto& uids = plan->uids;
    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = uids.q;
    bound[std::string(K_TOKEN)] = uids.k;
    bound[std::string(V_TOKEN)] = uids.v;
    bound[std::string(G_TOKEN)] = uids.g;
    bound[std::string(BETA_TOKEN)] = uids.beta;
    bound[std::string(CU_SEQLENS_TOKEN)] = uids.cuSeqlens;
    bound[std::string(O_TOKEN)] = uids.o;
    const auto bindOptional = [&](std::string_view token, const std::optional<int64_t>& uid) {
        if(uid.has_value())
        {
            bound[std::string(token)] = *uid;
        }
    };
    bindOptional(INITIAL_STATE_TOKEN, uids.initialState);
    bindOptional(FINAL_STATE_TOKEN, uids.finalState);
    bindOptional(A_LOG_TOKEN, uids.aLog);
    bindOptional(DT_BIAS_TOKEN, uids.dtBias);
    bindOptional(STATE_INDICES_TOKEN, uids.stateIndices);
    bound[std::string(TOKENS_TOKEN)] = plan->tokens;
    bound[std::string(SEQUENCES_TOKEN)] = plan->sequences;
    bound[std::string(HEADS_OUT_TOKEN)] = plan->headsOut;
    bound[std::string(KEY_DIM_TOKEN)] = plan->keyDim;
    bound[std::string(VALUE_DIM_TOKEN)] = plan->valueDim;
    return bound;
}

/// Whether @p kernel was built for @p plan: its dtype, K and V, a state addressing the
/// graph's states admit, and a V split that tiles V into 32-row slices.
bool kernelServesPlan(const GdnPlan& plan, const KernelDefinition& kernel)
{
    if(kernel.getStringMetadata(std::string(DTYPE_FIELD)) != plan.dtype
       || kernel.getIntMetadata(std::string(HEAD_DIM_FIELD)) != plan.keyDim
       || kernel.getIntMetadata(std::string(HEAD_DIM_V_FIELD)) != plan.valueDim
       || kernel.getIntMetadata(std::string(DV_SPLIT_FIELD)) * V_SLICE != plan.valueDim
       || kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD)) <= 0)
    {
        return false;
    }
    const auto strided = kernel.getIntMetadata(std::string(STATE_STRIDED_FIELD));
    return strided == 1 || (strided == 0 && plan.stateVectorizable);
}

/**
 * @brief Kernel-scoped applicability: the object's dtype, K and V, and -- for a
 *        `state_strided` 0 object -- states with a unit-stride K, whole-vector strides and
 *        16-byte aligned bases.
 */
bool flydslGdnKernelMatches(const MatchContext& context,
                            const BoundTokens& /*bound*/,
                            const KernelDefinition& kernel)
{
    const auto plan = planGdn(context);
    return plan.has_value() && kernelServesPlan(*plan, kernel);
}

/// The descriptors' priority: a vector-state object (100) above a strided one (10).
double flydslGdnScore(const MatchContext& /*context*/,
                      const BoundTokens& /*bound*/,
                      const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The compiled kernel plus everything read from the graph, resolved once.
class PreparedFlydslGdn : public PreparedDispatch
{
public:
    PreparedFlydslGdn(IngestorKernelCode code, GdnPlan plan)
        : _code(std::move(code))
        , _plan(std::move(plan))
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const GdnPlan& plan() const
    {
        return _plan;
    }

private:
    IngestorKernelCode _code;
    GdnPlan _plan;
};

/**
 * @brief The argument list this pack launches with, mirroring launch() below one for one:
 *        twelve pointers (q, k, v, g, beta, cu_seqlens, a_log, dt_bias, state_indices,
 *        initial_state, final_state, o), 29 int32s (GdnPlan::ints) and the f32 scale.
 *        Names are empty and offsets zero because neither is compared for this producer;
 *        see requireSignatureMatch.
 */
const std::vector<KernelArgument>& flydslGdnKernelSignature()
{
    static const std::vector<KernelArgument> s_signature = [] {
        const KernelArgument buffer{"global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
        const KernelArgument value{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
        std::vector<KernelArgument> signature(POINTER_COUNT, buffer);
        // The int32s and the f32 scale are all four-byte values.
        signature.insert(signature.end(), INT_COUNT + 1, value);
        return signature;
    }();
    return s_signature;
}

template <size_t... P, size_t... I>
void launchGdn(const compilation::IRunnableKernel& kernel,
               hipStream_t stream,
               const std::array<void*, POINTER_COUNT>& pointers,
               const std::array<int32_t, INT_COUNT>& ints,
               const float& scale,
               std::index_sequence<P...> /*pointerSlots*/,
               std::index_sequence<I...> /*intSlots*/)
{
    kernel.launch(stream, pointers[P]..., ints[I]..., scale);
}

/**
 * @brief Sizes and launches a flyDSL GDN kernel. Everything graph/kernel-derived resolves
 *        at prepare(); launch() only resolves buffers and launches.
 */
class FlydslGdnDispatchHandler : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    FlydslGdnDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                             const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The state lives in registers: no scratch.
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        auto plan = planGdn(context);
        if(!plan.has_value() || !kernelServesPlan(*plan, kernel))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_gdn: prepare() given a graph or kernel its matchers would decline");
        }
        const auto blockThreads = kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD));
        const auto dvSplit = kernel.getIntMetadata(std::string(DV_SPLIT_FIELD));

        // Build defines are ignored on the kpack path; built from the arch alone, as the
        // other flyDSL packs do.
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, flydslGdnKernelSignature());

        // `gdn_sequence_head_vslice`: one workgroup per (sequence, V slice) by output head.
        code.setBlockSize(static_cast<unsigned int>(blockThreads), 1, 1);
        code.setGridSize(static_cast<unsigned int>(plan->sequences * dvSplit),
                         static_cast<unsigned int>(plan->headsOut),
                         1);

        return std::make_unique<PreparedFlydslGdn>(std::move(code), std::move(*plan));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedGdn = dynamic_cast<const PreparedFlydslGdn&>(prepared);
        const auto& plan = preparedGdn.plan();
        const auto& uids = plan.uids;

        const auto pointer = [&](int64_t uid) {
            return hipdnn_plugin_sdk::findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers).ptr;
        };
        // An absent operand is never touched: the kernel gives its buffer zero records.
        const auto optionalPointer = [&](const std::optional<int64_t>& uid) -> void* {
            return uid.has_value() ? pointer(*uid) : nullptr;
        };

        // Changing this argument list means changing flydslGdnKernelSignature() with it.
        const std::array<void*, POINTER_COUNT> pointers{pointer(uids.q),
                                                        pointer(uids.k),
                                                        pointer(uids.v),
                                                        pointer(uids.g),
                                                        pointer(uids.beta),
                                                        pointer(uids.cuSeqlens),
                                                        optionalPointer(uids.aLog),
                                                        optionalPointer(uids.dtBias),
                                                        optionalPointer(uids.stateIndices),
                                                        optionalPointer(uids.initialState),
                                                        optionalPointer(uids.finalState),
                                                        pointer(uids.o)};
        launchGdn(preparedGdn.kernelForStream(handle.getStream()),
                  handle.getStream(),
                  pointers,
                  plan.ints,
                  plan.scale,
                  std::make_index_sequence<POINTER_COUNT>{},
                  std::make_index_sequence<INT_COUNT>{});
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& flydslGdnKpackModuleCache()
{
    // Process-lifetime, and this pack's own: a cache key is (archive, toc_key, arch), so a
    // cache shared with another pack would answer that pack's lookups too.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

} // namespace

void resetFlydslGdnModuleCache()
{
    flydslGdnKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer.
const FlydslGdnDispatchHandler& flydslGdnDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslGdnKpackModuleCache());
    static const FlydslGdnDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslGdnSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslGdnGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslGdnKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslGdnScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslGdnDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
