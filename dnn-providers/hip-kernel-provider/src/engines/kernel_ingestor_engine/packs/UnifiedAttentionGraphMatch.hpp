// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file UnifiedAttentionGraphMatch.hpp
 * @brief The graph- and record-reading helpers the gfx950 unified-attention packs
 *        (hipkernel:Gfx950AttentionUnified2D and hipkernel:Gfx950AttentionUnified3D) share.
 *
 * Both packs serve a single dense BSHD SDPA-forward node through the same paged-KV kernel
 * family, so they read the graph, classify its mask and read a candidate's metadata the
 * same way. Those rules live here once, so a matcher fix is one edit; what differs per
 * engine (the decode rule, the geometry check, the launch) stays in each pack. Decline
 * lines name their engine through DeclinedBy.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::unified_attention
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// hipDNN tensor axes. The LOGICAL order is always (B, H, S, D) regardless of layout.
inline constexpr uint32_t BATCH_AXIS = 0;
inline constexpr uint32_t HEAD_AXIS = 1;
inline constexpr uint32_t SEQ_AXIS = 2;
inline constexpr uint32_t HEAD_SIZE_AXIS = 3;
inline constexpr uint32_t SDPA_RANK = 4;

inline const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// The node this engine's matchers read, or nullptr if the graph is not a single
/// SDPA-forward node.
inline const data_objects::SdpaAttributes* sdpaNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::SdpaAttributes)
    {
        return nullptr;
    }

    return &node.attributesAs<data_objects::SdpaAttributes>();
}

/// Total over an UNVALIDATED graph: rank, stride/dim agreement and positive extents,
/// checked before anything indexes an axis.
inline bool isWellFormedOperand(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return false;
    }

    for(const auto dim : *dims)
    {
        if(dim <= 0)
        {
            return false;
        }
    }

    return !tensor.virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor);
}

/**
 * @brief Is this tensor's memory BSHD -- token-major, head varying fastest -- and gap-free?
 *
 * The kernel reads Q and writes O as `[total_q, Hq, D]` rows and reads K/V as a paged cache
 * `[blocks, block_size, Hkv, D]`; with the identity block table those are exactly the BSHD
 * layouts of a contiguous graph tensor. A differently-strided tensor is read as if it were
 * this one, so it must be declined.
 *
 * Unit-extent axes are exempt: a stride multiplies an index that is always 0 when the extent
 * is 1. An expected stride too large for int64_t is one no stride can equal, so that axis
 * fails unless it is unit-extent.
 */
inline bool hasBshdStrides(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();

    const int64_t heads = dims->Get(HEAD_AXIS);
    const int64_t sequence = dims->Get(SEQ_AXIS);
    const int64_t headSize = dims->Get(HEAD_SIZE_AXIS);

    const auto axisOk = [&](uint32_t axis, std::optional<int64_t> expected) {
        return dims->Get(axis) == 1 || (expected.has_value() && strides->Get(axis) == *expected);
    };

    return axisOk(BATCH_AXIS, pack_features::checkedProduct({sequence, heads, headSize}))
           && axisOk(HEAD_AXIS, headSize)
           && axisOk(SEQ_AXIS, pack_features::checkedProduct({heads, headSize}))
           && axisOk(HEAD_SIZE_AXIS, 1);
}

// ---------------------------------------------------------------------------
// Decline logging
//
// hipDNN reports a declined graph to its caller only as "No engine configurations available
// for the graph", so each decline logs its cause at INFO. graph_match runs on every catalog
// miss, so every piece of that work sits inside HIPDNN_PLUGIN_LOG_INFO's argument: the macro
// tests the level first, and with INFO off nothing below is called.
// ---------------------------------------------------------------------------

/// The head of a decline line: "<engine> declined the graph [<cause>]: ". Each pack wraps
/// this in its own `Declined{cause}` that fills in its engine name.
struct DeclinedBy
{
    std::string_view engine;
    std::string_view cause;
};

inline std::ostream& operator<<(std::ostream& out, const DeclinedBy& declined)
{
    return out << declined.engine << " declined the graph [" << declined.cause << "]: ";
}

/// Whether any node of the graph is an SDPA-forward node. Only such a graph hears why it
/// was declined: graph_match sees every graph the catalog misses on.
inline bool hasSdpaForwardNode(const MatchContext& context)
{
    for(const auto& node : context.graph.nodeWrappers())
    {
        if(node->attributesType() == data_objects::NodeAttributes::SdpaAttributes)
        {
            return true;
        }
    }
    return false;
}

/// "Q (uid 1): dims [..], strides [..]". Null dims or strides print as [].
inline std::string describeOperand(std::string_view role,
                                   const data_objects::TensorAttributes& tensor)
{
    using hipdnn_data_sdk::utilities::vecToString;
    using hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector;
    std::ostringstream out;
    out << role << " (uid " << tensor.uid() << "): dims "
        << vecToString(convertFlatBufferVectorToStdVector(tensor.dims())) << ", strides "
        << vecToString(convertFlatBufferVectorToStdVector(tensor.strides()));
    return out.str();
}

/// Why @p tensor is not one the kernel can address. Checks the rank itself rather than
/// trusting that the caller validated it.
inline std::string notBshdReason(std::string_view role,
                                 const data_objects::TensorAttributes& tensor)
{
    std::ostringstream out;
    out << describeOperand(role, tensor)
        << " is not dense BSHD (token-major, head fastest); the kernel addresses Q/O as "
           "[tokens, heads, head_size] and K/V as a gap-free paged cache and takes no stride "
           "arguments";
    return out.str();
}

/// The mask kinds these engines serve, in the numbering of rocKE's AttentionMaskType and of
/// the catalog's `mask_type` field. No variant carries a non-zero sliding_window, so a
/// windowed mask has no spelling here.
enum class MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL = 1,
    BOTTOM_RIGHT_CAUSAL = 2
};

/**
 * @brief Which mask the graph is asking for, or nullopt for one this engine lacks.
 *
 * Deprecated causal booleans are merged with the bounds and alignment by
 * asm_sdpa_engine::plan_utils::resolveDiagonalBand, then classified by getMaskType. Both
 * booleans set, invalid bounds, or any band other than unmasked or causal, is declined: no
 * shipped variant carries a non-zero sliding_window, and the right bound is -1 or 0.
 */
inline std::optional<MaskType> maskTypeFor(const data_objects::SdpaAttributes& attributes,
                                           std::string_view engine)
{
    using asm_sdpa_engine::plan_utils::getMaskType;
    using asm_sdpa_engine::plan_utils::resolveDiagonalBand;
    using PlanMaskType = asm_sdpa_engine::plan_utils::MaskType;

    PlanMaskType planMask{};
    try
    {
        const auto band = resolveDiagonalBand(attributes);
        planMask = getMaskType(band.leftBound, band.rightBound, band.alignment);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "mask"}) << e.what());
        return std::nullopt;
    }

    switch(planMask)
    {
    case PlanMaskType::NO_MASK:
        return MaskType::NO_MASK;
    case PlanMaskType::TOP_LEFT_CAUSAL:
        return MaskType::TOP_LEFT_CAUSAL;
    case PlanMaskType::BOTTOM_RIGHT_CAUSAL:
        return MaskType::BOTTOM_RIGHT_CAUSAL;
    case PlanMaskType::SLIDING_WINDOW:
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "mask"})
                               << "left_bound " << attributes.left_bound().value_or(-1)
                               << ", right_bound " << attributes.right_bound().value_or(-1)
                               << " is a window; this catalog is causal or unmasked only and no "
                                  "shipped variant carries a sliding_window");
        return std::nullopt;
    default:
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "mask"})
                               << "mask type " << static_cast<int>(planMask)
                               << " is not supported");
        return std::nullopt;
    }
}

/// The kernel's dtype spelling for a graph dtype, or nullopt for one it cannot be built for.
inline std::optional<std::string> supportedDataTypeName(data_objects::DataType dataType)
{
    if(dataType == data_objects::DataType::BFLOAT16)
    {
        return std::string("BF16");
    }
    if(dataType == data_objects::DataType::HALF)
    {
        return std::string("FP16");
    }
    return std::nullopt;
}

/// A candidate's integer metadata field, or nullopt when it is absent or holds another
/// type. Total: never throws, so a malformed record declines instead of faulting a match.
inline std::optional<int64_t> integerMetadata(const KernelDefinition& kernel,
                                              std::string_view field)
{
    const auto it = kernel.metadata.find(std::string(field));
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<int64_t>(&it->second);
    if(value == nullptr)
    {
        return std::nullopt;
    }
    return *value;
}

/// A candidate's string metadata field, or nullopt when it is absent or holds another type.
/// Total, like integerMetadata().
inline std::optional<std::string> stringMetadata(const KernelDefinition& kernel,
                                                 std::string_view field)
{
    const auto it = kernel.metadata.find(std::string(field));
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<std::string>(&it->second);
    if(value == nullptr)
    {
        return std::nullopt;
    }
    return *value;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::unified_attention
