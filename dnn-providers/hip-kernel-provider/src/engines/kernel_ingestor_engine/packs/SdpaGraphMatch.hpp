// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file SdpaGraphMatch.hpp
 * @brief The graph-reading helpers every ingestor pack that serves a single dense BSHD
 *        SDPA node shares: finding the node and its operands, the operand checks made before
 *        anything indexes an axis, the problem read from Q and K, the f32 compute-type rule
 *        for the packs that apply it, the declared-alignment, checked-product and
 *        overlap checks and the limits the FlyDSL SDPA families bound by, and the
 *        decline-line head.
 *
 * Pack-family rules (mask kinds, dtypes, the decode rule) stay with the family, e.g.
 * UnifiedAttentionGraphMatch.hpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::sdpa_graph
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// hipDNN tensor axes. The LOGICAL order is always (B, H, S, D) regardless of layout.
inline constexpr uint32_t BATCH_AXIS = 0;
inline constexpr uint32_t HEAD_AXIS = 1;
inline constexpr uint32_t SEQ_AXIS = 2;
inline constexpr uint32_t HEAD_SIZE_AXIS = 3;
inline constexpr uint32_t SDPA_RANK = 4;

/// The largest value a 32-bit offset or extent holds.
inline constexpr int64_t INT32_LIMIT = 2147483647LL;

/// The largest softmax scale magnitude every FlyDSL SDPA kernel serves. Real graphs use
/// 1/sqrt(d) or a temperature near 1; far above this each family overflows f32 at a different
/// point (the CDNA causal objects' running-max floor times scale * log2(e) passes FLT_MAX near
/// 2.4e8), so one bound here makes all of them decline alike rather than diverge.
inline constexpr double MAX_SOFTMAX_SCALE = 1048576.0; // 2^20

inline const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// Whether @p tensor promises a base pointer aligned to @p bytes. The backend enforces the
/// alignment a graph declares and no more, so a kernel that assumes a wider one must read the
/// declaration: a smaller one would hand it a pointer its vector accesses cannot take.
inline bool declaresAlignment(const data_objects::TensorAttributes& tensor, int64_t bytes)
{
    return tensor.alignment() >= bytes && tensor.alignment() % bytes == 0;
}

/// The product of @p factors, or nullopt when it does not fit in int64_t. Every product of
/// graph-controlled extents goes through this: an unchecked one is signed overflow, and a
/// wrapped value can pass a bound it should fail. A factor of 0 or below is a factor like any
/// other (pack_features::checkedProduct rejects it).
inline std::optional<int64_t> checkedProduct(std::initializer_list<int64_t> factors)
{
    int64_t product = 1;
    for(const int64_t factor : factors)
    {
        if(__builtin_mul_overflow(product, factor, &product))
        {
            return std::nullopt;
        }
    }
    return product;
}

/// Whether no two elements of @p tensor share an address, which an output needs: concurrent
/// workgroups writing one location race. Its axes of more than one element, ordered by
/// stride, must each step past everything the axes inside it span. With @p ignoreBatch the
/// batch axis is left out (a ragged tensor's batches are placed by its offsets). Conservative:
/// a layout interleaving two axes inside one another's gaps declines too.
inline bool elementsDoNotOverlap(const data_objects::TensorAttributes& tensor,
                                 bool ignoreBatch = false)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != strides->size())
    {
        return false;
    }
    std::vector<std::pair<int64_t, int64_t>> axes; // (stride, extent)
    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        if(dims->Get(axis) > 1 && !(ignoreBatch && axis == 0))
        {
            axes.emplace_back(strides->Get(axis), dims->Get(axis));
        }
    }
    std::sort(axes.begin(), axes.end());
    int64_t span = 1; // elements the inner axes reach, from the first
    for(const auto& [stride, extent] : axes)
    {
        if(stride < span)
        {
            return false;
        }
        const auto reach = checkedProduct({extent - 1, stride});
        if(!reach.has_value() || *reach > std::numeric_limits<int64_t>::max() - span)
        {
            return false;
        }
        span += *reach;
    }
    return true;
}

/// The node an SDPA pack's matchers read, or nullptr if the graph is not a single
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
 * A kernel that takes no stride arguments reads a differently-strided tensor as if it were
 * this one, so it must be declined. Callers must have checked isWellFormedOperand().
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

/// The graph facts a pack's matchers and prepare() need, derived once from the tensors.
struct SdpaShape
{
    int64_t batch = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t headSize = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

/// The unified attention packs' name for it.
using UnifiedAttentionProblem = SdpaShape;

/// The graph's shape, read from Q and K. Callers must have checked both with
/// isWellFormedOperand().
inline SdpaShape problemFor(const data_objects::TensorAttributes& q,
                            const data_objects::TensorAttributes& k)
{
    SdpaShape problem;
    problem.batch = q.dims()->Get(BATCH_AXIS);
    problem.numQueryHeads = q.dims()->Get(HEAD_AXIS);
    problem.seqLenQ = q.dims()->Get(SEQ_AXIS);
    problem.headSize = q.dims()->Get(HEAD_SIZE_AXIS);
    problem.numKvHeads = k.dims()->Get(HEAD_AXIS);
    problem.seqLenKv = k.dims()->Get(SEQ_AXIS);
    problem.dataType = q.data_type();
    return problem;
}

/**
 * @brief Whether a kernel that computes the softmax, accumulates and keeps the statistics in
 *        f32 meets the node's compute type.
 *
 * f32 is at least the precision FLOAT, HALF or BFLOAT16 asks for, and UNSET leaves it to the
 * engine. DOUBLE asks for more, and an integer or FP8 compute type for a different
 * computation, so those decline. Callers must have found the node with sdpaNode().
 */
inline bool computesInFloat(const MatchContext& context)
{
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    return computeType == data_objects::DataType::UNSET
           || computeType == data_objects::DataType::FLOAT
           || computeType == data_objects::DataType::HALF
           || computeType == data_objects::DataType::BFLOAT16;
}

// ---------------------------------------------------------------------------
// Decline logging
//
// hipDNN reports a declined graph to its caller only as "No engine configurations available
// for the graph", so each decline logs its cause at INFO. graph_match runs on every catalog
// miss, so every piece of that work sits inside HIPDNN_PLUGIN_LOG_INFO's argument: the macro
// tests the level first, and with INFO off nothing below is called.
// ---------------------------------------------------------------------------

/// The head of a decline line: "<engine> declined the graph [<cause>]: ".
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

} // namespace hip_kernel_provider::kernel_ingestor_engine::sdpa_graph
