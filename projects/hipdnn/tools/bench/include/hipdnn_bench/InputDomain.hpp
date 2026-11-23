// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_bench/NumericalValidation.hpp>
#include <hipdnn_bench/VariantPackBuilder.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_tools/TensorSpan.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/// @file InputDomain.hpp
/// @brief What the bench must write into a graph's inputs: the tensors the validation fill must
///        keep positive, and the contents of those that decide how much work runs.
namespace hipdnn_bench
{

/// @brief Whether a pointwise @p mode is undefined on negative values (and on zero, which the
///        fill never writes): LOG, SQRT and RSQRT.
inline bool undefinedOnNegatives(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    return mode == PointwiseMode::LOG || mode == PointwiseMode::SQRT
           || mode == PointwiseMode::RSQRT;
}

/// @brief The uids of the tensors an operation undefined on negative values reads in @p graph.
///
/// The fill keeps those it writes positive. One it does not write -- a tensor the graph
/// computes -- can be negative, so a NaN there may be the correct answer.
inline std::set<int64_t>
    positiveDomainReads(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph)
{
    std::set<int64_t> reads;
    if(graph.nodes() == nullptr)
    {
        return reads;
    }
    for(const auto* node : *graph.nodes())
    {
        const auto* pointwise = node->attributes_as_PointwiseAttributes();
        if(pointwise != nullptr && undefinedOnNegatives(pointwise->operation()))
        {
            reads.insert(pointwise->in_0_tensor_uid());
        }
    }
    return reads;
}

/// @brief positiveDomainReads() of the serialized graph (or graph-and-plan container) @p data.
inline std::set<int64_t> positiveDomainReads(const uint8_t* data, size_t size)
{
    const auto wrapper
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(data,
                                                                                         size);
    return positiveDomainReads(wrapper.getGraph());
}

/// @brief What the bench writes into the inputs whose contents decide how much work runs.
struct DeclaredContents
{
    /// Contents by uid, one value per element of the tensor's strided span: logical element i
    /// (row-major over its dims) at its stride offset, zeros in the gaps.
    std::map<int64_t, std::vector<int64_t>> values;
    /// Inputs whose contents decide the work but cannot be declared, with why. A run that
    /// reads one fails rather than timing another problem in its place.
    std::map<int64_t, std::string> refused;
};

namespace detail
{
namespace fb = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::heuristics::detail::tensor;

inline const fb::TensorAttributes* tensor(const fb::Graph& graph,
                                          flatbuffers::Optional<int64_t> uid)
{
    return uid.has_value() ? tensor(graph, *uid) : nullptr;
}

/// Extent @p axis of @p value, when it has one and it is positive.
inline std::optional<int64_t> extent(const fb::TensorAttributes* value, size_t axis)
{
    if(value == nullptr || value->dims() == nullptr || value->dims()->size() <= axis
       || value->dims()->Get(static_cast<flatbuffers::uoffset_t>(axis)) <= 0)
    {
        return std::nullopt;
    }
    return value->dims()->Get(static_cast<flatbuffers::uoffset_t>(axis));
}

/// Elements of @p value, when every extent is known and non-negative.
inline std::optional<int64_t> elementCount(const fb::TensorAttributes* value)
{
    if(value == nullptr || value->dims() == nullptr)
    {
        return std::nullopt;
    }
    int64_t count = 1;
    for(const auto dim : *value->dims())
    {
        if(dim < 0)
        {
            return std::nullopt;
        }
        count *= dim;
    }
    return count;
}

/// @p logical (row-major over @p value's dims) placed at the elements' stride offsets across
/// the strided span the bench allocates, zeros in the gaps. Nullopt when the strides do not
/// describe a span, or two elements the strides overlap would need different values.
inline std::optional<std::vector<int64_t>> placed(const fb::TensorAttributes& value,
                                                  const std::vector<int64_t>& logical)
{
    if(value.strides() == nullptr || value.strides()->empty())
    {
        return logical; // Packed as written.
    }
    const std::vector<int64_t> dims(value.dims()->begin(), value.dims()->end());
    const std::vector<int64_t> strides(value.strides()->begin(), value.strides()->end());
    const auto span = hipdnn_tools::elementSpan(dims, strides);
    if(span == 0)
    {
        return std::nullopt;
    }
    std::vector<int64_t> image(static_cast<size_t>(span), 0);
    std::vector<bool> written(image.size(), false);
    std::vector<int64_t> index(dims.size(), 0);
    for(const auto element : logical)
    {
        size_t offset = 0;
        for(size_t axis = 0; axis < dims.size(); ++axis)
        {
            offset += static_cast<size_t>(index[axis] * strides[axis]);
        }
        if(written[offset] && image[offset] != element)
        {
            return std::nullopt;
        }
        image[offset] = element;
        written[offset] = true;
        for(size_t axis = dims.size(); axis-- > 0;)
        {
            if(++index[axis] < dims[axis])
            {
                break;
            }
            index[axis] = 0;
        }
    }
    return image;
}

/// Collects contents; a uid two rules give different contents, or one rule cannot give any,
/// is refused.
class ContentsBuilder
{
public:
    /// Element i of @p value is @p first + i x @p step.
    void ramp(const fb::TensorAttributes* value, int64_t step, int64_t first = 0)
    {
        if(value == nullptr)
        {
            return;
        }
        const auto count = elementCount(value);
        if(!count)
        {
            refuse(value, "its extents are unknown");
            return;
        }
        std::vector<int64_t> logical(static_cast<size_t>(*count));
        for(size_t index = 0; index < logical.size(); ++index)
        {
            logical[index] = first + (static_cast<int64_t>(index) * step);
        }
        set(value, logical);
    }

    /// Every element of @p value is @p constant.
    void fill(const fb::TensorAttributes* value, int64_t constant)
    {
        ramp(value, 0, constant);
    }

    /// Element i of @p value (row-major over its dims) is @p logical[i].
    void set(const fb::TensorAttributes* value, const std::vector<int64_t>& logical)
    {
        if(value == nullptr || _contents.refused.count(value->uid()) != 0)
        {
            return;
        }
        const auto image = placed(*value, logical);
        if(!image)
        {
            refuse(value, "its strides do not place one value per element");
            return;
        }
        const auto [found, inserted] = _contents.values.emplace(value->uid(), *image);
        if(!inserted && found->second != *image)
        {
            refuse(value, "the operations reading it declare different contents");
        }
    }

    /// No contents make a run of @p value's readers the declared problem, because @p why.
    void refuse(const fb::TensorAttributes* value, const std::string& why)
    {
        if(value == nullptr)
        {
            return;
        }
        _contents.values.erase(value->uid());
        _contents.refused.emplace(value->uid(), why);
    }

    DeclaredContents take()
    {
        return std::move(_contents);
    }

private:
    DeclaredContents _contents;
};

/// Every element of @p value is @p length, or @p value is refused when that is unknown.
inline void declareLength(const fb::TensorAttributes* value,
                          std::optional<int64_t> length,
                          const char* unknown,
                          ContentsBuilder& contents)
{
    if(length)
    {
        contents.fill(value, *length);
    }
    else
    {
        contents.refuse(value, unknown);
    }
}

/// Valid lengths: every element Q's declared sequence extent (axis 2 of [B, H, S, D]), and
/// @p keys for K and V.
template <typename TAttention>
void declareLengths(const fb::Graph& graph,
                    const TAttention& op,
                    std::optional<int64_t> keys,
                    ContentsBuilder& contents)
{
    declareLength(tensor(graph, op.seq_len_q_tensor_uid()),
                  extent(tensor(graph, op.q_tensor_uid()), 2),
                  "Q's sequence extent is unknown",
                  contents);
    declareLength(tensor(graph, op.seq_len_kv_tensor_uid()),
                  keys,
                  "the keys per sequence cannot be inferred",
                  contents);
}

/// SDPA forward: lengths, cumulative lengths and page tables from the keys the work model
/// counts (hipdnn_plugin_sdk::heuristics::detail::keyExtent), and a block mask keeping every
/// block.
inline void declareAttention(const fb::Graph& graph,
                             const fb::SdpaAttributes& op,
                             ContentsBuilder& contents)
{
    const auto extentKv = hipdnn_plugin_sdk::heuristics::detail::keyExtent(graph, op);
    const auto keys = extentKv ? std::optional<int64_t>{extentKv->keys} : std::optional<int64_t>{};
    const auto queries = extent(tensor(graph, op.q_tensor_uid()), 2);
    declareLengths(graph, op, keys, contents);
    if(const auto* cumulative = tensor(graph, op.cu_seq_len_q_tensor_uid()); queries)
    {
        contents.ramp(cumulative, *queries);
    }
    else
    {
        contents.refuse(cumulative, "Q's sequence extent is unknown");
    }
    if(const auto* cumulative = tensor(graph, op.cu_seq_len_kv_tensor_uid()); keys)
    {
        contents.ramp(cumulative, *keys);
    }
    else
    {
        contents.refuse(cumulative, "the keys per sequence cannot be inferred");
    }
    // Every sequence its own pages: entry i of a table is block i, which needs a cache of at
    // least as many blocks as the table has entries.
    for(const auto& [uid, cache] : {std::pair{op.page_table_k_tensor_uid(), op.k_tensor_uid()},
                                    std::pair{op.page_table_v_tensor_uid(), op.v_tensor_uid()}})
    {
        const auto* table = tensor(graph, uid);
        const auto blocks = extent(tensor(graph, cache), 0);
        const auto entries = elementCount(table);
        if(blocks && entries && *entries <= *blocks)
        {
            contents.ramp(table, 1);
        }
        else
        {
            contents.refuse(table, "its cache holds fewer blocks than the table has entries");
        }
    }
    // UINT8 [B, H, ceil(Sq / 128), ceil(ceil(Skv / 128) / 8)], one bit per 128 x 128 tile
    // (SEM-025): every byte 0xFF keeps every block, when the mask covers every tile.
    constexpr int64_t TILE = 128;
    constexpr int64_t BITS = 8;
    const auto* mask = tensor(graph, op.block_mask_tensor_uid());
    const auto tileRows = extent(mask, 2);
    const auto tileBytes = extent(mask, 3);
    const bool narrow
        = (queries && tileRows && *tileRows < (*queries + TILE - 1) / TILE)
          || (keys && tileBytes && *tileBytes < ((*keys + TILE - 1) / TILE + BITS - 1) / BITS);
    if(mask != nullptr && mask->data_type() != fb::DataType::UINT8)
    {
        contents.refuse(mask, "a block mask is UINT8, one bit per tile");
    }
    else if(narrow)
    {
        contents.refuse(mask, "it covers fewer tiles than the declared queries and keys");
    }
    else
    {
        contents.fill(mask, 0xFF);
    }
}

/// Group g starts at g x rows / groups: every declared row routed, groups as even as they go.
inline void declareRouting(const fb::TensorAttributes* offsets,
                           std::optional<int64_t> rows,
                           ContentsBuilder& contents)
{
    const auto groups = elementCount(offsets);
    if(!rows || !groups || *groups == 0)
    {
        contents.refuse(offsets, "the routed rows or the groups are unknown");
        return;
    }
    std::vector<int64_t> logical(static_cast<size_t>(*groups));
    for(size_t group = 0; group < logical.size(); ++group)
    {
        logical[group] = static_cast<int64_t>(group) * *rows / *groups;
    }
    contents.set(offsets, logical);
}

/// GDN and KDA `cu_seqlens` [N + 1]: sequence n starts at n x T / N, so every one of q's T
/// tokens is in some sequence, as the count assumes.
template <typename TRecurrence>
void declareSequences(const fb::Graph& graph, const TRecurrence& op, ContentsBuilder& contents)
{
    const auto* offsets = tensor(graph, op.cu_seqlens_tensor_uid());
    const auto tokens = extent(tensor(graph, op.q_tensor_uid()), 0);
    const auto bounds = elementCount(offsets);
    if(!tokens || !bounds || *bounds < 2)
    {
        contents.refuse(offsets, "the tokens or the sequences are unknown");
        return;
    }
    const auto sequences = *bounds - 1;
    std::vector<int64_t> logical(static_cast<size_t>(*bounds));
    for(size_t index = 0; index < logical.size(); ++index)
    {
        logical[index] = static_cast<int64_t>(index) * *tokens / sequences;
    }
    contents.set(offsets, logical);
}

/// A ragged tensor's batches where the dense layout puts them: offset i is i batch strides,
/// in units of its multiplier.
inline void declareRaggedOffsets(const fb::Graph& graph, ContentsBuilder& contents)
{
    if(graph.tensors() == nullptr)
    {
        return;
    }
    for(const auto* ragged : *graph.tensors())
    {
        const auto* offsets = tensor(graph, ragged->ragged_offset_tensor_uid());
        const auto multiplier = ragged->ragged_offset_multiplier();
        if(offsets == nullptr)
        {
            continue;
        }
        if(ragged->strides() == nullptr || ragged->strides()->empty() || multiplier < 1
           || ragged->strides()->Get(0) % multiplier != 0)
        {
            contents.refuse(offsets,
                            "its ragged tensor's batch stride is not a multiple of its "
                            "ragged_offset_multiplier");
            continue;
        }
        contents.ramp(offsets, ragged->strides()->Get(0) / multiplier);
    }
}
} // namespace detail

/// @brief The inputs of @p graph whose contents decide how much of a node's declared work
///        runs, each with the contents under which the run is the problem the graph declares:
///        the one `graph.flops` counts (RFC 0019 section 6.8).
///
/// Without them a timing buffer's zeros give valid lengths and offsets of 0 and a block mask
/// keeping nothing, and the validation fill's 1s and 2s lengths of a token or two, so a label
/// would time a near-empty problem against the declared problem's work. Contents:
/// - a ragged offset tensor: element i = i x the ragged tensor's batch stride / its multiplier;
/// - SDPA forward and backward `seq_len_q` / `seq_len_kv`: every element Q's sequence extent or
///   the keys per sequence (a paged cache's inferred keys);
/// - SDPA `cu_seq_len_q` / `cu_seq_len_kv`: element i = i x that length;
/// - SDPA `page_table_k` / `page_table_v`: element i = block i;
/// - SDPA `block_mask`: every byte 0xFF, every block kept;
/// - MoE grouped matmul `first_token_offset` (both directions): group g starts at
///   g x rows / groups, rows being the declared routed rows;
/// - GDN and KDA `cu_seqlens`: sequence n starts at n x T / N.
/// An operand one of these rules cannot declare (an unknown extent, a ragged stride its
/// multiplier does not divide, a cache with fewer blocks than its table has entries, a block
/// mask that is not UINT8 or covers fewer tiles than the declared queries and keys, two rules
/// disagreeing) is in `refused`, and a run reading it fails.
inline DeclaredContents declaredContents(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph)
{
    detail::ContentsBuilder contents;
    detail::declareRaggedOffsets(graph, contents);
    if(graph.nodes() == nullptr)
    {
        return contents.take();
    }
    for(const auto* node : *graph.nodes())
    {
        if(const auto* sdpa = node->attributes_as_SdpaAttributes())
        {
            detail::declareAttention(graph, *sdpa, contents);
        }
        else if(const auto* backward = node->attributes_as_SdpaBackwardAttributes())
        {
            const auto keys = hipdnn_plugin_sdk::heuristics::detail::denseKeys(graph, *backward);
            detail::declareLengths(graph,
                                   *backward,
                                   keys ? std::optional<int64_t>{keys->keys}
                                        : std::optional<int64_t>{},
                                   contents);
        }
        else if(const auto* moe = node->attributes_as_MoeGroupedMatmulAttributes())
        {
            // Gathering routes the token index's rows, the other modes the token's.
            const auto* rows = moe->mode() == detail::fb::MoeGroupedMatmulMode::GATHER
                                   ? detail::tensor(graph, moe->token_index_tensor_uid())
                                   : detail::tensor(graph, moe->token_tensor_uid());
            detail::declareRouting(detail::tensor(graph, moe->first_token_offset_tensor_uid()),
                                   detail::extent(rows, 1),
                                   contents);
        }
        else if(const auto* moeBackward = node->attributes_as_MoeGroupedMatmulBwdAttributes())
        {
            detail::declareRouting(
                detail::tensor(graph, moeBackward->first_token_offset_tensor_uid()),
                detail::extent(detail::tensor(graph, moeBackward->token_tensor_uid()), 1),
                contents);
        }
        else if(const auto* gdn = node->attributes_as_GdnAttributes())
        {
            detail::declareSequences(graph, *gdn, contents);
        }
        else if(const auto* kda = node->attributes_as_KdaAttributes())
        {
            detail::declareSequences(graph, *kda, contents);
        }
    }
    return contents.take();
}

/// @brief declaredContents() of the serialized graph (or graph-and-plan container) @p data.
inline DeclaredContents declaredContents(const uint8_t* data, size_t size)
{
    const auto wrapper
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(data,
                                                                                         size);
    return declaredContents(wrapper.getGraph());
}

/// @brief Gives every tensor of @p plan memory from @p buffers and writes the inputs in
///        @p declared as it states them, so every run, timed or validated, is the problem the
///        graph declares.
///
/// An input of @p plan that @p declared refuses, or whose declared contents its buffer cannot
/// hold, is an error rather than another problem timed in its place. @p buffers provides
/// `void* add(const TensorRequirement&)` (zero-filled memory, null when exhausted) and
/// `bool upload(const TensorRequirement&, void*, const std::vector<uint8_t>&)`.
template <typename TBuffers>
hipdnn_frontend::Error allocateVariantPack(const VariantPackPlan& plan,
                                           TBuffers& buffers,
                                           std::unordered_map<int64_t, void*>& variantPack,
                                           const DeclaredContents& declared)
{
    if(!plan.error.empty())
    {
        return {hipdnn_frontend::ErrorCode::INVALID_VALUE, plan.error};
    }
    for(const auto& tensor : plan.tensors)
    {
        void* pointer = buffers.add(tensor);
        if(pointer == nullptr)
        {
            return {hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Out of device memory for '" + tensor.name + "' ("
                        + std::to_string(tensor.bytes) + " bytes)"};
        }
        variantPack[tensor.uid] = pointer;
        if(tensor.produced)
        {
            continue;
        }
        if(const auto refused = declared.refused.find(tensor.uid);
           refused != declared.refused.end())
        {
            return {hipdnn_frontend::ErrorCode::INVALID_VALUE,
                    "Cannot declare the contents of '" + tensor.name + "': " + refused->second};
        }
        const auto contents = declared.values.find(tensor.uid);
        if(contents == declared.values.end())
        {
            continue;
        }
        const auto image = detail::declaredImage(
            tensor.dataType, static_cast<size_t>(tensor.bytes), contents->second);
        if(image.empty())
        {
            return {hipdnn_frontend::ErrorCode::INVALID_VALUE,
                    "Cannot write the declared contents of '" + tensor.name + "' ("
                        + std::to_string(contents->second.size()) + " values into "
                        + std::to_string(tensor.bytes) + " bytes)"};
        }
        if(!buffers.upload(tensor, pointer, image))
        {
            return {hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Could not write the declared contents of '" + tensor.name + "'"};
        }
    }
    return {};
}

/// @brief Writes the validation fill into every input of @p variantPack that
///        allocateVariantPack() did not declare.
///
/// Results keep their zero fill so a kernel that writes nothing is detectable. Types this
/// build cannot encode exactly also keep zeros rather than risk filling NaNs. The inputs in
/// @p positive -- read by an operation undefined on negative values -- are filled positive.
/// A declared input keeps its declared contents, written afresh with each candidate's buffers.
///
/// @p fills keeps each input's image (a function of seed, uid and size) across one problem's
/// candidates, uploaded afresh into each candidate's buffers, so a kernel that scribbles on an
/// input cannot pass its damage on. @p buffers provides `upload` as for allocateVariantPack().
template <typename TBuffers>
hipdnn_frontend::Error fillGraphInputs(const VariantPackPlan& plan,
                                       const std::unordered_map<int64_t, void*>& variantPack,
                                       TBuffers& buffers,
                                       uint64_t seed,
                                       const std::set<int64_t>& positive,
                                       const DeclaredContents& declared,
                                       std::map<int64_t, std::vector<uint8_t>>& fills)
{
    for(const auto& tensor : plan.tensors)
    {
        const auto buffer = variantPack.find(tensor.uid);
        if(tensor.produced || buffer == variantPack.end() || declared.values.count(tensor.uid) != 0)
        {
            continue; // A result, something not in the pack at all, or declared contents.
        }
        const auto bytes = static_cast<size_t>(tensor.bytes);
        auto cached = fills.find(tensor.uid);
        // Empty is the declined-type answer and is reused as such; a size that moved between
        // candidates is not the same image, so it is generated again rather than reused.
        if(cached == fills.end() || (!cached->second.empty() && cached->second.size() != bytes))
        {
            cached = fills
                         .insert_or_assign(tensor.uid,
                                           detail::inputFillImage(tensor.dataType,
                                                                  bytes,
                                                                  seed,
                                                                  tensor.uid,
                                                                  positive.count(tensor.uid) != 0))
                         .first;
        }
        const auto& image = cached->second;
        if(!image.empty() && !buffers.upload(tensor, buffer->second, image))
        {
            return {hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Could not fill a candidate's input"};
        }
    }
    return {};
}

} // namespace hipdnn_bench
