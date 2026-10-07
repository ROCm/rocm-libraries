// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
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

namespace hip_kernel_provider::kernel_ingestor_engine
{
using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace
{
constexpr uint32_t BATCH_AXIS = 0, HEAD_AXIS = 1, SEQ_AXIS = 2, HEAD_SIZE_AXIS = 3;
constexpr uint32_t SDPA_RANK = 4;
constexpr int64_t UNBOUNDED = -1;
constexpr int64_t MAX_CAPACITY = 65536;
const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// The node this engine's matchers read, or nullptr if the graph is not a single
/// SDPA-forward node.
const data_objects::SdpaAttributes* sdpaNode(const MatchContext& context)
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

/// The product of @p factors, or nullopt when it does not fit in int64_t.
///
/// Every product of graph-controlled extents goes through this. The extents are whatever
/// the graph claims, so an unchecked product is signed overflow -- undefined, and in
/// practice a wrapped value that can equal a stride or pass a bound it should fail.
std::optional<int64_t> checkedProduct(std::initializer_list<int64_t> factors)
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

/// Total over an UNVALIDATED graph: rank, stride/dim agreement and positive extents,
/// checked before anything indexes an axis.
bool isWellFormedOperand(const data_objects::TensorAttributes& tensor)
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
 * @brief Is this tensor's memory BSHD -- token-major, head varying fastest?
 *
 * The kernel bakes this layout and there are no stride kernargs -- the builder computes
 * strides from `Hq * D` and `Hkv * D` -- so a differently-strided tensor is read as if it
 * were this one.
 *
 * Unit-extent axes are exempt: a stride multiplies an index that is always 0 when the
 * extent is 1. A single-head tensor is byte-identically BSHD and BHSD while the two
 * spellings disagree on strides[H], so a strict compare would decline a graph the kernel
 * serves perfectly.
 *
 * An expected stride too large for int64_t is one no stride can equal, so that axis
 * fails unless it is unit-extent.
 */
bool hasBshdStrides(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();

    const int64_t heads = dims->Get(HEAD_AXIS);
    const int64_t sequence = dims->Get(SEQ_AXIS);
    const int64_t headSize = dims->Get(HEAD_SIZE_AXIS);

    const auto axisOk = [&](uint32_t axis, std::optional<int64_t> expected) {
        return dims->Get(axis) == 1 || (expected.has_value() && strides->Get(axis) == *expected);
    };

    return axisOk(BATCH_AXIS, checkedProduct({sequence, heads, headSize}))
           && axisOk(HEAD_AXIS, headSize) && axisOk(SEQ_AXIS, checkedProduct({heads, headSize}))
           && axisOk(HEAD_SIZE_AXIS, 1);
}

/// The mask kinds this engine serves. No variant in this catalog carries a non-zero
/// sliding_window, so a windowed mask has no spelling here.
enum class MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL = 1,
    BOTTOM_RIGHT_CAUSAL = 2
};

/**
 * @brief Which mask the graph is asking for.
 *
 * A real bound wins over the deprecated booleans: a graph that sets a boolean AND
 * carries a bound is asking for a windowed mask.
 */
std::optional<MaskType> maskTypeFor(const data_objects::SdpaAttributes& attributes)
{
    const bool topLeftDeprecated = attributes.causal_mask();
    const bool bottomRightDeprecated = attributes.causal_mask_bottom_right();

    if(topLeftDeprecated && bottomRightDeprecated)
    {
        return std::nullopt;
    }

    const int64_t left
        = attributes.left_bound().has_value() ? attributes.left_bound().value() : UNBOUNDED;
    const int64_t right
        = attributes.right_bound().has_value() ? attributes.right_bound().value() : UNBOUNDED;

    // A non-zero right bound creates a bidirectional window the kernel cannot serve:
    // the compiled kernel is hard-causal (upper mask only) and has no right-bound field.
    // Decline early so the graph is not silently served with wrong numerics.
    if(right != UNBOUNDED && right != 0)
    {
        return std::nullopt;
    }

    // A bounded left edge is a window whatever the booleans say, and no shipped variant
    // carries a non-zero sliding_window. Serving one on a causal binary would apply the
    // wrong mask with no error.
    if(left != UNBOUNDED)
    {
        return std::nullopt;
    }

    if(topLeftDeprecated)
    {
        return MaskType::TOP_LEFT_CAUSAL;
    }
    if(bottomRightDeprecated)
    {
        return MaskType::BOTTOM_RIGHT_CAUSAL;
    }

    // Both bounds are now either unset or zero: unset on the right is an unmasked graph,
    // zero is a diagonal with no band, whose alignment picks the causal corner.
    if(right == UNBOUNDED)
    {
        return MaskType::NO_MASK;
    }
    return attributes.diagonal_alignment() == data_objects::DiagonalAlignment::BOTTOM_RIGHT
               ? MaskType::BOTTOM_RIGHT_CAUSAL
               : MaskType::TOP_LEFT_CAUSAL;
}

// One step per execution. The caller owns the cache and supplies its visible length.
// Fixed builder parameters: BF16, D128, Hq8/Hkv2, page64, one wave, one sequence.
std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto* a = sdpaNode(context);
    if(a == nullptr)
    {
        return std::nullopt;
    }
    if(context.graph.getNodeWrapper(0).computeDataType() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    const auto& attributes = *a;
    const auto* q = findTensor(context, a->q_tensor_uid());
    const auto* k = findTensor(context, a->k_tensor_uid());
    const auto* v = findTensor(context, a->v_tensor_uid());
    const auto* o = findTensor(context, a->o_tensor_uid());
    for(const auto* t : {q, k, v, o})
    {
        if(t == nullptr || !isWellFormedOperand(*t) || !hasBshdStrides(*t)
           || t->data_type() != data_objects::DataType::BFLOAT16 || t->dims()->Get(0) != 1
           || t->dims()->Get(3) != 128)
        {
            return std::nullopt;
        }
    }
    const int64_t capacity = k->dims()->Get(2);
    if(q->dims()->Get(1) != 8 || o->dims()->Get(1) != 8 || q->dims()->Get(2) != 1
       || o->dims()->Get(2) != 1 || k->dims()->Get(1) != 2 || v->dims()->Get(1) != 2
       || v->dims()->Get(2) != capacity || capacity > MAX_CAPACITY || capacity % 64 != 0)
    {
        return std::nullopt;
    }
    if(!a->seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    const auto* length = findTensor(context, a->seq_len_kv_tensor_uid().value());
    if(length == nullptr || length->data_type() != data_objects::DataType::INT32
       || !isWellFormedOperand(*length))
    {
        return std::nullopt;
    }
    for(const auto dim : *length->dims())
    {
        if(dim != 1)
        {
            return std::nullopt;
        }
    }
    const auto mask = maskTypeFor(*a);
    if(!mask.has_value() || *mask == MaskType::TOP_LEFT_CAUSAL)
    {
        return std::nullopt;
    }
    // Decline optional attributes the fixed kernel cannot honour.

    // Additive attention bias.
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Device-resident scale: the ABI takes `scale` as an f32 kernarg.
    if(attributes.scale_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Query lengths are fixed; only the KV length is dynamic.
    if(attributes.seq_len_q_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Dropout.
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        return std::nullopt;
    }
    // Paged KV.
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value())
    {
        return std::nullopt;
    }
    // Block-sparse, and attention SINKS.
    if(attributes.block_mask_tensor_uid().has_value()
       || attributes.sink_token_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // FP8 quantization.
    if(attributes.descale_q_tensor_uid().has_value()
       || attributes.descale_k_tensor_uid().has_value()
       || attributes.descale_v_tensor_uid().has_value()
       || attributes.descale_s_tensor_uid().has_value()
       || attributes.scale_s_tensor_uid().has_value() || attributes.scale_o_tensor_uid().has_value()
       || attributes.amax_s_tensor_uid().has_value() || attributes.amax_o_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Auxiliary softmax outputs. generate_stats is optional<bool>; explicit false is fine.
    if(attributes.stats_tensor_uid().has_value() || attributes.max_tensor_uid().has_value()
       || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value()
       || (attributes.generate_stats().has_value() && attributes.generate_stats().value()))
    {
        return std::nullopt;
    }
    // ALiBi slopes are unsupported; the supplied KV length handles padding.
    if(attributes.alibi_mask())
    {
        return std::nullopt;
    }
    // mma_core_mode is the MMA operand precision. This kernel's MFMA operands are the
    // graph's own fp16/bf16 inputs, so UNSET (the provider's choice), HALF and BFLOAT16
    // describe what it runs. The mode is not compared with the graph dtype: HALF is
    // accepted on bf16 graphs too, because the cuDNN-compat shim writes HALF whenever the
    // caller leaves the field unset. An allow-list, so an enum value added later declines
    // until judged.
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }
    // `implementation` is an execution-strategy hint. AUTO leaves the choice to the provider.
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        return std::nullopt;
    }

    // The softmax scale is a REQUIRED launch argument with no default.
    if(!attributes.attn_scale_value().has_value())
    {
        return std::nullopt;
    }
    const float scale = attributes.attn_scale_value().value();
    if(!std::isfinite(scale))
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound["q"] = a->q_tensor_uid();
    bound["k"] = a->k_tensor_uid();
    bound["v"] = a->v_tensor_uid();
    bound["o"] = a->o_tensor_uid();
    bound["length"] = a->seq_len_kv_tensor_uid().value();
    bound["capacity"] = capacity;
    int32_t bits = 0;
    std::memcpy(&bits, &scale, sizeof(bits));
    bound["scale"] = static_cast<int64_t>(bits);
    return bound;
}

bool kernelMatches(const MatchContext& /*unused*/,
                   const BoundTokens& /*unused*/,
                   const KernelDefinition& kernel)
{
    const auto it = kernel.metadata.find("dtype");
    if(it == kernel.metadata.end())
    {
        return false;
    }
    const auto* dtype = std::get_if<std::string>(&it->second);
    return dtype != nullptr && *dtype == "BF16";
}

int64_t read(const BoundTokens& bound, std::string_view key)
{
    const auto value = tryGetBoundInt(bound, key);
    if(!value)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "AttentionDecode: missing binding " + std::string(key));
    }
    return *value;
}

std::vector<KernelArgument> signature()
{
    return {{"global_buffer", 8, 0, "output_ptr"},
            {"global_buffer", 8, 8, "query_ptr"},
            {"global_buffer", 8, 16, "key_cache_ptr"},
            {"global_buffer", 8, 24, "value_cache_ptr"},
            {"global_buffer", 8, 32, "sink_ptr"},
            {"global_buffer", 8, 40, "block_tables_ptr"},
            {"global_buffer", 8, 48, "seq_lens_ptr"},
            {"global_buffer", 8, 56, "alibi_slopes_ptr"},
            {"global_buffer", 8, 64, "qq_bias_ptr"},
            {"global_buffer", 8, 72, "query_start_len_ptr"},
            {"by_value", 4, 80, "scale"},
            {"by_value", 4, 84, "k_scale"},
            {"by_value", 4, 88, "v_scale"},
            {"by_value", 4, 92, "out_scale"},
            {"by_value", 4, 96, "softcap"},
            {"by_value", 4, 100, "num_seqs"},
            {"by_value", 4, 104, "block_table_stride"},
            {"by_value", 4, 108, "qq_bias_stride_0"}};
}

class PreparedAttentionDecode : public PreparedDispatch
{
public:
    PreparedAttentionDecode(IngestorKernelCode kernelCode, BoundTokens tokens)
        : code(std::move(kernelCode))
        , bound(std::move(tokens))
        , table(static_cast<size_t>(read(bound, "capacity") / 64) + 4, 0)
    {
        table[1] = 1; // cu_q = [0,1]; table starts at byte 16.
        std::iota(table.begin() + 4, table.end(), 0);
        const auto bits = static_cast<int32_t>(read(bound, "scale"));
        std::memcpy(&scale, &bits, sizeof(scale));
    }
    IngestorKernelCode code;
    BoundTokens bound;
    std::vector<int32_t> table;
    float scale = 0;
};

void checkHip(hipError_t status)
{
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       std::string("AttentionDecode: ")
                                                           + hipGetErrorString(status));
    }
}

class AttentionDecodeDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit AttentionDecodeDispatchHandler(const compilation::KpackKernelLoader& loader)
        : _loader(loader)
    {
    }
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& bound,
                          const KernelDefinition& /*kernel*/) const override
    {
        return static_cast<size_t>(read(bound, "capacity") / 64 + 4) * sizeof(int32_t);
    }
    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        if(!graphMatches(context) || !kernelMatches(context, bound, kernel))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "AttentionDecode: unsupported graph/kernel");
        }
        auto code = buildIngestorKernelCode(_loader, context, kernel, signature());
        code.setBlockSize(64, 1, 1);
        code.setGridSize(2, 1, 1); // KV heads, query blocks, z; same on gfx942 and gfx950.
        return std::make_unique<PreparedAttentionDecode>(std::move(code), bound);
    }
    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* buffers,
                uint32_t count,
                void* workspace) const override
    {
        const auto& p = dynamic_cast<const PreparedAttentionDecode&>(prepared);
        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "AttentionDecode: workspace is required");
        }
        const auto ptr = [&](std::string_view key) {
            auto* result
                = hipdnn_plugin_sdk::findDeviceBuffer(read(p.bound, key), buffers, count).ptr;
            if(result == nullptr)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM, "AttentionDecode: null tensor buffer");
            }
            return result;
        };
        auto* q = ptr("q");
        auto* k = ptr("k");
        auto* v = ptr("v");
        auto* o = ptr("o");
        auto* length = ptr("length");
        auto* offsets = static_cast<int32_t*>(workspace);
        // Prepared storage is immutable and lives as long as the plan. Each execution
        // has its own workspace, so concurrent streams share no mutable adapter state.
        checkHip(hipMemcpyAsync(workspace,
                                p.table.data(),
                                p.table.size() * sizeof(int32_t),
                                hipMemcpyHostToDevice,
                                handle.getStream()));
        const void* unused = nullptr;
        // Caller supplies 1 <= length <= capacity and zero-initializes the full K/V
        // cache before each new sequence. Length masks scores, not full-tile V loads.
        p.code.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    o,
                    q,
                    k,
                    v,
                    unused,
                    offsets + 4,
                    length,
                    unused,
                    unused,
                    offsets,
                    p.scale,
                    1.0F,
                    1.0F,
                    1.0F,
                    0.0F,
                    int32_t{1},
                    static_cast<int32_t>(p.table.size() - 4),
                    int32_t{0});
    }

private:
    const compilation::KpackKernelLoader& _loader;
};
compilation::KpackModuleCache& moduleCache()
{
    static compilation::KpackModuleCache s_cache;
    return s_cache;
}
} // namespace
void resetAttentionDecodeModuleCache()
{
    moduleCache().clear();
}
void registerAttentionDecodeSymbols(SymbolScope<Handle>& scope)
{
    static const compilation::KpackKernelLoader s_loader(moduleCache());
    static const AttentionDecodeDispatchHandler s_handler(s_loader);
    scope.add("hipkernel.attention_decode.graph_match", &graphMatches);
    scope.add("hipkernel.attention_decode.kernel_match", &kernelMatches);
    scope.add("hipkernel.attention_decode.dispatch", &s_handler);
}
} // namespace hip_kernel_provider::kernel_ingestor_engine
#endif
