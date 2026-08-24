// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
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

/**
 * @file AttentionDenseNative.cpp
 * @brief The attention_dense engine's native half: applicability, selection, workspace,
 *        and dispatch for a rocKE-produced dense SDPA kernel.
 *
 * The kernel is `kernels/gfx942/attention_dense.py:build_gfx942_attention_dense`, a dense
 * flash-attention prefill whose ABI is four pointers plus a scalar:
 *
 *     (q_ptr, k_ptr, v_ptr, o_ptr, scale)
 *
 * matching `SdpaAttributes`' four required tensor uids plus `attn_scale_value`. The
 * descriptor names a `kpack` source, so `buildIngestorKernelCode()` loads an already
 * compiled code object and nothing in this file compiles.
 *
 * A rocKE kernel is compiled for one spec: `batch`, `seqlen_q`, `seqlen_kv`,
 * `num_query_heads`, `num_kv_heads`, `head_size`, `causal` and `dtype` are constants in
 * the code object and appear in its symbol name
 * (`rocke_attention_dense_d128_hq8_kv8_bn64_bf16_sq256_sk256_causal_lazyrs`), which splits
 * the checks in two:
 *
 *   * `graph_match` refuses topology and the modes the kernel cannot express. It runs
 *     once per (graph, device) and gates the whole engine.
 *   * `kernel_match` compares the baked geometry per candidate. A kernel built for one
 *     shape and launched on another reads past its tile and returns a wrong answer
 *     without faulting, so this is correctness rather than preference.
 *
 * The descriptor's `metadata` is the only place the baked spec is visible to the runtime,
 * so the packer carries it through verbatim and this file compares against it field by
 * field.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings. A
// descriptor cannot reference a C++ constant; the loader pre-flights every symbol a
// descriptor names, so a mismatch is reported at load rather than as a compile error.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.attention_dense.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.attention_dense.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.attention_dense.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.attention_dense.dispatch";

// KMD field names. Each mirrors a field of the rocKE spec that is baked into the binary.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view SEQLEN_Q_FIELD = "seqlen_q";
constexpr std::string_view SEQLEN_KV_FIELD = "seqlen_kv";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view BLOCK_N_FIELD = "block_n";

constexpr std::string_view QUERY_TOKEN = "attention_dense.query.uid";
constexpr std::string_view KEY_TOKEN = "attention_dense.key.uid";
constexpr std::string_view VALUE_TOKEN = "attention_dense.value.uid";
constexpr std::string_view OUTPUT_TOKEN = "attention_dense.output.uid";

/// SDPA operands are rank 4: [batch, heads, sequence, head_size].
constexpr uint32_t SDPA_RANK = 4;
constexpr int BATCH_AXIS = 0;
constexpr int HEAD_COUNT_AXIS = 1;
constexpr int SEQUENCE_AXIS = 2;
constexpr int HEAD_SIZE_AXIS = 3;

/// Launch geometry, restated here because the descriptor does not carry it: the kernel's
/// thread count is `(block_m / 32) * 64` and its grid is
/// `(ceil(seqlen_q / block_m), num_query_heads, batch)` -- see `attention_dense_grid` and
/// `attention_dense_block` in the rocKE module, whose shipped `_BLOCK_M` is 256.
constexpr int64_t DENSE_BLOCK_M = 256;
constexpr unsigned int DENSE_BLOCK_THREADS = 512;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched SDPA graph binds, in the kernel's argument order.
struct AttentionDenseBinding
{
    int64_t query = 0;
    int64_t key = 0;
    int64_t value = 0;
    int64_t output = 0;
};

/// The problem geometry read off the graph, in the same vocabulary the rocKE spec uses.
struct AttentionDenseProblem
{
    int64_t batch = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t headSize = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// The node this engine's matchers read, or nullptr when the graph is not a single SDPA
/// node. Shared so applicability and the operand read cannot disagree on which node they
/// read.
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

/// Does this rank-4 operand carry the BSHD strides the kernel bakes in?
///
/// The kernel derives its addresses from compile-time constants -- a token step is
/// `Hq * D`, a head step is `D` -- so an operand in any other memory order is still
/// indexed as BSHD, reads the wrong elements, and yields wrong values without faulting.
///
/// Dims are `[B, H, S, D]`, hipDNN's canonical SDPA order, while memory order is BSHD, so
/// the expected strides interleave heads within a token: batch `S*H*D`, head `D`, token
/// `H*D`, element 1.
bool hasBshdStrides(const data_objects::TensorAttributes* tensor)
{
    const auto* dims = tensor->dims();
    const auto* strides = tensor->strides();

    const int64_t heads = dims->Get(HEAD_COUNT_AXIS);
    const int64_t sequence = dims->Get(SEQUENCE_AXIS);
    const int64_t headSize = dims->Get(HEAD_SIZE_AXIS);

    return strides->Get(BATCH_AXIS) == sequence * heads * headSize
           && strides->Get(HEAD_COUNT_AXIS) == headSize
           && strides->Get(SEQUENCE_AXIS) == heads * headSize && strides->Get(HEAD_SIZE_AXIS) == 1;
}

/// Rank-4, positive extents, real memory. Runs on an unvalidated graph, so must be total:
/// a caller can present a tensor the frontend would have rejected.
bool isSupportedOperand(const data_objects::TensorAttributes* tensor)
{
    if(tensor == nullptr)
    {
        return false;
    }

    const auto* dims = tensor->dims();
    const auto* strides = tensor->strides();
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

    return !tensor->virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor);
}

/// Reads the problem geometry, or nullopt when this is not a graph the engine reads.
///
/// `o` is not required to carry populated dims: its shape is inferred by the frontend and
/// is not reliably populated while matching runs. Its layout is checked in `prepare()`
/// instead.
std::optional<AttentionDenseProblem> attentionProblem(const MatchContext& context)
{
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }

    const auto* query = findTensor(context, attributes->q_tensor_uid());
    const auto* key = findTensor(context, attributes->k_tensor_uid());
    const auto* value = findTensor(context, attributes->v_tensor_uid());
    if(!isSupportedOperand(query) || !isSupportedOperand(key) || !isSupportedOperand(value))
    {
        return std::nullopt;
    }

    // Layout. Safe to read only now that rank and extents are known good above.
    if(!hasBshdStrides(query) || !hasBshdStrides(key) || !hasBshdStrides(value))
    {
        return std::nullopt;
    }
    // Present and addressable, but its dims are the frontend's to fill in.
    if(findTensor(context, attributes->o_tensor_uid()) == nullptr)
    {
        return std::nullopt;
    }

    AttentionDenseProblem problem;
    problem.batch = query->dims()->Get(BATCH_AXIS);
    problem.numQueryHeads = query->dims()->Get(HEAD_COUNT_AXIS);
    problem.numKvHeads = key->dims()->Get(HEAD_COUNT_AXIS);
    problem.seqLenQ = query->dims()->Get(SEQUENCE_AXIS);
    problem.seqLenKv = key->dims()->Get(SEQUENCE_AXIS);
    problem.headSize = query->dims()->Get(HEAD_SIZE_AXIS);
    problem.dataType = query->data_type();

    // K and V must agree with Q and with each other on everything the kernel bakes in.
    // Reading only Q would let a mismatched pair through to a kernel that indexes both
    // with Q's geometry.
    if(key->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize
       || value->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        return std::nullopt;
    }
    if(value->dims()->Get(HEAD_COUNT_AXIS) != problem.numKvHeads
       || value->dims()->Get(SEQUENCE_AXIS) != problem.seqLenKv)
    {
        return std::nullopt;
    }
    if(key->dims()->Get(BATCH_AXIS) != problem.batch
       || value->dims()->Get(BATCH_AXIS) != problem.batch)
    {
        return std::nullopt;
    }
    if(key->data_type() != problem.dataType || value->data_type() != problem.dataType)
    {
        return std::nullopt;
    }

    // GQA: the kernel derives its group size by integer division, so a non-divisible
    // pair would silently drop the remainder heads.
    if(problem.numKvHeads <= 0 || problem.numQueryHeads % problem.numKvHeads != 0)
    {
        return std::nullopt;
    }

    return problem;
}

/**
 * @brief Engine-level applicability: is this a single dense SDPA node the engine can
 *        launch at all?
 *
 * `buildCatalog` runs this once and returns an empty catalog on `nullopt`, so everything
 * refused here is refused for every pack in the engine. Each check is a property of the
 * operation rather than of any one kernel.
 *
 * `SdpaAttributes` carries two dozen optional operands that all default to absent, so
 * every mode the kernel does not implement is refused explicitly: an unchecked mode is
 * accepted and then silently not performed.
 */
std::optional<BoundTokens> attentionDenseGraphMatches(const MatchContext& context)
{
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }

    const auto problem = attentionProblem(context);
    if(!problem.has_value())
    {
        return std::nullopt;
    }

    // Masks and biases the dense kernel does not emit.
    if(attributes->alibi_mask() || attributes->padding_mask()
       || attributes->attn_mask_tensor_uid().has_value()
       || attributes->block_mask_tensor_uid().has_value()
       || attributes->sink_token_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Paged KV belongs to the tiled family, not the dense arm.
    if(attributes->page_table_k_tensor_uid().has_value()
       || attributes->page_table_v_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Variable-length batches: `supports_attention_dense` rejects varlen and ragged, so
    // per-sequence lengths have no kernel behind them.
    if(attributes->seq_len_q_tensor_uid().has_value()
       || attributes->seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Dropout in any of its spellings.
    if((attributes->dropout_probability().has_value()
        && attributes->dropout_probability().value() != 0.0F)
       || attributes->dropout_mask_tensor_uid().has_value()
       || attributes->dropout_scale_tensor_uid().has_value()
       || attributes->seed_tensor_uid().has_value() || attributes->offset_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Auxiliary outputs: the kernel writes O and nothing else.
    if((attributes->generate_stats().has_value() && attributes->generate_stats().value())
       || attributes->stats_tensor_uid().has_value() || attributes->max_tensor_uid().has_value()
       || attributes->sum_exp_tensor_uid().has_value()
       || attributes->rng_dump_tensor_uid().has_value()
       || attributes->amax_s_tensor_uid().has_value()
       || attributes->amax_o_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Quantized pipelines.
    if(attributes->descale_q_tensor_uid().has_value()
       || attributes->descale_k_tensor_uid().has_value()
       || attributes->descale_v_tensor_uid().has_value()
       || attributes->descale_s_tensor_uid().has_value()
       || attributes->scale_s_tensor_uid().has_value()
       || attributes->scale_o_tensor_uid().has_value()
       || attributes->mma_core_mode() != data_objects::DataType::UNSET)
    {
        return std::nullopt;
    }

    // The scale reaches the kernel as a launch scalar, so it must be a host value. The
    // kernel's ABI has no argument for a scale living in device memory.
    if(!attributes->attn_scale_value().has_value() || attributes->scale_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(QUERY_TOKEN)] = attributes->q_tensor_uid();
    bound[std::string(KEY_TOKEN)] = attributes->k_tensor_uid();
    bound[std::string(VALUE_TOKEN)] = attributes->v_tensor_uid();
    bound[std::string(OUTPUT_TOKEN)] = attributes->o_tensor_uid();
    return bound;
}

std::string dataTypeName(data_objects::DataType dataType)
{
    return data_objects::EnumNameDataType(dataType);
}

/// The descriptor's dtype spelling for a graph dtype, or nullopt for one the kernel
/// cannot be built for. rocKE spells these `bf16`/`fp16` in its spec; the descriptor
/// carries hipDNN's own enum spelling so this comparison stays in one vocabulary.
std::optional<std::string> descriptorDtypeFor(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::BFLOAT16:
    case data_objects::DataType::HALF:
        return dataTypeName(dataType);
    default:
        return std::nullopt;
    }
}

/**
 * @brief Kernel-scoped applicability: does this kernel's baked spec fit the problem?
 *
 * Every field compared here is a compile-time constant in the rocKE code object, so a
 * candidate that disagrees indexes the wrong extents and produces a wrong answer without
 * faulting. `graph_match` cannot make this call: it runs before any candidate is in hand.
 */
bool attentionDenseKernelMatches(const MatchContext& context,
                                 const BoundTokens& /*bound*/,
                                 const KernelDefinition& kernel)
{
    const auto problem = attentionProblem(context);
    if(!problem.has_value())
    {
        return false;
    }

    const auto dtype = descriptorDtypeFor(problem->dataType);
    if(!dtype.has_value())
    {
        return false;
    }

    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const int64_t causal
        = attributes->causal_mask() || attributes->causal_mask_bottom_right() ? 1 : 0;

    return kernel.getStringMetadata(std::string(DTYPE_FIELD)) == *dtype
           && kernel.getIntMetadata(std::string(HEAD_SIZE_FIELD)) == problem->headSize
           && kernel.getIntMetadata(std::string(NUM_QUERY_HEADS_FIELD)) == problem->numQueryHeads
           && kernel.getIntMetadata(std::string(NUM_KV_HEADS_FIELD)) == problem->numKvHeads
           && kernel.getIntMetadata(std::string(SEQLEN_Q_FIELD)) == problem->seqLenQ
           && kernel.getIntMetadata(std::string(SEQLEN_KV_FIELD)) == problem->seqLenKv
           && kernel.getIntMetadata(std::string(CAUSAL_FIELD)) == causal;
}

/// Ranks the survivors. Geometry is part of the match, so every candidate reaching here
/// fits exactly and the only axis left is the KV tile: a larger `block_n` amortises the
/// per-tile loop over more keys. `block_n` is also the knob the UED exposes.
double attentionDenseScore(const MatchContext& /*context*/,
                           const BoundTokens& /*bound*/,
                           const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.getIntMetadata(std::string(BLOCK_N_FIELD)));
}

/**
 * @brief Re-reads the operand bindings a match established.
 *
 * @throws HipdnnPluginException if the graph is not one this matcher accepts.
 */
AttentionDenseBinding attentionDenseBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "attention_dense dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };

    return {read(QUERY_TOKEN), read(KEY_TOKEN), read(VALUE_TOKEN), read(OUTPUT_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The loaded kernel plus everything its launch needs, resolved once and owning nothing
/// that points back into the MatchContext it was prepared from.
class PreparedAttentionDense : public PreparedDispatch
{
public:
    PreparedAttentionDense(std::unique_ptr<compilation::ICompiledProgram> program,
                           std::unique_ptr<compilation::IRunnableKernel> kernel,
                           AttentionDenseBinding binding,
                           float scale)
        : _program(std::move(program))
        , _kernel(std::move(kernel))
        , _binding(binding)
        , _scale(scale)
    {
    }

    const compilation::IRunnableKernel& kernel() const
    {
        return *_kernel;
    }

    const AttentionDenseBinding& binding() const
    {
        return _binding;
    }

    float scale() const
    {
        return _scale;
    }

private:
    // The runnable kernel is a view into its program's module; both are held for the
    // plan's lifetime.
    std::unique_ptr<compilation::ICompiledProgram> _program;
    std::unique_ptr<compilation::IRunnableKernel> _kernel;
    AttentionDenseBinding _binding;
    float _scale;
};

/// A 1x1x1x1 NCHW-ordered tensor, existing only to satisfy KernelCompileOptions.
///
/// That constructor unconditionally classifies its tensor's layout as NCHW or NHWC and
/// throws on anything else, and it dereferences the pointer, so nullptr is not an
/// alternative. This engine's operands are BSHD (stride order 0,2,1,3), which is neither.
/// The KPACK branch of buildIngestorKernelCode never reads the options it produces -- the
/// code object is already compiled -- so a stand-in is sufficient.
///
/// An embedded_source kernel in this pack would need the real tensor instead, in a layout
/// the classifier accepts.
const data_objects::TensorAttributes& layoutNeutralTensor()
{
    static const std::vector<int64_t> s_dims{1, 1, 1, 1};
    static const std::vector<int64_t> s_strides{1, 1, 1, 1};
    static const flatbuffers::DetachedBuffer s_buffer = [] {
        flatbuffers::FlatBufferBuilder builder;
        auto attrs = data_objects::CreateTensorAttributesDirect(builder,
                                                                /*uid=*/0,
                                                                /*name=*/nullptr,
                                                                data_objects::DataType::BFLOAT16,
                                                                &s_strides,
                                                                &s_dims);
        builder.Finish(attrs);
        return builder.Release();
    }();
    return *flatbuffers::GetRoot<data_objects::TensorAttributes>(s_buffer.data());
}

/**
 * @brief The native dispatch behind this pack's UDD: loads the rocKE code object out of
 *        its kpack, sizes the launch, and runs it.
 *
 * Splits per RFC 0017 §8.5 like the other packs: `prepare()` resolves everything while
 * `context` is valid, `launch()` only resolves device pointers and launches, so nothing
 * mutates once prepared and concurrent execution is safe.
 */
class AttentionDenseDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    /// @param kpackLoader Same must-outlive contract. Which of the two is consulted is
    /// the selected kernel's source kind, decided in buildIngestorKernelCode -- for this
    /// engine that is always the kpack path.
    AttentionDenseDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                  const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        // The kernel keeps K/V tiles in LDS and the online-softmax state in registers;
        // its ABI has no scratch pointer, so there is nothing to size.
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = attentionDenseBinding(bound);

        const auto problem = attentionProblem(context);
        if(!problem.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "attention_dense dispatch prepared for a graph its matcher would decline");
        }

        const auto* attributes = sdpaNode(context);
        if(attributes == nullptr || !attributes->attn_scale_value().has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "attention_dense dispatch prepared for a graph carrying no scale value");
        }

        // O's layout is checked here rather than in the matcher: its shape is inferred by
        // the frontend and is not reliably populated while matching runs. By dispatch time
        // it is populated, and the kernel writes O with the same baked BSHD addressing it
        // reads its inputs with, so a mismatch would silently corrupt the output.
        const auto* output = findTensor(context, attributes->o_tensor_uid());
        if(output == nullptr || !isSupportedOperand(output) || !hasBshdStrides(output))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "attention_dense writes its output as BSHD (batch, sequence, head, "
                "element), and this graph's output tensor declares a different layout");
        }

        // Built from a stand-in tensor rather than the real query tensor; see
        // layoutNeutralTensor().
        const compilation::KernelCompileOptions options(&layoutNeutralTensor(),
                                                        context.deviceProperties.gcnArchName);

        auto code
            = buildIngestorKernelCode(_kernelCompiler, _kpackLoader, context, kernel, options);

        // Launch geometry, mirroring `attention_dense_grid` / `attention_dense_block`:
        // one CTA per (query block, query head, batch), each of `block_m / 32` wave64s.
        // The extents come from the graph, which `kernel_match` has already proven equal
        // to the kernel's baked ones.
        const int64_t queryBlocks = (problem->seqLenQ + DENSE_BLOCK_M - 1) / DENSE_BLOCK_M;

        code.kernel->setBlockSize(DENSE_BLOCK_THREADS, 1, 1);
        code.kernel->setGridSize(static_cast<unsigned int>(queryBlocks),
                                 static_cast<unsigned int>(problem->numQueryHeads),
                                 static_cast<unsigned int>(problem->batch));

        return std::make_unique<PreparedAttentionDense>(std::move(code.program),
                                                        std::move(code.kernel),
                                                        binding,
                                                        attributes->attn_scale_value().value());
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedDense = dynamic_cast<const PreparedAttentionDense&>(prepared);
        const auto& binding = preparedDense.binding();

        const auto query
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.query, deviceBuffers, numDeviceBuffers);
        const auto key
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.key, deviceBuffers, numDeviceBuffers);
        const auto value
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.value, deviceBuffers, numDeviceBuffers);
        const auto output
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.output, deviceBuffers, numDeviceBuffers);

        // Argument order is the rocKE ABI: q, k, v, o, scale.
        const float scale = preparedDense.scale();
        preparedDense.kernel().launch(
            handle.getStream(), query.ptr, key.ptr, value.ptr, output.ptr, scale);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

/// This pack's kpack module cache, process-lifetime. Internal linkage, unlike the
/// pointwise pack's, which is exposed so a test can observe two dispatches over one
/// (archive, toc_key, arch) sharing a single hipModule_t. Widen it when something outside
/// this file needs it.
compilation::KpackModuleCache& attentionDenseKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning
/// pointer to it, but a provider's Container is created and destroyed per handle, so it
/// (and the compiler and loader it holds) must outlive every Container.
const AttentionDenseDispatchHandler& attentionDenseDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(attentionDenseKpackModuleCache());
    static const AttentionDenseDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerAttentionDenseSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &attentionDenseGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &attentionDenseKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &attentionDenseScore);
    scope.add(std::string(DISPATCH_SYMBOL), &attentionDenseDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
