// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a pack that
// names a kpack archive nothing staged.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaCommon.hpp"

/**
 * @file FlydslSdpaAiwNative.cpp
 * @brief The engine of AOTriton's FlyDSL gfx1201 attention kernel: its kernel matcher, score
 *        and dispatch, and the one function that registers them.
 *
 * The kernel is vendored under `flydsl/kernels_src/kernels/attention_rdna4/` (its README
 * lists the modifications that bring it to the RDNA kernel's coverage) and built for
 * gfx12-generic. A kernel family of its own -- another argument list, another launch, its
 * own tuning table -- so an engine of its own, `hipkernel:flydsl_sdpa_aiw`, beside the RDNA
 * prefill and decode engines. It serves the same graphs they do, so it applies their graph
 * matcher (FlydslSdpaRdnaCommon.hpp) under its own symbol; which engine runs a graph several
 * serve is the heuristic's to decide, not a rule here.
 *
 * Each object bakes dtype, its head-dim tile (exact, or padded: any head_dim up to the tile
 * that is a multiple of 8, the real one a runtime argument), V's head_dim, whether a mask or
 * window applies, the additive mask and its dtype, and the attention sink. Batch, both
 * sequence lengths, the three head counts, every stride, the window, the scale and the LSE
 * layout are kernel arguments.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

using namespace flydsl_sdpa_rdna;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_aiw.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_aiw.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_sdpa_aiw.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_aiw.dispatch";

/// 1 for an object that reads an f32 additive mask, 0 for one in the query's dtype.
constexpr std::string_view BIAS_F32_FIELD = "bias_f32";
/// 1 for an object that reads the mask's key axis with any stride, 0 for a contiguous one.
constexpr std::string_view BIAS_KSTRIDED_FIELD = "bias_kstrided";
/// The workgroup size the object was built for (its tuning table's).
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";

/// The kernel's grid is (query heads, query tiles, batch): the y and z extents are 16-bit
/// in practice on the dispatch path, and x is 32-bit.
constexpr int64_t GRID_YZ_LIMIT = 65535;

/// Whether the graph needs the masked object: any bound, on either side. The non-masked
/// object applies no window at all.
bool needsMask(const FlydslSdpaRdnaBinding& binding)
{
    return binding.rightBound != UNBOUNDED || binding.leftBound != UNBOUNDED;
}

/// The bound the kernel treats as "no bound on this side": past every key and query, so a
/// row's range reaches the whole sequence, and small enough that i + bound stays in int32.
int64_t unboundedWindow(const SdpaProblem& problem)
{
    return problem.seqLenQ + problem.seqLenKv;
}

/// hipDNN's bounds as the kernel's top-left window: key j is kept iff
/// i - window_left <= j <= i + window_right. hipDNN keeps kv in
/// [q + align - left, q + align + right], with align = Skv - Sq at the bottom-right corner.
std::array<int64_t, 2> windowFor(const FlydslSdpaRdnaBinding& binding, const SdpaProblem& problem)
{
    const int64_t align = binding.alignBottomRight != 0 ? problem.seqLenKv - problem.seqLenQ : 0;
    const int64_t unbounded = unboundedWindow(problem);
    const int64_t left = binding.leftBound == UNBOUNDED ? unbounded : binding.leftBound - align;
    const int64_t right = binding.rightBound == UNBOUNDED ? unbounded : binding.rightBound + align;
    return {left, right};
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked metadata fit the graph?
 *
 * dtype; the head-dim tile (exact, or padded up to it); V's head_dim; whether a bound
 * applies; the mask, its dtype and its key stride; the sink. The window, the head counts,
 * every stride and the LSE layout are runtime arguments.
 */
bool flydslSdpaAiwKernelMatches(const MatchContext& context,
                                const BoundTokens& bound,
                                const KernelDefinition& kernel)
{
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const auto* q = findTensor(context, attributes->q_tensor_uid());
    const auto* k = findTensor(context, attributes->k_tensor_uid());
    const auto* v = findTensor(context, attributes->v_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr)
    {
        return false;
    }
    auto problem = problemFor(*q, *k);
    problem.numVHeads = v->dims()->Get(HEAD_AXIS);
    problem.headDimV = v->dims()->Get(HEAD_DIM_AXIS);

    const auto dataTypeName = kernelDataTypeName(problem.dataType);
    if(!dataTypeName.has_value() || stringMetadata(kernel, DTYPE_FIELD) != *dataTypeName)
    {
        return false;
    }

    // An exact object serves its head_dim; a padded one any multiple of 8 up to its tile,
    // with V as wide as Q and K. Padded tiers rank below the exact objects.
    const auto bakedHeadDim = integerMetadata(kernel, HEAD_DIM_FIELD);
    const auto headDimMax = integerMetadata(kernel, HEAD_DIM_MAX_FIELD);
    const auto bakedHeadDimV = integerMetadata(kernel, HEAD_DIM_V_FIELD);
    if(!bakedHeadDim.has_value() || !headDimMax.has_value() || !bakedHeadDimV.has_value())
    {
        return false;
    }
    if(*bakedHeadDim == GENERIC_HEAD_DIM)
    {
        if(problem.headDim % GENERIC_HEAD_DIM_ALIGN != 0 || problem.headDim > *headDimMax
           || problem.headDimV != problem.headDim)
        {
            return false;
        }
    }
    else if(*bakedHeadDim != problem.headDim || *bakedHeadDimV != problem.headDimV)
    {
        return false;
    }

    const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
    const auto threads = integerMetadata(kernel, BLOCK_THREADS_FIELD);
    if(!blockM.has_value() || *blockM <= 0 || !threads.has_value() || *threads <= 0)
    {
        return false;
    }
    const auto queryTiles = (problem.seqLenQ + *blockM - 1) / *blockM;
    if(queryTiles > GRID_YZ_LIMIT || problem.batch > GRID_YZ_LIMIT
       || problem.numHeads > INT32_LIMIT)
    {
        return false;
    }

    const auto binding = flydslSdpaRdnaBinding(bound);
    if(integerMetadata(kernel, CAUSAL_FIELD) != (needsMask(binding) ? 1 : 0))
    {
        return false;
    }

    // The additive mask: an object built for it, in its dtype; a key axis the object can
    // read (contiguous, unless the object reads any stride).
    const auto hasBias = integerMetadata(kernel, HAS_BIAS_FIELD);
    if(hasBias != (binding.bias == NO_BIAS ? 0 : 1))
    {
        return false;
    }
    if(binding.bias != NO_BIAS)
    {
        const auto* biasTensor = findTensor(context, binding.bias);
        if(biasTensor == nullptr)
        {
            return false;
        }
        const auto strides = servableBiasStrides(*biasTensor,
                                                 problem.batch,
                                                 problem.numHeads,
                                                 problem.seqLenQ,
                                                 problem.seqLenKv,
                                                 problem.dataType);
        const bool f32 = biasTensor->data_type() == data_objects::DataType::FLOAT;
        if(!strides.has_value() || integerMetadata(kernel, BIAS_F32_FIELD) != (f32 ? 1 : 0)
           || ((*strides)[3] != 1 && integerMetadata(kernel, BIAS_KSTRIDED_FIELD) != 1))
        {
            return false;
        }
    }

    return integerMetadata(kernel, HAS_SINK_FIELD) == (binding.sink == NO_SINK ? 0 : 1);
}

/// Ranks candidates that survived kernelMatches by their descriptor `priority`: an exact
/// head_dim above every padded tier, the narrower tier above the wider.
double flydslSdpaAiwScore(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/**
 * @brief The kernel's 50 parameters, mirroring launch() one for one (generators/gen_sdpa_aiw.py
 *        `_PARAMS`, with their names in each object's manifest).
 *
 * Q, K, V, B, O, LSE and four varlen pointers; varlen_bits, num_seqlens, max_seqlen_q,
 * max_seqlen_k, window_left, window_right (i32); two dropout pointers and an i64 offset, two
 * more pointers; idropout_p (i32), dropout_scale (f32), num_head_q, num_head_k, hdim_qk,
 * hdim_vo (i32), sm_scale (f32); the (batch, head, sequence) strides of Q, K, V and O and the
 * mask's batch, head and query strides (i64); SINK and its head stride; the mask's key
 * stride; num_head_v (i32); the LSE's batch, head and sequence strides (i64).
 */
const std::vector<KernelArgument>& flydslSdpaAiwSignature()
{
    static const std::vector<KernelArgument> s_signature = [] {
        const KernelArgument pointer{"global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
        const KernelArgument i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
        const KernelArgument f32{"by_value", static_cast<uint32_t>(sizeof(float)), 0, ""};
        const KernelArgument i64{"by_value", static_cast<uint32_t>(sizeof(int64_t)), 0, ""};
        std::vector<KernelArgument> signature(10, pointer);
        signature.insert(signature.end(), 6, i32);
        signature.insert(signature.end(), {pointer, pointer, i64, pointer, pointer, i32, f32});
        signature.insert(signature.end(), 4, i32);
        signature.push_back(f32);
        signature.insert(signature.end(), 15, i64);
        signature.insert(signature.end(), {pointer, i64, i64, i32, i64, i64, i64});
        return signature;
    }();
    return s_signature;
}

/// One operand's (batch, head, sequence) element strides, in the kernel's slot order.
std::array<int64_t, 3> bhsStrides(const data_objects::TensorAttributes& tensor)
{
    const auto* strides = tensor.strides();
    return {strides->Get(BATCH_AXIS), strides->Get(HEAD_AXIS), strides->Get(SEQ_AXIS)};
}

/// The compiled kernel plus everything launch() needs.
class PreparedFlydslSdpaAiw : public PreparedDispatch
{
public:
    PreparedFlydslSdpaAiw(IngestorKernelCode code,
                          FlydslSdpaRdnaBinding binding,
                          SdpaProblem problem,
                          std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand,
                          std::array<std::array<int64_t, 3>, 5> strides,
                          BiasStrides biasStrides,
                          std::array<int64_t, 2> window)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _scaleOperand(scaleOperand)
        , _strides(strides)
        , _biasStrides(biasStrides)
        , _window(window)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslSdpaRdnaBinding& binding() const
    {
        return _binding;
    }

    const SdpaProblem& problem() const
    {
        return _problem;
    }

    const std::optional<hipdnn_plugin_sdk::ScalarOperand>& scaleOperand() const
    {
        return _scaleOperand;
    }

    /// Q, K, V, O and the LSE (zeros when there is none), each (batch, head, sequence).
    const std::array<std::array<int64_t, 3>, 5>& strides() const
    {
        return _strides;
    }

    /// The mask's (batch, head, query, key) strides; zeros when there is none.
    const BiasStrides& biasStrides() const
    {
        return _biasStrides;
    }

    /// The kernel's (window_left, window_right).
    const std::array<int64_t, 2>& window() const
    {
        return _window;
    }

private:
    IngestorKernelCode _code;
    FlydslSdpaRdnaBinding _binding;
    SdpaProblem _problem;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _scaleOperand;
    std::array<std::array<int64_t, 3>, 5> _strides;
    BiasStrides _biasStrides;
    std::array<int64_t, 2> _window;
};

/**
 * @brief The native dispatch behind this pack's UDD. Everything graph/kernel-derived
 *        resolves once at prepare(); launch() only resolves buffers and the scale.
 */
class FlydslSdpaAiwDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    FlydslSdpaAiwDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                 const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The kernel's only scratch is LDS and registers.
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
        const auto binding = flydslSdpaRdnaBinding(bound);
        const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
        const auto threads = integerMetadata(kernel, BLOCK_THREADS_FIELD);
        if(!blockM.has_value() || *blockM <= 0 || !threads.has_value() || *threads <= 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_aiw: kernel '" + toString(kernel.kernelId)
                    + "' declares no usable block_m/block_threads");
        }

        const auto& q = requireTensor(context, binding.q);
        const auto& k = requireTensor(context, binding.k);
        const auto& v = requireTensor(context, binding.v);
        const auto& o = requireTensor(context, binding.o);
        auto problem = problemFor(q, k);
        problem.numVHeads = v.dims()->Get(HEAD_AXIS);
        problem.headDimV = v.dims()->Get(HEAD_DIM_AXIS);

        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, flydslSdpaAiwSignature());
        // `heads_by_qtiles_by_batch`: one workgroup per (query head, query tile, batch).
        const auto queryTiles = (problem.seqLenQ + *blockM - 1) / *blockM;
        code.setBlockSize(static_cast<unsigned int>(*threads), 1, 1);
        code.setGridSize(static_cast<unsigned int>(problem.numHeads),
                         static_cast<unsigned int>(queryTiles),
                         static_cast<unsigned int>(problem.batch));

        std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand;
        if(binding.scaleSource == ScaleSource::TENSOR)
        {
            scaleOperand = hipdnn_plugin_sdk::makeScalarOperand(
                context.graph.getTensorMap(), binding.scale, "scale");
        }

        BiasStrides biasStrides{0, 0, 0, 0};
        if(binding.bias != NO_BIAS)
        {
            const auto strides = servableBiasStrides(requireTensor(context, binding.bias),
                                                     problem.batch,
                                                     problem.numHeads,
                                                     problem.seqLenQ,
                                                     problem.seqLenKv,
                                                     problem.dataType);
            if(!strides.has_value())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "flydsl_sdpa_aiw: the bound mask is not one the kernel can read");
            }
            biasStrides = *strides;
        }

        // The LSE through its own strides; a nonzero query stride is what selects them in
        // the kernel, and with one query row any nonzero value addresses it.
        std::array<int64_t, 3> lse{0, 0, 0};
        if(binding.stats != NO_STATS)
        {
            lse = bhsStrides(requireTensor(context, binding.stats));
            lse[2] = std::max<int64_t>(lse[2], 1);
        }

        return std::make_unique<PreparedFlydslSdpaAiw>(
            std::move(code),
            binding,
            problem,
            scaleOperand,
            std::array<std::array<int64_t, 3>, 5>{
                bhsStrides(q), bhsStrides(k), bhsStrides(v), bhsStrides(o), lse},
            biasStrides,
            windowFor(binding, problem));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedSdpa = dynamic_cast<const PreparedFlydslSdpaAiw&>(prepared);
        const auto& binding = preparedSdpa.binding();
        const auto& problem = preparedSdpa.problem();
        const float scale
            = launchScale(binding, preparedSdpa.scaleOperand(), deviceBuffers, numDeviceBuffers);

        const auto q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers);
        const auto k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers);
        const auto v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers);
        const auto o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers);
        // The kernel writes the LSE only through a non-null pointer, and reads the mask and
        // the sink only in the objects kernel_match pairs with a bound one.
        void* stats = optionalBuffer(binding.stats, deviceBuffers, numDeviceBuffers);
        void* bias = optionalBuffer(binding.bias, deviceBuffers, numDeviceBuffers);
        void* sink = optionalBuffer(binding.sink, deviceBuffers, numDeviceBuffers);
        void* none = nullptr;

        const auto& s = preparedSdpa.strides();
        const auto& b = preparedSdpa.biasStrides();
        const auto& window = preparedSdpa.window();
        const auto i32 = [](int64_t value) { return static_cast<int32_t>(value); };
        const int32_t zero = 0;
        const int64_t zero64 = 0;
        const float noDropout = 1.0F;

        // Changing this argument list means changing flydslSdpaAiwSignature() with it.
        preparedSdpa.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    q.ptr,
                    k.ptr,
                    v.ptr,
                    bias,
                    o.ptr,
                    stats,
                    none, // seqinfo_q0: dense, no varlen
                    none,
                    none,
                    none,
                    zero, // varlen_bits
                    zero, // num_seqlens
                    i32(problem.seqLenQ),
                    i32(problem.seqLenKv),
                    i32(window[0]),
                    i32(window[1]),
                    none, // philox_seed_ptr: no dropout
                    none,
                    zero64,
                    none,
                    none,
                    zero, // idropout_p
                    noDropout,
                    i32(problem.numHeads),
                    i32(problem.numKvHeads),
                    i32(problem.headDim),
                    i32(problem.headDimV),
                    scale,
                    s[0][0],
                    s[0][1],
                    s[0][2],
                    s[1][0],
                    s[1][1],
                    s[1][2],
                    s[2][0],
                    s[2][1],
                    s[2][2],
                    s[3][0],
                    s[3][1],
                    s[3][2],
                    b[0],
                    b[1],
                    b[2],
                    sink,
                    binding.sinkStride,
                    b[3],
                    i32(problem.numVHeads),
                    s[4][0],
                    s[4][1],
                    s[4][2]);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& flydslSdpaAiwKpackModuleCache()
{
    // Process-lifetime, and this pack's own: a cache key is (archive, toc_key, arch), so a
    // cache shared with another pack would answer that pack's lookups too.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetFlydslSdpaAiwModuleCache()
{
    flydslSdpaAiwKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer
/// to it.
const FlydslSdpaAiwDispatchHandler& flydslSdpaAiwDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslSdpaAiwKpackModuleCache());
    static const FlydslSdpaAiwDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslSdpaAiwSymbols(SymbolScope<Handle>& scope)
{
    // The RDNA engines' graph matcher: all three serve the same graphs.
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydsl_sdpa_rdna::flydslSdpaRdnaGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslSdpaAiwKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslSdpaAiwScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslSdpaAiwDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
