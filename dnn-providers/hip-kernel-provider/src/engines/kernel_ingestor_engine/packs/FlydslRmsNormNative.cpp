// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// THROWAWAY POC pack (M2). Not for release, not for a PR.
//
// The flyDSL RMSNorm pack: a FAMILY of kernel instances (one HSACO per hidden size N),
// proving hipDNN can register N instances and SELECT the one whose baked N matches the
// graph's hidden size. The SAME pack serves two families:
//   * the TOY family (de-risk): one-thread-per-row rmsnorm_0, uniform (ptr,i32) ABI.
//   * the REAL family (M2 swap): FlyDSL build_rmsnorm_module bf16 output, rmsnorm_kernel_0,
//     the 2D-tensor kernarg ABI decoded from the HSACO's AMDGPU metadata.
// Which family a kernel belongs to is data, not code: each kernelDescriptor's metadata
// carries `hsaco` (filename), `symbol` (entry point), `block_threads` (launch block.x), and
// `abi` ("toy" | "real2d"). Descriptors that omit them fall back to the toy defaults, so the
// original toy proof keeps passing untouched.
//
// De-risk shortcuts (all deliberate, all thrown away later):
//   * Carrier op: a single Pointwise ADD node is (ab)used as the graph vehicle. in_0 = x,
//     in_1 = weight, out_0 = out. We ignore the ADD semantics entirely; the node just
//     ships three tensor uids + a hidden size. The real Pointwise pack never competes --
//     it only matches 1-element graphs, and this pack requires N >= 2.
//   * Dispatch raw-loads the matched kernel's HSACO from $FLYDSL_RMSNORM_HSACO_DIR. It never
//     touches buildIngestorKernelCode/kpack -- the descriptor's embedded_source is a placeholder.
//
// REAL 2D ABI (build_rmsnorm_module bf16, from the HSACO AMDGPU .args + stage-01 MLIR):
//   kernel rmsnorm_kernel(Input, Gamma, Rstd, Output); Rstd slot reuses the Gamma ptr.
//   kernarg_segment_size = 80. 2D [M,N] tensor descriptor = {i32 dim0, i32 dim1, i64 row_stride}
//   (layout (M,N):(row_stride,1), row-major => row_stride = N); 1D tensor descriptor = {i32 size}.
//   Offsets: Input.ptr@0 desc@8(16) Gamma.ptr@24 size@32(4) Rstd.ptr@40 size@48(4)
//            Output.ptr@56 desc@64(16). grid=(M,1,1) block=(block_threads=256,1,1).

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <fstream>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

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

constexpr std::string_view N_FIELD = "N";
// Optional per-kernel metadata (absent => toy defaults, so the toy proof is unchanged).
constexpr std::string_view HSACO_FIELD = "hsaco";               // literal HSACO filename
constexpr std::string_view SYMBOL_FIELD = "symbol";             // kernel entry point
constexpr std::string_view BLOCK_FIELD = "block_threads";       // launch block.x
constexpr std::string_view ABI_FIELD = "abi";                   // "toy" | "real2d"

// Tokens the graph match binds and the dispatch reads back. Internal to this pack.
constexpr std::string_view X_TOKEN = "flydsl_rmsnorm.x.uid";
constexpr std::string_view W_TOKEN = "flydsl_rmsnorm.w.uid";
constexpr std::string_view OUT_TOKEN = "flydsl_rmsnorm.out.uid";
constexpr std::string_view N_TOKEN = "flydsl_rmsnorm.n";
constexpr std::string_view ROWS_TOKEN = "flydsl_rmsnorm.rows";

// Toy-family defaults, used when a descriptor omits the optional metadata above.
constexpr const char* TOY_SYMBOL = "rmsnorm_0";
constexpr int64_t TOY_BLOCK_THREADS = 1;
constexpr const char* ABI_REAL2D = "real2d";

constexpr const char* FLYDSL_HSACO_DIR_ENV = "FLYDSL_RMSNORM_HSACO_DIR";

constexpr uint32_t MIN_SUPPORTED_RANK = 4;
constexpr uint32_t MAX_SUPPORTED_RANK = 5;
constexpr int64_t MIN_HIDDEN_SIZE = 2; // keeps this pack disjoint from 1-element pointwise

// ---------------------------------------------------------------------------
// Optional-metadata helpers (throwaway: silently fall back to the toy defaults)
// ---------------------------------------------------------------------------

std::optional<std::string> tryGetStringMeta(const KernelDefinition& kernel, std::string_view field)
{
    const auto value = kernel.tryGetMetadata(std::string(field));
    if(!value.has_value())
    {
        return std::nullopt;
    }
    if(const auto* s = std::get_if<std::string>(&value.value()))
    {
        return *s;
    }
    return std::nullopt;
}

int64_t getIntMetaOr(const KernelDefinition& kernel, std::string_view field, int64_t dflt)
{
    const auto value = kernel.tryGetMetadata(std::string(field));
    if(!value.has_value())
    {
        return dflt;
    }
    if(const auto* i = std::get_if<int64_t>(&value.value()))
    {
        return *i;
    }
    return dflt;
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

struct RmsNormBinding
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t out = 0;
    int64_t n = 0;
    int64_t rows = 0;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

const data_objects::PointwiseAttributes* pointwiseNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::PointwiseAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::PointwiseAttributes>();
}

/// Reads (rows, N) off a row-major [.., N] tensor: N is the last extent, rows the product
/// of the leading extents. Total so it can run on an unvalidated graph.
bool readShape(const data_objects::TensorAttributes& tensor, int64_t& rows, int64_t& n)
{
    const auto* dims = tensor.dims();
    if(dims == nullptr || dims->size() < MIN_SUPPORTED_RANK || dims->size() > MAX_SUPPORTED_RANK)
    {
        return false;
    }
    const uint32_t rank = dims->size();
    n = dims->Get(rank - 1);
    if(n < MIN_HIDDEN_SIZE)
    {
        return false;
    }
    rows = 1;
    for(uint32_t i = 0; i + 1 < rank; ++i)
    {
        rows *= dims->Get(i);
    }
    return rows >= 1;
}

/// Graph-scoped applicability: a single binary Pointwise ADD node over a [.., N] tensor,
/// N >= 2. Binds x/w/out uids plus the (N, rows) the dispatch launches with.
std::optional<BoundTokens> flydslRmsNormGraphMatches(const MatchContext& context)
{
    const auto* attributesPtr = pointwiseNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    if(attributes.operation() != data_objects::PointwiseMode::ADD)
    {
        return std::nullopt;
    }
    if(!attributes.in_1_tensor_uid().has_value() || attributes.in_2_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, attributes.in_0_tensor_uid());
    const auto* w = findTensor(context, attributes.in_1_tensor_uid().value());
    const auto* out = findTensor(context, attributes.out_0_tensor_uid());
    if(x == nullptr || w == nullptr || out == nullptr)
    {
        return std::nullopt;
    }
    if(x->virtual_() || w->virtual_() || out->virtual_())
    {
        return std::nullopt;
    }

    int64_t rows = 0;
    int64_t n = 0;
    if(!readShape(*x, rows, n))
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(X_TOKEN)] = attributes.in_0_tensor_uid();
    bound[std::string(W_TOKEN)] = attributes.in_1_tensor_uid().value();
    bound[std::string(OUT_TOKEN)] = attributes.out_0_tensor_uid();
    bound[std::string(N_TOKEN)] = n;
    bound[std::string(ROWS_TOKEN)] = rows;
    return bound;
}

/// Kernel-scoped applicability: does this instance's baked N equal the graph's hidden size?
/// This is the family selector -- the whole point of M2.
bool flydslRmsNormKernelMatches(const MatchContext& /*context*/,
                                const BoundTokens& bound,
                                const KernelDefinition& kernel)
{
    const auto graphN = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, N_TOKEN);
    if(!graphN.has_value())
    {
        return false;
    }
    return kernel.getIntMetadata(std::string(N_FIELD)) == *graphN;
}

double flydslRmsNormScore(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel)
{
    // One survivor per shape after kernel_match; any deterministic value orders it.
    return static_cast<double>(kernel.getIntMetadata(std::string(N_FIELD)));
}

RmsNormBinding rmsNormBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl rmsnorm dispatch is missing bound token '" + std::string(token) + "'");
        }
        return *value;
    };
    return {read(X_TOKEN), read(W_TOKEN), read(OUT_TOKEN), read(N_TOKEN), read(ROWS_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch (raw-load the HSACO whose baked N the family match selected)
// ---------------------------------------------------------------------------

class PreparedRmsNorm : public PreparedDispatch
{
public:
    PreparedRmsNorm(hipModule_t mod,
                    hipFunction_t fn,
                    RmsNormBinding binding,
                    bool real2d,
                    uint32_t blockThreads)
        : _mod(mod)
        , _fn(fn)
        , _binding(binding)
        , _real2d(real2d)
        , _blockThreads(blockThreads)
    {
    }

    ~PreparedRmsNorm() override
    {
        if(_mod != nullptr)
        {
            static_cast<void>(hipModuleUnload(_mod));
        }
    }

    PreparedRmsNorm(const PreparedRmsNorm&) = delete;
    PreparedRmsNorm& operator=(const PreparedRmsNorm&) = delete;

    hipFunction_t function() const
    {
        return _fn;
    }
    const RmsNormBinding& binding() const
    {
        return _binding;
    }
    bool real2d() const
    {
        return _real2d;
    }
    uint32_t blockThreads() const
    {
        return _blockThreads;
    }

private:
    hipModule_t _mod = nullptr;
    hipFunction_t _fn = nullptr;
    RmsNormBinding _binding;
    bool _real2d = false;
    uint32_t _blockThreads = 1;
};

class FlydslRmsNormDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = rmsNormBinding(bound);

        // Sanity: the selected kernel's baked N must equal the graph's -- kernel_match
        // guarantees it, but this makes a wiring bug loud instead of silently wrong.
        const int64_t kernelN = kernel.getIntMetadata(std::string(N_FIELD));
        if(kernelN != binding.n)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl rmsnorm: selected kernel N=" + std::to_string(kernelN)
                    + " != graph N=" + std::to_string(binding.n));
        }

        // Metadata drives which HSACO/symbol/ABI/block to use. Absent => toy defaults.
        const auto abiMeta = tryGetStringMeta(kernel, ABI_FIELD);
        const bool real2d = abiMeta.has_value() && abiMeta.value() == ABI_REAL2D;
        const std::string symbol =
            tryGetStringMeta(kernel, SYMBOL_FIELD).value_or(std::string(TOY_SYMBOL));
        const auto blockThreads = static_cast<uint32_t>(
            getIntMetaOr(kernel, BLOCK_FIELD, TOY_BLOCK_THREADS));

        // getenv is deprecated in the Windows CRT; the data SDK wraps the platform call.
        const std::string dirValue = hipdnn_data_sdk::utilities::getEnv(FLYDSL_HSACO_DIR_ENV, "");
        const char* dir = dirValue.empty() ? nullptr : dirValue.c_str();
        if(dir == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string("flyDSL rmsnorm dispatch needs ") + FLYDSL_HSACO_DIR_ENV + " set");
        }
        const auto hsacoName = tryGetStringMeta(kernel, HSACO_FIELD)
                                   .value_or("rmsnorm_toy_n" + std::to_string(binding.n)
                                             + "_gfx950.hsaco");
        const std::string path = std::string(dir) + "/" + hsacoName;

        std::vector<char> blob;
        {
            // std::fopen is deprecated in the Windows CRT; a stream reads the same bytes.
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if(!file)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM, std::string("cannot open HSACO: ") + path);
            }
            const auto nbytes = static_cast<std::streamsize>(file.tellg());
            file.seekg(0);
            blob.resize(static_cast<size_t>(nbytes));
            if(!file.read(blob.data(), nbytes))
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "short read on HSACO");
            }
        }

        hipModule_t mod = nullptr;
        if(hipModuleLoadData(&mod, blob.data()) != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                          "hipModuleLoadData failed for " + path);
        }
        hipFunction_t fn = nullptr;
        if(hipModuleGetFunction(&fn, mod, symbol.c_str()) != hipSuccess)
        {
            static_cast<void>(hipModuleUnload(mod));
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hipModuleGetFunction failed for symbol " + symbol);
        }
        return std::make_unique<PreparedRmsNorm>(mod, fn, binding, real2d, blockThreads);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& rms = dynamic_cast<const PreparedRmsNorm&>(prepared);
        const auto& b = rms.binding();

        const auto x = hipdnn_plugin_sdk::findDeviceBuffer(b.x, deviceBuffers, numDeviceBuffers);
        const auto w = hipdnn_plugin_sdk::findDeviceBuffer(b.w, deviceBuffers, numDeviceBuffers);
        const auto out = hipdnn_plugin_sdk::findDeviceBuffer(b.out, deviceBuffers, numDeviceBuffers);

        void* x_ptr = x.ptr;
        void* w_ptr = w.ptr;
        void* out_ptr = out.ptr;

        // Widest ABI is the real 2D one at 80 bytes; the toy uses the first 44.
        std::array<unsigned char, 80> args{};
        size_t argsz = 0;
        unsigned int gridX = 0;

        if(rms.real2d())
        {
            // Real FlyDSL build_rmsnorm_module ABI. kernel(Input, Gamma, Rstd, Output),
            // Rstd reuses the Gamma ptr. 2D desc = {i32 M, i32 N, i64 row_stride}; 1D = {i32 N}.
            const int32_t m32 = static_cast<int32_t>(b.rows);
            const int32_t n32 = static_cast<int32_t>(b.n);
            const int64_t rowStride = b.n; // contiguous row-major [M,N]
            std::memcpy(args.data() + 0, &x_ptr, sizeof(void*));       // Input.ptr
            std::memcpy(args.data() + 8, &m32, sizeof(int32_t));       // Input dim0 = M
            std::memcpy(args.data() + 12, &n32, sizeof(int32_t));      // Input dim1 = N
            std::memcpy(args.data() + 16, &rowStride, sizeof(int64_t)); // Input row_stride
            std::memcpy(args.data() + 24, &w_ptr, sizeof(void*));      // Gamma.ptr
            std::memcpy(args.data() + 32, &n32, sizeof(int32_t));      // Gamma size = N
            std::memcpy(args.data() + 40, &w_ptr, sizeof(void*));      // Rstd.ptr (= Gamma)
            std::memcpy(args.data() + 48, &n32, sizeof(int32_t));      // Rstd size = N
            std::memcpy(args.data() + 56, &out_ptr, sizeof(void*));    // Output.ptr
            std::memcpy(args.data() + 64, &m32, sizeof(int32_t));      // Output dim0 = M
            std::memcpy(args.data() + 68, &n32, sizeof(int32_t));      // Output dim1 = N
            std::memcpy(args.data() + 72, &rowStride, sizeof(int64_t)); // Output row_stride
            argsz = 80;
            gridX = static_cast<unsigned int>(b.rows); // one block per row
        }
        else
        {
            // Toy 0.3.x ABI: (ptr, i32 size) per Tensor, kernel is rmsnorm(out, x, w).
            const int32_t xn = static_cast<int32_t>(b.rows * b.n);
            const int32_t wn = static_cast<int32_t>(b.n);
            std::memcpy(args.data() + 0, &out_ptr, sizeof(void*));
            std::memcpy(args.data() + 8, &xn, sizeof(int32_t));
            std::memcpy(args.data() + 16, &x_ptr, sizeof(void*));
            std::memcpy(args.data() + 24, &xn, sizeof(int32_t));
            std::memcpy(args.data() + 32, &w_ptr, sizeof(void*));
            std::memcpy(args.data() + 40, &wn, sizeof(int32_t));
            argsz = 44;
            gridX = static_cast<unsigned int>(b.rows); // one thread per row
        }

        std::array<void*, 5> config{HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                    args.data(),
                                    HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                    &argsz,
                                    HIP_LAUNCH_PARAM_END};

        if(hipModuleLaunchKernel(rms.function(),
                                 gridX, 1, 1,
                                 rms.blockThreads(), 1, 1,
                                 0,
                                 handle.getStream(), nullptr, config.data())
           != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                          "hipModuleLaunchKernel failed");
        }
    }
};

const FlydslRmsNormDispatchHandler& flydslRmsNormDispatchHandler()
{
    static const FlydslRmsNormDispatchHandler s_dispatchHandler;
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

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
