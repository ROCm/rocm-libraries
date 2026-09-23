// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KernelCompileOptions.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdGeometry.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace conv = gfx950_conv_fwd;

namespace
{

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_conv_fwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_conv_fwd.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_conv_fwd.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_conv_fwd.dispatch";
constexpr std::string_view X_TOKEN = "gfx950_conv_fwd.x.uid";
constexpr std::string_view W_TOKEN = "gfx950_conv_fwd.w.uid";
constexpr std::string_view Y_TOKEN = "gfx950_conv_fwd.y.uid";

struct Binding
{
    int64_t x;
    int64_t w;
    int64_t y;
};

struct MatchedProblem
{
    Binding binding;
    conv::Problem problem;
    conv::Geometry geometry;
    data_objects::DataType dtype;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

bool isDeviceTensor(const data_objects::TensorAttributes* tensor)
{
    return tensor != nullptr && !tensor->virtual_()
           && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor)
           && !tensor->ragged_offset_tensor_uid().has_value() && tensor->alignment() >= 16
           && tensor->alignment() % 16 == 0
           && (tensor->data_type() == data_objects::DataType::HALF
               || tensor->data_type() == data_objects::DataType::BFLOAT16);
}

bool hasPackedNhwcStrides(const data_objects::TensorAttributes* tensor)
{
    if(tensor == nullptr || tensor->dims() == nullptr || tensor->strides() == nullptr
       || tensor->dims()->size() != 4 || tensor->strides()->size() != 4)
    {
        return false;
    }
    const auto* dims = tensor->dims();
    const std::array<int64_t, 4> extents{dims->Get(0), dims->Get(1), dims->Get(2), dims->Get(3)};
    if(!conv::tensorByteSize(extents))
    {
        return false;
    }
    // hipDNN [N,C,H,W] -> NHWC; filter [K,C,Y,X] -> KYXC; output [N,K,Ho,Wo] -> NHWK.
    // The byte-size guard above bounds every multiplication here.
    const std::array<int64_t, 4> expected{
        extents[1] * extents[2] * extents[3], 1, extents[1] * extents[3], extents[1]};
    for(flatbuffers::uoffset_t axis = 0; axis < 4; ++axis)
    {
        // A unit-extent axis never contributes to an address, whatever its stride.
        if(extents[axis] != 1 && tensor->strides()->Get(axis) != expected[axis])
        {
            return false;
        }
    }
    return true;
}

bool hasSpatialRankTwo(const flatbuffers::Vector<int64_t>* values)
{
    return values != nullptr && values->size() == 2;
}

std::optional<MatchedProblem> matchProblem(const MatchContext& context)
{
    if(!context.graph.isValid() || context.graph.getGraph().is_override_shape_enabled()
       || !hipdnn_plugin_sdk::archMatches(
           context.deviceProperties.gcnArchName, "gfx950", hipdnn_plugin_sdk::ArchMatchMode::PREFIX)
       || context.deviceProperties.warpSize != 64 || context.graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes
       || node.attributes() == nullptr || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    const auto& attributes = node.attributesAs<data_objects::ConvolutionFwdAttributes>();
    if(attributes.conv_mode() != data_objects::ConvMode::CROSS_CORRELATION
       || !hasSpatialRankTwo(attributes.pre_padding())
       || !hasSpatialRankTwo(attributes.post_padding()) || !hasSpatialRankTwo(attributes.stride())
       || !hasSpatialRankTwo(attributes.dilation()))
    {
        return std::nullopt;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < 2; ++axis)
    {
        if(attributes.pre_padding()->Get(axis) != attributes.post_padding()->Get(axis))
        {
            return std::nullopt;
        }
    }

    const Binding binding{
        attributes.x_tensor_uid(), attributes.w_tensor_uid(), attributes.y_tensor_uid()};
    const auto* x = findTensor(context, binding.x);
    const auto* w = findTensor(context, binding.w);
    const auto* y = findTensor(context, binding.y);
    if(binding.x == binding.w || binding.x == binding.y || binding.w == binding.y
       || !isDeviceTensor(x) || !isDeviceTensor(w) || !isDeviceTensor(y) || !hasPackedNhwcStrides(x)
       || !hasPackedNhwcStrides(w) || x->data_type() != w->data_type()
       || x->data_type() != y->data_type())
    {
        return std::nullopt;
    }
    // Output dims/strides may not yet be inferred. Validate them in prepare(), before
    // loading any code or computing device addresses; they do not define this problem.
    const auto* xDims = x->dims();
    const auto* wDims = w->dims();
    if(xDims->Get(1) != wDims->Get(1))
    {
        return std::nullopt; // A smaller filter channel count encodes grouped convolution.
    }
    const conv::Problem problem{xDims->Get(0),
                                xDims->Get(1),
                                wDims->Get(0),
                                xDims->Get(2),
                                xDims->Get(3),
                                wDims->Get(2),
                                wDims->Get(3),
                                attributes.stride()->Get(0),
                                attributes.stride()->Get(1),
                                attributes.pre_padding()->Get(0),
                                attributes.pre_padding()->Get(1),
                                attributes.dilation()->Get(0),
                                attributes.dilation()->Get(1)};
    const auto geometry = conv::deriveGeometry(problem);
    if(!geometry)
    {
        return std::nullopt;
    }
    return MatchedProblem{binding, problem, *geometry, x->data_type()};
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto matched = matchProblem(context);
    if(!matched)
    {
        return std::nullopt;
    }
    return BoundTokens{{std::string(X_TOKEN), matched->binding.x},
                       {std::string(W_TOKEN), matched->binding.w},
                       {std::string(Y_TOKEN), matched->binding.y}};
}

template <typename T>
bool metadataEquals(const KernelDefinition& kernel, const char* field, const T& expected)
{
    const auto it = kernel.metadata.find(field);
    if(it == kernel.metadata.end())
    {
        return false;
    }
    const auto* value = std::get_if<T>(&it->second);
    return value != nullptr && *value == expected;
}

bool kernelFits(const MatchedProblem& matched, const KernelDefinition& kernel)
{
    if(kernel.source.kind != KernelSourceKind::KPACK)
    {
        return false;
    }
    const auto& p = matched.problem;
    const std::array<std::pair<const char*, int64_t>, 13> fields{{{"N", p.n},
                                                                  {"C", p.c},
                                                                  {"K", p.k},
                                                                  {"Hi", p.hi},
                                                                  {"Wi", p.wi},
                                                                  {"Y", p.y},
                                                                  {"X", p.x},
                                                                  {"sH", p.strideH},
                                                                  {"sW", p.strideW},
                                                                  {"pH", p.padH},
                                                                  {"pW", p.padW},
                                                                  {"dH", p.dilationH},
                                                                  {"dW", p.dilationW}}};
    for(const auto& field : fields)
    {
        if(!metadataEquals(kernel, field.first, field.second))
        {
            return false;
        }
    }
    const std::string dtype = matched.dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
    // These are the two packaged dispatcher configurations. Keep the complete tuning
    // contract explicit: a new pipeline or atom needs its launch/ABI reviewed here.
    // rocKE's default_vector_sizes picks vec_c > 1 for even K in both storage types;
    // is_valid_spec permits the default epilogue only with scalar output stores.
    return metadataEquals(kernel, "dtype", dtype)
           && metadataEquals(kernel, "layout", std::string("NHWC"))
           && metadataEquals(kernel, "groups", int64_t{1})
           && metadataEquals(kernel, "tile_m", int64_t{64})
           && metadataEquals(kernel, "tile_n", int64_t{64})
           && (metadataEquals(kernel, "tile_k", int64_t{64})
               || metadataEquals(kernel, "tile_k", int64_t{128}))
           && metadataEquals(kernel, "warp_m", int64_t{2})
           && metadataEquals(kernel, "warp_n", int64_t{2})
           && metadataEquals(kernel, "warp_tile_m", int64_t{32})
           && metadataEquals(kernel, "warp_tile_n", int64_t{32})
           && metadataEquals(kernel, "warp_tile_k", int64_t{16})
           && metadataEquals(kernel, "wave_size", int64_t{64})
           && metadataEquals(kernel, "pipeline", std::string("mem"))
           && (metadataEquals(kernel, "epilogue", std::string("cshuffle"))
               || (p.k % 2 != 0 && metadataEquals(kernel, "epilogue", std::string("default"))))
           && conv::launchGeometry(p, matched.geometry, 64, 64, 2, 2, 64).has_value();
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto matched = matchProblem(context);
    return matched && kernelFits(*matched, kernel);
}

double score(const MatchContext& /*context*/,
             const BoundTokens& /*bound*/,
             const KernelDefinition& kernel)
{
    // Deterministic fallback only. BenchmarkPlan replaces this order with measurements
    // when benchmarking is enabled. tile_k=64 is the gfx950 dispatcher's default.
    return metadataEquals(kernel, "tile_k", int64_t{64}) ? 1.0 : 0.0;
}

void requireBindings(const BoundTokens& bound, const Binding& binding)
{
    if(tryGetBoundInt(bound, X_TOKEN) != std::optional<int64_t>(binding.x)
       || tryGetBoundInt(bound, W_TOKEN) != std::optional<int64_t>(binding.w)
       || tryGetBoundInt(bound, Y_TOKEN) != std::optional<int64_t>(binding.y))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950_conv_fwd dispatch requires this graph matcher's tensor bindings");
    }
}

void requireOutput(const MatchContext& context, const MatchedProblem& matched)
{
    const auto* output = findTensor(context, matched.binding.y);
    if(!hasPackedNhwcStrides(output) || output->dims()->Get(0) != matched.problem.n
       || output->dims()->Get(1) != matched.problem.k
       || output->dims()->Get(2) != matched.geometry.ho
       || output->dims()->Get(3) != matched.geometry.wo)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950_conv_fwd requires packed NHWK output with the inferred convolution dimensions");
    }
}

class PreparedConvFwd : public PreparedDispatch
{
public:
    PreparedConvFwd(IngestorKernelCode code, Binding binding, std::array<int32_t, 3> bytes)
        : _code(std::move(code))
        , _binding(binding)
        , _bytes(bytes)
    {
    }

    const compilation::IRunnableKernel& kernel() const
    {
        return *_code.kernel;
    }
    const Binding& binding() const
    {
        return _binding;
    }
    const std::array<int32_t, 3>& bytes() const
    {
        return _bytes;
    }

private:
    // Holds both the program/module and its kernel view; owns no MatchContext data.
    IngestorKernelCode _code;
    Binding _binding;
    std::array<int32_t, 3> _bytes;
};

class ConvFwdDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvFwdDispatchHandler(const compilation::IKernelCompiler& compiler,
                           const compilation::KpackKernelLoader& loader)
        : _compiler(compiler)
        , _loader(loader)
    {
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0; // The kernel uses registers and statically allocated LDS only.
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto matched = matchProblem(context);
        if(!matched || !kernelFits(*matched, kernel))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_fwd graph or packaged kernel does not satisfy the convolution "
                "contract");
        }
        requireBindings(bound, matched->binding);
        requireOutput(context, *matched);

        // KPACK ignores compile options. Use a canonical stand-in because arbitrary
        // strides on unit-extent axes are valid here but confuse the NCHW/NHWC sorter
        // in KernelCompileOptions. The real tensor has already passed our layout check.
        flatbuffers::FlatBufferBuilder builder;
        const std::vector<int64_t> dims{2, 2, 2, 2};
        const std::vector<int64_t> strides{8, 1, 4, 2};
        builder.Finish(data_objects::CreateTensorAttributesDirect(
            builder, 0, nullptr, matched->dtype, &strides, &dims));
        const compilation::KernelCompileOptions options(
            flatbuffers::GetRoot<data_objects::TensorAttributes>(builder.GetBufferPointer()),
            context.deviceProperties.gcnArchName);
        auto code = buildIngestorKernelCode(_compiler, _loader, context, kernel, options);
        const auto launch = conv::launchGeometry(matched->problem,
                                                 matched->geometry,
                                                 kernel.getIntMetadata("tile_m"),
                                                 kernel.getIntMetadata("tile_n"),
                                                 kernel.getIntMetadata("warp_m"),
                                                 kernel.getIntMetadata("warp_n"),
                                                 kernel.getIntMetadata("wave_size"));
        // kernelFits checked this exact geometry before loading the archive.
        code.kernel->setBlockSize(launch->blockX, 1, 1);
        code.kernel->setGridSize(launch->gridX, launch->gridY, launch->gridZ);
        code.kernel->setSharedMemBytes(0);
        return std::make_unique<PreparedConvFwd>(
            std::move(code), matched->binding, matched->geometry.tensorBytes);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& convPrepared = dynamic_cast<const PreparedConvFwd&>(prepared);
        if(deviceBuffers == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "gfx950_conv_fwd requires device buffers");
        }
        const auto& binding = convPrepared.binding();
        const auto a
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto b
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto d
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);
        for(const auto* ptr : {a.ptr, b.ptr, d.ptr})
        {
            if(ptr == nullptr || reinterpret_cast<uintptr_t>(ptr) % 16 != 0)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx950_conv_fwd requires non-null device pointers aligned to 16 bytes");
            }
        }
        const auto& bytes = convPrepared.bytes();
        if(!conv::bufferRangesDoNotOverlap({reinterpret_cast<uintptr_t>(a.ptr),
                                            reinterpret_cast<uintptr_t>(b.ptr),
                                            reinterpret_cast<uintptr_t>(d.ptr)},
                                           bytes))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_fwd requires nonoverlapping input, filter, and output storage");
        }
        // build_implicit_gemm_conv, conv_implicit_gemm.py: A*, B*, D*, A_bytes:i32,
        // B_bytes:i32, D_bytes:i32. These are storage byte sizes, not element counts.
        convPrepared.kernel().launch(
            handle.getStream(), a.ptr, b.ptr, d.ptr, bytes[0], bytes[1], bytes[2]);
    }

private:
    const compilation::IKernelCompiler& _compiler;
    const compilation::KpackKernelLoader& _loader;
};

compilation::KpackModuleCache& moduleCache()
{
    static compilation::KpackModuleCache s_cache;
    return s_cache;
}

const ConvFwdDispatchHandler& dispatchHandler()
{
    // The registries retain non-owning pointers; all three dependencies live for the
    // process, including across provider handle destruction and recreation.
    static const HipMlopsKernelCompiler s_compiler;
    static const compilation::KpackKernelLoader s_loader(moduleCache());
    static const ConvFwdDispatchHandler s_handler(s_compiler, s_loader);
    return s_handler;
}

} // namespace

void registerGfx950ConvFwdSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &score);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
}

void resetGfx950ConvFwdModuleCache()
{
    moduleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
