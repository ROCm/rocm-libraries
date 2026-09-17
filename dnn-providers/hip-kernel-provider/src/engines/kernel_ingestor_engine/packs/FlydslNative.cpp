// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// THROWAWAY POC pack. Not for release, not for a PR.
//
// The flyDSL pack's native half: the four symbols the installed flyDSL descriptors
// (flydsl_poc_scratch/flydsl_descriptors/flydsl/*.json) name -- graph_match, kernel_match,
// score, dispatch -- plus the one function that registers them. This is what closes the
// PHASE_A gap: with these symbols present, hipDNN's kernel-ingestor engine can SELECT the
// flyDSL engine through the public API (get_ranked_engine_ids -> "hipkernel:Flydsl"), not
// just execute a directly-instantiated handler.
//
// Matching mirrors PointwiseNative.cpp (single-node binary pointwise ADD over 1-element
// tensors, uniform dtype). Dispatch mirrors TestFlydslRawDispatchHandler.cpp: it IGNORES
// the KernelDefinition's source kind and raw-loads a build-time flyDSL -> HSACO code
// object with hipModuleLoadData, so nothing here touches buildIngestorKernelCode/kpack.
// The HSACO path comes from the FLYDSL_HSACO_PATH env var (throwaway shortcut).

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
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
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
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl.dispatch";

constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view DTYPE_FIELD = "dtype";

// Tokens the graph match binds and the dispatch reads back. Internal to this pack.
constexpr std::string_view INPUT_A_TOKEN = "flydsl.input_a.uid";
constexpr std::string_view INPUT_B_TOKEN = "flydsl.input_b.uid";
constexpr std::string_view OUTPUT_TOKEN = "flydsl.output.uid";

// The build-time flyDSL vadd HSACO symbol, and its 0.3.x kernarg ABI: each Tensor arg is
// (global_buffer ptr, by_value i32 size) => 44 bytes. This is a 1-element add, so every
// runtime size is 1.
constexpr const char* FLYDSL_SYMBOL = "vadd_0";
constexpr const char* FLYDSL_HSACO_ENV = "FLYDSL_HSACO_PATH";

constexpr uint32_t MIN_SUPPORTED_RANK = 4;
constexpr uint32_t MAX_SUPPORTED_RANK = 5;

// ---------------------------------------------------------------------------
// Matching (mirrors PointwiseNative.cpp)
// ---------------------------------------------------------------------------

struct FlydslBinding
{
    int64_t inputA = 0;
    int64_t inputB = 0;
    int64_t output = 0;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// Runs on an unvalidated graph, so must be total: every dim must be extent 1.
bool isSingleElement(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || strides->size() != dims->size()
       || dims->size() < MIN_SUPPORTED_RANK || dims->size() > MAX_SUPPORTED_RANK)
    {
        return false;
    }
    for(const auto dim : *dims)
    {
        if(dim != 1)
        {
            return false;
        }
    }
    return true;
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

std::optional<data_objects::DataType> graphDataType(const MatchContext& context)
{
    const auto* attributes = pointwiseNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }
    const auto* input = findTensor(context, attributes->in_0_tensor_uid());
    if(input == nullptr)
    {
        return std::nullopt;
    }
    return input->data_type();
}

std::string dataTypeName(data_objects::DataType dataType)
{
    return data_objects::EnumNameDataType(dataType);
}

/// Graph-scoped applicability: a single-node binary pointwise ADD over 1-element tensors of
/// uniform dtype -- the only graph the flyDSL vadd HSACO can serve. Returns the operand
/// bindings the dispatch reads back.
std::optional<BoundTokens> flydslGraphMatches(const MatchContext& context)
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

    // Binary: a second operand is required, a third would be a different operation.
    if(!attributes.in_1_tensor_uid().has_value() || attributes.in_2_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    const auto* inputA = findTensor(context, attributes.in_0_tensor_uid());
    const auto* inputB = findTensor(context, attributes.in_1_tensor_uid().value());
    const auto* output = findTensor(context, attributes.out_0_tensor_uid());
    if(inputA == nullptr || inputB == nullptr || output == nullptr)
    {
        return std::nullopt;
    }

    if(!isSingleElement(*inputA) || !isSingleElement(*inputB) || !isSingleElement(*output))
    {
        return std::nullopt;
    }

    if(inputA->virtual_() || inputB->virtual_() || output->virtual_())
    {
        return std::nullopt;
    }

    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(inputA)
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(inputB)
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(output))
    {
        return std::nullopt;
    }

    if(inputA->data_type() != inputB->data_type() || inputA->data_type() != output->data_type())
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(INPUT_A_TOKEN)] = attributes.in_0_tensor_uid();
    bound[std::string(INPUT_B_TOKEN)] = attributes.in_1_tensor_uid().value();
    bound[std::string(OUTPUT_TOKEN)] = attributes.out_0_tensor_uid();
    return bound;
}

/// Kernel-scoped applicability: does this kernel's baked dtype match the graph's?
bool flydslKernelMatches(const MatchContext& context,
                         const BoundTokens& /*bound*/,
                         const KernelDefinition& kernel)
{
    const auto dataType = graphDataType(context);
    if(!dataType.has_value())
    {
        return false;
    }
    return kernel.getStringMetadata(std::string(DTYPE_FIELD)) == dataTypeName(*dataType);
}

double flydslScore(const MatchContext& /*context*/,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));
}

FlydslBinding flydslBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    return {read(INPUT_A_TOKEN), read(INPUT_B_TOKEN), read(OUTPUT_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch (mirrors TestFlydslRawDispatchHandler.cpp: raw-load the HSACO)
// ---------------------------------------------------------------------------

class PreparedFlydsl : public PreparedDispatch
{
public:
    PreparedFlydsl(hipModule_t mod, hipFunction_t fn, FlydslBinding binding)
        : _mod(mod)
        , _fn(fn)
        , _binding(binding)
    {
    }

    ~PreparedFlydsl() override
    {
        if(_mod != nullptr)
        {
            static_cast<void>(hipModuleUnload(_mod));
        }
    }

    PreparedFlydsl(const PreparedFlydsl&) = delete;
    PreparedFlydsl& operator=(const PreparedFlydsl&) = delete;

    hipFunction_t function() const
    {
        return _fn;
    }

    const FlydslBinding& binding() const
    {
        return _binding;
    }

private:
    hipModule_t _mod = nullptr;
    hipFunction_t _fn = nullptr;
    FlydslBinding _binding;
};

class FlydslDispatchHandler : public IKernelDispatchHandler<Handle>
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
                                              const KernelDefinition& /*kernel*/) const override
    {
        const char* hsacoPath = std::getenv(FLYDSL_HSACO_ENV);
        if(hsacoPath == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string("flyDSL dispatch needs ") + FLYDSL_HSACO_ENV + " set to the HSACO path");
        }

        std::vector<char> blob;
        {
            FILE* f = std::fopen(hsacoPath, "rb");
            if(f == nullptr)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    std::string("cannot open HSACO: ") + hsacoPath);
            }
            std::fseek(f, 0, SEEK_END);
            const long n = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            blob.resize(static_cast<size_t>(n));
            const size_t got = std::fread(blob.data(), 1, static_cast<size_t>(n), f);
            std::fclose(f);
            if(got != static_cast<size_t>(n))
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "short read on HSACO");
            }
        }

        hipModule_t mod = nullptr;
        if(hipModuleLoadData(&mod, blob.data()) != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                          "hipModuleLoadData failed");
        }
        hipFunction_t fn = nullptr;
        if(hipModuleGetFunction(&fn, mod, FLYDSL_SYMBOL) != hipSuccess)
        {
            static_cast<void>(hipModuleUnload(mod));
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                          "hipModuleGetFunction failed");
        }
        return std::make_unique<PreparedFlydsl>(mod, fn, flydslBinding(bound));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& fly = dynamic_cast<const PreparedFlydsl&>(prepared);
        const auto& binding = fly.binding();

        const auto a
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.inputA, deviceBuffers, numDeviceBuffers);
        const auto b
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.inputB, deviceBuffers, numDeviceBuffers);
        const auto out
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.output, deviceBuffers, numDeviceBuffers);

        // flyDSL 0.3.x ABI: (ptr, i32 size) per Tensor, 44 bytes. 1 element -> size 1.
        std::array<unsigned char, 44> args{};
        const int32_t elems = 1;
        void* out_ptr = out.ptr;
        void* a_ptr = a.ptr;
        void* b_ptr = b.ptr;
        std::memcpy(args.data() + 0, &out_ptr, sizeof(void*));
        std::memcpy(args.data() + 8, &elems, sizeof(int32_t));
        std::memcpy(args.data() + 16, &a_ptr, sizeof(void*));
        std::memcpy(args.data() + 24, &elems, sizeof(int32_t));
        std::memcpy(args.data() + 32, &b_ptr, sizeof(void*));
        std::memcpy(args.data() + 40, &elems, sizeof(int32_t));
        size_t argsz = args.size();

        std::array<void*, 5> config{HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                    args.data(),
                                    HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                    &argsz,
                                    HIP_LAUNCH_PARAM_END};

        if(hipModuleLaunchKernel(fly.function(), 1, 1, 1, 1, 1, 1, 0, handle.getStream(), nullptr,
                                 config.data())
           != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                          "hipModuleLaunchKernel failed");
        }
    }
};

/// Process-lifetime handler; the registry holds a non-owning pointer to it.
const FlydslDispatchHandler& flydslDispatchHandler()
{
    static const FlydslDispatchHandler s_dispatchHandler;
    return s_dispatchHandler;
}

} // namespace

void registerFlydslSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
