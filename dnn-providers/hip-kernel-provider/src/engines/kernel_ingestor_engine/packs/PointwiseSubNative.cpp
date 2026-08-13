// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

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

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "core/Handle.hpp"
#include "core/Utils.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/// Mirrors PointwiseAddNative.cpp: same shape, different operation, symbols, kernel.
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

constexpr std::string_view GRAPH_MATCHER_SYMBOL_NAME = "hipkernel.pointwise_sub.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL_NAME = "hipkernel.pointwise_sub.kernel_match";
constexpr std::string_view SCORE_SYMBOL_NAME = "hipkernel.pointwise_sub.score";
constexpr std::string_view DISPATCH_SYMBOL_NAME = "hipkernel.pointwise_sub.dispatch";

constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view INPUT_A_TOKEN = "pointwise_sub.input_a.uid";
constexpr std::string_view INPUT_B_TOKEN = "pointwise_sub.input_b.uid";
constexpr std::string_view OUTPUT_TOKEN = "pointwise_sub.output.uid";

constexpr size_t LARGE_BLOCK_WORKSPACE_BYTES = 1024;
constexpr int64_t LARGE_BLOCK_SIZE = 256;

constexpr uint32_t MIN_SUPPORTED_RANK = 4;
constexpr uint32_t MAX_SUPPORTED_RANK = 5;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// Order is load-bearing: subtraction does not commute.
struct PointwiseSubBinding
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

bool hasSupportedLayout(const data_objects::TensorAttributes& tensor)
{
    try
    {
        static_cast<void>(core::utils::isChannelLastLayout(&tensor));
        return true;
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
    {
        return false;
    }
}

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

    return hasSupportedLayout(tensor);
}

std::optional<data_objects::DataType> graphDataType(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return std::nullopt;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::PointwiseAttributes)
    {
        return std::nullopt;
    }

    const auto& attributes = node.attributesAs<data_objects::PointwiseAttributes>();
    const auto* input = findTensor(context, attributes.in_0_tensor_uid());
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

bool pointwiseSubGraphMatches(const MatchContext& context, BoundTokens& bound)
{
    if(context.deviceId == hipdnn_plugin_sdk::ingestor::NO_DEVICE)
    {
        return false;
    }

    if(context.graph.nodeCount() != 1)
    {
        return false;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::PointwiseAttributes)
    {
        return false;
    }

    const auto& attributes = node.attributesAs<data_objects::PointwiseAttributes>();
    if(attributes.operation() != data_objects::PointwiseMode::SUB)
    {
        return false;
    }

    if(!attributes.in_1_tensor_uid().has_value() || attributes.in_2_tensor_uid().has_value())
    {
        return false;
    }

    const auto* inputA = findTensor(context, attributes.in_0_tensor_uid());
    const auto* inputB = findTensor(context, attributes.in_1_tensor_uid().value());
    const auto* output = findTensor(context, attributes.out_0_tensor_uid());
    if(inputA == nullptr || inputB == nullptr || output == nullptr)
    {
        return false;
    }

    if(!isSingleElement(*inputA) || !isSingleElement(*inputB) || !isSingleElement(*output))
    {
        return false;
    }

    if(inputA->virtual_() || inputB->virtual_() || output->virtual_())
    {
        return false;
    }

    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(inputA)
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(inputB)
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(output))
    {
        return false;
    }

    if(inputA->data_type() != inputB->data_type() || inputA->data_type() != output->data_type())
    {
        return false;
    }

    // Minuend then subtrahend; swapping is silently wrong, not a failure.
    bound[std::string(INPUT_A_TOKEN)] = attributes.in_0_tensor_uid();
    bound[std::string(INPUT_B_TOKEN)] = attributes.in_1_tensor_uid().value();
    bound[std::string(OUTPUT_TOKEN)] = attributes.out_0_tensor_uid();
    return true;
}

bool pointwiseSubKernelMatches(const MatchContext& context, const KernelDefinition& kernel)
{
    const auto dataType = graphDataType(context);
    if(!dataType.has_value())
    {
        return false;
    }

    return kernel.getStringMetadata(std::string(DTYPE_FIELD)) == dataTypeName(*dataType);
}

double pointwiseSubScore(const KernelDefinition& kernel, const MatchContext& /*context*/)
{
    return static_cast<double>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));
}

PointwiseSubBinding pointwiseSubBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "pointwise sub dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };

    return {read(INPUT_A_TOKEN), read(INPUT_B_TOKEN), read(OUTPUT_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

class PreparedPointwiseSub : public PreparedDispatch
{
public:
    PreparedPointwiseSub(std::unique_ptr<compilation::ICompiledProgram> program,
                         std::unique_ptr<compilation::IRunnableKernel> kernel,
                         PointwiseSubBinding binding)
        : _program(std::move(program))
        , _kernel(std::move(kernel))
        , _binding(binding)
    {
    }

    const compilation::IRunnableKernel& kernel() const
    {
        return *_kernel;
    }

    const PointwiseSubBinding& binding() const
    {
        return _binding;
    }

private:
    std::unique_ptr<compilation::ICompiledProgram> _program;
    std::unique_ptr<compilation::IRunnableKernel> _kernel;
    PointwiseSubBinding _binding;
};

std::string elementTypeFor(const KernelDefinition& kernel)
{
    const auto& dtype = kernel.getStringMetadata(std::string(DTYPE_FIELD));
    if(dtype == "FLOAT")
    {
        return "float";
    }
    if(dtype == "HALF")
    {
        return "_Float16";
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "kernel '" + toString(kernel.kernelId) + "' declares unsupported dtype '" + dtype + "'");
}

const data_objects::TensorAttributes& firstInput(const MatchContext& context,
                                                 const PointwiseSubBinding& binding)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(binding.inputA);
    if(it == tensors.end() || it->second == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched pointwise sub has no tensor for uid " + std::to_string(binding.inputA));
    }
    return *it->second;
}

class PointwiseSubDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    explicit PointwiseSubDispatchHandler(const compilation::IKernelCompiler& kernelCompiler)
        : _kernelCompiler(kernelCompiler)
    {
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        return kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)) == LARGE_BLOCK_SIZE
                   ? LARGE_BLOCK_WORKSPACE_BYTES
                   : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = pointwiseSubBinding(bound);

        const auto blockSize
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));

        compilation::KernelCompileOptions options(&firstInput(context, binding),
                                                  context.deviceProperties.gcnArchName);
        options.add("HIP_PLUGIN_POINTWISE_SUB_TYPE", elementTypeFor(kernel));
        options.add("HIP_PLUGIN_POINTWISE_SUB_BLOCK_SIZE", blockSize);

        auto program = _kernelCompiler.compile(kernel.source.sourceFile, options);
        auto runnableKernel = program->getKernel(kernel.source.entryPoint);

        runnableKernel->setBlockSize(blockSize, 1, 1);
        runnableKernel->setGridSize(1, 1, 1);

        return std::make_unique<PreparedPointwiseSub>(
            std::move(program), std::move(runnableKernel), binding);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedSub = dynamic_cast<const PreparedPointwiseSub&>(prepared);
        const auto& binding = preparedSub.binding();

        const auto inputA
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.inputA, deviceBuffers, numDeviceBuffers);
        const auto inputB
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.inputB, deviceBuffers, numDeviceBuffers);
        const auto output
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.output, deviceBuffers, numDeviceBuffers);

        preparedSub.kernel().launch(handle.getStream(), inputA.ptr, inputB.ptr, output.ptr);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
};

const PointwiseSubDispatchHandler& pointwiseSubDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const PointwiseSubDispatchHandler s_dispatchHandler(s_kernelCompiler);
    return s_dispatchHandler;
}

} // namespace

void registerPointwiseSubSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL_NAME), &pointwiseSubGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL_NAME), &pointwiseSubKernelMatches);
    scope.add(std::string(SCORE_SYMBOL_NAME), &pointwiseSubScore);
    scope.add(std::string(DISPATCH_SYMBOL_NAME), &pointwiseSubDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
