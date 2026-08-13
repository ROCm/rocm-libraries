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

/**
 * @file PointwiseSubNative.cpp
 * @brief The pointwise-subtract pack's native half: matching, scoring, dispatch, and
 *        the one function that registers them.
 *
 * The second pack, and the reason it exists: a new engine must cost its own two files
 * and one row in IngestorPacks.cpp's table, and nothing else. If adding this had needed
 * an edit to any shared file, the seam would be wrong.
 *
 * Deliberately a near-copy of PointwiseAddNative.cpp differing in operation, symbol
 * names, and kernel. The duplication is real and is the floor RFC 0017 §6's expression
 * language removes: matching and dispatch are the two escape hatches a descriptor
 * cannot yet express, so every pack restates them until it can. Nothing here is shared
 * with the add pack, because after ALMIOPEN-2401 an installed descriptor set has no way
 * to share code with another one.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The symbol names this file implements. PointwiseSubDescriptors.cpp declares the same
// strings; they are the contract between the two halves.
constexpr std::string_view GRAPH_MATCHER_SYMBOL_NAME = "hipkernel.pointwise_sub.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL_NAME = "hipkernel.pointwise_sub.kernel_match";
constexpr std::string_view SCORE_SYMBOL_NAME = "hipkernel.pointwise_sub.score";
constexpr std::string_view DISPATCH_SYMBOL_NAME = "hipkernel.pointwise_sub.dispatch";

// KMD fields this pack's kernels vary along, and the tokens matching binds for dispatch
// to read back.
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view INPUT_A_TOKEN = "pointwise_sub.input_a.uid";
constexpr std::string_view INPUT_B_TOKEN = "pointwise_sub.input_b.uid";
constexpr std::string_view OUTPUT_TOKEN = "pointwise_sub.output.uid";

/// Scratch reported by the larger-block kernel, mirroring the add pack so a two-engine
/// workspace query has a defined maximum.
constexpr size_t LARGE_BLOCK_WORKSPACE_BYTES = 1024;
constexpr int64_t LARGE_BLOCK_SIZE = 256;

/// The tensor ranks this pack accepts; compile options derive layout from the tensor
/// and reject anything outside this range.
constexpr uint32_t MIN_SUPPORTED_RANK = 4;
constexpr uint32_t MAX_SUPPORTED_RANK = 5;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched pointwise-subtract graph binds, in argument order.
/// Subtraction does not commute, so which operand is which is load-bearing here in a
/// way it is not for the add pack.
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

/// True when the tensor's stride order is channel-first or channel-last, the only
/// orders the dispatch path's compile options can classify.
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

/// True when the tensor is a supported rank and layout holding exactly one element.
bool isSingleElement(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    if(dims == nullptr || dims->size() < MIN_SUPPORTED_RANK || dims->size() > MAX_SUPPORTED_RANK)
    {
        return false;
    }

    int64_t elements = 1;
    for(const auto dim : *dims)
    {
        elements *= dim;
    }
    if(elements != 1)
    {
        return false;
    }

    return hasSupportedLayout(tensor);
}

/// The graph's element type, from the first input; the matcher below requires every
/// operand to agree, so any of them would answer the same.
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

/**
 * @brief Graph-scoped applicability: is this a single-node pointwise SUB over
 *        1-element tensors?
 *
 * Evaluated once per (graph, device); a failure disqualifies every kernel in the pack.
 * The operation check is what separates this pack from the add pack on an otherwise
 * identical graph.
 */
bool pointwiseSubGraphMatches(const MatchContext& context, BoundTokens& bound)
{
    // No device, no launch: every fact this matcher would select on -- including the
    // device properties the compile is configured from -- is meaningless without one.
    if(context.deviceId == hipdnn_plugin_sdk::ingestor::NO_DEVICE)
    {
        return false;
    }

    // Exactly one node: this pack's kernel serves one complete graph.
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

    // Binary subtract: a second operand is required, a third would be a different
    // operation.
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

    // One element each: this pack's kernel indexes element 0 and nothing else.
    if(!isSingleElement(*inputA) || !isSingleElement(*inputB) || !isSingleElement(*output))
    {
        return false;
    }

    // A virtual tensor has no device buffer for findDeviceBuffer to resolve at launch.
    if(inputA->virtual_() || inputB->virtual_() || output->virtual_())
    {
        return false;
    }

    // A 1-element rank-4 tensor is also the shape of a pass-by-value scalar, whose
    // variant-pack slot holds a host pointer, not a device one.
    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(inputA)
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(inputB)
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(output))
    {
        return false;
    }

    // Uniform dtype across operands; mixed precision is a different kernel.
    if(inputA->data_type() != inputB->data_type() || inputA->data_type() != output->data_type())
    {
        return false;
    }

    // Binds operand uids in minuend-then-subtrahend order. Getting these the wrong way
    // round is silently wrong rather than a failure, which is why the E2E test compares
    // against a CPU reference rather than only asserting the engine was selected.
    bound[std::string(INPUT_A_TOKEN)] = attributes.in_0_tensor_uid();
    bound[std::string(INPUT_B_TOKEN)] = attributes.in_1_tensor_uid().value();
    bound[std::string(OUTPUT_TOKEN)] = attributes.out_0_tensor_uid();
    return true;
}

/**
 * @brief Kernel-scoped applicability: does this kernel's dtype match the graph's?
 *
 * Evaluated once per candidate kernel. Without it, an f32 graph could reach an f16
 * binary and return wrong numbers rather than failing.
 */
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
    // A stand-in for a trained model: prefers the larger block size.
    return static_cast<double>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));
}

/**
 * @brief Re-reads the operand bindings a match established.
 *
 * @throws HipdnnPluginException if the graph is not one this matcher accepts.
 */
PointwiseSubBinding pointwiseSubBinding(const BoundTokens& bound)
{
    // Every token was written by the graph matcher that admitted this graph; a missing
    // one means the catalog was built by a matcher other than ours. With two packs in
    // the provider that is a live possibility rather than a theoretical one.
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

/// The compiled kernel plus the operand uids it launches with: everything read from the
/// graph, resolved once, owning nothing that points back into it.
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
    // The runnable kernel is a view into its program's module, so the program must
    // outlive it; both are held here for the plan's lifetime.
    std::unique_ptr<compilation::ICompiledProgram> _program;
    std::unique_ptr<compilation::IRunnableKernel> _kernel;
    PointwiseSubBinding _binding;
};

/// The C++ type the kernel is compiled for, from the kernel's dtype metadata.
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

    // Unreachable via matching, which admits only dtypes this pack declares.
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

/**
 * @brief The native dispatch behind this pack's UDD: sizes and launches a pointwise
 *        subtract.
 *
 * Splits per RFC 0017 §8.5: everything derived from the graph and chosen kernel
 * resolves once at plan build; execute only resolves device pointers by uid and
 * launches. A plan may execute concurrently from several threads, so nothing here
 * mutates after preparation.
 */
class PointwiseSubDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
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
        // Reads the operand uids the matcher bound rather than re-deriving them.
        const auto binding = pointwiseSubBinding(bound);

        const auto blockSize
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));

        compilation::KernelCompileOptions options(&firstInput(context, binding),
                                                  context.deviceProperties);
        options.add("HIP_PLUGIN_POINTWISE_SUB_TYPE", elementTypeFor(kernel));
        options.add("HIP_PLUGIN_POINTWISE_SUB_BLOCK_SIZE", blockSize);

        // The only KernelSourceKind this dispatch handler knows how to load.
        auto program = _kernelCompiler.compile(kernel.source.sourceFile, options);
        auto runnableKernel = program->getKernel(kernel.source.entryPoint);

        // One element, so one workgroup; block size comes from kernel metadata.
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

/// This pack's dispatch handler.
///
/// Process-lifetime: the registry holds a non-owning pointer to it while a provider's
/// Container is created and destroyed per handle, so it must outlive every Container.
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
